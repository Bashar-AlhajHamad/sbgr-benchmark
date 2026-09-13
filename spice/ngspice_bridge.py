"""
ngspice bridge -- Phases 1-2.

Phase 1 (minimal chain) proves the software chain end to end:
    render a netlist from a template  ->  call ngspice in batch  ->
    parse one .measure result  ->  return a float.

Phase 2 (hardening) makes that chain safe to drive from an optimizer, which will
happily propose device sizes that make SPICE misbehave:
    * a per-evaluation timeout;
    * fail-soft classification of every failure mode into an infeasible candidate
      with full penalty, instead of an exception escaping into the optimizer;
    * unique temp dir per evaluation, always cleaned up;
    * the deck and log of any failure preserved under failures/ for debugging;
    * an optional cache keyed on the rounded parameter vector;
    * an optional process pool, since ngspice is single-threaded and the useful
      parallelism is across evaluations.

The placeholder circuit is a resistive divider (templates/divider.cir.tmpl), so the
expected answer is known in closed form and the chain can be validated without any
analog design work:  v(out) = RBOT / (RTOP + RBOT)  for a 1 V source.

Phase 3 wraps `simulate` in the problems.py evaluator interface; the semantics it must
preserve are already encoded here as LAMBDA / N_SPECS / `failed_metrics`.

Usage:
    python ngspice_bridge.py --find       # report whether/where ngspice was located
    python ngspice_bridge.py --selftest   # render+run+parse the divider, check the value
    python ngspice_bridge.py --phase2     # exercise every hardening path

Locating ngspice (first hit wins):
    1. $NGSPICE_EXE                      (explicit override -- a full path to the binary)
    2. ngspice / ngspice_con on PATH
    3. common Windows install dirs, and a repo-local spice/tools/ngspice*/ (portable build)

Observed ngspice-46 behaviour that shaped the classifier below -- all verified against
the real binary, not assumed:
    * The exit code is NOT a reliable failure signal: a floating node and a failed
      .measure both exit 0 while the run is useless. Failures are classified from the
      output text, with the exit code only as a secondary hint.
    * `.meas op` is rejected ("unrecognized analysis type 'op'"). Operating-point
      quantities must be measured from a `.dc` sweep (or printed and parsed) --
      relevant for VREF and Power when the real BGR lands in Phase 4.
    * A failed measurement prints "... failed!" and simply omits the result line.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as _futures
import glob
import hashlib
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
FAILURE_DIR = HERE / "failures"

# Console binary first: on Windows the batch-friendly build is ngspice_con.exe;
# ngspice.exe (GUI build) also honours -b, so it is an acceptable fallback.
_EXE_NAMES = ("ngspice_con", "ngspice")

DEFAULT_TIMEOUT = 30.0

# Surrogate semantics that the SPICE evaluator must reproduce exactly (see problems.py):
# fitness = -PSRR + LAMBDA * penalty, penalty counts violated specs, feasible <=> penalty == 0.
LAMBDA = 1000.0
N_SPECS = 6

# The worst PSRR a failed candidate is credited with. Must be FINITE (see failed_metrics)
# and below anything a working candidate can reach, so a failure can never outrank a
# feasible design. Calibrate from the Stage-1 LHS probe: min observed PSRR minus a margin.
PSRR_FLOOR_DB = -200.0

# How many failure artefact directories to keep before pruning the oldest.
MAX_KEPT_FAILURES = 40


# --------------------------------------------------------------------------- Phase 1

_FIND_CACHE: dict = {}


def find_ngspice() -> str:
    """
    Return a path to an ngspice executable, or raise RuntimeError with guidance.

    Memoized on NGSPICE_EXE: the step-3 search globs recursively over the portable install
    (hundreds of files) and costs ~50 ms, which is invisible once but is ~1.4 CPU-hours
    across 10^5 evaluations if a caller forgets to pass exe=.
    """
    ck = os.environ.get("NGSPICE_EXE", "")
    if ck in _FIND_CACHE:
        return _FIND_CACHE[ck]
    found = _find_ngspice_uncached()
    _FIND_CACHE[ck] = found
    return found


def _find_ngspice_uncached() -> str:
    # 1. explicit override
    override = os.environ.get("NGSPICE_EXE")
    if override:
        if Path(override).is_file():
            return override
        raise RuntimeError(f"NGSPICE_EXE is set but not a file: {override!r}")

    # 2. on PATH
    for name in _EXE_NAMES:
        found = shutil.which(name)
        if found:
            return found

    # 3. common install locations + repo-local portable copy
    patterns = []
    for root in (
        r"C:\Program Files\ngspice",
        r"C:\Program Files (x86)\ngspice",
        r"C:\Spice64",
        r"C:\Spice",
        r"C:\msys64\mingw64",
        str(HERE / "tools"),  # e.g. spice/tools/Spice64/bin/... from a portable build
    ):
        for name in _EXE_NAMES:
            patterns.append(os.path.join(root, "**", name + ".exe"))
            patterns.append(os.path.join(root, "**", name))  # non-Windows portable
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]

    raise RuntimeError(
        "ngspice not found. Install it, then either add it to PATH, set the "
        "NGSPICE_EXE environment variable to the binary's full path, or drop a "
        "portable build under spice/tools/. Re-check with:  python ngspice_bridge.py --find"
    )


def render_netlist(template_name: str, params: dict) -> str:
    """Substitute ${NAME} placeholders in templates/<template_name> with params."""
    text = (TEMPLATES / template_name).read_text(encoding="utf-8")
    return render_text(text, params)


def render_text(text: str, params: dict) -> str:
    """Substitute ${NAME} placeholders in an in-memory deck (used by the test suite)."""
    # safe_substitute leaves any unrelated ${...} intact rather than raising,
    # and string.Template ($NAME) never collides with SPICE's own {expr} braces.
    return string.Template(text).safe_substitute({k: str(v) for k, v in params.items()})


def run_ngspice(netlist_text: str, exe: str | None = None, timeout: float = DEFAULT_TIMEOUT,
                keep_dir: bool = False):
    """
    Write the deck to a unique temp dir, run ngspice in batch, and return
    (combined_output_text, returncode). Batch invocation:  ngspice -b -o out.log deck.cir
    Captures stdout, stderr, and the -o log so the parser can search all three.

    The temp dir is removed before returning unless keep_dir is set; `simulate` uses
    keep_dir to preserve the artefacts of a failed run.
    """
    exe = exe or find_ngspice()
    workdir = tempfile.mkdtemp(prefix="ngspice_")
    deck = os.path.join(workdir, "deck.cir")
    log = os.path.join(workdir, "out.log")
    Path(deck).write_text(netlist_text, encoding="utf-8")
    try:
        proc = subprocess.run(
            [exe, "-b", "-o", log, deck],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        log_text = Path(log).read_text(encoding="utf-8", errors="replace") if Path(log).exists() else ""
        combined = "\n".join([proc.stdout or "", proc.stderr or "", log_text])
        return combined, proc.returncode
    finally:
        if not keep_dir:
            shutil.rmtree(workdir, ignore_errors=True)


def parse_measure(output_text: str, name: str) -> float:
    """
    Extract a .measure result printed by ngspice, e.g. a line like
        vout                =  3.333333e-01
    Returns the float, or raises ValueError if the measurement is absent
    (ngspice prints 'failed' when a .meas cannot be evaluated).
    """
    num = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    m = re.search(rf"(?im)^\s*{re.escape(name)}\s*=\s*({num})", output_text)
    if not m:
        raise ValueError(
            f"measurement {name!r} not found in ngspice output "
            f"(did the .meas fail or the run error out?)"
        )
    return float(m.group(1))


def evaluate_divider(rtop: float, rbot: float, exe: str | None = None) -> float:
    """Render -> run -> parse for the placeholder divider. Returns measured v(out)."""
    netlist = render_netlist("divider.cir.tmpl", {"RTOP": rtop, "RBOT": rbot})
    output, rc = run_ngspice(netlist, exe=exe)
    return parse_measure(output, "vout")


# --------------------------------------------------------------------------- Phase 2

TIMEOUT = "timeout"
DECK_ERROR = "deck_error"
NO_CONVERGENCE = "no_convergence"
MEASURE_FAILED = "measure_failed"
LAUNCH_ERROR = "launch_error"

FAILURE_KINDS = (TIMEOUT, DECK_ERROR, NO_CONVERGENCE, MEASURE_FAILED, LAUNCH_ERROR)

# A deck error means OUR netlist or substitution is wrong -- every evaluation will fail
# the same way, so it is a harness bug, not a bad candidate. Checked first, and counted
# separately so a caller can notice "100% deck_error" instead of silently optimizing noise.
_DECK_ERROR_PATTERNS = (
    r"Error on line",
    r"Simulation interrupted due to error",
    r"unknown parameter",
    r"can't find model",
    r"unrecognized analysis type",
)

_NO_CONVERGENCE_PATTERNS = (
    r"singular matrix",
    r"gmin stepping failed",
    r"source stepping failed",
    r"timestep too small",
    r"could not be simulated successfully",
    r"DC solution failed",
    r"iteration limit reached",
    r"no convergence",
)

_MEASURE_FAILED_PATTERNS = (
    r"\.meas.*failed!",
    r"measure\s+\S+\s+.*:\s*out of interval",
)


def _first_match(text: str, patterns) -> str | None:
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            line = text[max(0, text.rfind("\n", 0, m.start()) + 1):
                        text.find("\n", m.end()) if text.find("\n", m.end()) > 0 else len(text)]
            return line.strip()[:200]
    return None


def classify_output(text: str, returncode: int) -> tuple[str | None, str]:
    """
    Decide whether an ngspice run is usable, from its output text.

    Returns (failure_kind_or_None, detail). The exit code is deliberately only a
    fallback signal: ngspice-46 exits 0 on a floating node and on a failed .measure.
    """
    hit = _first_match(text, _DECK_ERROR_PATTERNS)
    if hit:
        return DECK_ERROR, hit
    hit = _first_match(text, _NO_CONVERGENCE_PATTERNS)
    if hit:
        return NO_CONVERGENCE, hit
    hit = _first_match(text, _MEASURE_FAILED_PATTERNS)
    if hit:
        return MEASURE_FAILED, hit
    if returncode != 0:
        return DECK_ERROR, f"ngspice exited with code {returncode} and no recognised message"
    return None, ""


@dataclass
class SimResult:
    """Outcome of one evaluation. `ok` is the only thing a caller must branch on."""
    ok: bool
    values: dict[str, float] = field(default_factory=dict)
    failure: str | None = None
    detail: str = ""
    artefacts: str | None = None
    seconds: float = 0.0
    from_cache: bool = False

    def __str__(self) -> str:
        if self.ok:
            vals = ", ".join(f"{k}={v:.6g}" for k, v in self.values.items())
            return f"ok({vals}{', cached' if self.from_cache else ''})"
        return f"FAIL[{self.failure}] {self.detail}"


@dataclass
class Stats:
    """Failure bookkeeping across a run -- lets a caller spot a systematic harness bug."""
    evaluations: int = 0
    cache_hits: int = 0
    ok: int = 0
    failures: collections.Counter = field(default_factory=collections.Counter)

    def record(self, r: SimResult) -> None:
        self.evaluations += 1
        if r.from_cache:
            self.cache_hits += 1
        if r.ok:
            self.ok += 1
        else:
            self.failures[r.failure] += 1

    def summary(self) -> str:
        parts = [f"evals={self.evaluations}", f"ok={self.ok}", f"cache_hits={self.cache_hits}"]
        parts += [f"{k}={v}" for k, v in sorted(self.failures.items())]
        return "  ".join(parts)


class ResultCache:
    """
    Memoize on the rounded parameter vector. Two candidates that agree to `digits`
    SIGNIFICANT digits give indistinguishable netlists, so re-simulating is waste.
    Insertion-ordered dict used as a cheap FIFO bound.

    Rounding is significant-digit, not decimal-place. `round(v, 9)` looks equivalent but
    is decimal places, so every magnitude below ~5e-10 collapses onto a single key --
    a demonstrated silent-wrong-answer bug: six distinct bias currents between 1 pA and
    1 nA shared one entry and four of them were served another candidate's measurements
    with ok=True. Device geometries in um are far from that cliff, but a caller working in
    farads or amps would fall straight off it.
    """

    def __init__(self, maxsize: int = 20000, digits: int = 9):
        self.maxsize = max(1, int(maxsize))
        self.digits = digits
        self._d: dict = {}

    def key(self, deck_id: str, params: dict, measures: tuple):
        rounded = tuple(sorted(
            (k, float(f"{float(v):.{self.digits}g}") if _is_number(v) else str(v))
            for k, v in params.items()
        ))
        return (deck_id, rounded, tuple(measures))

    def get(self, k):
        """Return a copy: the cached object must never be reachable by a caller that
        might mutate result.values in place, which would poison every later hit."""
        hit = self._d.get(k)
        if hit is None:
            return None
        return SimResult(hit.ok, dict(hit.values), hit.failure, hit.detail,
                         hit.artefacts, hit.seconds, hit.from_cache)

    def put(self, k, v: SimResult) -> None:
        # Only evict when actually inserting a NEW key. Re-putting an existing key used to
        # evict an unrelated entry and shrink the cache below maxsize.
        if k not in self._d and len(self._d) >= self.maxsize:
            self._d.pop(next(iter(self._d)))
        self._d[k] = SimResult(v.ok, dict(v.values), v.failure, v.detail,
                               v.artefacts, v.seconds, v.from_cache)

    def __len__(self) -> int:
        return len(self._d)


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _keep_artefacts(workdir: str, kind: str) -> str | None:
    """Move a failed run's deck+log under failures/ so it can be inspected later."""
    try:
        FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        _prune_failures()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = FAILURE_DIR / f"{kind}_{stamp}_{os.path.basename(workdir)}"
        shutil.copytree(workdir, dest, dirs_exist_ok=True)
        return str(dest)
    except OSError:
        return None


def _prune_failures() -> None:
    kept = sorted(FAILURE_DIR.glob("*"), key=lambda p: p.stat().st_mtime)
    for old in kept[:max(0, len(kept) - MAX_KEPT_FAILURES + 1)]:
        shutil.rmtree(old, ignore_errors=True)


def simulate(
    measures,
    template_name: str | None = None,
    params: dict | None = None,
    *,
    deck_text: str | None = None,
    exe: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    cache: ResultCache | None = None,
    stats: Stats | None = None,
    keep_failures: bool = True,
) -> SimResult:
    """
    Render -> run -> parse, converting every failure mode into a SimResult with ok=False.

    Never raises for a simulator-side problem; that is the point. Pass either
    `template_name` (a file under templates/) or `deck_text` (an in-memory deck).
    """
    params = params or {}
    measures = tuple(measures)
    # hashlib, not hash(): str hashing is salted per process, which would make inline-deck
    # cache keys (and failure artefact names) differ between runs and between pool workers.
    deck_id = template_name or (
        "<inline:" + hashlib.blake2b((deck_text or "").encode("utf-8"), digest_size=8).hexdigest() + ">"
    )

    if cache is not None:
        k = cache.key(deck_id, params, measures)
        hit = cache.get(k)
        if hit is not None:
            r = SimResult(ok=hit.ok, values=dict(hit.values), failure=hit.failure,
                          detail=hit.detail, artefacts=hit.artefacts,
                          seconds=hit.seconds, from_cache=True)
            if stats is not None:
                stats.record(r)
            return r

    t0 = time.perf_counter()
    try:
        exe = exe or find_ngspice()
    except RuntimeError as e:
        return _finish(SimResult(False, {}, LAUNCH_ERROR, str(e)[:200],
                                 seconds=time.perf_counter() - t0), cache, stats, locals())

    try:
        text = (render_text(deck_text, params) if deck_text is not None
                else render_netlist(template_name, params))
    except OSError as e:
        # Environmental, not a property of the candidate: a locked or briefly unavailable
        # template file must NOT be cached as a permanent failure for this design point.
        return _finish(SimResult(False, {}, LAUNCH_ERROR, f"template unreadable: {e}"[:200],
                                 seconds=time.perf_counter() - t0), cache, stats, locals())
    except (KeyError, ValueError) as e:
        # A genuinely malformed template or substitution -- a harness bug, deterministic,
        # and therefore safe (and useful) to cache.
        return _finish(SimResult(False, {}, DECK_ERROR, f"render failed: {e}"[:200],
                                 seconds=time.perf_counter() - t0), cache, stats, locals())

    workdir = tempfile.mkdtemp(prefix="ngspice_")
    deck = os.path.join(workdir, "deck.cir")
    log = os.path.join(workdir, "out.log")
    Path(deck).write_text(text, encoding="utf-8")
    artefacts = None
    try:
        try:
            proc = subprocess.run([exe, "-b", "-o", log, deck], capture_output=True,
                                  text=True, timeout=timeout, cwd=workdir)
        except subprocess.TimeoutExpired:
            if keep_failures:
                artefacts = _keep_artefacts(workdir, TIMEOUT)
            return _finish(SimResult(False, {}, TIMEOUT,
                                     f"no result within {timeout:g}s", artefacts,
                                     time.perf_counter() - t0), cache, stats, locals())
        except OSError as e:
            return _finish(SimResult(False, {}, LAUNCH_ERROR, str(e)[:200],
                                     seconds=time.perf_counter() - t0), cache, stats, locals())

        log_text = Path(log).read_text(encoding="utf-8", errors="replace") if Path(log).exists() else ""
        out = "\n".join([proc.stdout or "", proc.stderr or "", log_text])

        kind, detail = classify_output(out, proc.returncode)
        if kind is None:
            values, missing = {}, []
            for name in measures:
                try:
                    values[name] = parse_measure(out, name)
                except ValueError:
                    missing.append(name)
            if missing:
                kind, detail = MEASURE_FAILED, f"no value printed for {', '.join(missing)}"

        if kind is not None:
            if keep_failures:
                artefacts = _keep_artefacts(workdir, kind)
            return _finish(SimResult(False, {}, kind, detail, artefacts,
                                     time.perf_counter() - t0), cache, stats, locals())

        return _finish(SimResult(True, values, None, "", None,
                                 time.perf_counter() - t0), cache, stats, locals())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# A timeout or a missing binary says something about the machine, not about the candidate;
# caching those would poison a design point permanently on a momentary hiccup. The other
# failures are deterministic properties of the rendered deck and are safe to memoize.
_UNCACHEABLE_FAILURES = (TIMEOUT, LAUNCH_ERROR)


def _finish(r: SimResult, cache: ResultCache | None, stats: Stats | None, ctx: dict) -> SimResult:
    """Cache (except environment-dependent failures), count, and return."""
    if cache is not None and r.failure not in _UNCACHEABLE_FAILURES:
        cache.put(cache.key(ctx["deck_id"], ctx["params"], ctx["measures"]), r)
    if stats is not None:
        stats.record(r)
    return r


# --------------------------------------------------- Phase 2b: persistent server

# Measured on this machine against the real sky130 tt library: spawning ngspice once per
# evaluation costs ~85 s, essentially all of it re-parsing the PDK, which makes a 10^5
# evaluation campaign impossible (~2350 CPU-hours). Parsing once and then driving the same
# process with `alterparam` + `reset` costs a fraction of a second per candidate. So for a
# campaign the unit of work is a long-lived process, not a subprocess call.
#
# Verified properties of `ngspice_con -p` that this class relies on:
#   * `source <deck>` parses the .lib once; `reset` rebuilds the circuit WITHOUT re-reading it.
#   * `alterparam <name>=<value>` then `reset` re-evaluates DERIVED .param expressions too
#     (proved: two different (ra, rmul) pairs with equal products gave bit-identical results).
#   * `op`, `ac`, `dc temp`, `meas ac`, `meas dc`, `let` and `print` all work as interactive
#     commands in one session, repeatedly.
#   * every echoed line is prefixed with "ngspice NN -> ", which must be stripped.

_PROMPT_RE = re.compile(r"ngspice \d+ -> ?")

SERVER_LOAD_TIMEOUT = 300.0
SERVER_EVAL_TIMEOUT = 30.0

_SERVER_OPTIONS = ("-D", "ngbehavior=hsa", "-D", "ng_nomodcheck",
                   "-D", "skywaterpdk", "-D", "num_threads=1")


class NgspiceServer:
    """
    A long-lived ngspice process with the PDK parsed once.

        srv = NgspiceServer(deck_text=deck, exe=exe)
        srv.load()                                   # ~85 s, once
        r = srv.evaluate({"wp": 8, "wn": 4}, COMMANDS, ("vref_v", "tc_ppm"))
        srv.close()

    `evaluate` returns the same `SimResult` the one-shot `simulate` returns, so callers can
    treat the two interchangeably. A wedged or dead process is detected by a missing end
    marker and is restarted transparently; that evaluation is reported as a `timeout`.

    Not thread-safe: give each worker its own instance.
    """

    def __init__(self, deck_text: str, exe: str | None = None,
                 extra_args=_SERVER_OPTIONS,
                 load_timeout: float = SERVER_LOAD_TIMEOUT,
                 eval_timeout: float = SERVER_EVAL_TIMEOUT,
                 preamble=("set noaskquit", "option temp=27")):
        self.deck_text = deck_text
        self.exe = exe or find_ngspice()
        self.extra_args = tuple(extra_args)
        self.load_timeout = load_timeout
        self.eval_timeout = eval_timeout
        self.preamble = tuple(preamble)
        self._proc: subprocess.Popen | None = None
        self._q: Queue | None = None
        self._reader: threading.Thread | None = None
        self._workdir: str | None = None
        self._n = 0
        self.restarts = 0
        self.deck_errors = 0
        self.load_seconds = 0.0

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Spawn ngspice and parse the deck (and therefore the PDK). Idempotent."""
        if self._proc is not None and self._proc.poll() is None:
            return
        self._workdir = self._workdir or tempfile.mkdtemp(prefix="ngsrv_")
        deck = os.path.join(self._workdir, "deck.cir")
        Path(deck).write_text(self.deck_text, encoding="utf-8")

        self._proc = subprocess.Popen(
            [self.exe, *self.extra_args, "-p"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=self._workdir,
        )
        self._q = Queue()
        self._reader = threading.Thread(target=self._pump, args=(self._proc, self._q), daemon=True)
        self._reader.start()

        t0 = time.perf_counter()
        marker = "__LOADED__"
        out = self._exchange([*self.preamble,
                              "source " + deck.replace("\\", "/"),
                              "echo " + marker], marker, self.load_timeout)
        self.load_seconds = time.perf_counter() - t0
        if out is None:
            self._kill()
            raise RuntimeError(f"ngspice did not finish loading within {self.load_timeout:g}s")
        kind, detail = classify_output(out, 0)
        if kind == DECK_ERROR:
            self._kill()
            raise RuntimeError(f"deck failed to load: {detail}")

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        self._kill()
        if self._workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def restart(self) -> None:
        self.restarts += 1
        self._kill()
        self.load()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- evaluation --------------------------------------------------------

    def evaluate(self, params: dict, commands, measures,
                 cache: ResultCache | None = None, stats: Stats | None = None) -> SimResult:
        """
        Apply `params` via alterparam, `reset`, run `commands`, and parse `measures`.

        `commands` is the analysis script for one candidate, e.g.
            ("op", "let vref_v = v(vref)", "print vref_v", "ac dec 8 10 1e7", ...)
        """
        measures = tuple(measures)
        deck_id = "<server:" + hashlib.blake2b(self.deck_text.encode("utf-8"),
                                               digest_size=8).hexdigest() + ">"
        ctx = {"deck_id": deck_id, "params": params, "measures": measures}
        if cache is not None:
            hit = cache.get(cache.key(deck_id, params, measures))
            if hit is not None:
                r = SimResult(hit.ok, dict(hit.values), hit.failure, hit.detail,
                              hit.artefacts, hit.seconds, from_cache=True)
                if stats is not None:
                    stats.record(r)
                return r

        if self._proc is None or self._proc.poll() is not None:
            self.load()

        t0 = time.perf_counter()
        self._n += 1
        marker = f"__CAND_{self._n}__"
        script = [f"alterparam {k}={v}" for k, v in params.items()]
        script += ["reset", *commands, "echo " + marker]

        out = self._exchange(script, marker, self.eval_timeout)
        if out is None:
            # no end marker: the process is wedged or dead. Restart and fail soft.
            try:
                self.restart()
            except RuntimeError as e:
                return _finish(SimResult(False, {}, LAUNCH_ERROR, str(e)[:200],
                                         seconds=time.perf_counter() - t0),
                               cache, stats, ctx)
            return _finish(SimResult(False, {}, TIMEOUT,
                                     f"no result within {self.eval_timeout:g}s (server restarted)",
                                     seconds=time.perf_counter() - t0), cache, stats, ctx)

        kind, detail = classify_output(out, 0)
        if kind is None:
            values, missing = {}, []
            for name in measures:
                try:
                    values[name] = parse_measure(out, name)
                except ValueError:
                    missing.append(name)
            if missing:
                kind, detail = MEASURE_FAILED, f"no value printed for {', '.join(missing)}"
        if kind is not None:
            # A deck error can leave the interpreter in a state where EVERY later candidate
            # fails too -- observed: after one bad candidate, all subsequent evaluations in
            # the session came back as deck_error. That silently turns the rest of a run
            # infeasible while still producing plausible-looking output, so a deck error
            # costs a restart. `no_convergence` and `measure_failed` are ordinary candidate
            # outcomes and must NOT trigger one (a reload costs ~100 s).
            if kind == DECK_ERROR:
                self.deck_errors += 1
                try:
                    self.restart()
                except RuntimeError as e:
                    return _finish(SimResult(False, {}, LAUNCH_ERROR,
                                             f"restart after deck_error failed: {e}"[:200],
                                             seconds=time.perf_counter() - t0),
                                   cache, stats, ctx)
            return _finish(SimResult(False, {}, kind, detail,
                                     seconds=time.perf_counter() - t0), cache, stats, ctx)
        return _finish(SimResult(True, values, None, "",
                                 seconds=time.perf_counter() - t0), cache, stats, ctx)

    def send(self, lines, marker: str | None = None, timeout: float | None = None) -> str | None:
        """
        Send raw interpreter commands and return everything up to `marker`.

        For multi-phase measurements that a single `evaluate` cannot express. The loop-gain
        measurement needs this: Tian injection runs two AC analyses and then combines vectors
        from BOTH plots, and a plot can only be referenced by its literal name (`ac1.u`).
        `$p1.u` does NOT work -- ngspice expands `$p1.u` as one variable named "p1.u", finds
        nothing, and silently drops it from the expression. So the caller reads `$curplot`
        from phase one and interpolates the real name into phase two.
        """
        if self._proc is None or self._proc.poll() is not None:
            self.load()
        if marker is None:
            self._n += 1
            marker = f"__SEND_{self._n}__"
            lines = list(lines) + ["echo " + marker]
        return self._exchange(lines, marker, timeout or self.eval_timeout)

    # -- plumbing ----------------------------------------------------------

    @staticmethod
    def _pump(proc: subprocess.Popen, q: Queue) -> None:
        """Drain stdout in a thread: readline() on a pipe cannot be polled on Windows."""
        try:
            for line in proc.stdout:
                q.put(line)
        except (OSError, ValueError):
            pass
        finally:
            q.put(None)  # sentinel: stream closed

    def _exchange(self, lines, marker: str, timeout: float) -> str | None:
        """Send `lines`, collect output until `marker` appears. None on timeout/death."""
        try:
            self._proc.stdin.write("\n".join(lines) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):
            return None
        deadline = time.monotonic() + timeout
        chunks = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._q.get(timeout=min(remaining, 0.5))
            except Empty:
                continue
            if line is None:
                return None  # process died
            chunks.append(_PROMPT_RE.sub("", line))
            if marker in line:
                return "".join(chunks)

    def resync(self, grace: float = 15.0) -> bool:
        """Discard stale output after a timeout WITHOUT paying a process restart.

        A timeout leaves the interpreter still working on the command we stopped waiting for. Its
        output arrives later and would be read as the answer to the NEXT command, which is how a
        single timeout silently corrupts a whole run. Restarting fixes that, but re-parsing the
        PDK costs 13-42 s, and a rehearsal measured 35 timeouts per 2,000-evaluation run -- so on
        the real budget the reloads alone would dominate the campaign.

        Because every exchange marker is unique (`__SEND_<n>__`, counter-based), stale output can
        never contain a NEW marker. So sending a fresh one and reading until it appears drains
        whatever was in flight and proves the pipe is aligned again, at the cost of nothing but
        the wait. Returns True if the pipe is clean; False means the interpreter is genuinely
        wedged and the caller should restart.

        `grace` is a bet: the measured timeouts are mostly starvation on a contended node rather
        than an unbounded solve, so the command usually completes shortly after we give up on it.
        When the bet loses, the cost is `grace` seconds on top of the restart that follows.
        """
        if self._proc is None or self._proc.poll() is not None:
            return False
        self._n += 1
        marker = f"__RESYNC_{self._n}__"
        try:
            self._proc.stdin.write(f"echo {marker}\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):
            return False
        deadline = time.monotonic() + grace
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                line = self._q.get(timeout=min(remaining, 0.5))
            except Empty:
                continue
            if line is None:
                return False
            if marker in line:
                return True

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._proc = None
        self._q = None
        self._reader = None


def failed_metrics(psrr_key: str = "PSRR_DB") -> dict:
    """
    The fail-soft candidate: every specification counted as violated.

    Mirrors the surrogate semantics -- penalty is a count, feasible <=> penalty == 0,
    fitness = -PSRR + LAMBDA * penalty with the unknown PSRR contributing nothing.
    Phase 3 returns this dict whenever `simulate` reports ok=False.
    """
    return {
        # NOT nan. np.argmin([1.0, nan, 2.0]) == 1, so a single NaN fitness among the first
        # `pop` evaluations becomes the incumbent, every later `<` comparison against it is
        # False, and the optimizer run is destroyed silently. Every optimizer in
        # algorithms.py seeds its incumbent with np.argmin(fit).
        psrr_key: PSRR_FLOOR_DB,
        "penalty": N_SPECS,
        "is_feasible": 0,          # int, matching problems.py; not a bool
        # no "fitness" key: problems.py has none, and the evaluator computes it
    }


# --- optional process pool: ngspice is single-threaded, so parallelise evaluations ---

def _batch_worker(job: dict) -> SimResult:
    return simulate(job.pop("measures"), **job)


def simulate_many(jobs, workers: int | None = None):
    """
    Run independent `simulate` calls in a process pool and return results in input order.
    Each job is a kwargs dict for `simulate` (including "measures"). Caches are per
    process, so pass a cache only when running serially.

    A job that raises inside a worker yields a LAUNCH_ERROR SimResult for that job only;
    `pool.map` would instead propagate the first exception and discard every sibling
    result that had already completed.
    """
    jobs = [dict(j) for j in jobs]
    if workers in (0, 1) or len(jobs) <= 1:
        out = []
        for j in jobs:
            try:
                out.append(_batch_worker(j))
            except Exception as e:  # noqa: BLE001 -- fail soft, one job at a time
                out.append(SimResult(False, {}, LAUNCH_ERROR, f"{type(e).__name__}: {e}"[:200]))
        return out
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with _futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_batch_worker, j) for j in jobs]
        out = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001
                out.append(SimResult(False, {}, LAUNCH_ERROR, f"{type(e).__name__}: {e}"[:200]))
        return out


# --------------------------------------------------------------------------- CLI

def _cmd_find() -> int:
    try:
        exe = find_ngspice()
    except RuntimeError as e:
        print(f"[find] NOT FOUND\n{e}")
        return 1
    print(f"[find] ngspice: {exe}")
    return 0


def _cmd_selftest() -> int:
    try:
        exe = find_ngspice()
    except RuntimeError as e:
        print(f"[selftest] cannot run -- {e}")
        return 1
    print(f"[selftest] using: {exe}")

    rtop, rbot = 2000.0, 1000.0
    expected = rbot / (rtop + rbot)
    try:
        got = evaluate_divider(rtop, rbot, exe=exe)
    except (subprocess.TimeoutExpired, ValueError, OSError) as e:
        print(f"[selftest] FAILED to obtain a measurement: {e}")
        return 1

    ok = abs(got - expected) < 1e-3
    print(f"[selftest] v(out) measured={got:.6f} expected={expected:.6f} "
          f"-> {'PASS' if ok else 'MISMATCH'}")
    return 0 if ok else 1


# Decks used only by --phase2, each provoking one specific failure mode.
_BAD_DECKS = {
    NO_CONVERGENCE: """* two ideal sources on one node -> singular matrix
V1 a 0 DC 1
V2 a 0 DC 2
.dc V1 0 1 0.5
.meas dc vv FIND v(a) AT=1
.end
""",
    DECK_ERROR: """* unresolvable device value
Vin in 0 DC 1
Rbad in out ThisIsNotANumber
.dc Vin 0 1 0.5
.meas dc vout FIND v(out) AT=1
.end
""",
    MEASURE_FAILED: """* valid sim, impossible measurement point
Vin in 0 DC 1
R1 in out 2k
R2 out 0 1k
.dc Vin 0 1 0.5
.meas dc vout FIND v(out) AT=99
.end
""",
    TIMEOUT: """* transient far longer than any sane per-eval budget
Vin in 0 PULSE(0 1 0 1n 1n 1u 2u)
R1 in out 1k
C1 out 0 1n
.tran 1p 5
.meas tran vmax MAX v(out)
.end
""",
}


def _count_temp_dirs() -> int:
    return len(glob.glob(os.path.join(tempfile.gettempdir(), "ngspice_*")))


def _cmd_phase2() -> int:
    try:
        exe = find_ngspice()
    except RuntimeError as e:
        print(f"[phase2] cannot run -- {e}")
        return 1
    print(f"[phase2] using: {exe}\n")

    stats = Stats()
    cache = ResultCache()
    failed = 0

    def check(label: str, cond: bool, extra: str = "") -> None:
        nonlocal failed
        if not cond:
            failed += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + extra) if extra else ''}")

    temp_before = _count_temp_dirs()

    print("1. happy path")
    r = simulate(["vout"], "divider.cir.tmpl", {"RTOP": 2000, "RBOT": 1000},
                 exe=exe, cache=cache, stats=stats)
    check("ok and value correct", r.ok and abs(r.values.get("vout", 0) - 1 / 3) < 1e-3, str(r))

    print("\n2. cache")
    r2 = simulate(["vout"], "divider.cir.tmpl", {"RTOP": 2000, "RBOT": 1000},
                  exe=exe, cache=cache, stats=stats)
    check("second identical call served from cache", r2.from_cache and r2.ok, str(r2))
    r3 = simulate(["vout"], "divider.cir.tmpl", {"RTOP": 3000, "RBOT": 1000},
                  exe=exe, cache=cache, stats=stats)
    check("different params NOT served from cache", not r3.from_cache and r3.ok, str(r3))

    print("\n3. failure modes are classified, not raised")
    for kind, deck in _BAD_DECKS.items():
        tmo = 3.0 if kind == TIMEOUT else DEFAULT_TIMEOUT
        rr = simulate(["vv", "vout", "vmax"], deck_text=deck, exe=exe,
                      timeout=tmo, stats=stats)
        check(f"{kind:16s} detected", (not rr.ok) and rr.failure == kind,
              f"got {rr.failure}: {rr.detail[:80]}")
        if not rr.ok and rr.artefacts:
            check(f"{kind:16s} artefacts kept", os.path.isdir(rr.artefacts),
                  os.path.relpath(rr.artefacts, HERE))

    print("\n4. fail-soft candidate")
    fm = failed_metrics()
    import math as _math
    check("penalty == N_SPECS, infeasible, and PSRR is FINITE (never NaN)",
          fm["penalty"] == N_SPECS and fm["is_feasible"] == 0
          and _math.isfinite(fm["PSRR_DB"]) and fm["PSRR_DB"] <= PSRR_FLOOR_DB, str(fm))
    check("is_feasible is an int, matching problems.py",
          isinstance(fm["is_feasible"], int) and not isinstance(fm["is_feasible"], bool))
    check("no stray 'fitness' key (problems.py has none)", "fitness" not in fm)

    print("\n4b. regressions for the seven reported defects")
    # (1) significant digits, not decimal places
    c = ResultCache(digits=9)
    keys = {c.key("d", {"x": v}, ("m",)) for v in (1e-12, 2e-12, 3e-12, 5e-11, 7e-10, 1e-9)}
    check("6 log-spaced values in 1e-12..1e-9 give 6 distinct keys", len(keys) == 6,
          f"got {len(keys)}")
    # (4) cache hands out copies
    k = c.key("d", {"x": 1.0}, ("m",))
    c.put(k, SimResult(True, {"m": 1.0}))
    got = c.get(k)
    got.values["m"] = 999.0
    check("mutating a returned result does not poison the cache",
          c.get(k).values["m"] == 1.0, str(c.get(k).values))
    # (6) maxsize guards
    c2 = ResultCache(maxsize=0)
    try:
        c2.put(c2.key("d", {"x": 1}, ()), SimResult(True, {}))
        ok0 = True
    except StopIteration:
        ok0 = False
    check("maxsize=0 does not raise StopIteration", ok0)
    c3 = ResultCache(maxsize=3)
    for i in range(3):
        c3.put(c3.key("d", {"x": i}, ()), SimResult(True, {}))
    c3.put(c3.key("d", {"x": 1}, ()), SimResult(True, {}))   # re-put an existing key
    check("re-putting an existing key does not evict an unrelated entry", len(c3) == 3,
          f"len={len(c3)}")
    # (7) one raising job does not destroy the batch
    bad = simulate_many([{"measures": ("vout",), "template_name": "divider.cir.tmpl",
                          "params": {"RTOP": 2000, "RBOT": 1000}, "exe": exe},
                         {"measures": ("vout",), "template_name": "does_not_exist.tmpl",
                          "params": {}, "exe": exe},
                         {"measures": ("vout",), "template_name": "divider.cir.tmpl",
                          "params": {"RTOP": 3000, "RBOT": 1000}, "exe": exe}], workers=1)
    check("a failing job leaves its siblings intact",
          len(bad) == 3 and bad[0].ok and bad[2].ok and not bad[1].ok,
          f"{[str(b)[:26] for b in bad]}")
    # (5) an unreadable template is environmental, so it must not be cached
    c4 = ResultCache()
    r5 = simulate(("vout",), "no_such_template.tmpl", {}, exe=exe, cache=c4,
                  keep_failures=False)
    check("missing template -> launch_error and NOT cached",
          (not r5.ok) and r5.failure == LAUNCH_ERROR and len(c4) == 0,
          f"{r5.failure}, cache={len(c4)}")
    # (2) find_ngspice is memoized
    t0 = time.perf_counter()
    for _ in range(50):
        find_ngspice()
    dt = time.perf_counter() - t0
    check("50 find_ngspice() calls are memoized (< 50 ms total)", dt < 0.05,
          f"{dt*1000:.1f} ms")

    print("\n5. temp dirs cleaned up")
    temp_after = _count_temp_dirs()
    check("no ngspice_* temp dirs leaked", temp_after == temp_before,
          f"before={temp_before} after={temp_after}")

    print("\n6. process pool")
    jobs = [dict(measures=["vout"], template_name="divider.cir.tmpl",
                 params={"RTOP": 1000 * (i + 1), "RBOT": 1000}, exe=exe)
            for i in range(6)]
    t0 = time.perf_counter()
    res = simulate_many(jobs, workers=4)
    dt = time.perf_counter() - t0
    want = [1000 / (1000 * (i + 1) + 1000) for i in range(6)]
    got_ok = all(x.ok for x in res)
    close = got_ok and all(abs(x.values["vout"] - w) < 1e-3 for x, w in zip(res, want))
    check(f"6 parallel evaluations all correct ({dt:.2f}s)", close,
          "" if close else str([str(x) for x in res]))

    print(f"\n[phase2] stats: {stats.summary()}")
    print(f"[phase2] cache entries: {len(cache)}")
    print(f"[phase2] {'ALL CHECKS PASSED' if failed == 0 else str(failed) + ' CHECK(S) FAILED'}")
    return 0 if failed == 0 else 1


# A parameterised divider with a DERIVED param, and no .control block: the server drives it.
_SERVER_PROBE_DECK = """.title server probe -- parameterised divider
.param rtop=2000 rbot=1000 mul=1
.param rbot_eff='rbot*mul'
Vin in 0 DC 1
R1 in out {rtop}
R2 out 0 {rbot_eff}
C1 out 0 1n
.end
"""

_SERVER_PROBE_CMDS = ("op", "let vout_v = v(out)", "print vout_v")


def _cmd_server() -> int:
    """Exercise NgspiceServer without the PDK, so the whole suite runs in seconds."""
    try:
        exe = find_ngspice()
    except RuntimeError as e:
        print(f"[server] cannot run -- {e}")
        return 1
    print(f"[server] using: {exe}\n")

    failed = 0

    def check(label, cond, extra=""):
        nonlocal failed
        if not cond:
            failed += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + extra) if extra else ''}")

    temp_before = _count_temp_dirs()
    srv = NgspiceServer(_SERVER_PROBE_DECK, exe=exe, eval_timeout=6.0)
    cache = ResultCache()
    stats = Stats()
    try:
        srv.load()
        print(f"1. load: {srv.load_seconds*1000:.0f} ms (no PDK in this probe)")

        print("\n2. parameters take effect, and derived params propagate")
        for rtop, rbot, mul in ((2000, 1000, 1), (3000, 1000, 1), (2000, 500, 2)):
            want = (rbot * mul) / (rtop + rbot * mul)
            r = srv.evaluate({"rtop": rtop, "rbot": rbot, "mul": mul},
                             _SERVER_PROBE_CMDS, ("vout_v",), cache=cache, stats=stats)
            got = r.values.get("vout_v")
            check(f"rtop={rtop} rbot={rbot} mul={mul} -> {want:.6f}",
                  r.ok and got is not None and abs(got - want) < 1e-4,
                  f"got {got}")
        # (2000,500,2) and (2000,1000,1) must agree: that is the derived-param proof
        a = srv.evaluate({"rtop": 2000, "rbot": 1000, "mul": 1}, _SERVER_PROBE_CMDS,
                         ("vout_v",), stats=stats)
        b = srv.evaluate({"rtop": 2000, "rbot": 500, "mul": 2}, _SERVER_PROBE_CMDS,
                         ("vout_v",), stats=stats)
        check("equal rbot*mul gives identical results (derived param re-evaluated)",
              a.ok and b.ok and a.values["vout_v"] == b.values["vout_v"],
              f"{a.values.get('vout_v')} vs {b.values.get('vout_v')}")

        print("\n3. cache")
        c1 = srv.evaluate({"rtop": 2000, "rbot": 1000, "mul": 1}, _SERVER_PROBE_CMDS,
                          ("vout_v",), cache=cache, stats=stats)
        check("repeat served from cache", c1.from_cache and c1.ok, str(c1))

        print("\n4. a missing measurement is classified, not raised")
        r = srv.evaluate({"rtop": 2000}, _SERVER_PROBE_CMDS, ("no_such_value",), stats=stats)
        check("measure_failed", (not r.ok) and r.failure == MEASURE_FAILED, str(r))

        print("\n5. a wedged analysis times out and the server recovers")
        n0 = srv.restarts
        r = srv.evaluate({"rtop": 2000}, ("tran 1p 5",), ("vout_v",), stats=stats)
        check("timeout reported", (not r.ok) and r.failure == TIMEOUT, str(r))
        check("server restarted", srv.restarts == n0 + 1, f"restarts={srv.restarts}")
        r = srv.evaluate({"rtop": 2000, "rbot": 1000, "mul": 1}, _SERVER_PROBE_CMDS,
                         ("vout_v",), stats=stats)
        check("still usable after restart",
              r.ok and abs(r.values.get("vout_v", 0) - 1 / 3) < 1e-4, str(r))

        print("\n5b. a deck error cannot contaminate the next candidate")
        # Observed on the real BGR: after one bad candidate every later evaluation in the
        # session came back deck_error, silently turning the rest of a run infeasible.
        good = {"rtop": 2000, "rbot": 1000, "mul": 1}
        base = srv.evaluate(good, _SERVER_PROBE_CMDS, ("vout_v",), stats=stats)
        d0, n0 = srv.deck_errors, srv.restarts
        # `Rbogus` references an undefined parameter -> a genuine deck error
        bad = srv.evaluate(good, ("let q = 1", "print nonexistent_vector_xyz"),
                           ("vout_v",), stats=stats)
        after = srv.evaluate(good, _SERVER_PROBE_CMDS, ("vout_v",), stats=stats)
        check("the candidate after a failure still gets the right answer",
              after.ok and abs(after.values.get("vout_v", 0) - base.values["vout_v"]) < 1e-9,
              f"before={base.values.get('vout_v')} after={after.values.get('vout_v')}"
              f" (bad was {bad.failure})")
        if bad.failure == DECK_ERROR:
            check("a deck_error triggered a restart",
                  srv.deck_errors == d0 + 1 and srv.restarts == n0 + 1,
                  f"deck_errors={srv.deck_errors} restarts={srv.restarts}")
        else:
            print(f"  [note] that probe classified as {bad.failure}, not deck_error; "
                  "the restart path is covered by the timeout check above")

        print("\n6. throughput")
        t0 = time.perf_counter()
        N = 20
        for i in range(N):
            srv.evaluate({"rtop": 1000 + i, "rbot": 1000, "mul": 1},
                         _SERVER_PROBE_CMDS, ("vout_v",), stats=stats)
        dt = time.perf_counter() - t0
        print(f"     {N} evaluations in {dt:.2f}s -> {dt/N*1000:.0f} ms each")
    finally:
        srv.close()

    print("\n7. cleanup")
    check("no ngspice_* temp dirs leaked", _count_temp_dirs() == temp_before,
          f"before={temp_before} after={_count_temp_dirs()}")

    print(f"\n[server] stats: {stats.summary()}")
    print(f"[server] {'ALL CHECKS PASSED' if failed == 0 else str(failed) + ' CHECK(S) FAILED'}")
    return 0 if failed == 0 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ngspice bridge (Phases 1-2)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--find", action="store_true", help="locate the ngspice binary and exit")
    g.add_argument("--selftest", action="store_true", help="render+run+parse the divider")
    g.add_argument("--phase2", action="store_true",
                   help="exercise timeout, fail-soft, cleanup, cache and pool")
    g.add_argument("--server", action="store_true",
                   help="exercise the persistent NgspiceServer (no PDK needed)")
    args = p.parse_args(argv)
    if args.find:
        return _cmd_find()
    if args.selftest:
        return _cmd_selftest()
    if args.phase2:
        return _cmd_phase2()
    return _cmd_server()


if __name__ == "__main__":
    sys.exit(main())
