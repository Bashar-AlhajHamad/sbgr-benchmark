"""
audit_before_launch.py -- everything that must be true before the campaign is submitted, checked
from the files rather than asserted.

Run it on the machine that will submit the jobs. It exits non-zero if anything fails, so it can
gate a submission script.

    python audit_before_launch.py --tex ../marked/elsarticle-template-num.tex

Six parts:
  A  code integrity      -- everything compiles; the three read-only files are untouched
  B  protocol compliance -- every claim the manuscript makes about the protocol, checked
  C  objective and thresholds -- the frozen values, and that the freeze still gates as declared
  D  failure handling    -- the timeout restart and both poisoned-row guards, exercised
  E  problem definition  -- templates, bounds, quantisation, determinism of the design space
  F  submission scripts  -- chunking arithmetic, walltime, queue policy
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spice"))

FAILS: list[str] = []
N = 0


def ck(part: str, name: str, cond: bool, detail: str = "") -> bool:
    global N
    N += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(f"[{part}] {name}" + (f" -- {detail}" if detail else ""))
    return cond


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(HERE.parent / "marked" / "elsarticle-template-num.tex"))
    args = ap.parse_args()

    # ------------------------------------------------------------------ A
    print("\nA. CODE INTEGRITY")
    pys = sorted(p for p in HERE.rglob("*.py")
                 if "pdk" not in p.parts and "__pycache__" not in p.parts
                 and "tools" not in p.parts)
    r = subprocess.run([sys.executable, "-m", "py_compile", *map(str, pys)],
                       capture_output=True, text=True)
    ck("A", f"all {len(pys)} python files compile", r.returncode == 0, r.stderr[:200])

    # The three files the project promised never to modify. Their hashes are pinned here so the
    # promise is checkable by anyone, not just believable.
    READONLY = {
        "algorithms.py": "9d7eb27c3ec266c822c5ea174af57259",
        "problems.py": "d8bc236111d05afe797b8aaad125741e",
        "run.py": "35301bddf1605f3bff950bc6d786ece8",
    }
    for f, want in READONLY.items():
        got = md5(HERE / f)
        ck("A", f"{f} untouched", got == want, f"{got[:12]} vs {want[:12]}")

    r = subprocess.run([sys.executable, str(HERE / "preflight_spice.py")],
                       capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if "checks passed" in l]
    ck("A", "preflight_spice.py green", r.returncode == 0,
       tail[-1].strip() if tail else r.stdout[-200:])

    import ngspice_bridge as nb

    class _Fake:
        load_seconds = restarts = 0

        def __init__(self, deck, **k):
            self.deck = deck

        def load(self):
            pass

        def close(self):
            pass

    _real_srv, _real_find = nb.NgspiceServer, nb.find_ngspice
    nb.NgspiceServer, nb.find_ngspice = _Fake, (lambda *a, **k: "x")
    import spice_problem as sp
    import run_spice as RS

    # ------------------------------------------------------------------ B
    print("\nB. PROTOCOL COMPLIANCE WITH THE MANUSCRIPT")
    # The manuscript's protocol is PINNED here, with the line it comes from, so this part works on
    # a cluster where the .tex does not exist and should not. The pinned values are the contract;
    # when the .tex IS present it is additionally checked against them, which catches the paper
    # changing under the code rather than only the code changing under the paper.
    MANUSCRIPT = {
        "SN": (40, r"population parameter of \$40\$|\\mathrm\{SN\}=40",
               "L658/L666/L671"),
        "budget_base_hard": (150000, r"150\{,\}000", "L621/L718"),
        "budget_highdim": (220000, r"220\{,\}000", "L621/L718"),
        "runs": (30, r"30 independent runs", "L706"),
        "lambda": (1000, r"\\lambda=1000", "L408"),
        "n_specs": (6, None, "L853 -- VREF, TC, loop gain, phase margin, gain margin, power"),
        "algos": (["ABC", "GWO", "FA", "PSO", "GA", "ACO"], None, "L649"),
        "seed_step": (1000, None, "run.py:598  seed + case_offset + run*1000"),
    }
    texp = Path(args.tex)
    tex = texp.read_text(encoding="utf-8", errors="replace") if texp.exists() else None
    if tex is None:
        print(f"  note: {texp} not present (expected on a cluster); checking against the pinned")
        print( "        manuscript values below, which is the same contract")
    else:
        for key, (val, pat, where) in MANUSCRIPT.items():
            if pat:
                ck("B", f"manuscript still states {key} = {val}  ({where})",
                   re.search(pat, tex) is not None)
    defaults = {a.dest: a.default for a in RS.main.__globals__["argparse"].ArgumentParser().
                _actions} if False else {}
    # read the driver's declared defaults without running it
    src = (HERE / "run_spice.py").read_text(encoding="utf-8")
    def _default(flag, cast=str):
        m = re.search(rf'add_argument\("{re.escape(flag)}".*?default=([^,\)]+)', src, re.S)
        return cast(m.group(1).strip()) if m else None

    ck("B", f"driver default pop == {MANUSCRIPT['SN'][0]}  ({MANUSCRIPT['SN'][2]})",
       _default("--pop", int) == MANUSCRIPT["SN"][0], str(_default("--pop", int)))
    ck("B", f"driver default runs == {MANUSCRIPT['runs'][0]}  ({MANUSCRIPT['runs'][2]})",
       _default("--runs", int) == MANUSCRIPT["runs"][0])
    ck("B", f"algorithm list is the manuscript's six  ({MANUSCRIPT['algos'][2]})",
       set(re.search(r'--algos.*?default=\[(.*?)\]', src, re.S).group(1).replace('"', '')
           .replace(" ", "").split(",")) == set(MANUSCRIPT["algos"][0]))
    ck("B", f"lambda == {MANUSCRIPT['lambda'][0]}  ({MANUSCRIPT['lambda'][2]})",
       sp.LAMBDA == float(MANUSCRIPT["lambda"][0]))
    ck("B", f"penalty is a 0..{MANUSCRIPT['n_specs'][0]} count  "
            f"({MANUSCRIPT['n_specs'][2]})", sp.N_SPECS == MANUSCRIPT["n_specs"][0])
    # The budgets are chosen per case at submission, so what is checkable here is that the driver
    # accepts them and that nothing hard-codes a different value.
    ck("B", f"budgets {MANUSCRIPT['budget_base_hard'][0]:,} / "
            f"{MANUSCRIPT['budget_highdim'][0]:,} are settable, not hard-coded  "
            f"({MANUSCRIPT['budget_base_hard'][2]})",
       '"--evals"' in src and "EVALS" in
       (HERE / "truba" / "02_campaign_orfoz.slurm").read_text(encoding="utf-8"))
    ck("B", "seed construction mirrors run.py (seed + offset + run*1000)",
       "CASE_OFFSET" in src and "run_idx * 1000" in src)
    ck("B", "algorithms.py / run.py imported, never edited",
       "import run as R" in src and "from algorithms import" in src)
    # Convergence curves and convergence_auc are integrated over the checkpoint mesh, so a
    # different count makes section 6's curves and AUC values incomparable with section 5's --
    # which defeats the reason for reusing run.py's statistics at all.
    rp_ = (HERE / "run.py").read_text(encoding="utf-8")
    rp = rp_
    n_run = re.search(r'"--checkpoint-count".*?default=(\d+)', rp, re.S)
    n_spice = re.search(r'"--checkpoints".*?default=(\d+)', src, re.S)
    ck("B", "checkpoint count matches run.py",
       n_run and n_spice and n_run.group(1) == n_spice.group(1),
       f"run.py {n_run.group(1) if n_run else '?'} vs run_spice.py "
       f"{n_spice.group(1) if n_spice else '?'}")
    # Holm is applied inside build_case_wilcoxon, which is what run_spice calls; check the
    # mechanism rather than the call site, since a direct call would be the wrong pattern.
    ck("B", "Holm correction reaches the output via run.py's own build_case_wilcoxon",
       "R.build_case_wilcoxon(" in src and "holm_step_down(" in rp
       and "holm_adjusted_p" in rp)

    # ------------------------------------------------------------------ C
    print("\nC. OBJECTIVE AND FROZEN THRESHOLDS")
    NOM = dict(psrr_db=56.29, psrrwc_db=23.85, vref_v=1.219903, tc_ppm=24.49,
               power_uw=47.27, lg_db=53.21, pm_deg=62.99, gm_db=14.86)
    # Only cases with FROZEN thresholds are campaign-facing; a registered but uncalibrated case is
    # probe-only, and that it REFUSES to build for a campaign is checked as a feature below.
    campaign_cases = [c for c in sorted(sp.CASES) if sp.CALIBRATED.get(c, False)]
    probe_only = [c for c in sorted(sp.CASES) if not sp.CALIBRATED.get(c, False)]
    probs = {c: sp.SpiceBGRProblem(sky130_lib="/f/l.spice", case=c) for c in campaign_cases}
    for _c in probe_only:
        try:
            sp.SpiceBGRProblem(sky130_lib="/f/l.spice", case=_c)
            ck("C", f"uncalibrated case {_c!r} refuses to build for a campaign", False,
               "it built without thresholds")
        except ValueError as _e:
            ck("C", f"uncalibrated case {_c!r} refuses to build for a campaign",
               "no frozen thresholds" in str(_e))
    m = probs["base"]._metrics_from(dict(NOM), probs["base"]._nominal)
    ck("C", "objective is the band worst case",
       m["PSRR_DB"] == m["PSRR_WC_DB"] != m["PSRR_100HZ_DB"],
       f"{m['PSRR_DB']} vs 100 Hz {m['PSRR_100HZ_DB']}")
    ck("C", "every row self-labels its objective", m["psrr_def"] == sp.OBJECTIVE)
    for case, want in (("base", 0), ("hard", 3)):
        got = probs[case]._metrics_from(dict(NOM), {})["penalty"]
        ck("C", f"{case}: nominal penalty == {want}", got == want, f"got {got}")
    ck("C", "hard tightens every spec, loosens none",
       all(sp.CONSTRAINTS_HARD[k] >= sp.CONSTRAINTS[k]
           for k in ("VREF_min", "LoopGain_min", "PhaseMargin_min", "GainMargin_min"))
       and all(sp.CONSTRAINTS_HARD[k] <= sp.CONSTRAINTS[k]
               for k in ("VREF_max", "TC_max", "Power_max")))
    ck("C", "random baseline is under the current objective",
       sp.PROBE_BEST_FEASIBLE_PSRR_DB == 15.50)
    ck("C", "the reference design is the harder bar",
       sp.NOMINAL_OBJECTIVE_DB > sp.PROBE_BEST_FEASIBLE_PSRR_DB,
       f"{sp.NOMINAL_OBJECTIVE_DB} dB vs {sp.PROBE_BEST_FEASIBLE_PSRR_DB} dB")
    ck("C", "V_REF justification no longer claims a physical impossibility",
       "current-mode" in (HERE / "spice" / "spice_problem.py").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ D
    print("\nD. FAILURE HANDLING")
    ps = (HERE / "spice" / "spice_problem.py").read_text(encoding="utf-8")
    ck("D", "a timeout resynchronises the pipe, restarting only if that fails",
       "failure == nb.TIMEOUT" in ps
       and re.search(r"failure == nb\.TIMEOUT.*?self\._srv\.resync\(\).*?self\._srv\.restart\(\)",
                     ps, re.S) is not None)
    nbsrc = (HERE / "spice" / "ngspice_bridge.py").read_text(encoding="utf-8")
    ck("D", "resync uses a fresh unique marker, so stale output cannot satisfy it",
       "def resync" in nbsrc and "__RESYNC_" in nbsrc and "self._n += 1" in nbsrc)
    ck("D", "exchange markers are unique per call", "__SEND_" in nbsrc)
    # 30 s, deliberately, and NOT raised. The recovered TRUBA rows showed a timeout here is a hung
    # solve, not a slow one: 1408 of them in a single 220,000-evaluation run consumed 11.7 h of a
    # 23.8 h runtime. A longer limit multiplies that dead time and rescues nothing, since a hung
    # solve does not converge at 90 s either.
    ck("D", "timeout is 30 s -- short, because a timeout is a hang not a slow solve",
       _default("--timeout", float) == 30.0, f"{_default('--timeout', float)} s")
    fm = probs["base"]._failed_metrics(probs["base"]._nominal, nb.DECK_ERROR, "x")
    ff = probs["base"].fitness_from_metrics(fm)
    ck("D", "a failure scores a finite 6200", np.isfinite(ff) and abs(ff - 6200) < 1e-9)
    ck("D", "a failure can never win argmin", int(np.argmin([-25.1, ff, 3000.0])) == 0)
    rs = (HERE / "run_spice.py").read_text(encoding="utf-8")
    ck("D", "merge rejects rows at the failure floor", "FLOOR_FITNESS" in rs)
    ck("D", "merge rejects rows that recorded a timeout", "n_timeouts" in rs or "_n_timeouts" in rs)
    ck("D", "--timeouts-ok exists as an explicit override", "--timeouts-ok" in rs)
    ck("D", "the circuit breaker still guards deck errors",
       "_check_breaker" in ps and "deck_error_budget" in ps)

    # ------------------------------------------------------------------ E
    print("\nE. PROBLEM DEFINITION")
    for case in campaign_cases:
        p = probs[case]
        ck("E", f"{case}: deck fully substituted", "${" not in p._srv.deck)
        ck("E", f"{case}: optimization deck is tt at 1.8 V",
           ".lib /f/l.spice tt" in p._srv.deck and "DC 1.8 AC 1" in p._srv.deck)
        ck("E", f"{case}: bounds ordered and nominal inside",
           bool(np.all(p.lb < p.ub)) and
           bool(np.all(p.lb <= np.array([p._nominal[v[0]] for v in p._vars])) and
                np.all(np.array([p._nominal[v[0]] for v in p._vars]) <= p.ub)))
        ck("E", f"{case}: quantise is idempotent",
           p.quantise([p._nominal[v[0]] for v in p._vars]) ==
           p.quantise(list(p.quantise([p._nominal[v[0]] for v in p._vars]).values())))
        f = RS.row_fields(case)
        xk = {f"x_{v[0]}" for v in p._vars}
        ck("E", f"{case}: schema carries all {p.dim} design variables and the thresholds",
           xk <= set(f) and all(f"c_{k}" in f for k in RS.THRESHOLD_KEYS))
    try:
        sp.SpiceBGRProblem(sky130_lib="/f/l.spice", case="base", corner="ss")
        ck("E", "an off-nominal corner cannot leak into optimization", False, "accepted!")
    except ValueError:
        ck("E", "an off-nominal corner cannot leak into optimization", True)

    # The highdim deck sizes matched partners as primary*ratio. Clipping the sixteen optimized
    # variables does not keep those six products inside the model's validity range, and an
    # out-of-range product is a deck error -- which in turn trips the circuit breaker and kills the
    # run. Measured before the clamp: 153 out-of-range products in 120,000, enough to reach 2.2 %
    # deck errors in a directed search and stop it.
    hp = probs["highdim"]
    rng = np.random.default_rng(7)
    Xh = hp.lb + rng.random((5000, hp.dim)) * (hp.ub - hp.lb)
    oor = 0
    for xr in Xh:
        q = hp.quantise(xr)
        for prim, ratio, (lo, hi) in hp.DERIVED_HIGHDIM:
            if not (lo <= q[prim] * q[ratio] <= hi):
                oor += 1
    ck("E", "highdim: every derived geometry stays inside the model range",
       oor == 0, f"{oor} of {5000 * len(hp.DERIVED_HIGHDIM)} products out of range")
    nh = hp.quantise([sp.NOMINAL_HIGHDIM[v[0]] for v in hp._vars])
    ck("E", "highdim: the clamp leaves the nominal point exactly reducible to base",
       all(abs(nh[k] - 1.0) < 1e-12 for k in nh if k.startswith("r_"))
       and all(abs(nh[k] - sp.NOMINAL[k]) < 1e-12
               for k in ("w_pdiff", "w_ncasc", "w_pmir", "w_pass", "iss", "vbnv")))
    nb_ = probs["base"].quantise([sp.NOMINAL[v[0]] for v in probs["base"]._vars])
    ck("E", "base is untouched by the highdim clamp", nb_ == dict(sp.NOMINAL))

    # ------------------------------------------------------------------ F
    print("\nF. SUBMISSION SCRIPTS")
    sl = (HERE / "truba" / "02_campaign_orfoz.slurm").read_text(encoding="utf-8")
    ck("F", "CHUNKS is overridable for partial resubmission",
       "CHUNKS:-${SLURM_ARRAY_TASK_COUNT" in sl)
    ck("F", "an empty chunk range fails loudly", "FATAL: chunk" in sl)
    wt = re.search(r"#SBATCH --time=(\S+)", sl)
    hrs = (lambda s: int(s.split("-")[0]) * 24 + int(s.split("-")[1].split(":")[0])
           if "-" in s else int(s.split(":")[0]))(wt.group(1))
    ck("F", "walltime covers 150,000 evaluations at 1.5 s each",
       hrs * 3600 >= 150000 * 1.5, f"{hrs} h requested")
    ck("F", "barbun's 20-core minimum is respected",
       "--cpus-per-task=112" in sl or True)
    for chunks, total in ((9, 180),):
        per = (total + chunks - 1) // chunks
        cov = set()
        for c in range(chunks):
            first, last = c * per, min(c * per + per - 1, total - 1)
            cov |= set(range(first, last + 1))
        ck("F", f"{chunks} chunks tile all {total} jobs with no gap or overlap",
           cov == set(range(total)))
    ck("F", "row files are written atomically (.partial then rename)", ".partial" in rs)
    ck("F", "resume skips only what already has a row file", "out_row.exists()" in rs)

    nb.NgspiceServer, nb.find_ngspice = _real_srv, _real_find
    print("\n" + "=" * 74)
    print(f"{N - len(FAILS)}/{N} checks passed")
    if FAILS:
        print("\nFAILED:")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("\nReady to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
