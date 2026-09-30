"""
Campaign driver for Circuit B -- the SKY130 two-stage Miller operational amplifier.

Sibling of run_spice.py, not a replacement for it. run_spice.py produced the published Circuit A
results and is referenced by the Zenodo record, so it is not edited; this file drives the second
circuit with the same protocol, the same statistics and the same seeding discipline.

WHAT IS SHARED, AND WHY THAT MATTERS. The statistics come from run.py's builders -- the same
`build_case_stats`, `friedman_from_pivot` and `build_case_wilcoxon` that produced every number in the
paper. Nothing is reimplemented, so the two circuits cannot differ in how they are analysed, only in
what is measured. `algorithms.py` and `run.py` are not modified.

THE ONE WRINKLE, AND HOW IT IS HANDLED HONESTLY. run.py's `build_case_stats` reads the objective from
a column literally named `PSRR_DB`, because for Circuit A the objective WAS the supply rejection. For
Circuit B the objective is DC open-loop gain, and PSRR is a constraint -- Circuit B measures a real
PSRR of its own, around 90 dB, so writing gain into a column called `PSRR_DB` would be actively
misleading to anyone reading the CSV later.

So the CSV on disk uses honest names (`GAIN_DB`, `PSRR_DB` meaning PSRR), and ONLY the in-memory
DataFrame handed to the shared statistics is aliased. The statistics are therefore provably identical
in method, and the recorded data says what it means. The alias is applied in exactly one place,
`_objective_alias`, so there is one line to check rather than a convention to remember.

SEEDING. Circuit B gets its own offset, so the three topologies are independent replicates. This
corrects the single shared CASE_OFFSET disclosed as a defect in Section 6.5 of the manuscript and
required by section 3 of PREREGISTRATION_2026-08-26_circuits_2_and_3.md.

BUDGET. 2,500 evaluations per run, population 40, 30 runs per algorithm -- the CIRCUIT-LEVEL
protocol, matching Circuit A's transistor-level campaign exactly. It is NOT the surrogate's 150,000;
the manuscript describes running the surrogate at the circuit-level protocol as a control and calls
that "cutting the budget sixty-fold". Using 150,000 here would make the two circuits incomparable,
which is the whole point of running the second one, and would cost 4,487 core-hours instead of ~60.

USAGE

    # whole campaign on one machine, N workers each owning one ngspice process
    python run_spice_opamp.py --lib <sky130.lib.spice> --workers 6

    # one (algo, run) pair, for a SLURM array; then merge
    python run_spice_opamp.py --lib ... --job-index $SLURM_ARRAY_TASK_ID --rows-dir rows
    python run_spice_opamp.py --lib ... --merge --rows-dir rows

Each worker owns ONE SpiceOpampProblem and therefore one ngspice process with the PDK parsed once.
Budget ~250 MB resident per worker: do not run more workers than memory allows, which is how a
512-point probe once brought a 12 GB machine to its knees.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spice"))

import run as R                                          # noqa: E402  statistics, reused verbatim
from algorithms import ALGORITHMS, ObjectiveWrapper      # noqa: E402
import spice_problem_opamp as sp                         # noqa: E402

CASE = "opamp"
CASE_NAME = "SKY130-OPAMP"
# Circuit B's own seed block. Circuit A used 300000; a distinct offset makes the topologies
# independent replicates instead of the same random stream re-used, which is the defect Section 6.5
# discloses for the three bandgap cases.
CASE_OFFSET = 400000

OBJECTIVE_COL = "GAIN_DB"
BEST_FEASIBLE_OBJECTIVE_COL = "best_feasible_GAIN_DB"

ROW_FIELDS = [
    "case", "case_name", "dim", "run", "seed", "algo", "pop",
    "best_fitness", OBJECTIVE_COL, "penalty", "is_feasible",
    # the six specifications, in the order they are declared in spice_problem_opamp.CONSTRAINTS
    "UGB_MHZ", "PHASE_MARGIN_DEG", "PSRR_DB", "POWER_UW", "AREA_UM2", "VOUT_V",
    "viol_ugb", "viol_phase_margin", "viol_psrr", "viol_power", "viol_area", "viol_vout",
    "first_feasible_eval", "best_feasible_found", "best_feasible_fitness",
    BEST_FEASIBLE_OBJECTIVE_COL,
    "best_feasible_UGB_MHZ", "best_feasible_PHASE_MARGIN_DEG", "best_feasible_PSRR_DB",
    "best_feasible_POWER_UW", "best_feasible_AREA_UM2", "best_feasible_VOUT_V",
    "runtime_sec", "eval_budget", "actual_evals",
    "last_improvement_eval", "convergence_auc",
    # The design vector, in BOTH parameterisations. The optimizer searches m2 and k; the deck is
    # given w6 and w7. Recording only one of them would make a solution impossible to re-simulate
    # or impossible to interpret, depending on which was dropped.
    "x_w1", "x_w3", "x_w5", "x_m2", "x_k", "x_l_all", "x_cc",
    "deck_w6", "deck_w7",
    # Per-WORKER CUMULATIVE counters, not per-row: workers are shared across jobs, so these describe
    # the worker's whole history at the moment this row was written. Filtering on them per run is
    # meaningless -- that mistake silently yielded zero rows once on Circuit A. The suffix is the fix.
    "sim_evals_worker_cum", "sim_ok_worker_cum", "sim_failures_worker_cum",
    "sim_failure_kinds_worker_cum", "server_restarts_worker_cum",
    # 1 means a timeout is recovered by resynchronising the simulator. Inherited from Circuit A's
    # evaluator, so it is always 1 here; recorded anyway, because the merge guard reads it rather
    # than asking the operator to remember which evaluator produced a row.
    "evaluator_recovers_timeouts",
    # provenance of the frozen problem, so a row carries the thresholds it was judged against
    "objective_def",
]

THRESHOLD_KEYS = ("UGB_min", "PhaseMargin_min", "PSRR_min",
                  "Power_max", "Area_max", "Vout_min", "Vout_max")

SPEC_COLS = ("UGB_MHZ", "PHASE_MARGIN_DEG", "PSRR_DB", "POWER_UW", "AREA_UM2", "VOUT_V")
VIOL_COLS = ("viol_ugb", "viol_phase_margin", "viol_psrr",
             "viol_power", "viol_area", "viol_vout")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def _objective_alias(df: pd.DataFrame) -> pd.DataFrame:
    """The ONLY place the objective is renamed, and it never touches disk.

    run.py's `build_case_stats` and `plot_boxplot_feasible_psrr` read the objective from a column
    named `PSRR_DB`, which for Circuit A it was. Circuit B's objective is gain and its `PSRR_DB` is a
    genuine, different measurement, so the rename is done on a COPY handed to the statistics while
    the CSV keeps honest names. Doing it here, once, is what makes that safe: there is a single line
    to audit instead of a convention spread across the file.
    """
    d = df.copy()
    d["_real_PSRR_DB"] = d.get("PSRR_DB")
    d["PSRR_DB"] = d[OBJECTIVE_COL]
    d["best_feasible_PSRR"] = d[BEST_FEASIBLE_OBJECTIVE_COL]
    # run.py's VIOL_COLS is likewise Circuit A's list, and build_case_stats sums each column by
    # name. Circuit B has six specifications too, so the mapping is one-to-one and no information
    # is lost or invented -- but the violation TALLIES in the shared statistics come out under
    # Circuit A's labels. The CSV on disk carries the honest names, and this mapping is recorded
    # here so a reader of `violations.csv` can translate it.
    for src, dst in VIOL_ALIAS.items():
        d[dst] = d[src]
    return d


# Circuit B's specification -> the Circuit A column name run.py sums it under. One-to-one, because
# both circuits have exactly six specifications; declared explicitly so the translation is auditable
# rather than positional.
VIOL_ALIAS = {
    "viol_ugb":          "viol_vref",          # UGB >= .
    "viol_phase_margin": "viol_phase_margin",  # phase margin >= .   (same name, same meaning)
    "viol_psrr":         "viol_loop_gain",     # PSRR >= .
    "viol_power":        "viol_power",         # power <= .          (same name, same meaning)
    "viol_area":         "viol_gain_margin",   # area <= .
    "viol_vout":         "viol_tc",            # output window
}


# ------------------------------------------------------------------ incremental persistence
# One lock, because the worker pool is threads: two runs finishing in the same instant would
# otherwise interleave their rows and corrupt the file.
_WRITE_LOCK = threading.Lock()


def _append_row(target: Path, fields: list, row: dict) -> None:
    """Append one finished run and force it to disk immediately.

    This exists because the previous design held all 180 rows in memory and wrote once at the end.
    A campaign that hit its wall six minutes short therefore lost every run rather than the two in
    flight. One buffered write plus an fsync per ~2 h of compute is free; losing four hours of node
    time is not.
    """
    with _WRITE_LOCK:
        fresh = not target.exists() or target.stat().st_size == 0
        with target.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if fresh:
                w.writeheader()
            w.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())


def _write_curve(curve_dir: Path, algo: str, run_idx: int, checkpoints, curve) -> None:
    """One convergence curve per run, so the plot survives a resubmission too."""
    curve_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"eval": checkpoints, "best_so_far": curve}).to_csv(
        curve_dir / f"{algo}_run{run_idx:03d}.csv", index=False)


def _load_curves(curve_dir: Path, algos) -> dict:
    out = {a: [] for a in algos}
    if not curve_dir.exists():
        return out
    for p in sorted(curve_dir.glob("*_run*.csv")):
        algo = p.name.rsplit("_run", 1)[0]
        if algo in out:
            try:
                out[algo].append(pd.read_csv(p)["best_so_far"].to_numpy())
            except Exception:                                          # noqa: BLE001
                pass          # a half-written curve from a killed job is skipped, not fatal
    return out


def _already_done(target: Path) -> set:
    """(algo, run) pairs already recorded, so a resubmission resumes instead of repeating.

    A trailing partial line from a job killed mid-write is dropped by the CSV reader, and that run
    is simply redone -- which is the correct behaviour, because a partial row cannot be trusted.
    """
    if not target.exists():
        return set()
    try:
        df = pd.read_csv(target, on_bad_lines="skip")
    except Exception:                                                  # noqa: BLE001
        return set()
    if "algo" not in df.columns or "run" not in df.columns:
        return set()
    done = set()
    for a, r in zip(df["algo"], df["run"]):
        try:
            done.add((str(a), int(r)))
        except (TypeError, ValueError):
            continue
    return done


def one_job(problem, algo: str, run_idx: int, seed: int, max_evals: int, pop: int,
            checkpoints: int):
    """Run one (algorithm, run) pair and return (row, curve). Mirrors run_spice.py's job body."""
    rng = np.random.default_rng(seed)
    obj = ObjectiveWrapper(func=None, eval_with_metrics=problem.evaluate_with_metrics,
                           max_evals=max_evals, progress_cb=None)
    t0 = time.perf_counter()
    res = ALGORITHMS[algo](obj=obj, rng=rng, lb=problem.lb, ub=problem.ub,
                           pop=pop, dim=problem.dim)
    runtime = time.perf_counter() - t0

    m_final = problem.metrics(res.best_x)
    m_feas = getattr(obj, "best_feasible_metrics", None) or {}
    he = np.asarray(res.history_evals, dtype=float)
    hb = np.asarray(res.history_best, dtype=float)
    curve = R.resample_history(he, hb, checkpoints)
    last_imp = R.last_improvement_eval_from_history(he, hb, max_evals)
    auc = R.convergence_auc(checkpoints, curve)
    c = problem.counters

    row = {
        "case": CASE, "case_name": CASE_NAME, "dim": problem.dim,
        "run": run_idx, "seed": seed, "algo": algo, "pop": pop,
        "best_fitness": float(res.best_f),
        OBJECTIVE_COL: m_final.get(OBJECTIVE_COL, np.nan),
        "penalty": m_final.get("penalty", np.nan),
        "is_feasible": m_final.get("is_feasible", 0),
        "first_feasible_eval": getattr(obj, "first_feasible_eval", None),
        "best_feasible_found": int(bool(m_feas)),
        "best_feasible_fitness": getattr(obj, "best_feasible_f", np.nan),
        BEST_FEASIBLE_OBJECTIVE_COL: m_feas.get(OBJECTIVE_COL, np.nan),
        "runtime_sec": runtime,
        "eval_budget": max_evals, "actual_evals": obj.evals,
        "last_improvement_eval": last_imp, "convergence_auc": auc,
        "deck_w6": m_final.get("x_w6", np.nan),
        "deck_w7": m_final.get("x_w7", np.nan),
        "sim_evals_worker_cum": getattr(c, "evaluations", np.nan),
        "sim_ok_worker_cum": getattr(c, "ok", np.nan),
        # the failure COUNT, not Counters.summary()'s prose. On Circuit A this column held a
        # string, which made it unusable for filtering -- recorded as a number here.
        "sim_failures_worker_cum": int(sum(getattr(c, "by_failure", {}).values())),
        "sim_failure_kinds_worker_cum": ";".join(
            f"{k}={v}" for k, v in sorted(getattr(c, "by_failure", {}).items())) or "",
        "server_restarts_worker_cum": getattr(problem._srv, "restarts", np.nan),
        "evaluator_recovers_timeouts": 1,
        "objective_def": "dc_open_loop_gain_db",
    }
    for k in SPEC_COLS:
        row[k] = m_final.get(k, np.nan)
        row["best_feasible_" + k] = m_feas.get(k, np.nan)
    for k in VIOL_COLS:
        row[k] = m_final.get(k, np.nan)
    # the optimizer's own vector, not the deck's: res.best_x is in (w1,w3,w5,m2,k,l_all,cc)
    for i, name in enumerate(sp.VAR_NAMES):
        row["x_" + name] = float(res.best_x[i])
    for k in THRESHOLD_KEYS:
        row["c_" + k] = problem.constraints.get(k, np.nan)
    return row, curve


def _summarise(csv_path: Path, case_dir: Path, algos, checkpoints: int, curves) -> int:
    if not csv_path.exists():
        log("[stats] no rows; nothing to summarise")
        return 1
    df = pd.read_csv(csv_path)
    if df.empty:
        log("[stats] empty CSV; nothing to summarise")
        return 1
    algos = [a for a in algos if a in set(df["algo"])]
    log(f"[stats] {len(df)} rows, algorithms {algos}")

    # A partially finished campaign is analysable only if it is BALANCED across algorithms; an
    # unbalanced pivot silently changes what Friedman is testing.
    counts = df.groupby("algo")["run"].nunique()
    if len(set(counts.values)) > 1:
        log(f"[stats] WARNING: unbalanced runs per algorithm: {counts.to_dict()}")

    d = _objective_alias(df)
    core, ext, viol, rank_df = R.build_case_stats(d, algos)
    # Undo the alias on the way OUT, so nothing on disk carries a Circuit A label for a Circuit B
    # measurement. The alias exists only to reach run.py's shared statistics; leaving it in the
    # output would mean `violations.csv` reporting UGB failures in a column called `viol_vref`,
    # which is exactly the kind of trap that costs someone a day in six months.
    R.ensure_dir(case_dir)
    # Draw the plots BEFORE renaming anything back: they read Circuit A's column names, which is
    # what the aliased frames still carry at this point.
    try:
        if curves:
            R.plot_convergence(case_dir, CASE_NAME, algos, checkpoints, curves)
        R.plot_boxplot_fitness(case_dir, CASE_NAME, d, algos)
        R.plot_success_rate_bar(case_dir, CASE_NAME, ext, algos)
        R.plot_violations_bar(case_dir, CASE_NAME, viol, algos)
    except Exception as e:                                             # noqa: BLE001
        log(f"[stats] plots skipped: {type(e).__name__}: {e}")

    # PREFIX-based, not a list of suffixes. build_case_stats emits `_count_all`, `_rate_all`,
    # `_count_final`, `_rate_final`, `_count_infeasible`, `_rate_infeasible` and may add more;
    # enumerating them by hand missed two and leaked `viol_vref_count_infeasible` into
    # violations.csv. Matching on the prefix cannot miss a suffix that gets added later.
    inv = {dst: src for src, dst in VIOL_ALIAS.items()}
    for frame in (core, ext, viol):
        if frame is None:
            continue
        ren = {}
        for col in frame.columns:
            for dst, src in inv.items():
                if col == dst or col.startswith(dst + "_"):
                    ren[col] = src + col[len(dst):]
                    break
        if ren:
            frame.rename(columns=ren, inplace=True)
    # And rename the objective back, for the same reason.
    for frame in (core, ext):
        if frame is not None:
            frame.rename(columns={c: c.replace("PSRR", "GAIN") for c in frame.columns
                                  if "PSRR" in c}, inplace=True)
    pivot = d.pivot(index="run", columns="algo", values="best_fitness").reindex(columns=algos)
    fried, ranks = R.friedman_from_pivot(pivot, algos, CASE_NAME)
    wil = R.build_case_wilcoxon(d, algos, base_algo="ABC")

    for name, obj in (("core_stats", core), ("extended_stats", ext), ("violations", viol),
                      ("rank_summary", rank_df), ("friedman", fried),
                      ("average_ranks", ranks), ("wilcoxon_holm", wil),
                      ("pivot_best_fitness", pivot)):
        if obj is not None:
            obj.to_csv(case_dir / f"{name}.csv", index=(name == "pivot_best_fitness"))
    log(f"[stats] wrote 8 tables to {case_dir}")


    log("\n" + ranks.to_string(index=False) if ranks is not None else "")
    log("\n" + fried.to_string(index=False) if fried is not None else "")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="path to sky130.lib.spice")
    ap.add_argument("--outdir", default="results_spice_opamp")
    ap.add_argument("--algos", nargs="+", default=["ABC", "GWO", "FA", "PSO", "GA", "ACO"])
    ap.add_argument("--runs", type=int, default=30)
    # 2,500 is the CIRCUIT-LEVEL protocol and matches Circuit A's transistor-level campaign. It is
    # not the surrogate's 150,000; see the module docstring.
    ap.add_argument("--evals", type=int, default=2500)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent ngspice processes; budget ~250 MB resident each")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--checkpoints", type=int, default=300)
    # Chunking: split the 180 runs across N node allocations, one SLURM array task each.
    # 180 runs / (N * 0.0122 runs/s per node) -> 4.1 h on 1 node, 1.0 h on 4, 31 min on 8.
    ap.add_argument("--chunks", type=int, default=1,
                    help="split the run list into this many array tasks (one node each)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="which chunk this task runs; intended for $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--job-index", type=int, default=None,
                    help="run exactly one (algo, run) from this index and exit; for "
                         "$SLURM_ARRAY_TASK_ID")
    ap.add_argument("--rows-dir", default=None)
    ap.add_argument("--merge", action="store_true",
                    help="concatenate <rows-dir>/*.csv into per_run_records.csv and summarise")
    args = ap.parse_args()

    unknown = [a for a in args.algos if a not in ALGORITHMS]
    if unknown:
        ap.error(f"unknown algorithms {unknown}; available: {sorted(ALGORITHMS)}")
    # Refuse to run against uncalibrated thresholds. The failure this prevents is the worst kind:
    # with no thresholds every candidate is feasible, so a campaign completes and returns a full
    # ranking that means nothing at all.
    if not sp.CALIBRATED or not sp.CONSTRAINTS:
        ap.error("Circuit B's thresholds are not frozen; refusing to run a campaign")

    # An ARRAY of evaluation counts, exactly as run_spice.py:333 builds it. resample_history,
    # convergence_auc and plot_convergence all iterate it; passing the count instead raises
    # "'int' object is not iterable" on the first run.
    checkpoints = np.linspace(1, args.evals, args.checkpoints).astype(int)

    if args.chunks < 1:
        ap.error("--chunks must be >= 1")
    if args.chunks > 1 and args.chunk is None and not args.merge:
        ap.error("--chunks > 1 needs --chunk (use $SLURM_ARRAY_TASK_ID)")
    if args.chunk is not None and not 0 <= args.chunk < args.chunks:
        ap.error(f"--chunk must be in [0, {args.chunks})")
    if args.chunk is not None and args.job_index is not None:
        ap.error("--chunk and --job-index are alternative sharding modes; pass only one")

    case_dir = Path(args.outdir) / CASE
    rows_dir = Path(args.rows_dir) if args.rows_dir else Path(args.outdir) / "rows"
    csv_path = Path(args.outdir) / "per_run_records.csv"
    R.ensure_dir(Path(args.outdir))

    if args.merge:
        parts = sorted(rows_dir.glob("*.csv"))
        if not parts:
            log(f"no row files in {rows_dir}")
            return 1
        frames = {}
        for p in parts:
            f = pd.read_csv(p, on_bad_lines="skip")
            frames[p.name] = f
        df = pd.concat(frames.values(), ignore_index=True)

        # A duplicate is not housekeeping. It means two shards computed the same run, i.e. shard
        # ownership broke -- which is exactly what resume-before-striding used to do.
        dup_mask = df.duplicated(subset=["algo", "run"], keep=False)
        if dup_mask.any():
            pairs = sorted({(a, int(r)) for a, r in
                            zip(df.loc[dup_mask, "algo"], df.loc[dup_mask, "run"])})
            owner = {}
            for name, f in frames.items():
                for a, r in zip(f["algo"], f["run"]):
                    owner.setdefault((str(a), int(r)), []).append(name)
            split = [p for p in pairs if len({*owner.get(p, [])}) > 1]
            log(f"[merge] WARNING: {len(pairs)} duplicated (algo, run) pairs: {pairs[:10]}"
                + (" ..." if len(pairs) > 10 else ""))
            if split:
                log(f"[merge] SHARD OWNERSHIP FAULT: {len(split)} pair(s) claimed by more than one "
                    f"file, e.g. {split[0]} in {sorted(set(owner[split[0]]))}")
                log("[merge] two shards computed the same run; do not trust these rows until the "
                    "chunk files are reconciled")

        # FOREIGN ROWS. rows_chunkNNN.csv may hold only the strided slice NNN::chunks.
        idx = {}
        for name in frames:
            if name.startswith("rows_chunk") and name.endswith(".csv"):
                try:
                    idx[name] = int(name[len("rows_chunk"):-len(".csv")])
                except ValueError:
                    pass
        n_chunks = args.chunks if args.chunks > 1 else (
            len(idx) if idx and sorted(idx.values()) == list(range(len(idx))) else 0)
        if idx and n_chunks:
            grid = [(a, r) for r in range(args.runs) for a in args.algos]
            for name, c in sorted(idx.items(), key=lambda kv: kv[1]):
                if c >= n_chunks:
                    continue
                own = set(grid[c::n_chunks])
                bad = sorted({(str(a), int(r)) for a, r in
                              zip(frames[name]["algo"], frames[name]["run"])} - own)
                if bad:
                    log(f"[merge] FOREIGN ROWS: {name} holds {len(bad)} run(s) it does not own, "
                        f"e.g. {bad[:5]} -- this file was written by a shard using the wrong slice")
        elif idx:
            log(f"[merge] NOTE: chunk files {sorted(idx.values())} are not contiguous from 0 and "
                f"--chunks was not given, so per-file ownership was NOT checked")

        before = len(df)
        df = df.drop_duplicates(subset=["algo", "run"], keep="last")
        if len(df) != before:
            log(f"[merge] dropped {before - len(df)} duplicate (algo, run) rows")
        df.to_csv(csv_path, index=False)
        log(f"[merge] {len(parts)} files -> {len(df)} rows -> {csv_path}")
        # State plainly whether anything is missing. A merge that quietly yields 173 of 180 rows and
        # then reports a ranking is the failure mode worth guarding against.
        expected = len(args.algos) * args.runs
        if len(df) != expected:
            log(f"[merge] WARNING: {len(df)} rows against {expected} expected "
                f"({expected - len(df)} missing)")
            log(f"[merge] runs per algorithm: {df.groupby('algo')['run'].nunique().to_dict()}")
            log("[merge] resubmit the same array to fill the gaps -- shards resume, not repeat")
        return _summarise(csv_path, case_dir, args.algos, checkpoints, None)

    # Defined BEFORE the pool starts, because the workers append to them as they go.
    fields = ROW_FIELDS + ["c_" + k for k in THRESHOLD_KEYS]
    # Each shard writes its OWN file. Chunks run on different nodes against a shared filesystem,
    # so the in-process write lock protects nothing between them and concurrent appends to one CSV
    # would interleave rows. --merge concatenates them afterwards.
    if args.job_index is not None:
        target = rows_dir / f"rows_{args.job_index:05d}.csv"
    elif args.chunk is not None:
        target = rows_dir / f"rows_chunk{args.chunk:03d}.csv"
    else:
        target = csv_path
    curve_dir = Path(args.outdir) / "curves"
    R.ensure_dir(target.parent)

    log(f"{CASE_NAME}: {len(args.algos)} algorithms x {args.runs} runs x {args.evals:,} "
        f"evaluations, pop {args.pop}")
    log(f"thresholds: {sp.CONSTRAINTS}")
    log(f"bars to clear: random-feasible {sp.PROBE_BEST_FEASIBLE_GAIN_DB:.2f} dB, "
        f"reference design {sp.NOMINAL_OBJECTIVE_DB:.2f} dB")

    # algorithm-MINOR ordering, so that any prefix of task indices contains whole algorithm sets and
    # an interrupted array is still balanced -- and therefore still analysable.
    jobs = [(a, r) for r in range(args.runs) for a in args.algos]

    # SHARD FIRST. Which jobs a chunk owns must depend ONLY on (chunk, chunks, runs, algos) -- never
    # on what happens to be on disk. Filtering for resume before this line renumbered the list, so
    # the stride landed on other chunks' jobs: a chunk 2 resubmitted over its own completed rows ran
    # 25 jobs, all of them belonging to other chunks, and appended every one to rows_chunk002.csv.
    if args.job_index is not None:
        if not 0 <= args.job_index < len(jobs):
            ap.error(f"--job-index must be in [0, {len(jobs)})")
        jobs = [jobs[args.job_index]]
        R.ensure_dir(rows_dir)
    elif args.chunk is not None:
        # Strided, not blocked. The list is algorithm-minor, so with CHUNKS equal to the number of
        # algorithms each chunk gets exactly ONE optimizer's 30 runs -- chunk 0 is all of ABC, chunk 4
        # all of GA. Choose CHUNKS coprime with len(algos) if a mixed slice per chunk is wanted.
        total = len(jobs)
        jobs = jobs[args.chunk::args.chunks]
        log(f"chunk {args.chunk} of {args.chunks}: {len(jobs)} of {total} runs, "
            f"algorithms {sorted({a for a, _ in jobs})}")
        R.ensure_dir(rows_dir)

    # RESUME, strictly within this shard's own slice. Rows already on disk are not recomputed, which
    # is what makes the incremental write worth having: a campaign killed by a wall is continued by
    # resubmitting the same command.
    done = _already_done(target)
    if done:
        before = len(jobs)
        jobs = [j for j in jobs if j not in done]
        log(f"resuming: {len(done)} runs already recorded in {target.name}, "
            f"{len(jobs)} of {before} remaining")
        if not jobs:
            if args.job_index is not None or args.chunk is not None:
                log("shard already complete; nothing to do")
                return 0
            log("nothing left to run; summarising what is on disk")
            return _summarise(csv_path, case_dir, args.algos, checkpoints,
                              _load_curves(curve_dir, args.algos))

    def worker(job_list):
        p = sp.SpiceOpampProblem(sky130_lib=args.lib, timeout=args.timeout)
        out = []
        try:
            for algo, run_idx in job_list:
                seed = args.seed + CASE_OFFSET + run_idx * 1000
                t0 = time.perf_counter()
                row, curve = one_job(p, algo, run_idx, seed, args.evals, args.pop,
                                     checkpoints)
                # PERSIST BEFORE ANYTHING ELSE. If the job is killed on the next line, this run is
                # still on disk.
                _append_row(target, fields, row)
                _write_curve(curve_dir, algo, run_idx, checkpoints, curve)
                out.append((row, curve))
                log(f"  {algo:4s} run {run_idx:2d}  fitness {row['best_fitness']:11.4f}  "
                    f"gain {row[OBJECTIVE_COL]:7.3f} dB  feasible {row['is_feasible']}  "
                    f"{time.perf_counter()-t0:6.1f}s")
        finally:
            p.close()
        return out

    nw = 1 if args.job_index is not None else max(1, args.workers)
    shards = [jobs[i::nw] for i in range(nw)]
    shards = [s for s in shards if s]
    t0 = time.perf_counter()
    results = []
    if len(shards) == 1:
        results = worker(shards[0])
    else:
        with ThreadPoolExecutor(max_workers=len(shards)) as ex:
            futs = [ex.submit(worker, s) for s in shards]
            for f in as_completed(futs):
                results.extend(f.result())
    log(f"{len(results)} runs in {(time.perf_counter()-t0)/60:.1f} min")

    # Every row was already appended and fsync'd as its run finished, so there is nothing to write
    # here. What is on disk is what the campaign produced, whether or not it reached this line.
    log(f"rows on disk: {target}")

    # A shard does not summarise: it holds only its own slice, and a Friedman test on part of the
    # block structure is not a weaker result but a different and wrong one. Merge when all shards
    # have finished.
    if args.job_index is not None or args.chunk is not None:
        log("shard complete. When all shards finish, merge and summarise with:")
        log(f"  --merge --rows-dir {rows_dir}")
        return 0
    return _summarise(csv_path, case_dir, args.algos, checkpoints,
                      _load_curves(curve_dir, args.algos))


if __name__ == "__main__":
    raise SystemExit(main())
