"""
run_spice.py -- campaign driver for the transistor-level SKY130 bandgap case study.

Why this file exists instead of a flag on run.py: run.py cannot drive this case even
unmodified. `--problem` is `choices=["sbgr"]` (run.py:524), `--cases` is
`choices=["base","hard","highdim"]` (run.py:525), and `case_offset` is a hard-coded dict
(run.py:597), so any new problem or case name is a parse error or a KeyError.

But run.py IS import-safe (its work is behind `if __name__ == "__main__"`), so this driver
imports it and reuses its statistics and plotting verbatim -- build_case_stats,
build_case_wilcoxon, friedman_from_pivot, holm_step_down, resample_history,
convergence_auc, plot_* -- and emits the identical row schema (run.py:665-713) that those
helpers consume. algorithms.py and run.py are never modified.

Design notes that came out of measurement, not taste:

  * Parallelism is at RUN level, not evaluation level. Each evaluation blocks in
    subprocess I/O, which releases the GIL, and the optimizer's own Python work is
    microseconds against a ~0.3 s simulation. THREADS, not processes: on Windows
    ProcessPoolExecutor uses spawn, re-imports __main__, and would force pickling the
    evaluator, its ngspice process and its RNG.
  * Each worker owns ONE SpiceBGRProblem, therefore one ngspice process with the PDK
    already parsed. Creating one per job would pay the ~70 s library load 180 times.
  * Every (algo, run) row is appended to the CSV the moment it finishes. A 10-hour campaign
    that loses everything to a Windows update is a bad trade for a few lines of code.
    Re-running skips whatever the CSV already contains.

Usage:
    python run_spice.py --lib <path to sky130.lib.spice> --outdir results_spice
    python run_spice.py --lib ... --algos ABC PSO --runs 3 --evals 300   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import re
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

import run as R                      # noqa: E402  statistics + plots, reused verbatim
from algorithms import ALGORITHMS, ObjectiveWrapper   # noqa: E402
import spice_problem as sp           # noqa: E402

CASE = "spice"
CASE_NAME = "SKY130-BGR"
# Matches the surrogate campaign's offsets in spirit: a distinct block so seeds never
# collide with the SBGR runs.
CASE_OFFSET = 300000

BASE_ROW_FIELDS = [
    # `pop` is recorded because it is a protocol parameter, and its absence is exactly how a
    # deviation from the published SN=40 went unnoticed for a whole campaign. Anything that
    # changes the comparison must appear in the row.
    "case", "case_name", "dim", "run", "seed", "algo", "pop",
    "best_fitness", "PSRR_DB", "penalty", "is_feasible",
    "VREF", "TC", "LOOP_GAIN_DB", "PHASE_MARGIN_DEG", "GAIN_MARGIN_DB", "POWER_UW",
    "viol_vref", "viol_tc", "viol_loop_gain", "viol_phase_margin", "viol_gain_margin",
    "viol_power",
    "first_feasible_eval", "best_feasible_found", "best_feasible_fitness",
    "best_feasible_PSRR", "best_feasible_VREF", "best_feasible_TC",
    "best_feasible_LOOP_GAIN_DB", "best_feasible_PHASE_MARGIN_DEG",
    "best_feasible_GAIN_MARGIN_DB", "best_feasible_POWER_UW",
    "runtime_sec", "eval_budget", "actual_evals",
    "last_improvement_eval", "convergence_auc",
    # SPICE-specific extras; run.py's helpers ignore unknown columns.
    # `sim_*` and `server_restarts` are per-WORKER CUMULATIVE counters, not per-row: workers are
    # shared across jobs, so these describe the worker's whole history at the moment this row was
    # written. Filtering on them per run is meaningless -- that mistake silently yielded zero
    # rows once. The names carry the suffix so it cannot happen again.
    "sim_evals_worker_cum", "sim_ok_worker_cum", "sim_failures_worker_cum",
    "server_restarts_worker_cum",
    # Which PSRR definition was the objective. Recorded because the objective was restated from
    # a single frequency to the band worst case, and a numeric column whose meaning changed
    # between campaigns without a label is unreadable afterwards.
    "psrr_def", "PSRR_WC_DB", "PSRR_100HZ_DB",
    "best_feasible_PSRR_WC_DB", "best_feasible_PSRR_100HZ_DB",
    # Which evaluator produced this row: 1 means a timeout is recovered by restarting the
    # simulator, 0 or absent means a timeout silently voided everything after it. The merge guard
    # reads this instead of asking the operator to remember.
    "evaluator_recovers_timeouts",
]

THRESHOLD_KEYS = ("VREF_min", "VREF_max", "TC_max", "LoopGain_min",
                  "PhaseMargin_min", "GainMargin_min", "Power_max")


def row_fields(case: str) -> list[str]:
    """The CSV schema for one case. Built per case, NOT a module constant, because the variable
    NAMES differ between cases: base/hard have 7, highdim has 16 with different names. A fixed
    list of base's names would have silently dropped every highdim design vector, because
    csv.DictWriter is created with extrasaction="ignore" -- the campaign would have completed and
    the designs simply would not be there.

    The design vector is recorded twice: `x_*` for the final best solution and `xf_*` for the best
    feasible one. Its absence in the first campaign is why those winning circuits can never be
    re-simulated, plotted or printed, and a transistor-level case study whose best design cannot
    be shown has nothing to show.

    The thresholds in force are recorded too. The first campaign wrote case='spice' and no
    thresholds, so base-vs-hard had to be reconstructed afterwards by replaying both candidate
    sets against the recorded metrics. That worked, but an artefact should not need detective
    work to interpret.
    """
    names = [v[0] for v in sp.CASES[case][1]]
    fields = list(BASE_ROW_FIELDS)
    fields += [f"x_{n}" for n in names]
    fields += [f"xf_{n}" for n in names]
    fields += [f"c_{k}" for k in THRESHOLD_KEYS]
    if len(fields) != len(set(fields)):
        dup = sorted({f for f in fields if fields.count(f) > 1})
        raise AssertionError(f"duplicate CSV fields for case {case!r}: {dup}")
    return fields

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def one_run(problem, algo: str, run_idx: int, seed: int, max_evals: int,
            pop: int, checkpoints: np.ndarray, case: str) -> tuple[dict, np.ndarray]:
    """One (algorithm, run) job. Mirrors run.py's inner loop exactly."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    obj = ObjectiveWrapper(func=None, eval_with_metrics=problem.evaluate_with_metrics,
                           max_evals=max_evals, progress_cb=None)
    res = ALGORITHMS[algo](obj=obj, rng=rng, lb=problem.lb, ub=problem.ub,
                           pop=pop, dim=problem.dim)
    runtime = time.perf_counter() - t0

    m_final = problem.metrics(res.best_x)
    m_feas = obj.best_feasible_metrics or {}
    he = np.asarray(res.history_evals, dtype=float)
    hb = np.asarray(res.history_best, dtype=float)
    curve = R.resample_history(he, hb, checkpoints)
    last_imp = R.last_improvement_eval_from_history(he, hb, max_evals)
    auc = R.convergence_auc(checkpoints, curve)

    row = {
        "case": case, "case_name": problem.name, "dim": problem.dim,
        "run": run_idx, "seed": seed, "algo": algo, "pop": int(pop),
        "best_fitness": float(res.best_f),
        "PSRR_DB": m_final.get("PSRR_DB", np.nan),
        "penalty": m_final.get("penalty", np.nan),
        "is_feasible": int(m_final.get("is_feasible", 0)),
        "VREF": m_final.get("VREF", np.nan),
        "TC": m_final.get("TC", np.nan),
        "LOOP_GAIN_DB": m_final.get("LOOP_GAIN_DB", np.nan),
        "PHASE_MARGIN_DEG": m_final.get("PHASE_MARGIN_DEG", np.nan),
        "GAIN_MARGIN_DB": m_final.get("GAIN_MARGIN_DB", np.nan),
        "POWER_UW": m_final.get("POWER_UW", np.nan),
        "viol_vref": int(m_final.get("viol_vref", 0)),
        "viol_tc": int(m_final.get("viol_tc", 0)),
        "viol_loop_gain": int(m_final.get("viol_loop_gain", 0)),
        "viol_phase_margin": int(m_final.get("viol_phase_margin", 0)),
        "viol_gain_margin": int(m_final.get("viol_gain_margin", 0)),
        "viol_power": int(m_final.get("viol_power", 0)),
        "first_feasible_eval": (float(obj.first_feasible_eval)
                                if obj.first_feasible_eval is not None else np.nan),
        "best_feasible_found": int(bool(m_feas)),
        # nan, not obj.best_feasible_f, when nothing feasible was found: the wrapper
        # initialises that field to +inf, and run.py:648 writes nan in the same situation.
        # An inf in a numeric CSV column poisons any later mean that forgets to filter on
        # best_feasible_found -- 2 rows of the first campaign carried one.
        "best_feasible_fitness": (float(obj.best_feasible_f) if m_feas else np.nan),
        "best_feasible_PSRR": m_feas.get("PSRR_DB", np.nan),
        "best_feasible_VREF": m_feas.get("VREF", np.nan),
        "best_feasible_TC": m_feas.get("TC", np.nan),
        "best_feasible_LOOP_GAIN_DB": m_feas.get("LOOP_GAIN_DB", np.nan),
        "best_feasible_PHASE_MARGIN_DEG": m_feas.get("PHASE_MARGIN_DEG", np.nan),
        "best_feasible_GAIN_MARGIN_DB": m_feas.get("GAIN_MARGIN_DB", np.nan),
        "best_feasible_POWER_UW": m_feas.get("POWER_UW", np.nan),
        "runtime_sec": float(runtime),
        "eval_budget": int(max_evals), "actual_evals": int(obj.evals),
        "last_improvement_eval": float(last_imp),
        "convergence_auc": float(auc),
        "sim_evals_worker_cum": problem.counters.evaluations,
        "sim_ok_worker_cum": problem.counters.ok,
        "sim_failures_worker_cum": ";".join(f"{k}={v}" for k, v in
                                           sorted(problem.counters.by_failure.items())),
        "server_restarts_worker_cum": problem._srv.restarts,
        "evaluator_recovers_timeouts": int(getattr(sp, "EVALUATOR_RECOVERS_TIMEOUTS", 0)),
        "psrr_def": m_final.get("psrr_def", ""),
        "PSRR_WC_DB": m_final.get("PSRR_WC_DB", np.nan),
        "PSRR_100HZ_DB": m_final.get("PSRR_100HZ_DB", np.nan),
        "best_feasible_PSRR_WC_DB": m_feas.get("PSRR_WC_DB", np.nan),
        "best_feasible_PSRR_100HZ_DB": m_feas.get("PSRR_100HZ_DB", np.nan),
    }
    # design vectors and thresholds, straight from the metrics dict the evaluator already built
    row.update({f"x_{k[2:]}": v for k, v in m_final.items() if k.startswith("x_")})
    row.update({f"xf_{k[2:]}": v for k, v in m_feas.items() if k.startswith("x_")})
    row.update({k: v for k, v in m_final.items() if k.startswith("c_")})
    return row, curve


def _summarise(df_path, case_dir, algo_names, checkpoints, curves) -> int:
    """Statistics and plots, every one of them run.py's own helper. Separated out so an
    already-complete campaign can be re-summarised from its CSV in seconds, without spawning
    workers or paying the library load."""
    if not Path(df_path).exists():
        log("[stats] no rows; nothing to summarise")
        return 1
    df = pd.read_csv(df_path)
    algos = [a for a in algo_names if a in set(df["algo"])]
    log(f"\n[stats] {len(df)} rows, algorithms: {algos}")

    # build_case_stats' FOURTH return is a per-algorithm RANK SUMMARY, not the
    # blocks x algorithms pivot that friedman_from_pivot expects. Mixing them up silently
    # yields a 6x3 frame, zero complete blocks, and an "Insufficient complete blocks" verdict
    # on a campaign that has 30 perfectly good blocks -- which is exactly what happened on the
    # first pass. run.py builds the pivot separately at run.py:728; do the same.
    core, ext, viol, rank_df = R.build_case_stats(df, algos)
    pivot = df.pivot(index="run", columns="algo", values="best_fitness").reindex(columns=algos)
    core.to_csv(case_dir / "core_stats.csv", index=False)
    ext.to_csv(case_dir / "extended_stats.csv", index=False)
    viol.to_csv(case_dir / "violations.csv", index=False)
    rank_df.to_csv(case_dir / "rank_summary.csv", index=False)
    pivot.to_csv(case_dir / "pivot_best_fitness.csv")
    log(core.to_string(index=False))

    if len(algos) >= 3:
        fried, ranks = R.friedman_from_pivot(pivot, algos, CASE_NAME)
        fried.to_csv(case_dir / "friedman.csv", index=False)
        ranks.to_csv(case_dir / "average_ranks.csv", index=False)
        log("\n" + fried.to_string(index=False))
        log("\n" + ranks.to_string(index=False))
    else:
        log("\n[stats] Friedman needs >= 3 algorithms; skipped")

    wil = R.build_case_wilcoxon(df, algos, base_algo="ABC")
    wil.to_csv(case_dir / "wilcoxon_holm.csv", index=False)
    log("\n" + wil.to_string(index=False))

    try:
        if curves and any(curves.values()):
            R.plot_convergence(case_dir, CASE_NAME, algos, checkpoints, curves)
        else:
            log("[plots] convergence skipped: curves live in memory only, so they cannot be "
                "rebuilt from the CSV")
        R.plot_boxplot_fitness(case_dir, CASE_NAME, df, algos)
        R.plot_boxplot_feasible_psrr(case_dir, CASE_NAME, df, algos)
        R.plot_success_rate_bar(case_dir, CASE_NAME, ext, algos)
        # reusable now that all six specs are live -- with fewer, the unmeasured ones would be
        # drawn as zero-height bars, visually asserting "never violated"
        R.plot_violations_bar(case_dir, CASE_NAME, viol, algos)
        log(f"[plots] written to {case_dir}")
    except Exception as e:  # noqa: BLE001
        log(f"[plots] failed: {type(e).__name__}: {e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="path to sky130.lib.spice")
    ap.add_argument("--outdir", default="results_spice")
    ap.add_argument("--case", default="base", choices=sorted(sp.CASES),
                    help="base = the calibrated nominal thresholds; hard = the same deck with "
                         "thresholds tightened by the pre-declared rule in spice_problem.py")
    ap.add_argument("--algos", nargs="+",
                    default=["ABC", "GWO", "FA", "PSO", "GA", "ACO"],
                    help="default is the same six as the surrogate campaign")
    ap.add_argument("--runs", type=int, default=30,
                    help="30 for parity with the surrogate campaign; 10 is the hard floor "
                         "because run.wilcoxon_signed_rank returns NaN below it")
    ap.add_argument("--evals", type=int, default=2500)
    ap.add_argument("--pop", type=int, default=40,
                    help="SN=40 is the published protocol (manuscript, tab:baseline_params). "
                         "It also sets ABC's scout limit floor(0.6*SN*D), so lowering it "
                         "starves the mechanism that distinguishes ABC -- do not change it "
                         "without saying so in the paper.")
    ap.add_argument("--workers", type=int, default=4,
                    help="4 measured best; 8 is SLOWER (hyperthread contention)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="per-evaluation ngspice timeout, seconds. Kept at 30 after the recovered "
                         "TRUBA rows showed what a timeout actually is here: not a slow "
                         "evaluation but a HANG. Typical cost is 0.15-0.39 s, so 30 s is already "
                         "80-200x that, and one run recorded 1408 timeouts out of 220,000 "
                         "evaluations -- 0.64 %, but 11.7 h of the 23.8 h runtime. Raising the "
                         "limit multiplies that dead time without rescuing any candidate, "
                         "because a hung solve does not finish at 90 s either. Cut it short and "
                         "restart instead.")
    ap.add_argument("--checkpoints", type=int, default=300,
                    help="300 to match run.py's --checkpoint-count (run.py:542). It was 60 here, "
                         "which would have sampled section 6's convergence curves on a five-times "
                         "coarser grid than section 5's and computed convergence_auc over a "
                         "different integration mesh -- the two sections' curves and AUC columns "
                         "would then not be directly comparable, which is the whole point of "
                         "reusing run.py's statistics.")
    # ---- cluster mode ----
    # On TRUBA the natural unit is a SLURM array task: one (algo, run) per task, one core, one
    # ngspice process. That is simpler and more robust than the thread pool -- a task that dies
    # takes nothing else with it, and the scheduler does the packing. Each task writes its OWN
    # row file, so there is no shared file to lock and no resume race between 540 concurrent
    # tasks. `--merge` then concatenates them and runs the identical statistics.
    ap.add_argument("--job-index", type=int, default=None,
                    help="run exactly one (algo, run) derived from this index and exit; "
                         "intended for $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--rows-dir", default=None,
                    help="directory for per-task row files (default <outdir>/rows)")
    ap.add_argument("--merge", action="store_true",
                    help="concatenate <rows-dir>/*.csv into per_run_records.csv and summarise")
    ap.add_argument("--timeouts-ok", action="store_true",
                    help="accept rows that recorded an ngspice timeout. Only valid if EVERY row "
                         "was produced by an evaluator carrying the restart-on-timeout fix; "
                         "before that fix a single timeout voids the rest of the run silently.")
    args = ap.parse_args()

    for a in args.algos:
        if a not in ALGORITHMS:
            ap.error(f"unknown algorithm {a!r}; available: {sorted(ALGORITHMS)}")

    outdir = Path(args.outdir)
    case_dir = outdir / args.case
    R.ensure_dir(case_dir)
    csv_path = outdir / "per_run_records.csv"
    checkpoints = np.linspace(1, args.evals, args.checkpoints).astype(int)
    fields = row_fields(args.case)
    rows_dir = Path(args.rows_dir) if args.rows_dir else outdir / "rows"

    # ---------------------------------------------------------------- merge mode
    if args.merge:
        # Row files only. The sibling *.curve.csv files hold the persisted convergence curves and
        # have a completely different schema -- concatenating them would silently produce garbage
        # rows that every downstream statistic would then treat as runs.
        files = sorted(f for f in rows_dir.glob("*.csv") if not f.name.endswith(".curve.csv"))
        if not files:
            ap.error(f"no row files in {rows_dir}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

        # Rebuild the per-algorithm convergence curves from disk so Section 6 can carry the same
        # figure as Section 5. Missing curves are not fatal -- the statistics do not depend on
        # them -- but they are reported, because a silently absent figure is how a section ends up
        # inconsistent with the one it is meant to be compared against.
        curves_from_disk: dict[str, list[np.ndarray]] = {a: [] for a in args.algos}
        n_curve = 0
        for f in files:
            cf = f.with_suffix(".curve.csv")
            if cf.exists():
                cdf = pd.read_csv(cf)
                algo_of = pd.read_csv(f)["algo"].iloc[0]
                if algo_of in curves_from_disk:
                    curves_from_disk[algo_of].append(cdf["best_so_far"].to_numpy())
                    n_curve += 1
        log(f"[merge] {n_curve}/{len(files)} convergence curves recovered from disk")
        # Every guard here exists because a merge that quietly drops or duplicates tasks produces
        # a plausible, wrong statistical result. 540 array tasks is too many to eyeball.
        expected = len(args.algos) * args.runs
        dups = df.groupby(["algo", "run"]).size()
        problems = []
        if len(files) != expected:
            problems.append(f"{len(files)} row files but {expected} (algo, run) pairs expected "
                            f"-- missing: "
                            f"{sorted(set((a, r) for r in range(args.runs) for a in args.algos) - set(zip(df.algo, df.run)))[:12]}")
        if (dups > 1).any():
            problems.append(f"duplicated (algo, run): {dups[dups > 1].to_dict()}")
        for col, want in (("case", args.case), ("pop", args.pop),
                          ("eval_budget", args.evals)):
            seen = sorted(set(df[col].dropna().tolist()))
            if len(seen) != 1 or str(seen[0]) != str(want):
                problems.append(f"{col} is {seen}, expected [{want}]")
        if "psrr_def" in df.columns and len(set(df.psrr_def.dropna())) > 1:
            problems.append(f"rows mix objective definitions: {set(df.psrr_def)}")

        # POISONED-SESSION DETECTOR. A row can look perfectly complete and be worthless. Observed
        # on TRUBA 2026-08-07: `runtime_sec = 64.5` for `actual_evals = 220000`, i.e. 0.3 ms per
        # evaluation, when the cheapest real ngspice evaluation measured anywhere in this project
        # is 128 ms. The mechanism is the timeout-desynchronisation bug (see spice_problem.metrics):
        # once the pipe desynchronises every evaluation fails, failures other than timeout ARE
        # cached, the optimizer collapses onto one point because every fitness is the same floor
        # value, and the remaining budget is consumed as instant cache hits. The row then reports
        # actual_evals == eval_budget with every column populated.
        #
        # Nothing else in the schema distinguishes such a row from a legitimate all-infeasible run,
        # so the check has to be on throughput. The floor is deliberately far below any measured
        # rate: 128 ms was the fastest single-process evaluation on an idle node, so 10 ms per
        # evaluation cannot be reached by any real campaign and cannot reject a genuine row.
        # The DEFINITIVE marker is the best fitness sitting exactly on the failure floor. That
        # value means the best candidate found in the entire run was a FAILED SIMULATION, i.e.
        # not one of 150,000-220,000 evaluations produced a measurable circuit. Uniform random
        # sampling of this box measures successfully about half the time on highdim and almost
        # always on base, and the initial population alone is 40 points, so a legitimate run
        # cannot end here. Throughput is reported alongside as corroboration, but it is NOT the
        # test: legitimate cache hits are free, so a converged run can average well under the
        # 128 ms of a real simulation without being poisoned.
        FLOOR_FITNESS = 1000.0 * 6 + 200.0        # LAMBDA*N_SPECS - PSRR_FLOOR_DB

        def _n_deck(s):
            m = re.search(r"\bdeck_error=(\d+)", str(s) if pd.notna(s) else "")
            return int(m.group(1)) if m else 0

        if "best_fitness" in df.columns:
            at_floor = df[np.isclose(df["best_fitness"], FLOOR_FITNESS, atol=1e-6)]
            if len(at_floor):
                # A row at the floor means no evaluation in the whole run produced a measurable
                # circuit. Under the pre-fix evaluator that is the poisoned-pipe signature and the
                # row is void. Under the fixed evaluator it can also be an honest, very bad run --
                # SPICE-Highdim reaches penalty 6 easily, only ~20 % of random points there
                # simulate at all, and an algorithm that never escapes that region legitimately
                # ends here. Blocking it would delete a real result; ignoring it would resurrect
                # the bug this guard exists for. The two are told apart by the failure
                # composition: poisoning shows up as deck_error dominating (one run recorded
                # deck_error=219945), an honestly bad run as measure_failed dominating.
                recov_f = (df["evaluator_recovers_timeouts"].fillna(0).astype(int)
                           if "evaluator_recovers_timeouts" in df.columns
                           else pd.Series(0, index=df.index))
                det, block = [], []
                for r in at_floor.itertuples():
                    ms = (r.runtime_sec / r.actual_evals * 1000
                          if getattr(r, "actual_evals", 0) else float("nan"))
                    nd = _n_deck(getattr(r, "sim_failures_worker_cum", ""))
                    frac = nd / max(1, int(getattr(r, "actual_evals", 1)))
                    tag = f"{r.algo}/run{r.run} ({ms:.1f} ms/eval, deck_error {frac:.1%})"
                    det.append(tag)
                    if recov_f.loc[r.Index] == 0 or frac > 0.5 or ms < 10.0:
                        block.append(tag)
                if block:
                    problems.append(
                        f"{len(block)} row(s) at the failure floor with the poisoned-session "
                        f"signature -- a pre-fix evaluator, or deck errors dominating, or a "
                        f"throughput no real simulation can reach: " + ", ".join(block)
                        + ". Delete those row files and re-submit; the driver redoes only the "
                          "missing (algo, run) pairs.")
                warn = [d for d in det if d not in block]
                if warn:
                    log(f"[merge] {len(warn)} run(s) ended at the failure floor but look honest "
                        f"(fixed evaluator, few deck errors, plausible throughput): "
                        + ", ".join(warn)
                        + ". Kept. This is a real 0 % outcome for those runs, not a fault.")

        # PARTIAL poisoning is the harder case and the floor test cannot see it. If a run finds a
        # good solution at evaluation 5,000 and is poisoned at 20,000, every later evaluation
        # returns the failure floor, never beats the incumbent, and the row looks entirely normal
        # while a seventh of the budget did the work.
        #
        # The signal is exact rather than statistical. In --job-index mode each SLURM task builds
        # ONE SpiceBGRProblem and runs ONE (algo, run) pair, so `sim_failures_worker_cum` is that
        # job's own tally, not a shared counter. A non-zero `timeout=` in it means the pipe
        # desynchronised at least once, and under the pre-fix evaluator everything after that point
        # is worthless. There is no way to recover how much was lost, so the row is rejected.
        #
        # Once the restart-on-timeout fix is deployed to every worker a timeout is harmless -- it
        # costs one PDK reload and the session continues -- so campaigns run entirely on the fixed
        # evaluator should pass --timeouts-ok. That flag is deliberately explicit: it is a claim
        # about which evaluator produced the rows, and only the person who submitted them knows it.
        def _n_timeouts(s):
            m = re.search(r"\btimeout=(\d+)", str(s) if pd.notna(s) else "")
            return int(m.group(1)) if m else 0

        if "sim_failures_worker_cum" in df.columns:
            n_to = df["sim_failures_worker_cum"].map(_n_timeouts)
            # Which evaluator produced each row is read from the row, not assumed. A row without
            # the marker predates it and is therefore pre-fix by definition.
            recov = (df["evaluator_recovers_timeouts"].fillna(0).astype(int)
                     if "evaluator_recovers_timeouts" in df.columns
                     else pd.Series(0, index=df.index))
            unsafe = df[(n_to > 0) & (recov == 0)]
            if len(unsafe) and not args.timeouts_ok:
                problems.append(
                    f"{len(unsafe)} row(s) recorded a timeout on an evaluator that does NOT "
                    f"recover from one: "
                    + ", ".join(f"{r.algo}/run{r.run}={_n_timeouts(r.sim_failures_worker_cum)}"
                                for r in unsafe.itertuples())
                    + ". There a single timeout desynchronises the ngspice pipe and every later "
                      "evaluation is void, with no way to tell how much of the budget survived. "
                      "Delete those row files and re-run them. (--timeouts-ok overrides, but only "
                      "do that if you know the evaluator was patched and simply did not record "
                      "the marker.)")
            # Timeouts on the fixed evaluator are survivable, but they are still lost compute and
            # a sudden rise means something changed on the machine. Report, never hide.
            safe_to = int(n_to[(recov == 1)].sum())
            if safe_to:
                log(f"[merge] {safe_to} timeouts across {int((n_to > 0).sum())} run(s), all on "
                    f"an evaluator that restarts and resynchronises -- survivable, but each one "
                    f"cost a hung solve plus a PDK reload")
        if problems:
            log("[merge] REFUSING to merge:")
            for p in problems:
                log("  - " + p)
            return 1
        df.to_csv(csv_path, index=False)
        log(f"[merge] {len(files)} row files -> {csv_path} ({len(df)} rows), all guards passed")
        return _summarise(csv_path, case_dir, args.algos, checkpoints, curves_from_disk)

    # ---------------------------------------------------------------- single array task
    if args.job_index is not None:
        total = len(args.algos) * args.runs
        if not 0 <= args.job_index < total:
            ap.error(f"--job-index must be in [0, {total}) for "
                     f"{len(args.algos)} algorithms x {args.runs} runs")
        # algorithm-minor so that a partially finished array is still BALANCED across algorithms:
        # any prefix of task indices contains whole (algo) sets, which is what makes an
        # interrupted campaign analysable instead of worthless.
        run_idx, algo = divmod(args.job_index, len(args.algos))
        algo = args.algos[algo]
        R.ensure_dir(rows_dir)
        out_row = rows_dir / f"{args.case}_{algo}_run{run_idx:03d}.csv"
        if out_row.exists():
            log(f"[task {args.job_index}] {out_row.name} already exists; nothing to do")
            return 0
        seed = args.seed + CASE_OFFSET + run_idx * 1000
        log(f"[task {args.job_index}] case={args.case} algo={algo} run={run_idx} seed={seed} "
            f"pop={args.pop} evals={args.evals}")
        problem = sp.SpiceBGRProblem(sky130_lib=args.lib, case=args.case,
                                     timeout=args.timeout)
        log(f"[task {args.job_index}] ngspice ready ({problem._srv.load_seconds:.0f}s)")
        t0 = time.perf_counter()
        try:
            row, _curve = one_run(problem, algo, run_idx, seed, args.evals, args.pop,
                                  checkpoints, args.case)
        finally:
            problem.close()
        # written to a temporary name first, then renamed: a task killed by the walltime limit
        # mid-write would otherwise leave a truncated row that --merge would happily accept
        tmp = out_row.with_suffix(".partial")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerow(row)
        tmp.replace(out_row)

        # PERSIST THE CONVERGENCE CURVE. In array mode each task is its own process, so a curve
        # held only in memory dies with it -- and _summarise then prints "curves live in memory
        # only" and skips the figure. Section 5 has a convergence figure; without this, Section 6
        # would have none, and the two sections would not be visually comparable. The curve is
        # `checkpoints` floats resampled by run.py's own resample_history, so writing it costs a
        # few kilobytes and makes the figure reproducible from the artefact rather than only from
        # a live run.
        cpath = out_row.with_suffix(".curve.csv")
        ctmp = cpath.with_suffix(".partial")
        pd.DataFrame({"eval": checkpoints, "best_so_far": _curve}).to_csv(ctmp, index=False)
        ctmp.replace(cpath)
        log(f"[task {args.job_index}] done in {time.perf_counter()-t0:.0f}s  "
            f"best={row['best_fitness']:.4f}  feas={row['is_feasible']}  "
            f"pen={row['penalty']}  -> {out_row.name}")
        return 0

    # ---- resume: skip whatever is already on disk ----
    #
    # The resume key is (algo, run) only, which is correct for continuing an interrupted
    # campaign and CATASTROPHIC across configurations: pointing a different --case or --pop at
    # a directory that already holds 180 rows makes every job "already complete", so the driver
    # prints statistics from the OLD configuration and exits in seconds having simulated
    # nothing. The result looks like a successful run. So refuse instead of guessing -- a
    # per-configuration --outdir is a one-line instruction, and a silently mislabelled campaign
    # is a retracted paper.
    done: set[tuple[str, int]] = set()
    if csv_path.exists():
        try:
            prev = pd.read_csv(csv_path)
            for col, want in (("case", args.case), ("pop", args.pop)):
                if col not in prev.columns:
                    ap.error(f"{csv_path} predates the {col!r} column, so it cannot be "
                             f"verified against this run. Use a fresh --outdir.")
                seen = sorted(set(prev[col].dropna().tolist()))
                cast = (lambda z: str(z)) if col == "case" else (lambda z: int(z))
                if [cast(s) for s in seen] != [cast(want)]:
                    ap.error(
                        f"{csv_path} already contains {col}={seen} but this run requests "
                        f"{col}={want!r}. Resume matches on (algo, run) only, so continuing "
                        f"would either skip every job or mix two configurations into one "
                        f"CSV -- and mixed rows make the pivot in _summarise ambiguous. "
                        f"Use a separate --outdir per configuration.")
            done = {(str(r.algo), int(r.run)) for r in prev.itertuples()}
            log(f"[resume] {len(done)} (algo, run) pairs already complete "
                f"for case={args.case} pop={args.pop}")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            log(f"[resume] could not read {csv_path}: {e}")

    # RUN-MAJOR, not algorithm-major. Algorithm-major finishes one algorithm before starting
    # the next, so stopping early leaves some algorithms with zero runs and the dataset is
    # statistically useless -- an all-or-nothing commitment. Run-major accumulates a BALANCED
    # design: at any moment the minimum run count across algorithms defines a complete,
    # analysable subset, so the campaign can be stopped whenever the evidence is sufficient.
    jobs = [(a, r) for r in range(args.runs) for a in args.algos
            if (a, r) not in done]
    if not jobs:
        # Nothing to simulate: fall through to the statistics section without spawning any
        # worker, so the stats and plots can be regenerated from an existing CSV in seconds
        # rather than paying ~85 s of library load per worker for nothing.
        log("[run] all jobs already complete -- regenerating statistics only")
        return _summarise(csv_path, case_dir, args.algos, checkpoints, None)
    log(f"[run] {len(jobs)} jobs | {args.evals} evals each | {args.workers} workers")
    # 0.416 s is the MEASURED steady-state cost per evaluation on the real circuit, after
    # the plot leak was fixed (before that it drifted from 0.70 to 1.10 s and any projection
    # was meaningless).
    log(f"[run] projected {len(jobs) * args.evals * 0.416 / 3600 / args.workers:.1f} h "
        f"wall at 0.416 s/eval")

    # ---- one problem (one ngspice process, PDK parsed once) per worker ----
    pool: queue.Queue = queue.Queue()
    problems = []
    t_load = time.perf_counter()
    for i in range(args.workers):
        p = sp.SpiceBGRProblem(sky130_lib=args.lib, case=args.case, timeout=args.timeout)
        problems.append(p)
        pool.put(p)
        log(f"[load] worker {i+1}/{args.workers} ready "
            f"({p._srv.load_seconds:.0f}s)")
    log(f"[load] all workers ready in {time.perf_counter()-t_load:.0f}s")

    write_lock = threading.Lock()
    new_file = not csv_path.exists()
    curves: dict[str, list[np.ndarray]] = {a: [] for a in args.algos}
    n_done = 0
    t0 = time.perf_counter()

    def work(algo: str, run_idx: int):
        p = pool.get()
        try:
            seed = args.seed + CASE_OFFSET + run_idx * 1000
            return one_run(p, algo, run_idx, seed, args.evals, args.pop, checkpoints,
                           args.case)
        finally:
            pool.put(p)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, a, r): (a, r) for a, r in jobs}
            for fut in as_completed(futs):
                algo, run_idx = futs[fut]
                try:
                    row, curve = fut.result()
                except Exception as e:  # noqa: BLE001
                    log(f"[FAIL] {algo} run {run_idx}: {type(e).__name__}: {e}")
                    continue
                curves[algo].append(curve)
                with write_lock:
                    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=fields,
                                           extrasaction="ignore")
                        if new_file:
                            w.writeheader()
                            new_file = False
                        w.writerow(row)
                n_done += 1
                el = time.perf_counter() - t0
                eta = el / n_done * (len(jobs) - n_done) / 3600
                log(f"[{n_done}/{len(jobs)}] {algo:4s} run {run_idx:2d}  "
                    f"best={row['best_fitness']:10.3f}  "
                    f"feas={row['is_feasible']}  "
                    f"pen={row['penalty']}  "
                    f"{row['runtime_sec']:6.1f}s   ETA {eta:.1f}h")
    finally:
        for p in problems:
            try:
                p.close()
            except Exception:  # noqa: BLE001
                pass

    return _summarise(csv_path, case_dir, args.algos, checkpoints, curves)


if __name__ == "__main__":
    sys.exit(main())
