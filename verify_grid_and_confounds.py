# -*- coding: utf-8 -*-
"""Is the base@20,000 result a measurement of the circuit, or of the frequency grid?

THE QUESTION. The campaign's objective is the worst-case rail rejection over 10 Hz - 10 MHz,
and it is evaluated as `meas ac m_wc MAX vm` on `ac dec 8 10 1e7` -- 49 logarithmically spaced
points (spice_problem.py:278, :536). A minimax objective on a DISCRETE grid is exploitable: an
optimizer with 20,000 evaluations can place a rejection notch BETWEEN two grid points and be
paid for rejection the circuit does not have. Nothing in the deck forbids it and nothing in the
campaign would report it.

WHY IT IS NOT PARANOIA HERE. The campaign reached 55.84 dB where the reference design scores
23.85 dB -- +32 dB of worst-case rejection from amplifier sizing alone, with the bandgap core
fixed at its trim point. Five of the six algorithms landed within 0.55 dB of each other and ACO
clustered at 55.62 +/- 0.26 dB over thirty runs. A tight cluster just under a common bound is
exactly what a genuine physical ceiling looks like AND exactly what grid-fitting looks like.
The two are not distinguishable from the campaign artefact, and this is the third time in this
project that a plausible objective turned out to reward something other than what it named
(single-frequency PSRR, then the pipe desynchronisation).

HOW THIS SEPARATES THEM. Every design is re-measured through the EXACT campaign code path --
`SpiceBGRProblem.metrics`, same deck, same corner, same supply, same quantisation -- with the
single AC sweep density changed from 8 to 256 points per decade. Nothing else differs.

  * worst case stable from 8 to 256 pts/decade  -> the objective is a property of the circuit,
                                                   the campaign stands, `hard` may launch
  * worst case collapses as the grid refines    -> the campaign optimized the grid; the
                                                   objective must be re-specified and base re-run

The tolerance is declared HERE, before the numbers exist, for the same reason the thresholds
were frozen before the probe: a bound chosen after seeing the data decides the result instead of
measuring it.

TWO OTHER THINGS THAT WOULD INVALIDATE THE RANKING RATHER THAN THE OBJECTIVE, checked as well:

  1. The merge reported 48,477 timeouts over 180 runs (1.35 % of evaluations). A timeout scores
     the failure floor, so an algorithm that explores into the hanging region is punished for
     exploring. If the timeouts are spread evenly the ranking is unaffected; if they are
     concentrated in one or two algorithms, part of the ranking is a property of the EVALUATOR's
     fragility and must be disclosed as such rather than read as an algorithm result. In
     --job-index mode each job is its own process, so `sim_failures_worker_cum` is that run's own
     tally (run_spice.py:79-84).

  2. Each recorded best design must re-simulate to its recorded metrics. If it does not, the row
     was written from a desynchronised pipe and the campaign is void regardless of everything
     else.

Usage on TRUBA:

    cd /arf/scratch/${USER}/sky130-bgr/code && source ../env.sh
    $SKY130_PY verify_grid_and_confounds.py --lib $SKY130_LIB --results ../results/base
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "spice"))
import spice_problem as sp                                             # noqa: E402

# Points per decade over 10 Hz - 10 MHz. 8 is the campaign's; 256 is 1537 points, dense enough
# that a notch narrow enough to hide would also be too narrow to matter physically.
DENSITIES = (8, 16, 32, 64, 128, 256)
CAMPAIGN_DENSITY = 8

# ---- declared before the measurement --------------------------------------------------------
GRID_TOLERANCE_DB = 1.0     # a best design may lose at most this much when the grid is refined
TIMEOUT_SKEW_FACTOR = 3.0   # worst algorithm's timeout rate over the best algorithm's

# Re-simulation tolerance, and it depends on WHERE this runs. On the machine that produced the
# rows the evaluator is deterministic and re-simulation is bit-identical, so anything above noise
# means a poisoned row. Off that machine the comparison also spans a compiler and a libm: the
# 2026-08-06 cross-build calibration measured 510 of 512 designs classifying identically between
# Windows/MSVC and Linux/gcc, with loop gain moving by ~0.05 dB on the two that differed. Using
# the tight bound cross-platform reports twelve failures for a difference that was measured and
# published a week earlier, which is a false alarm, not a finding.
# Measured, not assumed. Re-simulating the twelve best/median designs of the 2,000-evaluation
# rehearsal on Windows reproduces PSRR_WC_DB to |delta| <= 0.211 dB, and LOOP_GAIN_DB drifts by
# up to 1.11 dB. The evaluator is deterministic WITHIN one session -- the same design evaluated
# first in a fresh session, again after six others, and again back-to-back gives 53.748790 dB
# every time, history sensitivity 0.00e+00 dB -- so this is the compiler and libm, not the pipe
# and not the sweep. At 1e-4 dB every design fails on a campaign that is sound, and it fails with
# the SAME signature as the pipe desynchronisation, which is the one confusion this script exists
# to prevent.
REPRO_TOLERANCE_SAME_DB = 1e-4
REPRO_TOLERANCE_CROSS_DB = 0.50

# The published surrogate ranking (manuscript tab:core_quant_results), for the transfer question.
PUBLISHED_RANKS = {"ABC": 2.300, "PSO": 3.000, "GA": 3.033,
                   "ACO": 3.100, "FA": 4.633, "GWO": 4.933}

fails: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def n_of(s, k: str) -> int:
    m = re.search(rf"\b{k}=(\d+)", str(s) if pd.notna(s) else "")
    return int(m.group(1)) if m else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--results", required=True, help="e.g. ../results/base")
    ap.add_argument("--case", default="base")
    # Generous: this is a diagnostic, not a campaign, and a dense sweep must never be scored as a
    # hang. Stage 0 measured the per-evaluation cost as FLAT in analysis size, so 256 pts/decade
    # should cost what 8 does -- but a limit that assumes that would be assuming the answer.
    ap.add_argument("--timeout", type=float, default=180.0)
    # Set this when re-simulating rows produced on a DIFFERENT build (e.g. checking a TRUBA
    # campaign from Windows). It only widens the reproduction tolerance; it does not touch the
    # grid check, which must pass identically everywhere.
    ap.add_argument("--cross-build", action="store_true")
    args = ap.parse_args()
    repro_tol = (REPRO_TOLERANCE_CROSS_DB if args.cross_build
                 else REPRO_TOLERANCE_SAME_DB)

    res = Path(args.results)
    d = pd.read_csv(res / "per_run_records.csv")
    # --case selects the THRESHOLDS the re-simulation scores against, and it defaults to "base"
    # while --results is a free path. Pointing --results at ../results/hard and forgetting
    # --case hard re-scores hard's designs against base's thresholds and prints penalties for a
    # specification that campaign was never run under -- silently, because base and hard share
    # the same seven variable names and differ ONLY in CONSTRAINTS. `hard` is the very next thing
    # this script gates, so the trap is one command away.
    cases_in_file = set(d["case"].astype(str))
    if cases_in_file != {args.case}:
        raise SystemExit(
            f"--case {args.case!r} does not match the artefact, whose case column holds "
            f"{sorted(cases_in_file)}. Re-run with the matching --case: otherwise every "
            f"penalty below is scored against the wrong thresholds.")
    missing_pub = sorted(set(d.algo) - set(PUBLISHED_RANKS))
    if missing_pub:
        raise SystemExit(f"no published rank recorded for {missing_pub}; add them to "
                         f"PUBLISHED_RANKS before this can report a transfer correlation.")
    names = [v[0] for v in sp.CASES[args.case][1]]
    # The binding threshold, read from the artefact rather than written as a literal. It was a
    # literal 50.0 until 2026-08-12, which is base's LoopGain_min: on `hard` (52.2) that printed
    # every cross-build flip margin 2.2 dB too large, turning nine designs sitting ON the bound
    # (0.0002-0.0493 dB) into nine designs sitting comfortably inside it -- i.e. it hid the very
    # explanation the disclosure exists to give. Same class of error as the --case trap guarded
    # above, which is why this now comes from the data.
    lg_mins = set(d.c_LoopGain_min.round(9))
    if len(lg_mins) != 1:
        raise SystemExit(f"c_LoopGain_min is not single-valued in the artefact: {sorted(lg_mins)}; "
                         f"the margin disclosure below assumes one threshold per campaign.")
    lg_min = float(next(iter(lg_mins)))
    if abs(lg_min - sp.CASES[args.case][3]["LoopGain_min"]) > 1e-9:
        raise SystemExit(
            f"artefact c_LoopGain_min={lg_min} disagrees with CASES[{args.case!r}] "
            f"LoopGain_min={sp.CASES[args.case][3]['LoopGain_min']}; the rows were scored against "
            f"a different specification than this script would re-score them against.")

    print("=" * 92)
    print(f"0. THE ARTEFACT   {res / 'per_run_records.csv'}")
    print("=" * 92)
    print(f"  {len(d)} rows, {d.algo.nunique()} algorithms, {d.run.nunique()} runs, "
          f"case={set(d.case)}, pop={set(d['pop'])}, budget={set(d.eval_budget)}")
    print(f"  objective={set(d.psrr_def)}   evaluator_recovers_timeouts="
          f"{set(d.evaluator_recovers_timeouts)}")
    ck("every row is the objective this analysis assumes",
       set(d.psrr_def) == {"PSRR_worst_case_10Hz_10MHz"})
    ck("PSRR_DB is the band worst case, not the 100 Hz diagnostic",
       bool(np.allclose(d.PSRR_DB, d.PSRR_WC_DB, atol=1e-9, equal_nan=True)),
       "if this fails, every fitness in the campaign is the wrong column")
    # The whole timeout argument in section 1 assumes a timeout cost one PDK reload. On a
    # pre-2026-08-07 row it instead voided everything after it, and the two are indistinguishable
    # from the counters alone.
    ck("every row came from the timeout-recovering evaluator",
       set(d.evaluator_recovers_timeouts) == {1}, f"{sorted(set(d.evaluator_recovers_timeouts))}")

    # ---------------------------------------------------------------- 1. failure attribution
    print("\n" + "=" * 92)
    print("1. IS THE RANKING CONTAMINATED BY WHERE THE SIMULATOR HANGS?")
    print("=" * 92)
    d["_timeout"] = d.sim_failures_worker_cum.map(lambda v: n_of(v, "timeout"))
    d["_deck"] = d.sim_failures_worker_cum.map(lambda v: n_of(v, "deck_error"))
    d["_meas"] = d.sim_failures_worker_cum.map(lambda v: n_of(v, "measure_failed"))
    print(f"  {'algo':5s} {'timeouts':>10s} {'% of evals':>11s} {'deck':>7s} {'meas_fail':>10s} "
          f"{'feasible':>9s} {'median dB':>10s} {'worst run dB':>13s} {'std':>7s}")
    rates = {}
    for a in sorted(set(d.algo)):
        s = d[d.algo == a]
        ev = float(s.actual_evals.sum())
        rate = 100.0 * s._timeout.sum() / ev if ev else float("nan")
        rates[a] = rate
        # dB columns come from PSRR_DB, NOT from -best_fitness: fitness carries the 1000*penalty
        # term, so an infeasible final solution prints something like -970.95 under a decibel
        # header. base@20k happens to have no infeasible finals; `hard` is designed so that it
        # will, and this is the script that gates it.
        print(f"  {a:5s} {int(s._timeout.sum()):10d} {rate:10.2f} % {int(s._deck.sum()):7d} "
              f"{int(s._meas.sum()):10d} {int(s.is_feasible.sum()):6d}/{len(s):<3d}"
              f"{s.PSRR_DB.median():10.2f} {s.PSRR_DB.min():13.2f} "
              f"{s.PSRR_DB.std():7.2f}")
    lo, hi = min(rates.values()), max(rates.values())
    # Against the MEAN, not against the best. An algorithm that never times out is the best
    # possible outcome, and dividing by its zero rate makes the ratio ~1e9 and condemns the
    # campaign for it: the guard would fire hardest exactly when the evaluator behaved best.
    mean_rate = sum(rates.values()) / len(rates)
    ratio = hi / mean_rate if mean_rate > 0 else 1.0
    ck("timeouts are not concentrated in one algorithm", ratio <= TIMEOUT_SKEW_FACTOR,
       f"worst {hi:.2f} % vs best {lo:.2f} % vs mean {mean_rate:.2f} %  "
       f"(worst/mean {ratio:.1f}x, allowed {TIMEOUT_SKEW_FACTOR:.0f}x)")
    ck("no deck errors anywhere", int(d._deck.sum()) == 0, f"{int(d._deck.sum())}")

    # measured ranks, recomputed here rather than trusted from the summary
    piv = d.pivot_table(index="run", columns="algo", values="best_fitness")
    meas = piv.rank(axis=1, method="average").mean().to_dict()
    print("\n  measured average rank vs the surrogate's published rank:")
    order = sorted(meas, key=lambda a: meas[a])
    for a in order:
        print(f"    {a:5s} SPICE {meas[a]:.3f}   published {PUBLISHED_RANKS[a]:.3f}")
    algs = sorted(meas)
    rs = pd.Series([meas[a] for a in algs]).rank()
    rp = pd.Series([PUBLISHED_RANKS[a] for a in algs]).rank()
    rho = float(np.corrcoef(rs, rp)[0, 1])
    print(f"\n  Spearman rho(published ordering, SPICE ordering) = {rho:+.3f}")
    print("  This number is the case study's headline. It is reported whatever it is; the")
    print("  pre-registration fixed the reading of all four outcomes before any data existed.")

    # ---------------------------------------------------------------- 2. designs to re-measure
    print("\n" + "=" * 92)
    print("2. RE-MEASURING THE WINNING DESIGNS")
    print("=" * 92)
    designs: list[tuple[str, np.ndarray, pd.Series | None]] = []
    for a in sorted(set(d.algo)):
        s = d[d.algo == a].sort_values("best_fitness")
        for tag, row in (("best", s.iloc[0]), ("median", s.iloc[len(s) // 2])):
            designs.append((f"{a}/{tag}/run{int(row.run)}",
                            np.array([float(row[f"x_{n}"]) for n in names]), row))
    designs.append(("REFERENCE(nominal)",
                    np.array([float(sp.CASES[args.case][2][n]) for n in names]), None))
    print(f"  {len(designs)} designs x {len(DENSITIES)} sweep densities "
          f"= {len(designs) * len(DENSITIES)} full evaluations")

    p = sp.SpiceBGRProblem(sky130_lib=args.lib, case=args.case, timeout=args.timeout)
    saved_sweep = sp._SWEEP_AC
    table: dict[str, dict[int, float]] = {lbl: {} for lbl, _, _ in designs}
    repro: list[tuple] = []
    oob: dict[str, dict[str, float]] = {}
    ok_all = True
    try:
        for dec in DENSITIES:
            sp._SWEEP_AC = f"ac dec {dec} 10 1e7"
            n_pts = dec * 6 + 1
            p._cache.clear()          # the key is the design, not the sweep -- must not reuse
            print(f"\n  --- ac dec {dec} 10 1e7   ({n_pts} points) ---")
            for lbl, x, row in designs:
                m = p.metrics(x)
                if not m.get("sim_ok"):
                    ok_all = False
                    print(f"    {lbl:26s}  SIM FAILED: {m.get('sim_failure')} "
                          f"{m.get('sim_detail', '')}")
                    continue
                table[lbl][dec] = float(m["PSRR_WC_DB"])
                print(f"    {lbl:26s}  worst {m['PSRR_WC_DB']:8.3f} dB   "
                      f"@100Hz {m['PSRR_100HZ_DB']:8.3f} dB   penalty {int(m['penalty'])}")
                if dec == CAMPAIGN_DENSITY and row is not None:
                    repro.append((lbl, float(row["PSRR_WC_DB"]), float(m["PSRR_WC_DB"]),
                                  int(row["penalty"]), int(m["penalty"]),
                                  float(row["LOOP_GAIN_DB"]), float(m["LOOP_GAIN_DB"])))

        # ---- what happens OUTSIDE the specified band ----------------------------------------
        # The objective is bounded at 10 MHz, so nothing constrains the design above it. A
        # reviewer will widen the band in one simulation, so measure it here first and report it
        # rather than be shown it. This is a disclosure, not a gate: all six algorithms are
        # scored on the same in-band objective, so out-of-band behaviour cannot bias the ranking.
        for tag, top in (("10 Hz-10 MHz", "1e7"), ("10 Hz-100 MHz", "1e8"),
                         ("10 Hz-1 GHz", "1e9")):
            sp._SWEEP_AC = f"ac dec 8 10 {top}"
            p._cache.clear()
            for lbl, x, row in designs:
                if "/median/" in lbl:
                    continue
                m = p.metrics(x)
                if m.get("sim_ok"):
                    oob.setdefault(lbl, {})[tag] = float(m["PSRR_WC_DB"])
    finally:
        sp._SWEEP_AC = saved_sweep
        p.close()
    ck("every design re-simulated at every density", ok_all)

    print(f"\n  reproduction at the campaign density (tolerance {repro_tol} dB, "
          f"{'cross-build' if args.cross_build else 'same-build'}):")
    print(f"    {'design':26s} {'recorded':>10s} {'now':>10s} {'d(PSRR)':>9s} "
          f"{'LG rec':>8s} {'LG now':>8s} {'d(LG)':>8s}  penalty")
    worst_d, flips = 0.0, []
    for lbl, rec, now, pen_r, pen_n, lg_r, lg_n in repro:
        worst_d = max(worst_d, abs(now - rec))
        flag = "" if pen_r == pen_n else f"  <-- {pen_r} -> {pen_n} FLIPPED"
        if pen_r != pen_n:
            flips.append((lbl, lg_r - lg_min))
        print(f"    {lbl:26s} {rec:10.4f} {now:10.4f} {now - rec:9.4f} "
              f"{lg_r:8.4f} {lg_n:8.4f} {lg_n - lg_r:8.4f}  {pen_r}->{pen_n}{flag}")
    ck("recorded designs re-simulate to their recorded objective",
       worst_d <= repro_tol, f"worst delta {worst_d:.4f} dB")
    if flips:
        print(f"\n  {len(flips)} design(s) changed feasibility across builds. Loop-gain margin")
        print(f"  each had on the campaign build: "
              + ", ".join(f"{l.split('/')[0]} {m:+.4f} dB" for l, m in flips))
        print("  A design whose margin is smaller than the cross-build spread is feasible on one")
        print("  simulator build and infeasible on another. That is a property of WHERE the")
        print("  design sits, not of the campaign.")

    # ---------------------------------------------------------------- 3. the verdict
    print("\n" + "=" * 92)
    print("3. GRID DEPENDENCE -- the decisive check")
    print("=" * 92)
    head = "  " + f"{'design':26s}" + "".join(f"{f'dec {k}':>11s}" for k in DENSITIES) + \
           f"{'loss':>9s}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    worst_loss, worst_lbl = 0.0, "no design lost anything on refinement"
    for lbl, _, _ in designs:
        r = table[lbl]
        if CAMPAIGN_DENSITY not in r or DENSITIES[-1] not in r:
            print(f"  {lbl:26s}  incomplete")
            continue
        loss = r[CAMPAIGN_DENSITY] - r[DENSITIES[-1]]
        if loss > worst_loss:
            worst_loss, worst_lbl = loss, lbl
        print(f"  {lbl:26s}" + "".join(f"{r.get(k, float('nan')):11.3f}" for k in DENSITIES)
              + f"{loss:9.3f}")
    print(f"\n  Worst loss on refinement: {worst_loss:.3f} dB  ({worst_lbl})")
    print(f"  `loss` = worst-case PSRR on the campaign's 49-point grid MINUS the worst case on")
    print(f"  {DENSITIES[-1] * 6 + 1} points. Positive means the coarse grid over-credited the design.")
    ck(f"the objective is grid-independent (loss <= {GRID_TOLERANCE_DB} dB, declared in advance)",
       worst_loss <= GRID_TOLERANCE_DB, f"worst {worst_loss:.3f} dB on {worst_lbl}")

    # ---------------------------------------------------------------- 4. outside the band
    if oob:
        print("\n" + "=" * 92)
        print("4. OUTSIDE THE SPECIFIED BAND -- a disclosure, not a gate")
        print("=" * 92)
        tags = ["10 Hz-10 MHz", "10 Hz-100 MHz", "10 Hz-1 GHz"]
        print("  " + f"{'design':26s}" + "".join(f"{t:>16s}" for t in tags))
        for lbl in sorted(oob):
            print(f"  {lbl:26s}" + "".join(f"{oob[lbl].get(t, float('nan')):16.3f}"
                                           for t in tags))
        print("\n  The objective stops at 10 MHz, so nothing constrains the design above it. If")
        print("  the in-band winner falls below the reference design once the band is widened,")
        print("  the improvement is bandwidth extension to the stated limit -- which is what was")
        print("  asked for -- and Section 6 must say so rather than claim broadband superiority.")

    # ---------------------------------------------------------------- 5. constraint margin
    print("\n" + "=" * 92)
    print("5. WHERE THE OPTIMUM SITS -- margin to the binding constraint")
    print("=" * 92)
    lgm = d.LOOP_GAIN_DB - d.c_LoopGain_min
    print(f"  {'algo':5s} {'SPICE rank':>11s} {'median margin dB':>17s} {'runs < 0.05 dB':>16s}")
    ranks = d.pivot_table(index="run", columns="algo",
                          values="best_fitness").rank(axis=1).mean()
    for a in sorted(set(d.algo), key=lambda z: ranks[z]):
        s = lgm[d.algo == a]
        print(f"  {a:5s} {ranks[a]:11.3f} {s.median():17.4f} "
              f"{int((s < 0.05).sum()):13d}/{len(s)}")
    rr = pd.Series({a: ranks[a] for a in sorted(set(d.algo))})
    mm = lgm.groupby(d.algo).median()
    print(f"\n  Spearman rho(rank, margin) = "
          f"{float(np.corrcoef(rr.rank(), mm[rr.index].rank())[0, 1]):+.4f}   "
          f"positive = the better an algorithm ranks, the less margin it keeps")
    print("  A design with less margin than the cross-build spread measured above is feasible on")
    print("  one simulator build and infeasible on another. That is the robustness measure the")
    print("  pre-registration fixed in advance, and the PVT pass is what settles it.")

    print("\n" + "=" * 92)
    budget = sorted(set(d.eval_budget))
    tag = f"{args.case}@{budget[0]:,}" if len(budget) == 1 else f"{args.case}"
    if fails:
        print(f"VERDICT: {tag} IS NOT SOUND.  " + "; ".join(fails))
    else:
        print(f"VERDICT: {tag} is sound on all three counts.")
    print("=" * 92)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
