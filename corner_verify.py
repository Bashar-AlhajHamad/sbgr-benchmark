"""
corner_verify.py -- re-simulate the campaign's final designs across process corners and supply,
WITHOUT re-optimization, and report how many survive.

WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL.

The paper's headline claim is about ROBUSTNESS: "ABC achieves superior robustness and feasibility
attainment". The optimization campaign runs at a single corner (tt, 1.8 V, 27 degC) because that
is the published problem. But optimization quality at one corner is not robustness in the sense
an analog designer means, so a case study that stops there does not test the paper's own thesis.

The need is concrete, not hypothetical. The objective was measured to reward a fragile trade:
designs that maximise rail rejection at low frequency do it by giving up rejection at high
frequency (92 dB at 100 Hz against 2.8 dB at 10 MHz on one measured design). There is no reason
to expect such a design to hold at ss or ff, and every reason to check.

Cost: ~17 core-hours against the campaign's ~17,460 -- one part in a thousand. At that ratio,
omitting it is an oversight rather than a trade-off.

THE MEASURE, FIXED IN ADVANCE (see the PRE-REGISTRATION block in the plan file, 2026-08-06):

    A design that is feasible at tt/1.8 V and infeasible at ANY of the 15 conditions counts as
    INFEASIBLE for the robustness measure.

That rule is declared before the data because it decides the outcome of the third pre-registered
hypothesis ("ABC does not lead on nominal fitness but its designs survive PVT better"). Choosing
it afterwards would decide the result instead of measuring it.

    python corner_verify.py --lib <sky130.lib.spice> --csv results/base/per_run_records.csv
    python corner_verify.py --lib ... --csv ... --job-index $SLURM_ARRAY_TASK_ID   # 15 tasks
    python corner_verify.py --csv ... --merge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spice"))

import spice_problem as sp   # noqa: E402

GRID = [(c, v) for c in sp.PVT_CORNERS for v in sp.PVT_SUPPLIES]
NOMINAL_CONDITION = ("tt", 1.80)


def designs_from_csv(csv_path: Path, case: str) -> pd.DataFrame:
    """Pull the final-best design vectors out of a campaign CSV.

    Uses the `x_*` columns, which exist only because the schema was extended after the first
    campaign shipped without them -- that campaign's winning circuits can never be re-simulated,
    which is precisely the failure this file depends on not repeating.
    """
    df = pd.read_csv(csv_path)
    names = [v[0] for v in sp.CASES[case][1]]
    cols = [f"x_{n}" for n in names]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} has no design vectors ({missing[:4]}...). This campaign predates the "
            f"schema that records them, so its designs cannot be re-simulated at all. Re-run it "
            f"with the current run_spice.py.")
    out = df[["algo", "run", "best_fitness", "is_feasible", "penalty"] + cols].copy()
    bad = out[cols].isna().any(axis=1)
    if bad.any():
        print(f"[warn] {int(bad.sum())} rows have an incomplete design vector "
              f"(simulation failures); they are carried through as not-verifiable")
    return out


def evaluate_condition(lib: str, case: str, corner: str, vdd: float,
                       designs: pd.DataFrame, timeout: float) -> pd.DataFrame:
    names = [v[0] for v in sp.CASES[case][1]]
    cols = [f"x_{n}" for n in names]
    p = sp.SpiceBGRProblem(sky130_lib=lib, case=case, timeout=timeout,
                           corner=corner, vdd=vdd, allow_off_nominal=True)
    print(f"[{corner} @ {vdd:.2f} V] ngspice ready ({p._srv.load_seconds:.0f}s), "
          f"{len(designs)} designs", flush=True)
    rows, t0 = [], time.perf_counter()
    try:
        for i, r in enumerate(designs.itertuples(index=False), 1):
            x = np.array([getattr(r, c) for c in cols], dtype=float)
            if not np.all(np.isfinite(x)):
                m = {"penalty": sp.N_SPECS, "is_feasible": 0, "PSRR_DB": sp.PSRR_FLOOR_DB,
                     "sim_failure": "no_design_vector"}
            else:
                m = p.metrics(x)
            rows.append({
                "corner": corner, "vdd": vdd, "algo": r.algo, "run": r.run,
                "feasible": int(m.get("is_feasible", 0)),
                "penalty": int(m.get("penalty", sp.N_SPECS)),
                "objective": float(m.get("PSRR_DB", np.nan)),
                "VREF": m.get("VREF", np.nan), "TC": m.get("TC", np.nan),
                "LOOP_GAIN_DB": m.get("LOOP_GAIN_DB", np.nan),
                "PHASE_MARGIN_DEG": m.get("PHASE_MARGIN_DEG", np.nan),
                "GAIN_MARGIN_DB": m.get("GAIN_MARGIN_DB", np.nan),
                "POWER_UW": m.get("POWER_UW", np.nan),
                "sim_failure": m.get("sim_failure") or "",
            })
            if i % 60 == 0:
                print(f"  {i}/{len(designs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    finally:
        p.close()
    return pd.DataFrame(rows)


def summarise(pvt: pd.DataFrame, algos: list[str]) -> pd.DataFrame:
    """Per-algorithm robustness, applying the pre-declared all-conditions rule."""
    nom = pvt[(pvt.corner == NOMINAL_CONDITION[0]) & (pvt.vdd == NOMINAL_CONDITION[1])]
    nom_ok = set(map(tuple, nom[nom.feasible == 1][["algo", "run"]].values))
    n_cond = pvt.groupby(["algo", "run"]).size().max()
    per = pvt.groupby(["algo", "run"]).agg(
        conditions=("feasible", "size"),
        n_feasible=("feasible", "sum"),
        worst_objective=("objective", "min"),
        worst_penalty=("penalty", "max"),
    ).reset_index()
    per["nominal_feasible"] = [int((a, r) in nom_ok)
                               for a, r in zip(per.algo, per["run"])]
    per["robust"] = ((per.n_feasible == per.conditions) & (per.nominal_feasible == 1)).astype(int)

    out = []
    for a in algos:
        s = per[per.algo == a]
        if not len(s):
            continue
        out.append({
            "algo": a,
            "n_designs": len(s),
            "nominal_feasible": int(s.nominal_feasible.sum()),
            "nominal_rate": float(s.nominal_feasible.mean()),
            # the headline: survives EVERY condition, the pre-declared rule
            "pvt_robust": int(s.robust.sum()),
            "pvt_robust_rate": float(s.robust.mean()),
            # how much of the nominal feasibility is lost to PVT
            "lost_to_pvt": int(s.nominal_feasible.sum() - s.robust.sum()),
            "mean_conditions_passed": float(s.n_feasible.mean()),
            "median_worst_objective": float(s.worst_objective.median()),
            "worst_case_worst_objective": float(s.worst_objective.min()),
        })
    return pd.DataFrame(out).sort_values("pvt_robust_rate", ascending=False), per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="merged per_run_records.csv of one case")
    ap.add_argument("--lib", help="path to sky130.lib.spice (not needed with --merge)")
    ap.add_argument("--case", default="base", choices=sorted(sp.CASES))
    ap.add_argument("--outdir", default=None, help="default <csv parent>/pvt")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--algos", nargs="+",
                    default=["ABC", "GWO", "FA", "PSO", "GA", "ACO"])
    ap.add_argument("--job-index", type=int, default=None,
                    help=f"run one of the {len(GRID)} (corner, supply) conditions and exit")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent / "pvt"
    outdir.mkdir(parents=True, exist_ok=True)
    parts = outdir / "conditions"
    parts.mkdir(exist_ok=True)

    if args.merge:
        files = sorted(parts.glob("*.csv"))
        if len(files) != len(GRID):
            print(f"[merge] REFUSING: {len(files)} condition files but {len(GRID)} expected; "
                  f"missing "
                  f"{[f'{c}@{v}' for c, v in GRID if not (parts / f'{c}_{v:.2f}.csv').exists()]}")
            return 1
        pvt = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        pvt.to_csv(outdir / "pvt_all_conditions.csv", index=False)
        summ, per = summarise(pvt, args.algos)
        per.to_csv(outdir / "pvt_per_design.csv", index=False)
        summ.to_csv(outdir / "pvt_summary.csv", index=False)
        print(f"\n{len(pvt)} (design, condition) simulations over {len(GRID)} conditions\n")
        print("pre-declared rule: feasible at tt/1.8 V but infeasible at ANY of the "
              f"{len(GRID)} conditions counts as INFEASIBLE\n")
        print(summ.to_string(index=False))
        # evaluate_condition writes "" on success (line 105), but the rows make a round trip
        # through CSV and pd.read_csv turns an empty field into NaN. `NaN != ""` is True, so the
        # naive test counted EVERY successful simulation as a failure and reported 2700 of 2700,
        # while groupby dropped the NaN keys and printed an empty breakdown. Cosmetic -- no
        # summary number was affected -- but an artefact that contradicts itself is one a reviewer
        # stops reading, so the test is now on emptiness rather than on inequality to "".
        fails = pvt[pvt.sim_failure.fillna("").astype(str).str.strip().ne("")]
        if len(fails):
            print(f"\nsimulation failures: {len(fails)} of {len(pvt)} "
                  f"({100*len(fails)/len(pvt):.2f} %)")
            print(fails.groupby(["corner", "sim_failure"]).size().to_string())
        print(f"\nwrote {outdir}")
        return 0

    if not args.lib:
        ap.error("--lib is required unless --merge")
    designs = designs_from_csv(csv_path, args.case)
    print(f"{len(designs)} designs from {csv_path}")

    todo = [GRID[args.job_index]] if args.job_index is not None else GRID
    if args.job_index is not None and not 0 <= args.job_index < len(GRID):
        ap.error(f"--job-index must be in [0, {len(GRID)})")
    for corner, vdd in todo:
        dst = parts / f"{corner}_{vdd:.2f}.csv"
        if dst.exists():
            print(f"[skip] {dst.name} already done")
            continue
        res = evaluate_condition(args.lib, args.case, corner, vdd, designs, args.timeout)
        tmp = dst.with_suffix(".partial")
        res.to_csv(tmp, index=False)
        tmp.replace(dst)
        print(f"[done] {dst.name}: {int(res.feasible.sum())}/{len(res)} feasible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
