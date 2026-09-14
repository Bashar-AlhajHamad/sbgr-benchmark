# -*- coding: utf-8 -*-
"""Write a corrected PVT summary beside each shipped one, without touching the original.

The shipped `pvt_summary.csv` reports `nominal_feasible` values that contradict the optimizer's own
record for 173 of 540 designs -- ABC appears as 0 of 30 feasible in hard and highdim where
`per_run_records.csv` says 30 of 30. 165 of the 173 are simulation failures scored identically to a
design that violates all six specifications, so the column is measuring "did the re-simulation
return anything" rather than "is the design feasible".

The original is left in place: it is the record of what was actually run, and overwriting it would
destroy the evidence for E7. This writes `pvt_summary_corrected.csv` alongside, carrying both
scorings explicitly rather than silently picking one:

  nominal_feasible          taken from per_run_records (the optimizer's own verdict, authoritative
                            at its own operating point)
  sim_failures / n_sims     the size of the confound, per algorithm
  mean_cond_passed_scored   failures counted as failed conditions -- the shipped convention
  mean_cond_passed_usable   failures excluded -- with n_usable so the reader sees when it is thin
  pvt_robust                designs passing all 15; identical under both conventions, and the only
                            PVT figure that survives every scoring choice

    python pvt_summary_fix.py                  # reads results/ beside this script
    python pvt_summary_fix.py --root <dir>     # if the campaign data lives elsewhere
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolved from this file's own location, not from the working directory and not from an absolute
# path on one machine. The campaign directories ship in results/ next to this script, so the default
# is correct for anyone who has the archive.
_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default=str(Path(__file__).resolve().parent / "results"),
                 help="directory holding the campaign folders (default: results/ beside this file)")
_args, _ = _ap.parse_known_args()
R = Path(_args.root)
if not R.is_dir():
    raise SystemExit(f"no such directory: {R}\n"
                     f"pass --root <dir> pointing at the folder that holds "
                     f"base_150k/, hard_150k/ and highdim_220k/")
ALG = ["ABC", "GWO", "FA", "PSO", "GA", "ACO"]
CAMP = ["base_150k", "hard_150k", "highdim_220k"]
# Five process corners x three supplies. Kept as a name so that changing the harness grid
# cannot leave this summary silently reporting robustness against the wrong denominator.
N_COND = 15

COLS = ["algo", "n_designs", "nominal_feasible", "nominal_rate",
        "n_sims", "sim_failures", "sim_failure_rate",
        "pvt_robust", "pvt_robust_rate",
        "mean_cond_passed_scored", "mean_cond_passed_usable", "n_usable",
        "cond_feasible_rate_usable", "median_worst_objective", "worst_case_worst_objective"]


def build(out: str) -> pd.DataFrame:
    a = pd.read_csv(R / out / "pvt" / "pvt_all_conditions.csv")
    d = pd.read_csv(R / out / "per_run_records.csv")
    a = a.assign(failed=a.sim_failure.fillna("").astype(str).str.strip().ne(""))

    rows = []
    for alg in ALG:
        s = a[a.algo == alg]
        dd = d[d.algo == alg]
        npass = s.groupby("run").feasible.sum()
        ok = s[~s.failed]
        okn = len(ok)
        cond_u = ok.groupby("run").feasible.mean()
        worst = s.groupby("run").objective.min()
        rows.append({
            "algo": alg,
            "n_designs": int(dd.shape[0]),
            "nominal_feasible": int(dd.is_feasible.sum()),
            "nominal_rate": round(float(dd.is_feasible.mean()), 6),
            "n_sims": int(len(s)),
            "sim_failures": int(s.failed.sum()),
            "sim_failure_rate": round(float(s.failed.mean()), 6),
            "pvt_robust": int((npass == N_COND).sum()),
            "pvt_robust_rate": round(float((npass == N_COND).mean()), 6),
            "mean_cond_passed_scored": round(float(npass.mean()), 4),
            "mean_cond_passed_usable": (round(float(N_COND * ok.feasible.mean()), 4) if okn else np.nan),
            "n_usable": okn,
            "cond_feasible_rate_usable": (round(float(ok.feasible.mean()), 6) if okn else np.nan),
            "median_worst_objective": round(float(worst.median()), 4),
            "worst_case_worst_objective": round(float(worst.min()), 4),
        })
    return pd.DataFrame(rows)[COLS]


if __name__ == "__main__":
    for out in CAMP:
        dst = R / out / "pvt" / "pvt_summary_corrected.csv"
        t = build(out)
        t.to_csv(dst, index=False)
        old = pd.read_csv(R / out / "pvt" / "pvt_summary.csv").set_index("algo")
        print(f"\n{'='*94}\n{out}  ->  {dst.name}\n{'='*94}")
        print(t.to_string(index=False))
        print(f"\n  nominal_feasible  shipped : "
              + "  ".join(f"{x} {int(old.loc[x,'nominal_feasible']):2d}" for x in ALG))
        print(f"  nominal_feasible  corrected: "
              + "  ".join(f"{x} {int(t.set_index('algo').loc[x,'nominal_feasible']):2d}" for x in ALG))
        print(f"  pvt_robust unchanged at 0 for every algorithm: "
              f"{bool((t.pvt_robust == 0).all())}")
