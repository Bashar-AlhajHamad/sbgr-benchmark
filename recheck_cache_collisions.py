# -*- coding: utf-8 -*-
"""Re-measure, with the evaluator cache effectively disabled, the runs whose recorded feasibility
and recorded metrics disagree.

THE DEFECT THIS RESOLVES. `spice_problem.py` keys its memoisation on the design vector rounded to
nine significant digits (line 596). Two vectors that differ only below that are one cache entry, so
the second one is scored with the first one's metrics. Ten runs across the two current-mode
campaigns carry a populated `best_feasible_*` block while another recorded metric set shows a
constraint violation, and the record alone cannot say which measurement belongs to which vector.

WHY A RE-RUN IS THE ONLY ANSWER. Keeping the fitness in the record preserves the ordering the
cached evaluator actually produced -- that part is sound and is what the campaign optimised against.
It does NOT establish that the final design vectors, re-simulated on their own, reproduce those
metrics. Only measuring them again, without the cache, does that.

WHAT IS AND IS NOT DISABLED. `cache_size=1` is used rather than 0: the eviction branch pops before
inserting, so a size of 0 raises StopIteration on the first miss. With size 1 each new vector evicts
the previous one, so no two distinct vectors can ever share an entry -- which is exactly the
collision being tested for.

THE DISTINCTION THIS SCRIPT MAKES, and which the paper must now make too:
  * feasibility the OPTIMIZER reached  -- what the run recorded, cache included
  * feasibility VERIFIED on the final vector -- what this script measures

    python recheck_cache_collisions.py --lib <sky130.lib.spice>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# The repository keeps this script beside results/; the cluster keeps it in code/ with results/ one
# level up. Resolve against whichever layout actually holds the data rather than assuming either.
ROOT = HERE if (HERE / "results").is_dir() else HERE.parent
sys.path.insert(0, str(HERE / "spice"))
import spice_problem as sp                                             # noqa: E402

# The ten runs, identified from the shipped records: a populated best_feasible block beside
# is_feasible == 0. Two on the published-specification core, eight on the recalibrated one.
SUSPECT = {
    "banba":     ("results/banba_150k/per_run_records.csv",     [("PSO", 9), ("PSO", 28)]),
    "banba_cal": ("results/banba_cal_150k/per_run_records.csv",
                  [("ACO", 2), ("ACO", 5), ("ACO", 9), ("ACO", 15), ("ACO", 29),
                   ("PSO", 11), ("PSO", 13), ("PSO", 27)]),
}
VIOL = ("viol_vref", "viol_tc", "viol_loop_gain", "viol_phase_margin",
        "viol_gain_margin", "viol_power")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default="results/cache_recheck.csv")
    args = ap.parse_args()

    rows = []
    for case, (csv, suspects) in SUSPECT.items():
        path = ROOT / csv
        if not path.exists():
            print(f"FATAL: {path} not found")
            return 1
        d = pd.read_csv(path)
        names = [v[0] for v in sp.CASES[case][1]]

        # cache_size=1 makes a collision impossible: every new vector evicts the previous entry.
        p = sp.SpiceBGRProblem(sky130_lib=args.lib, case=case, timeout=args.timeout, cache_size=1)
        try:
            print(f"\n=== {case}  ({len(suspects)} runs, cache disabled) ===", flush=True)
            for algo, run in suspects:
                r = d[(d.algo == algo) & (d.run == run)]
                if r.empty:
                    print(f"  {algo}/run{run}: NOT FOUND in {csv}")
                    continue
                r = r.iloc[0]
                x = np.array([float(r[f"x_{n}"]) for n in names])
                m = p.metrics(x)
                viol = [k[5:] for k in VIOL if int(m.get(k, 0)) == 1]
                ok = int(m.get("is_feasible", 0)) == 1
                rows.append({
                    "case": case, "algo": algo, "run": int(run),
                    "recorded_is_feasible": int(r["is_feasible"]),
                    "recorded_best_feasible_found": int(r.get("best_feasible_found", -1)),
                    "recorded_best_fitness": float(r["best_fitness"]),
                    "remeasured_is_feasible": int(ok),
                    "remeasured_PSRR_DB": float(m.get("PSRR_DB", float("nan"))),
                    "remeasured_violations": "+".join(viol) or "none",
                    "agrees_with_record": int(int(r["is_feasible"]) == int(ok)),
                })
                print(f"  {algo}/run{run:<3d} recorded is_feasible={int(r['is_feasible'])}  "
                      f"re-measured={int(ok)}  violations={'+'.join(viol) or 'none'}", flush=True)
        finally:
            p.close()

    out = pd.DataFrame(rows)
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / args.out, index=False)

    n_agree = int(out.agrees_with_record.sum())
    print("\n" + "=" * 78)
    print(f"  {len(out)} runs re-measured without the cache")
    print(f"  agree with the recorded feasibility flag : {n_agree}/{len(out)}")
    print(f"  disagree                                 : {len(out) - n_agree}/{len(out)}")
    if n_agree == len(out):
        print("\n  Every final vector reproduces its recorded feasibility. The cache collision")
        print("  affected intermediate scoring only; no reported success rate changes.")
    else:
        print("\n  At least one final vector does NOT reproduce its recorded feasibility.")
        print("  The affected success rates and any statistic derived from them must be updated,")
        print("  and the paper must separate optimizer-reached from verified feasibility.")
    print(f"\n  written: {args.out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
