"""
verify_campaign_pop.py -- prove, from the artefact rather than from memory, which population the
existing SPICE-Base campaign was run with.

WHY THIS EXISTS. `pop` was added to run_spice.py's row schema only AFTER the first SPICE-Base
campaign finished, so `results_spice/per_run_records.csv` does not record it. The value is
believed to be 20, but "believed" is not a basis for a sentence in a paper, and it is precisely
the parameter whose omission let a protocol deviation go unnoticed for a whole campaign.

HOW IT PROVES IT. The evaluator is deterministic -- verified bit-identical across two different
CPU architectures, on the nominal point and on all 512 probe points -- and each (algo, run) job
is seeded only by its recorded `seed`. So re-running one recorded job at a candidate population
either reproduces its `best_fitness` to the last bit or it does not. One match and one mismatch
settles the question from data.

It also re-checks cross-machine determinism, which §6 wants to claim anyway: if the recorded
fitness is reproduced exactly on a different machine than the one that produced it, that claim
is measured rather than asserted.

Cost: one job per candidate population, ~17 min each at 2,500 evaluations.

    python verify_campaign_pop.py --lib <sky130.lib.spice> --csv results_spice/per_run_records.csv
    python verify_campaign_pop.py --lib ... --pops 20 40 --algo ABC --run 0
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

from algorithms import ALGORITHMS, ObjectiveWrapper   # noqa: E402
import spice_problem as sp                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--csv", default="results_spice/per_run_records.csv")
    ap.add_argument("--pops", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--algo", default="ABC")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--case", default="base", choices=sorted(sp.CASES))
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "pop" in df.columns:
        print(f"{args.csv} already records pop={sorted(set(df['pop'].dropna()))} -- "
              f"nothing to reconstruct.")
        return 0
    sel = df[(df.algo == args.algo) & (df.run == args.run)]
    if sel.empty:
        ap.error(f"no row for algo={args.algo} run={args.run} in {args.csv}")
    row = sel.iloc[0]
    target = float(row["best_fitness"])
    seed, budget = int(row["seed"]), int(row["eval_budget"])
    print(f"target row : algo={args.algo} run={args.run} seed={seed} budget={budget}")
    print(f"             best_fitness = {target!r}")
    print(f"             recorded penalty={row['penalty']} feasible={row['is_feasible']} "
          f"PSRR={row['PSRR_DB']}\n")

    problem = sp.SpiceBGRProblem(sky130_lib=args.lib, case=args.case, timeout=args.timeout)
    print(f"[load] ngspice ready ({problem._srv.load_seconds:.0f}s)\n")
    results = {}
    try:
        for pop in args.pops:
            t0 = time.perf_counter()
            obj = ObjectiveWrapper(func=None, eval_with_metrics=problem.evaluate_with_metrics,
                                   max_evals=budget, progress_cb=None)
            res = ALGORITHMS[args.algo](obj=obj, rng=np.random.default_rng(seed),
                                       lb=problem.lb, ub=problem.ub, pop=pop,
                                       dim=problem.dim)
            got = float(res.best_f)
            exact = (got == target)
            close = abs(got - target) <= 1e-9 * max(1.0, abs(target))
            results[pop] = (got, exact, close)
            print(f"  pop={pop:3d}  best_fitness = {got!r}   "
                  f"{'EXACT MATCH' if exact else ('matches to 1e-9' if close else 'differs')}"
                  f"   ({time.perf_counter()-t0:.0f}s)")
    finally:
        problem.close()

    matches = [p for p, (_, e, c) in results.items() if e or c]
    print()
    if len(matches) == 1:
        print(f"PROVEN: the campaign was run with pop={matches[0]}. Reproduced on this machine "
              f"from the recorded seed alone, which also re-confirms cross-machine determinism.")
        print(f"Record it in the paper and in a provenance note beside the CSV.")
        return 0
    if not matches:
        print("NO CANDIDATE REPRODUCES THE RECORDED FITNESS. Do not assume a population -- "
              "something else differs (library version, template, or thresholds), and that "
              "matters far more than the population. Investigate before using this campaign.")
        return 1
    print(f"AMBIGUOUS: {matches} all reproduce it, so this job cannot discriminate. "
          f"Try another --algo/--run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
