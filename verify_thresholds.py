"""
verify_thresholds.py -- re-run the Latin-hypercube probe on THIS machine and check that the
FROZEN thresholds still pass all four calibration gates.

This VERIFIES; it does not re-derive. The distinction matters more than the code does. The
thresholds in spice_problem.CONSTRAINTS were fixed from the nominal operating point before any
optimization run was executed, and the sentence "they were not adjusted afterwards" is the
whole answer to a reviewer who asks why the published 65-nm numbers were not used. Re-tuning
them after seeing anything -- probe or campaign -- would destroy that argument. So if a gate
fails here, the correct response is to report it, not to move a threshold.

The one legitimate reason to run this: the original probe was executed on a different machine
during development. ngspice is deterministic and the thresholds carry 5-6 % margins, so the
result should be identical, but re-running it here (about 30 s on a fast machine) means every
number in the case-study section comes from the single machine the paper documents.

    python verify_thresholds.py --lib <path to sky130.lib.spice>
    python verify_thresholds.py --lib ... --n 512 --seed 20260804   # the original settings
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "spice"))
import spice_problem as sp  # noqa: E402

# The four gates, exactly as they were applied when the thresholds were frozen.
JOINT_LO, JOINT_HI = 0.005, 0.05
MARG_LO, MARG_HI = 0.10, 0.60
SHARE_MAX = 0.80

VIOL = {"viol_vref": "VREF window", "viol_tc": "TC max",
        "viol_loop_gain": "loop gain min", "viol_phase_margin": "phase margin min",
        "viol_gain_margin": "gain margin min", "viol_power": "power max"}

# What the original probe measured, for comparison. Any large discrepancy means the toolchain
# did not transfer identically, which is far more important than the gates themselves.
ORIGINAL = {"joint": 0.0215, "n_feasible": 11, "n": 512,
            "best_feasible_psrr": 50.49,
            "marginal": {"VREF window": 0.465, "TC max": 0.287, "loop gain min": 0.418,
                         "phase margin min": 0.471, "gain margin min": 0.580,
                         "power max": 0.301}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260804,
                    help="the seed the original calibration used; keep it to compare")
    ap.add_argument("--out", default="results_spice/lhs_probe.csv")
    args = ap.parse_args()

    try:
        from scipy.stats import qmc
    except Exception as e:  # noqa: BLE001
        print(f"scipy.stats.qmc is required: {e}")
        return 1

    p = sp.SpiceBGRProblem(sky130_lib=args.lib, timeout=60.0)
    print(f"library parsed in {p._srv.load_seconds:.0f}s")
    print("frozen thresholds:")
    for k, v in p.constraints.items():
        print(f"  {k:18s} {v}")
    print()
    try:
        X = qmc.scale(qmc.LatinHypercube(d=p.dim, seed=args.seed).random(args.n),
                      p.lb, p.ub)
        rows, t0 = [], time.perf_counter()
        for i, x in enumerate(X):
            rows.append(p.metrics(x))
            if (i + 1) % 128 == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1}/{args.n}  {el:5.0f}s  {el/(i+1)*1000:4.0f} ms/eval",
                      flush=True)
        total = time.perf_counter() - t0
    finally:
        p.close()

    n = len(rows)
    ok_rows = [r for r in rows if r["sim_ok"]]
    marg = {lbl: 1 - np.mean([r[k] for r in rows]) for k, lbl in VIOL.items()}
    feas = sum(1 for r in rows if r["is_feasible"])
    joint = feas / n
    infe = [r for r in rows if not r["is_feasible"]]
    share = {lbl: (np.mean([r[k] for r in infe]) if infe else 0.0)
             for k, lbl in VIOL.items()}

    print(f"\n=== {n} points in {total:.0f}s ({total/n*1000:.0f} ms/eval), "
          f"{len(ok_rows)}/{n} simulated ===")

    print("\n=== THE FOUR GATES ===")
    x_nom = np.array([sp.NOMINAL[v] for v in sp.VAR_NAMES], dtype=float)
    # the nominal was already evaluated during verify_setup; recompute cheaply here
    p2 = sp.SpiceBGRProblem(sky130_lib=args.lib, timeout=60.0)
    try:
        m_nom = p2.metrics(x_nom)
    finally:
        p2.close()
    g1 = m_nom["penalty"] == 0
    g2 = JOINT_LO <= joint <= JOINT_HI
    g3 = all(MARG_LO <= r <= MARG_HI for r in marg.values())
    g4 = all(s <= SHARE_MAX for s in share.values())
    print(f"  1. nominal design feasible          : {'PASS' if g1 else 'FAIL'} "
          f"(penalty {m_nom['penalty']})")
    print(f"  2. joint feasibility in [0.5 %, 5 %]: {'PASS' if g2 else 'FAIL'} "
          f"({100*joint:.2f} %, {feas}/{n})")
    print(f"  3. marginal rates in [10 %, 60 %]   : {'PASS' if g3 else 'FAIL'} "
          f"({100*min(marg.values()):.0f}-{100*max(marg.values()):.0f} %)")
    print(f"  4. no spec above 80 % of infeasible : {'PASS' if g4 else 'FAIL'} "
          f"(max {100*max(share.values()):.0f} %)")

    print("\n=== does this machine reproduce the original calibration? ===")
    print(f"  {'spec':20s} {'here':>8s} {'original':>10s} {'delta':>8s}")
    for lbl in marg:
        o = ORIGINAL["marginal"].get(lbl)
        d = (marg[lbl] - o) if o is not None else float("nan")
        print(f"  {lbl:20s} {100*marg[lbl]:7.1f} % {100*o:9.1f} % {100*d:+7.1f} pp")
    print(f"  {'joint feasibility':20s} {100*joint:7.2f} % "
          f"{100*ORIGINAL['joint']:9.2f} % {100*(joint-ORIGINAL['joint']):+7.2f} pp")
    fp = [r["PSRR_DB"] for r in rows if r["is_feasible"]]
    if fp:
        print(f"  best feasible PSRR   {max(fp):7.2f}   "
              f"{ORIGINAL['best_feasible_psrr']:9.2f}   "
              f"{max(fp)-ORIGINAL['best_feasible_psrr']:+7.2f} dB")
        print(f"\n  -> the campaign must beat {max(fp):.2f} dB or the optimization added "
              f"nothing over random sampling")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(outp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  wrote {outp.resolve()}")

    allok = g1 and g2 and g3 and g4
    print("\n" + "=" * 68)
    if allok:
        print("ALL FOUR GATES PASS on this machine. The frozen thresholds stand.")
        print("Launch:  py run_spice.py --lib <lib> --outdir results_spice --workers 4")
    else:
        print("A GATE FAILED. Report it -- do NOT adjust a threshold. The thresholds were")
        print("frozen before any optimization run, and that commitment is worth more than a")
        print("prettier pass rate. Send me this output.")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
