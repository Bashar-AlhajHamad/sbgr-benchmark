# -*- coding: utf-8 -*-
"""The control the PVT sweep needs: does the PUBLISHED design survive PVT either?

WHY THIS EXISTS. The PVT pass over the base@20,000 campaign returned pvt_robust = 0 for all six
algorithms: not one of 180 designs that is feasible at tt/1.8 V stays feasible across all five
process corners at all three supplies. Read alone, that number invites the wrong conclusion --
"the optimizers produce fragile designs".

But every design in the campaign was optimized at ONE corner, because that is the published
problem. So 0/180 may be a property of the FORMULATION rather than of the optimizers, and the
only way to tell is to put a design nobody optimized through the same 15 conditions. The Kuijk
reference design this benchmark was built from is exactly that design: it was published, it hits
its three published figures, and no optimizer ever touched it.

  * reference survives 15/15  -> the optimizers really did trade robustness for objective, and
                                 0/180 is about them
  * reference fails too       -> single-corner optimization cannot produce corner-robust designs
                                 on this circuit, the campaign inherited that from the problem
                                 statement, and Section 6 must say so before a reviewer does

ABC's and ACO's best designs run beside it, because they bracket the mechanism: ACO's median
loop-gain margin is 0.0003 dB and ABC's is 0.7082 dB, and if margin is what buys corner survival
then their curves should separate here.

    python pvt_reference_control.py --lib <sky130.lib.spice> [--csv <per_run_records.csv>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "spice"))
import spice_problem as sp                                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--case", default="base")
    ap.add_argument("--csv", default=None,
                    help="a merged per_run_records.csv; adds each algorithm's best design")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    names = [v[0] for v in sp.CASES[args.case][1]]
    nominal = sp.CASES[args.case][2]

    designs: list[tuple[str, np.ndarray]] = [
        ("REFERENCE(published)", np.array([float(nominal[n]) for n in names]))]
    if args.csv:
        d = pd.read_csv(args.csv)
        for a in sorted(set(d.algo)):
            b = d[d.algo == a].sort_values("best_fitness").iloc[0]
            designs.append((f"{a}/best/run{int(b.run)}",
                            np.array([float(b[f"x_{n}"]) for n in names])))

    grid = [(c, v) for c in sp.PVT_CORNERS for v in sp.PVT_SUPPLIES]
    print(f"{len(designs)} designs x {len(grid)} conditions "
          f"({len(sp.PVT_CORNERS)} corners x {len(sp.PVT_SUPPLIES)} supplies)")
    print("pre-declared rule (2026-08-06): feasible at tt/1.8 V and infeasible at ANY of the "
          f"{len(grid)} conditions counts as INFEASIBLE\n")

    passed: dict[str, int] = {lbl: 0 for lbl, _ in designs}
    worst: dict[str, float] = {lbl: float("inf") for lbl, _ in designs}
    detail: dict[str, list[str]] = {lbl: [] for lbl, _ in designs}

    for corner, vdd in grid:
        # allow_off_nominal is the deliberate switch: an OPTIMIZATION run must refuse to leave
        # tt/1.8 V, and only the verification pass is permitted to.
        p = sp.SpiceBGRProblem(sky130_lib=args.lib, case=args.case, timeout=args.timeout,
                               corner=corner, vdd=vdd, allow_off_nominal=True)
        try:
            line = f"  {corner} @ {vdd:.2f} V  "
            for lbl, x in designs:
                m = p.metrics(x)
                ok = int(m.get("is_feasible", 0)) == 1
                obj = float(m.get("PSRR_DB", float("nan")))
                passed[lbl] += int(ok)
                worst[lbl] = min(worst[lbl], obj)
                if not ok:
                    v = [k[5:] for k in ("viol_vref", "viol_tc", "viol_loop_gain",
                                         "viol_phase_margin", "viol_gain_margin", "viol_power")
                         if int(m.get(k, 0)) == 1]
                    detail[lbl].append(f"{corner}@{vdd:.2f}:{'+'.join(v) or 'sim'}")
                line += f"{lbl.split('/')[0]}={'OK ' if ok else 'no '}"
            print(line, flush=True)
        finally:
            p.close()

    print(f"\n{'design':24s} {'conditions passed':>18s} {'worst objective dB':>20s}")
    for lbl, _ in designs:
        print(f"  {lbl:22s} {passed[lbl]:12d}/{len(grid)} {worst[lbl]:20.3f}")

    print("\nwhich specification fails, per design:")
    for lbl, _ in designs:
        if detail[lbl]:
            print(f"  {lbl}:")
            print("    " + "  ".join(detail[lbl][:15]))
        else:
            print(f"  {lbl}: passes every condition")

    ref = passed["REFERENCE(published)"]
    print("\n" + "=" * 78)
    if ref == len(grid):
        print("The published design survives all 15 conditions. The campaign's 0/180 is therefore")
        print("a property of the OPTIMIZERS -- they traded corner robustness for objective.")
    else:
        print(f"The published design also fails ({ref}/{len(grid)} conditions passed). The")
        print("campaign's 0/180 is therefore inherited from the PROBLEM STATEMENT: optimizing at")
        print("a single corner cannot produce corner-robust designs on this circuit, and no")
        print("optimizer in the comparison was asked to. Section 6 must state this before")
        print("reporting any per-algorithm robustness number.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
