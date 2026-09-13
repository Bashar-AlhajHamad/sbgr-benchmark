"""
verify_anchor.py -- THE GATE. Does this machine reproduce the validated circuit measurements?

Run this on TRUBA's `debug` queue before submitting anything. It is the only thing standing
between us and 17,460 core-hours spent on numbers we cannot trust.

WHY A DIFFERENT MACHINE CAN GIVE DIFFERENT ANSWERS. The validated numbers below were measured
with a Windows ngspice build. On TRUBA the binary is compiled by a different compiler against a
different libm on a different microarchitecture (Xeon Platinum 8480+, AVX-512). ngspice solves a
nonlinear circuit by Newton iteration to a tolerance, so a last-digit difference in an
exponential or a division can move which point it converges to. Usually the effect is invisible;
occasionally it is not. Either way it is measurable in minutes, and guessing is not an option
when the frozen thresholds sit only a few percent away from the nominal operating point.

WHAT HAPPENS IF IT FAILS. The thresholds are NOT re-tuned. That freeze, dated 2026-08-04 and
made before any optimization run, is the entire answer to the reviewer who asks why the published
65-nm numbers were not used, and re-tuning it after seeing a new machine's output would destroy
the claim. A failure here is a decision point to be taken explicitly and disclosed, not a
calibration step.

    python verify_anchor.py --lib <sky130.lib.spice>
    python verify_anchor.py --lib ... --probe 512      # also re-derive the random baseline
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "spice"))

import ngspice_bridge as nb    # noqa: E402
import spice_problem as sp     # noqa: E402

# Validated on the Windows reference machines (i7 dev box and i9-14900HX), bit-identical on both,
# and matching the source design's three published figures (1220 mV / 24 ppm/degC / 47 uW).
ANCHOR = {
    "VREF":             (1.219903, 0.0005, "V     published 1.220"),
    "TC":               (24.49,    1.0,    "ppm/C published 24"),
    "POWER_UW":         (47.27,    1.0,    "uW    published 47"),
    "LOOP_GAIN_DB":     (53.21,    0.5,    "dB    validated against a closed form"),
    "PHASE_MARGIN_DEG": (62.99,    1.0,    "deg   Tian vs closed form agreed to 0.02 deg"),
    "GAIN_MARGIN_DB":   (14.86,    0.5,    "dB"),
    "PSRR_100HZ_DB":    (56.29,    0.5,    "dB    single-frequency diagnostic"),
    "PSRR_WC_DB":       (23.85,    1.0,    "dB    THE OBJECTIVE: worst case over the band"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--probe", type=int, default=0,
                    help="also run an N-point Latin-hypercube probe to re-derive the "
                         "random-search baseline under the current objective (512 = the "
                         "original calibration size)")
    ap.add_argument("--probe-seed", type=int, default=20260804,
                    help="the ORIGINAL calibration seed, so the design points are identical to "
                         "the 2026-08-04 probe and only the scoring changes")
    ap.add_argument("--timing", type=int, default=200,
                    help="evaluations used to measure this machine's cost per evaluation")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    print("=" * 78)
    print("ANCHOR GATE")
    print("=" * 78)
    print(f"  host      : {platform.node()}")
    print(f"  platform  : {platform.platform()}")
    print(f"  processor : {platform.processor() or 'n/a'}")
    print(f"  python    : {sys.version.split()[0]}   numpy {np.__version__}")
    exe = nb.find_ngspice()
    print(f"  ngspice   : {exe}")
    try:
        import subprocess
        v = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
        # ngspice leads with an asterisk banner, so line 0 is "******". Pick the line that
        # actually names the version -- this string goes into §6 as provenance.
        lines = [ln.strip() for ln in ((v.stdout or "") + (v.stderr or "")).splitlines()
                 if ln.strip()]
        ver = next((ln for ln in lines if "ngspice" in ln.lower()
                    and any(c.isdigit() for c in ln)), lines[0] if lines else "?")
        print(f"  version   : {ver}")
    except Exception as e:  # noqa: BLE001
        print(f"  version   : could not query ({e})")
    print(f"  objective : {sp.OBJECTIVE}")
    print()

    fails = []
    t0 = time.perf_counter()
    p = sp.SpiceBGRProblem(sky130_lib=args.lib, case="base", timeout=args.timeout)
    print(f"[load] PDK parsed in {p._srv.load_seconds:.1f}s "
          f"(total {time.perf_counter()-t0:.1f}s)\n")

    # ---------------------------------------------------------------- 1. the anchor
    print("1. nominal operating point")
    x = np.array([sp.NOMINAL[v[0]] for v in sp.VARS], dtype=float)
    m = p.metrics(x)
    if m.get("sim_failure"):
        print(f"   FATAL: the nominal design did not simulate: {m['sim_failure']} "
              f"{m.get('sim_detail','')}")
        return 1
    w = max(len(k) for k in ANCHOR)
    for k, (want, tol, note) in ANCHOR.items():
        got = float(m.get(k, float("nan")))
        ok = abs(got - want) <= tol
        print(f"   {'PASS' if ok else 'FAIL'}  {k:<{w}}  {got:>12.6g}  vs {want:<10.6g} "
              f"tol {tol:<7.4g} {note}")
        if not ok:
            fails.append(f"{k}: {got:.6g} vs {want:.6g} (tol {tol:g})")

    # ---------------------------------------------------------------- 2. threshold gates
    print("\n2. the frozen thresholds still gate as declared")
    for case, want in (("base", 0), ("hard", 3)):
        q = sp.SpiceBGRProblem(sky130_lib=args.lib, case=case, timeout=args.timeout) \
            if case != "base" else p
        mm = q.metrics(x)
        got = int(mm.get("penalty", -1))
        ok = got == want
        print(f"   {'PASS' if ok else 'FAIL'}  {case:5s}: nominal penalty {got} "
              f"(declared {want})"
              + ("" if ok else "   <-- the 2026-08-04 freeze does NOT hold on this machine"))
        if not ok:
            fails.append(f"{case} nominal penalty {got}, declared {want}")
        if case != "base":
            q.close()

    # ---------------------------------------------------------------- 3. determinism
    print("\n3. determinism: the same design must give the same numbers twice")
    p._cache.clear()
    m2 = p.metrics(x)
    keys = [k for k in ANCHOR]
    same = all(float(m[k]) == float(m2[k]) for k in keys)
    print(f"   {'PASS' if same else 'FAIL'}  re-simulated with a cleared cache: "
          f"{'bit-identical' if same else 'DIFFERS'}")
    if not same:
        for k in keys:
            if float(m[k]) != float(m2[k]):
                print(f"        {k}: {float(m[k])!r} then {float(m2[k])!r}")
        fails.append("the evaluator is not deterministic on this machine")

    # ---------------------------------------------------------------- 4. cost per evaluation
    print(f"\n4. cost per evaluation on this machine ({args.timing} evaluations)")
    rng = np.random.default_rng(11)
    t0, n_ok = time.perf_counter(), 0
    for _ in range(args.timing):
        xr = p.lb + rng.random(p.dim) * (p.ub - p.lb)
        mm = p.metrics(xr)
        n_ok += 0 if mm.get("sim_failure") else 1
    dt = (time.perf_counter() - t0) / args.timing
    print(f"   {dt*1000:.0f} ms per evaluation   ({n_ok}/{args.timing} simulated cleanly)")
    print(f"   reference: 416 ms on the i9-14900HX  ->  this machine is "
          f"{dt/0.416:.2f}x that cost")
    for label, budget in (("base/hard", 150_000), ("highdim", 220_000)):
        h = budget * dt / 3600
        ok = h < 72
        print(f"   {'PASS' if ok else 'FAIL'}  {label:9s} {budget:,} evals -> {h:.1f} h per "
              f"array task, against the 3-day (72 h) queue limit"
              + ("" if ok else "   <-- REDUCE THE BUDGET OR CHANGE QUEUE"))
        if not ok:
            fails.append(f"{label} at {budget:,} evaluations needs {h:.0f} h > 72 h")
    print(f"   whole campaign: 540 tasks -> "
          f"{(2*180*150_000 + 180*220_000) * dt / 3600:,.0f} core-hours")

    # ---------------------------------------------------------------- 5. random baseline
    if args.probe:
        print(f"\n5. Latin-hypercube probe, {args.probe} points, seed {args.probe_seed}")
        # MUST be scipy.stats.qmc.LatinHypercube, matching verify_thresholds.py:75 verbatim --
        # that is what the 2026-08-04 calibration used. A hand-rolled numpy LHS with the same seed
        # produces a DIFFERENT point set, and the first version of this file did exactly that,
        # which made the printed comparison against the original 11/512 meaningless while looking
        # authoritative. Same generator, same seed, same box => the same 512 designs, so any
        # difference in the result is a difference in the SIMULATOR or the OBJECTIVE, which is the
        # only thing worth measuring here.
        from scipy.stats import qmc
        print("   generator: scipy.stats.qmc.LatinHypercube, identical to the 2026-08-04")
        print("   calibration, so the 512 design points are the same and any difference is")
        print("   attributable to the simulator build or the objective definition")
        X = qmc.scale(qmc.LatinHypercube(d=p.dim, seed=args.probe_seed).random(args.probe),
                      p.lb, p.ub)
        rec, t0 = [], time.perf_counter()
        for i, xi in enumerate(X, 1):
            mm = p.metrics(xi)
            rec.append(mm)
            if i % 128 == 0:
                print(f"     {i}/{args.probe}  ({time.perf_counter()-t0:.0f}s)", flush=True)
        feas = [r for r in rec if r.get("is_feasible")]
        okm = [r for r in rec if not r.get("sim_failure")]
        pens = np.array([r.get("penalty", 6) for r in rec])
        print(f"   simulated cleanly     : {len(okm)}/{args.probe} "
              f"({100*len(okm)/args.probe:.1f} %)")
        print(f"   jointly feasible      : {len(feas)}/{args.probe} "
              f"({100*len(feas)/args.probe:.2f} %)   "
              f"{'IN the 0.5-5 % calibration band' if 0.005 <= len(feas)/args.probe <= 0.05 else 'OUTSIDE the 0.5-5 % band'}")
        # The constraints do not involve PSRR, so on the IDENTICAL point set this count must
        # reproduce the 2026-08-04 figure of 11/512 exactly. A difference means the simulator
        # build moved a marginal design across a threshold -- worth knowing, and reportable, but
        # not a reason to touch the thresholds.
        if args.probe == 512 and args.probe_seed == 20260804:
            print(f"   2026-08-04 reference  : 11/512 (2.15 %) on the Windows build"
                  + ("   -- REPRODUCED" if len(feas) == 11 else
                     f"   -- DIFFERS by {len(feas)-11:+d} design(s) at the threshold margin"))
        print("   penalty histogram     : "
              + "  ".join(f"{k}:{int((pens == k).sum())}" for k in range(7)))
        for spec, col, lo in (("VREF", "viol_vref", None), ("TC", "viol_tc", None),
                              ("LoopGain", "viol_loop_gain", None),
                              ("PhaseMargin", "viol_phase_margin", None),
                              ("GainMargin", "viol_gain_margin", None),
                              ("Power", "viol_power", None)):
            rate = 100 * (1 - np.mean([r.get(col, 1) for r in rec]))
            print(f"     {spec:12s} marginal pass rate {rate:5.1f} %"
                  + ("" if 10 <= rate <= 60 else "   <-- outside the 10-60 % band"))
        if feas:
            best = max(float(r["PSRR_DB"]) for r in feas)
            print(f"\n   >>> RANDOM-SEARCH BASELINE under {sp.OBJECTIVE}: {best:.2f} dB")
            print(f"   >>> paste into spice_problem.py:")
            print(f"   >>>     PROBE_BEST_FEASIBLE_PSRR_DB = {best:.2f}")
            print("   The campaign must beat this, or the optimization added nothing over "
                  "uniform sampling.")
        else:
            print("\n   NO feasible probe point. Under this objective the feasible region is not "
                  "reachable by uniform sampling; report the fact and do not invent a baseline.")

    p.close()
    print("\n" + "=" * 78)
    if fails:
        print("GATE FAILED -- do NOT submit the campaign:")
        for f in fails:
            print("  - " + f)
        print("\nThe thresholds are NOT to be re-tuned to make this pass. The 2026-08-04 freeze")
        print("predates every optimization run and is the answer to the goalpost objection.")
        print("Decide explicitly, and disclose whatever is decided.")
        return 1
    print("GATE PASSED -- this machine reproduces the validated circuit measurements.")
    print("The campaign may be submitted. Record the environment block printed above in §6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
