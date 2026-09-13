"""512-point Latin-hypercube probe of the Brokaw amplifier box, run THROUGH SpiceBGRProblem itself.

Not a stand-in for the evaluator: the probe drives the same class the campaign will drive, so the
quantiser, the result cache, the deck-error breaker and the ngspice timeout recovery are exercised
here rather than assumed. Anything that would break during a run breaks now, at a cost of minutes.

Thresholds are set wide open, so nothing is judged yet -- the point is to record raw metrics. What
they are used for afterwards:
    pick_reference.py   chooses the anchor (the nominal amplifier point)
    calibrate.py        derives the thresholds and checks the four acceptance gates

THE CORE IS ALREADY FIXED and is not probed. V_REF = V_BE1 + 2*(r1/r2)*V_T*ln(n_npn) depends only on
the core, given that the loop forces the two collectors equal. It was trimmed by direct sweep:
r1 = 115.6 k places V_REF at 1.219892 V, matching the Kuijk anchor of 1.2199 V so the two
voltage-mode cases share an anchor voltage; and rl = 260 k because the PTAT branch current makes the
drop across rl grow with temperature -- at rl = 430 k the NPNs entered saturation at the hot end and
TC measured 397 ppm/degC instead of 18. Both are baked into the deck.

THE PROVISIONAL NOMINAL POINT DOES NOT SIMULATE: widths copied from the Kuijk nominal with vbpv = 0.45
fail with a convergence error in the temperature sweep. That is expected of a guess rather than a
measurement, and the same happened on the Banba deck, where the provisional point gave a 38 deg phase
margin against a 60 deg requirement. Finding a point that works is what this probe is for, so the
failure is reported below rather than allowed to block the probe.

Seed 20260901 differs from Circuit B's (20260826), Circuit C's (20260826) and Banba's (20260831), so
the probes are independent samples rather than the same unit hypercube points re-scored.

    python lhs_probe.py                # writes lhs_probe.csv
"""
import csv
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "spice"))
import spice_problem as sp                                      # noqa: E402
from scipy.stats import qmc                                     # noqa: E402

LIB = (CODE / "spice/pdk/volare/sky130/versions"
       "/c6d73a35f524070e85faff4a6a9eef49553ebc2b/sky130A/libs.tech/combined/sky130.lib.spice")

WIDE = dict(VREF_min=-1e9, VREF_max=1e9, TC_max=1e9, LoopGain_min=-1e9,
            PhaseMargin_min=-1e9, GainMargin_min=-1e9, Power_max=1e9)
METRICS = ("PSRR_DB", "VREF", "TC", "LOOP_GAIN_DB", "PHASE_MARGIN_DEG",
           "GAIN_MARGIN_DB", "POWER_UW", "PSRR_100HZ_DB")
N, SEED = 512, 20260901
OUT = Path(__file__).with_name("lhs_probe.csv")


def main() -> int:
    names = [v[0] for v in sp.VARS_BROKAW]
    s = qmc.LatinHypercube(d=len(sp.VARS_BROKAW), seed=SEED).random(N)
    # float(), not the numpy scalar. repr() of a numpy scalar under NumPy 2.x is `np.float64(...)`,
    # which SPICE cannot parse: the .param assignment fails silently and ngspice exits, which the
    # harness records as a TIMEOUT. That cost hours on Circuit B; the cast is one character.
    X = [[float(lo + s[i][j] * (hi - lo))
          for j, (_n, lo, hi, _u, _st) in enumerate(sp.VARS_BROKAW)] for i in range(N)]

    p = sp.SpiceBGRProblem(sky130_lib=str(LIB), case="brokaw", constraints=WIDE,
                           require_calibrated=False, timeout=90.0)
    print(f"evaluator up; {N} points, seed {SEED}, dim {p.dim}", flush=True)

    mref = p.metrics([sp.NOMINAL_BROKAW[n] for n in names])
    print("provisional nominal point through the evaluator:")
    if mref.get("sim_failure"):
        print(f"    FAILS: {mref['sim_failure']} -- expected; see the module docstring")
    else:
        for k in METRICS:
            v = mref.get(k)
            print(f"    {k:18s} {v:12.4f}" if isinstance(v, (int, float)) else f"    {k:18s} {v}")
    print(flush=True)

    t0, rows, fails = time.time(), [], 0
    for i, x in enumerate(X):
        m = p.metrics(x)
        if not m.get("sim_ok"):
            fails += 1
        rows.append({**{n: xi for n, xi in zip(names, x)},
                     **{k: m.get(k) for k in METRICS},
                     "sim_ok": m.get("sim_ok"), "sim_failure": m.get("sim_failure")})
        if (i + 1) % 32 == 0:
            el = time.time() - t0
            print(f"  {i + 1:4d}/{N}   {el:6.1f}s   {el / (i + 1):.2f}s/point   "
                  f"failures {fails}", flush=True)

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows, {fails} simulation failures)")
    print(p.counters.summary())
    p.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
