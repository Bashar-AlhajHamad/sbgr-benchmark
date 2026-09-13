"""512-point Latin-hypercube probe of the Banba amplifier box, run THROUGH SpiceBGRProblem itself.

Not a stand-in for the evaluator: the probe drives the same class the campaign will drive, so the
quantiser, the result cache, the deck-error breaker and the ngspice timeout recovery are exercised
here rather than assumed. Anything that would break during a run breaks now, at a cost of minutes.

Thresholds are set wide open, so nothing is judged yet -- the point is to record raw metrics. What
they are used for afterwards:
    pick_reference.py   chooses the reference design (the nominal amplifier point)
    calibrate.py        derives the thresholds and checks the four acceptance gates

THE CORE IS ALREADY FIXED and is not probed. The loop forces nA = nB, so the branch current is
I_Q = V_T*ln(n_pnp)/r_ptat and V_REF = r_ref*(V_EB/r_ctat + I_Q): both are core-only quantities,
independent of every amplifier variable. They were trimmed once by direct sweep -- r_ptat = 24.8 k
for minimum TC, r_ref = 159.90 k for V_REF = 800.47 mV, inside the PUBLISHED 798-802 mV window -- and
baked into the deck. Probing them again would only re-measure a settled quantity.

Seed 20260831 is the date this probe was written. It differs from Circuit B's (20260826) and Circuit
C's (20260826) so that the three are independent samples rather than the same unit hypercube points
re-scored through different evaluators.

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

# Wide open on every specification: this run measures, it does not judge.
WIDE = dict(VREF_min=-1e9, VREF_max=1e9, TC_max=1e9, LoopGain_min=-1e9,
            PhaseMargin_min=-1e9, GainMargin_min=-1e9, Power_max=1e9)
METRICS = ("PSRR_DB", "VREF", "TC", "LOOP_GAIN_DB", "PHASE_MARGIN_DEG",
           "GAIN_MARGIN_DB", "POWER_UW", "PSRR_100HZ_DB")
N, SEED = 512, 20260831
OUT = Path(__file__).with_name("lhs_probe.csv")


def main() -> int:
    names = [v[0] for v in sp.VARS_BANBA]
    s = qmc.LatinHypercube(d=len(sp.VARS_BANBA), seed=SEED).random(N)
    # float(), not the numpy scalar. repr() of a numpy scalar under NumPy 2.x is `np.float64(...)`,
    # which SPICE cannot parse: the .param assignment fails silently and ngspice exits, which the
    # harness records as a TIMEOUT. That cost hours on Circuit B; the cast is one character.
    X = [[float(lo + s[i][j] * (hi - lo))
          for j, (_n, lo, hi, _u, _st) in enumerate(sp.VARS_BANBA)] for i in range(N)]

    p = sp.SpiceBGRProblem(sky130_lib=str(LIB), case="banba", constraints=WIDE,
                           require_calibrated=False, timeout=90.0)
    print(f"evaluator up; {N} points, seed {SEED}, dim {p.dim}", flush=True)

    mref = p.metrics([sp.NOMINAL_BANBA[n] for n in names])
    print("provisional nominal point through the evaluator:")
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
