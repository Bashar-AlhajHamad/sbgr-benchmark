"""512-point Latin-hypercube probe of Circuit C's design box, run THROUGH SpiceLdoProblem itself.

Not a stand-in for the evaluator: the probe drives the same class the campaign will drive, so the
quantiser, the result cache, the deck-error breaker and the ngspice timeout recovery are exercised
here rather than assumed. Anything that would break during a run breaks now, at a cost of minutes.

Thresholds are set wide open so nothing is judged yet -- the point is to record the raw metrics.

Seed 20260826 is the pre-registration date, matching Circuit B's probe seed but a DIFFERENT design
box and a different evaluator, so the two are independent samples of different problems rather than
the same points re-scored.

    python lhs_probe.py                # writes lhs_probe.csv
    python calibrate.py                # prints the thresholds and the four acceptance gates
"""
import csv
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "spice"))
import spice_problem_ldo as L                                  # noqa: E402
from scipy.stats import qmc                                    # noqa: E402

LIB = (CODE / "spice/pdk/volare/sky130/versions"
       "/c6d73a35f524070e85faff4a6a9eef49553ebc2b/sky130A/libs.tech/combined/sky130.lib.spice")
WIDE = dict(Vout_min=-1e9, Vout_max=1e9, LoadReg_max=1e9, LineReg_max=1e9,
            Iq_max=1e9, Dropout_max=1e9, Peaking_max=1e9)
METRICS = ("PSRR_DB", "VOUT_V", "LOADREG_MV_MA", "LINEREG_MV_V",
           "IQ_UA", "DROPOUT_MV", "PEAKING_DB", "AREA_UM2")
N, SEED = 512, 20260826
OUT = Path(__file__).with_name("lhs_probe.csv")


def main() -> int:
    if not L.NOMINAL:
        print("Circuit C has no reference design yet; run pick_reference.py and freeze NOMINAL first")
        return 1

    s = qmc.LatinHypercube(d=len(L.VARS), seed=SEED).random(N)
    # float(), not the numpy scalar. repr() of a numpy scalar under NumPy 2.x is `np.float64(...)`,
    # which SPICE cannot parse: the `.param` fails silently and ngspice exits, which the harness
    # records as a TIMEOUT. That cost hours on Circuit B; the cast is one character.
    X = [[float(lo + s[i][j] * (hi - lo))
          for j, (_n, lo, hi, _u, _st) in enumerate(L.VARS)] for i in range(N)]

    p = L.SpiceLdoProblem(sky130_lib=str(LIB), constraints=WIDE, require_calibrated=False,
                          timeout=60.0)
    print(f"evaluator up; {N} points, seed {SEED}", flush=True)

    mref = p.metrics([L.NOMINAL[n] for n in L.VAR_NAMES])
    print("reference design through the evaluator:")
    for k in METRICS:
        print(f"    {k:16s} {mref[k]:12.4f}")
    print(flush=True)

    t0, rows = time.time(), []
    for i, x in enumerate(X):
        rows.append(p.metrics(x))
        if (i + 1) % 64 == 0:
            print(f"  {i+1}/{N}  ({time.time()-t0:.0f}s)  {p.counters.summary()}", flush=True)
    p.close()
    print(f"\n{N} points in {(time.time()-t0)/60:.1f} min   {p.counters.summary()}", flush=True)

    xk = ["x_" + n for n in L.VAR_NAMES]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i"] + list(L.VAR_NAMES) + xk + ["sim_ok", "sim_failure"] + list(METRICS))
        for i, m in enumerate(rows):
            w.writerow(
                [i] + [f"{X[i][j]:.6f}" for j in range(len(L.VARS))]
                + [("" if m.get(k) is None else f"{m[k]:.6f}") for k in xk]
                + [m["sim_ok"], m.get("sim_failure") or ""]
                + [("" if m.get(k) is None or m[k] != m[k] else f"{m[k]:.6f}") for k in METRICS])
    print(f"wrote {OUT}", flush=True)

    ok = [m for m in rows if m["sim_ok"]]
    print(f"\nmeasurable {len(ok)}/{N} = {100*len(ok)/N:.1f} %")
    fails: dict = {}
    for m in rows:
        if not m["sim_ok"]:
            d = (m.get("sim_detail") or "")[:44]
            fails[d] = fails.get(d, 0) + 1
    if fails:
        print("failure reasons:")
        for k, v in sorted(fails.items(), key=lambda z: -z[1]):
            print(f"  {v:4d}  {k}")

    import statistics as S
    print(f"\n  {'metric':16s} {'min':>11s} {'p10':>11s} {'median':>11s} {'p90':>11s} "
          f"{'max':>11s} {'ref':>11s}")
    for k in METRICS:
        v = sorted(m[k] for m in ok if m.get(k) is not None and m[k] == m[k])
        if not v:
            continue
        q = lambda t: v[min(len(v) - 1, int(t * len(v)))]          # noqa: E731
        print(f"  {k:16s} {v[0]:11.3f} {q(.10):11.3f} {S.median(v):11.3f} {q(.90):11.3f} "
              f"{v[-1]:11.3f} {mref[k]:11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
