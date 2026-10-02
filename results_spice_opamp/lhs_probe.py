"""The threshold-calibration probe for Circuit B, run THROUGH SpiceOpampProblem itself.

Deliberately not a stand-in for the evaluator: the probe drives the same class the campaign will
drive, so the quantiser, the derived-geometry clamp, the result cache, the deck-error breaker and the
ngspice timeout recovery are all exercised here rather than assumed. Anything that would break during
a run breaks now, at a cost of three minutes.

Thresholds are set wide open so nothing is judged yet -- the point is to record the raw metrics.
512 points, Latin hypercube, seed 20260826 (the pre-registration date, distinct from Circuit A's
20260804 so the two probes are independent samples rather than the same points re-scored).
"""
import csv, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
CODE = Path(__file__).resolve().parents[1]  # the code directory, resolved relative to this script
sys.path.insert(0, str(CODE / "spice"))
import spice_problem_opamp as O
from scipy.stats import qmc

LIB = (CODE / "spice/pdk/volare/sky130/versions"
       "/c6d73a35f524070e85faff4a6a9eef49553ebc2b/sky130A/libs.tech/combined/sky130.lib.spice")
WIDE = dict(UGB_min=-1e9, PhaseMargin_min=-1e9, Power_max=1e9, Area_max=1e9,
            PSRR_min=-1e9, Vout_min=-1e9, Vout_max=1e9)
N, SEED = 512, 20260826
METRICS = ("GAIN_DB", "UGB_MHZ", "PHASE_MARGIN_DEG", "POWER_UW", "VOUT_V", "AREA_UM2", "PSRR_DB")

s = qmc.LatinHypercube(d=len(O.VARS), seed=SEED).random(N)
X = [[float(lo + s[i][j]*(hi-lo)) for j, (_n, lo, hi, _u, _st) in enumerate(O.VARS)]
     for i in range(N)]

p = O.SpiceOpampProblem(sky130_lib=str(LIB), constraints=WIDE, require_calibrated=False,
                        timeout=25.0)
print(f"evaluator up; {N} points, seed {SEED}", flush=True)

t0 = time.time()
rows = []
mref = p.metrics([O.NOMINAL[n] for n in O.VAR_NAMES])
print("reference design through the evaluator:", flush=True)
for k in METRICS:
    print(f"    {k:18s} {mref[k]:12.4f}"
          f"   (recorded {O.REFERENCE_METRICS.get(k.split('_')[0].title().replace('Db','').replace('Uw','').replace('Um2','').replace('V','') or 'Gain', float('nan'))})"
          if False else f"    {k:18s} {mref[k]:12.4f}", flush=True)

for i, x in enumerate(X):
    m = p.metrics(x)
    rows.append(m)
    if (i+1) % 64 == 0:
        print(f"  {i+1}/{N}  ({time.time()-t0:.0f}s)  {p.counters.summary()}", flush=True)
p.close()
print(f"\n{N} points in {time.time()-t0:.0f}s", flush=True)
print("counters:", p.counters.summary(), flush=True)

OUT = Path(__file__).with_name("probe_final.csv")
with OUT.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    xk = ["x_" + n for n in ("w1", "w3", "w5", "w6", "w7", "l_all", "cc")]
    w.writerow(["i"] + list(O.VAR_NAMES) + xk + ["sim_ok", "sim_failure"] + list(METRICS))
    for i, m in enumerate(rows):
        w.writerow([i] + [f"{X[i][j]:.6f}" for j in range(len(O.VARS))]
                   + [("" if m.get(k) is None else f"{m[k]:.6f}") for k in xk]
                   + [m["sim_ok"], m.get("sim_failure") or ""]
                   + [("" if m.get(k) is None or m[k] != m[k] else f"{m[k]:.6f}") for k in METRICS])
print(f"wrote {OUT}", flush=True)

ok = [m for m in rows if m["sim_ok"]]
print(f"\nmeasurable {len(ok)}/{N} = {100*len(ok)/N:.1f} %", flush=True)
fails = {}
for m in rows:
    if not m["sim_ok"]:
        fails[m.get("sim_failure")] = fails.get(m.get("sim_failure"), 0) + 1
if fails:
    print("failures:", ", ".join(f"{k}={v}" for k, v in sorted(fails.items())), flush=True)
import statistics as S
print(f"\n  {'metric':20s} {'min':>10s} {'p10':>10s} {'median':>10s} {'p90':>10s} {'max':>10s} {'ref':>10s}")
for k in METRICS:
    v = sorted(m[k] for m in ok if m.get(k) is not None and m[k] == m[k])
    if not v: continue
    q = lambda t: v[min(len(v)-1, int(t*len(v)))]
    print(f"  {k:20s} {v[0]:10.3f} {q(.10):10.3f} {S.median(v):10.3f} {q(.90):10.3f} "
          f"{v[-1]:10.3f} {mref[k]:10.3f}")
print("done", flush=True)
