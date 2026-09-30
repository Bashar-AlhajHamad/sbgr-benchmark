"""Choose Circuit C's reference design by a criterion declared BEFORE the search that finds it.

Circuit B needed a revision because its first anchor sat at the 75th percentile of the power
distribution: gate 1 forbids a threshold that excludes the reference design, so that anchor made the
power threshold untightenable and 81 % of the box passed it. The lesson is applied here in advance --
the anchor must be a GOOD design on every axis, not merely a working one, and it must not sit at a
box extreme.

THE CRITERION, REVISED. The first version minimised quiescent current subject to the other
specifications. That was wrong, and the calibration caught it: random sampling reached 75.41 dB of
PSRR while the resulting anchor achieved only 64.21 dB, so "beat the reference design" -- the claim
this benchmark exists to make -- was satisfied by chance. For Circuits A and B the reference beats
random search (23.85 dB against 15.50, and 83.49 dB against 81.18); an anchor that random search
outperforms is not a reference design, it is just a point.

The error was choosing the anchor by a CONSTRAINT (Iq) when the objective is PSRR. A reference design
has to be a competent design on the quantity being optimised.

THE CRITERION USED, fixed here, above the code that uses it:

    MAXIMISE PSRR, subject to
        LoadReg <=  5 mV/mA
        LineReg <=  5 mV/V
        Iq      <= 40 uA          a real budget, not merely whatever falls out
        Dropout <= 200 mV         it must actually be a LOW-dropout regulator
        Peaking <=  1 dB          a well-damped loop
        Vout    within +/-1 %     tighter than the +/-2 % specification window, so the anchor does
                                  not sit on the edge of its own specification
        wpass   <= 80 um          NOT at the 100 um box ceiling: an anchor at a box extreme is what
                                  forced Circuit B's revision

This is what a reference design is: the best objective a competent engineer reaches while meeting
every specification. It is still a DESIGN and not an optimisation result -- it comes from a coarse
hand-set grid, not from a search of the continuous box -- so the campaign has a real bar to clear
rather than a foregone one.
"""
import itertools, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
CODE = Path(r"C:\University\ABC\claude\code")
sys.path.insert(0, str(CODE / "spice"))
import spice_problem_ldo as L

LIB = (CODE / "spice/pdk/volare/sky130/versions"
       "/c6d73a35f524070e85faff4a6a9eef49553ebc2b/sky130A/libs.tech/combined/sky130.lib.spice")
WIDE = dict(Vout_min=-1e9, Vout_max=1e9, LoadReg_max=1e9, LineReg_max=1e9,
            Iq_max=1e9, Dropout_max=1e9, Peaking_max=1e9)
SEED_NOM = dict(w1=10.0, w3=10.0, w5=4.0, wpass=60.0, l_all=1.0, rdiv=100.0, cc=5.0)

p = L.SpiceLdoProblem(sky130_lib=str(LIB), constraints=WIDE, nominal=SEED_NOM,
                      require_calibrated=False, timeout=60.0)
print("evaluator up\n", flush=True)

def meets(m):
    return (m["sim_ok"] and m["LOADREG_MV_MA"] <= 5.0 and m["LINEREG_MV_V"] <= 5.0
            and m["IQ_UA"] <= 40.0 and m["DROPOUT_MV"] <= 200.0
            and m["PEAKING_DB"] <= 1.0
            and abs(m["VOUT_V"] - L.VOUT_TARGET) <= 0.01 * L.VOUT_TARGET)

GRID = dict(w1=(5.0, 10.0, 20.0), w3=(5.0, 10.0), w5=(2.0, 4.0, 8.0),
            wpass=(40.0, 60.0, 80.0), l_all=(0.5, 1.0, 1.5), rdiv=(100.0, 200.0, 400.0),
            cc=(2.0, 5.0, 10.0))
keys = list(GRID)
combos = list(itertools.product(*(GRID[k] for k in keys)))
print(f"{len(combos)} grid candidates", flush=True)

best, n_ok, n_meet, t0 = None, 0, 0, time.time()
for i, vals in enumerate(combos):
    x = dict(zip(keys, vals))
    m = p.metrics([x[k] for k in L.VAR_NAMES])
    if m["sim_ok"]:
        n_ok += 1
    if meets(m):
        n_meet += 1
        if best is None or m["PSRR_DB"] > best[1]["PSRR_DB"]:
            best = (dict(x), dict(m))
            print(f"  [{i+1}/{len(combos)}] PSRR {m['PSRR_DB']:6.2f} dB  Iq {m['IQ_UA']:6.2f} uA  "
                  f"drop {m['DROPOUT_MV']:5.0f} mV  load {m['LOADREG_MV_MA']:5.2f}  "
                  f"line {m['LINEREG_MV_V']:5.2f}  peak {m['PEAKING_DB']:4.2f}  "
                  f"Vout {m['VOUT_V']:.4f}   {x}", flush=True)
    if (i + 1) % 200 == 0:
        print(f"  ...{i+1}/{len(combos)}  simulated {n_ok}, meeting criterion {n_meet}  "
              f"({time.time()-t0:.0f}s)", flush=True)

print(f"\n{len(combos)} candidates in {(time.time()-t0)/60:.1f} min; "
      f"{n_ok} simulated, {n_meet} met the criterion", flush=True)
print(f"counters: {p.counters.summary()}", flush=True)
p.close()

if best is None:
    print("\nNO candidate meets the declared criterion -- report that, do not relax it silently")
else:
    x, m = best
    print("\nREFERENCE DESIGN (lowest Iq meeting the declared criterion)")
    print("  " + "  ".join(f"{k}={x[k]:g}" for k in L.VAR_NAMES))
    for k, u in (("PSRR_DB","dB"),("VOUT_V","V"),("IQ_UA","uA"),("LOADREG_MV_MA","mV/mA"),
                 ("LINEREG_MV_V","mV/V"),("DROPOUT_MV","mV"),("PEAKING_DB","dB"),
                 ("AREA_UM2","um^2")):
        print(f"    {k:15s} {m[k]:10.4f}  {u}")
    print("\n  paste-ready:")
    print("  NOMINAL = dict(" + ", ".join(f"{k}={x[k]:g}" for k in L.VAR_NAMES) + ")")
    print("  REFERENCE_METRICS = dict(" + ", ".join(
        f"{a}={m[b]:.4f}" for a, b in (("PSRR","PSRR_DB"),("Vout","VOUT_V"),("Iq","IQ_UA"),
                                       ("LoadReg","LOADREG_MV_MA"),("LineReg","LINEREG_MV_V"),
                                       ("Dropout","DROPOUT_MV"),("Peaking","PEAKING_DB"))) + ")")
print("done", flush=True)
