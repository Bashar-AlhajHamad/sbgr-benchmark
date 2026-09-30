"""Choose Circuit B's reference design by a criterion declared BEFORE looking at the result.

Why re-choose it. The hand-derived reference burns 66.9 uW, roughly the 75th percentile of the design
box. Because gate 1 forbids a threshold that excludes the reference design, Power_max cannot be set
tighter than 66.9 uW, so 81 % of the box passes it and Power stops being a real specification. A
power-hungry anchor is a defect in the problem setup, not a cosmetic one -- no designer picks a
reference two-stage amplifier that wastes current.

THE CRITERION, fixed here before the search runs:

    minimise supply power, subject to
        gain        >= 60 dB          a two-stage amplifier's ordinary target
        UGB         >= 5 MHz
        phase margin>= 60 deg         the classic stability target, and Circuit A's own PM threshold
        Vout        in [0.6, 1.2] V   comfortably centred, tighter than the specification window
                                      so the anchor is not sitting on the edge of its own spec

This is ordinary design practice with a declared objective, and it fixes only where the ANCHOR sits.
It does not touch the optimizers' problem: all six algorithms face identical thresholds whatever the
anchor is, so this cannot affect fairness, only difficulty.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, ".")
import opamp_probe as P
import ngspice_bridge as nb

MODEL_W = (0.42, 100.0)

def derive(w1, w3, w5, m2, k, l_all, cc):
    w6 = min(max(m2 * w3, MODEL_W[0]), MODEL_W[1])
    w7 = min(max(k * w5 * m2 / 2.0, MODEL_W[0]), MODEL_W[1])
    return dict(w1=round(w1, 6), w3=round(w3, 6), w5=round(w5, 6), w6=round(w6, 6),
                w7=round(w7, 6), l_all=round(l_all, 6), cc=round(cc, 6))

srv = nb.NgspiceServer(deck_text=P.deck_text(), eval_timeout=25.0)
srv.load(); print(f"PDK parse {srv.load_seconds:.1f}s", flush=True)

def meas(x):
    out = srv.send([f"alterparam {kk}={float(x[kk]):.9g}" for kk in P.VAR_NAMES] + P.COMMANDS)
    if out is None:
        srv.resync() or srv.restart(); return None
    kind, _ = nb.classify_output(out, 0)
    v = P._grab(out, "gain_db", "ugb", "ph_lf", "ph_ugb", "supp_db", "v(out)", "i(vdd)")
    if kind or v["gain_db"] is None or v["v(out)"] is None or v["i(vdd)"] is None: return None
    if None in (v["ugb"], v["ph_lf"], v["ph_ugb"]): return None
    pm = 180 - ((v["ph_lf"] - v["ph_ugb"]) % 360)
    area = (2*x["w1"] + 2*x["w3"] + 2*x["w5"] + x["w6"] + x["w7"]) * x["l_all"]
    return dict(Gain=v["gain_db"], UGB=v["ugb"]/1e6, PhaseMargin=pm,
                Power=abs(v["i(vdd)"])*1.8e6, Vout=v["v(out)"], Area=area,
                PSRR=v["gain_db"]-v["supp_db"])

def meets(m):
    return (m and m["Gain"] >= 60.0 and m["UGB"] >= 5.0 and m["PhaseMargin"] >= 60.0
            and 0.6 <= m["Vout"] <= 1.2)

# grid over the bias-setting variables; w1 and w3 fixed at the hand-derived values, since the
# criterion is about CURRENT, and m2/k/w5 are what set it
best, n, t0 = None, 0, time.time()
for w5 in (4.0, 6.0, 8.0, 10.0):
    for m2 in (1.5, 2.0, 2.5, 3.0, 4.0):
        for k in (0.80, 0.85, 0.90, 0.95):
            for cc in (1.0, 2.0, 3.0):
                for l_all in (0.5, 1.0, 1.5):
                    x = derive(10.0, 10.0, w5, m2, k, l_all, cc)
                    m = meas(x); n += 1
                    if meets(m) and (best is None or m["Power"] < best[1]["Power"]):
                        best = (x, m, dict(w5=w5, m2=m2, k=k, cc=cc, l_all=l_all))
                        print(f"  new best: {m['Power']:6.2f} uW  gain {m['Gain']:5.1f} dB"
                              f"  ugb {m['UGB']:6.3f} MHz  PM {m['PhaseMargin']:5.1f}"
                              f"  vout {m['Vout']:.3f}   (w5={w5} m2={m2} k={k} cc={cc} L={l_all})",
                              flush=True)
srv.close()
print(f"\nevaluated {n} candidates in {time.time()-t0:.0f}s", flush=True)
if best is None:
    print("NO candidate meets the declared criterion -- loosen it explicitly, or report this")
else:
    x, m, p = best
    print("\nREFERENCE DESIGN (lowest power meeting the declared criterion)")
    print("  parameters :", "  ".join(f"{a}={b:g}" for a, b in p.items()))
    print("  widths     :", "  ".join(f"{a}={x[a]:g}" for a in P.VAR_NAMES))
    for a in ("Gain","UGB","PhaseMargin","Power","Vout","Area","PSRR"):
        print(f"  {a:12s} {m[a]:10.4f}")
print("done", flush=True)
