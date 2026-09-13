"""Freeze TWO threshold sets for the current-mode (Banba) case, and check the four gates on each.

WHY TWO SETS. They answer different objections, and running both makes the answer to each checkable
instead of asserted.

  (A) PUBLISHED-FAITHFUL. Five of the six 65-nm thresholds verbatim, TC alone re-anchored. This is
      the set the Kuijk deck could not use: its V_REF window (798-802 mV) is unreachable when the
      resistor network pins V_REF at the bandgap voltage, and its gain-margin floor (20 dB) excludes
      the Kuijk reference design, which reaches only 14.86 dB. Both become usable on a current-mode
      core. Cost: two specifications end up loose (see gate 3 below), which we disclose rather than
      calibrate away.

  (B) CALIBRATED. One declared knob T, mechanically mapped, so that no specification is decorative.
      Same rule as Circuits B and C. Cost: the thresholds are ours again, not the published ones.

Reporting BOTH turns "did you pick the thresholds that suited you?" from an accusation into a measured
question: does the optimizer ranking change between (A) and (B)? If it does not, that is evidence the
ranking is a property of the circuit. If it does, that is a finding about benchmark sensitivity. Either
way the answer is in the paper rather than in the authors' good intentions.

THE TC PROBLEM, and a defect it exposes in our own table. The published TC is <= 8 ppm/degC. Across
508 simulated points the best this design box reaches is 10.56 ppm, so no measured point clears the
published value and a campaign under it would return 0 % success for every algorithm and measure
nothing. A 512-point sample cannot establish that the feasible region is empty, and the manuscript
claims only that the sample did not reach the published value. TC is therefore re-anchored to 26 ppm,
exactly as the Kuijk case did (also from 8). tab:spice_specs marks three of the six published
thresholds as untransplantable -- V_REF, gain margin and TC. An earlier draft marked only the first
two; that undercount was corrected in the
manuscript independently of anything decided here.

A point that fails to simulate counts as INFEASIBLE, not as missing data. Dropping such points would
flatter every threshold, because the unmeasurable corners of the box are the violent ones.

    python calibrate.py
"""
import csv
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
CSV = Path(__file__).with_name("lhs_probe.csv")

VREF_TARGET = 0.800
# name -> (direction, label, unit). "lo" = lower is better, "hi" = higher is better.
SPECS = [
    ("TC", "lo", "TC_max", "ppm/degC"),
    ("LOOP_GAIN_DB", "hi", "LoopGain_min", "dB"),
    ("PHASE_MARGIN_DEG", "hi", "PhaseMargin_min", "deg"),
    ("GAIN_MARGIN_DB", "hi", "GainMargin_min", "dB"),
    ("POWER_UW", "lo", "Power_max", "uW"),
]
PUBLISHED = {"VREF_min": 0.798, "VREF_max": 0.802, "TC_max": 26.0, "LoopGain_min": 40.0,
             "PhaseMargin_min": 60.0, "GainMargin_min": 20.0, "Power_max": 400.0}
ORIGIN = {"VREF_min": "published", "VREF_max": "published", "TC_max": "re-anchored from 8",
          "LoopGain_min": "published", "PhaseMargin_min": "published",
          "GainMargin_min": "published", "Power_max": "published"}
BAND = (0.5, 5.0)                 # joint feasibility, per cent
MARGINAL = (10.0, 60.0)           # per-specification pass rate, per cent


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def load():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    ok, bad = [], 0
    for r in rows:
        if r.get("sim_ok") == "1" and all(num(r, k) is not None for k, *_ in SPECS) \
                and num(r, "VREF") is not None:
            ok.append(r)
        else:
            bad += 1
    return rows, ok, bad


def penalty(r, C):
    v = num(r, "VREF")
    p = int(not (C["VREF_min"] <= v <= C["VREF_max"]))
    for k, d, label, _u in SPECS:
        x = num(r, k)
        p += int(x > C[label]) if d == "lo" else int(x < C[label])
    return p


def evaluate(C, rows, ok, bad, anchor, title, origin=None):
    # origin is passed explicitly rather than read from the module-level ORIGIN, which is keyed by
    # threshold name and therefore matches for every row of either set. Set B derives all seven of
    # its thresholds from the box, so it passes {} and every row prints "calibrated".
    origin = ORIGIN if origin is None else origin
    n = len(rows)
    print(f"\n{'=' * 94}\n{title}\n{'=' * 94}")
    print(f"  {'threshold':16s} {'value':>12s} {'unit':>10s}  {'marginal pass':>14s}  origin")
    marg = {}
    v = num(anchor, "VREF")
    nv = sum(1 for r in ok if C["VREF_min"] <= num(r, "VREF") <= C["VREF_max"])
    marg["VREF window"] = 100 * nv / n
    print(f"  {'VREF window':16s} {C['VREF_min']:.4f}-{C['VREF_max']:.4f}{'V':>4s}"
          f"  {100 * nv / n:12.1f} %  {origin.get('VREF_min', 'calibrated')}")
    for k, d, label, unit in SPECS:
        cnt = sum(1 for r in ok
                  if (num(r, k) <= C[label] if d == "lo" else num(r, k) >= C[label]))
        marg[label] = 100 * cnt / n
        print(f"  {label:16s} {C[label]:12.4f} {unit:>10s}  {100 * cnt / n:12.1f} %  "
              f"{origin.get(label, 'calibrated')}")

    feas = [r for r in ok if penalty(r, C) == 0]
    joint = 100 * len(feas) / n
    ap = penalty(anchor, C)
    print(f"\n  joint feasibility        {len(feas):4d}/{n}  ({joint:.2f} %)"
          f"      unmeasurable points counted infeasible: {bad}")
    print(f"  anchor penalty           {ap}")

    # which specification drives infeasibility
    infeas = [r for r in ok if penalty(r, C) > 0]
    if infeas:
        share = {}
        vb = sum(1 for r in infeas if not (C["VREF_min"] <= num(r, "VREF") <= C["VREF_max"]))
        share["VREF window"] = 100 * vb / len(infeas)
        for k, d, label, _u in SPECS:
            c = sum(1 for r in infeas
                    if (num(r, k) > C[label] if d == "lo" else num(r, k) < C[label]))
            share[label] = 100 * c / len(infeas)
        worst = max(share, key=share.get)

    print("\n  GATES")
    g1 = ap == 0
    g2 = BAND[0] <= joint <= BAND[1]
    off = {k: v for k, v in marg.items() if not (MARGINAL[0] <= v <= MARGINAL[1])}
    g3 = not off
    g4 = share[worst] <= 80.0 if infeas else True
    print(f"    1  anchor feasible                        {'PASS' if g1 else 'FAIL'}")
    print(f"    2  joint feasibility in [{BAND[0]}, {BAND[1]}] %          "
          f"{'PASS' if g2 else 'FAIL'}   ({joint:.2f} %)")
    print(f"    3  every marginal rate in [{MARGINAL[0]:.0f}, {MARGINAL[1]:.0f}] %       "
          f"{'PASS' if g3 else 'FAIL'}"
          + ("" if g3 else "   outside: " + ", ".join(f"{k} {v:.1f} %" for k, v in off.items())))
    print(f"    4  no specification drives > 80 % of it   {'PASS' if g4 else 'FAIL'}"
          f"   ({worst} {share[worst]:.1f} %)" if infeas else "")
    return dict(C=C, joint=joint, gates=(g1, g2, g3, g4), marginal=marg)


def quantile(vals, q):
    """q in [0,1]; linear interpolation on the sorted sample."""
    s = sorted(vals)
    if not s:
        return None
    i = q * (len(s) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def calibrated_set(ok, anchor, T):
    """Thresholds that let T per cent of the measured box pass each specification, clamped so the
    anchor still passes. One knob; moving T moves all six together."""
    C = {}
    dev = [abs(num(r, "VREF") - VREF_TARGET) for r in ok]
    half = quantile(dev, T / 100.0)
    half = max(half, abs(num(anchor, "VREF") - VREF_TARGET) * 1.001)
    C["VREF_min"], C["VREF_max"] = VREF_TARGET - half, VREF_TARGET + half
    for k, d, label, _u in SPECS:
        vals = [num(r, k) for r in ok]
        if d == "lo":
            thr = quantile(vals, T / 100.0)
            C[label] = max(thr, num(anchor, k))
        else:
            thr = quantile(vals, 1.0 - T / 100.0)
            C[label] = min(thr, num(anchor, k))
    return C


def main() -> int:
    rows, ok, bad = load()
    print(f"probe: {len(rows)} points, {len(ok)} measurable, {bad} counted infeasible")

    # the anchor: best feasible point under the published-faithful set (see pick_reference.py)
    anchor = None
    for r in ok:
        if penalty(r, PUBLISHED) == 0 and (anchor is None
                                           or num(r, "PSRR_DB") > num(anchor, "PSRR_DB")):
            anchor = r
    if anchor is None:
        print("no feasible anchor under the published-faithful set")
        return 1
    print(f"anchor PSRR {num(anchor, 'PSRR_DB'):.4f} dB, VREF {num(anchor, 'VREF'):.5f} V")

    A = evaluate(dict(PUBLISHED), rows, ok, bad, anchor,
                 "SET A -- PUBLISHED-FAITHFUL (five of six thresholds verbatim)")

    # --- set B: sweep the single knob, then apply the declared preference ---
    print(f"\n{'=' * 94}\nSET B -- CALIBRATED: sweeping the single knob T\n{'=' * 94}")
    print(f"  {'T %':>6s} {'joint %':>9s} {'marginals in band?':>20s}")
    cands = []
    for T in (20, 25, 30, 35, 40, 45, 50, 55, 60):
        C = calibrated_set(ok, anchor, T)
        feas = [r for r in ok if penalty(r, C) == 0]
        joint = 100 * len(feas) / len(rows)
        marg = {}
        nv = sum(1 for r in ok if C["VREF_min"] <= num(r, "VREF") <= C["VREF_max"])
        marg["VREF window"] = 100 * nv / len(rows)
        for k, d, label, _u in SPECS:
            c = sum(1 for r in ok
                    if (num(r, k) <= C[label] if d == "lo" else num(r, k) >= C[label]))
            marg[label] = 100 * c / len(rows)
        inband = all(MARGINAL[0] <= v <= MARGINAL[1] for v in marg.values())
        ok_joint = BAND[0] <= joint <= BAND[1]
        print(f"  {T:6d} {joint:9.2f} {('yes' if inband else 'no'):>20s}"
              f"{'   <- joint in band' if ok_joint else ''}")
        if inband and ok_joint:
            cands.append((T, joint, C))

    if not cands:
        print("\n  no T satisfies both gate 2 and gate 3; reporting the T nearest the band centre")
        centre = math.sqrt(BAND[0] * BAND[1])
        allT = []
        for T in (20, 25, 30, 35, 40, 45, 50, 55, 60):
            C = calibrated_set(ok, anchor, T)
            j = 100 * len([r for r in ok if penalty(r, C) == 0]) / len(rows)
            allT.append((abs(j - centre), T, C))
        allT.sort()
        Tbest, Cbest = allT[0][1], allT[0][2]
    else:
        centre = math.sqrt(BAND[0] * BAND[1])
        cands.sort(key=lambda z: abs(z[1] - centre))
        Tbest, Cbest = cands[0][0], cands[0][2]
    print(f"\n  declared preference: marginals in band, then joint nearest the geometric centre "
          f"of [{BAND[0]}, {BAND[1]}] % = {math.sqrt(BAND[0] * BAND[1]):.2f} %  ->  T = {Tbest}")

    B = evaluate(Cbest, rows, ok, bad, anchor, f"SET B -- CALIBRATED at T = {Tbest} %",
                 origin={})

    print(f"\n{'=' * 94}\nPASTE INTO spice_problem.py\n{'=' * 94}")
    for nm, res in (("CONSTRAINTS_BANBA", A), ("CONSTRAINTS_BANBA_CAL", B)):
        print(f"\n{nm} = {{")
        for k in ("VREF_min", "VREF_max", "TC_max", "LoopGain_min", "PhaseMargin_min",
                  "GainMargin_min", "Power_max"):
            print(f"    {k!r}: {res['C'][k]:.6g},")
        print("}")
    print(f"\n# gates  A: {A['gates']}   B: {B['gates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
