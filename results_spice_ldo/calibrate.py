"""Freeze Circuit C's six thresholds from the 512-point probe. One declared knob, mechanical mapping.

THE RULE is the corrected section 4 of PREREGISTRATION_2026-08-26_circuits_2_and_3.md, the same one
used for Circuit B, and it has exactly one degree of freedom:

    declare ONE target marginal pass rate T. For each scaled specification, set the threshold to the
    empirical quantile of the probe that lets T % of the box pass; clamp so the reference design
    still passes. Then require joint feasibility in [0.5 %, 5 %]. T is the only knob and moving it
    moves all six thresholds together.

Why not the "uniform relative slack" the pre-registration originally described: applied to Circuit B
it needed a scalar of 1.04 before joint feasibility entered the band, at which three thresholds went
NEGATIVE and four of six specifications became impossible to fail. It also mis-described its own
precedent -- Circuit A's frozen slacks are 0.12 %, 6.2 %, 16.4 %, 6.0 %, 4.8 % and 5.8 %, per
specification. The correction is recorded in the pre-registration rather than applied silently.

Candidate selection among T values is declared here, before the numbers are seen: prefer a T whose
marginal rates all lie in [10 %, 60 %]; among those take the one whose joint feasibility is nearest
the geometric centre of [0.5 %, 5 %].

THE OUTPUT WINDOW took three attempts, and each failure is worth recording because each was caught by
a different gate. At a declared +/-2 % all 452 measurable probe points passed -- not a specification
at all, only a restatement that the simulation had succeeded. At a declared +/-0.1 % it bound at 34 %
but excluded the reference design, breaking gate 1. Calibrated by quantile and clamped to the anchor,
it landed at exactly the anchor's own error and STILL excluded it, by 35 microvolts, because the clamp
used a four-decimal rounding of a full-precision measurement. It is now fixed at +/-0.2 % -- the next
round figure above the anchor's 2.235 mV error, an ordinary LDO accuracy spec, admitting the anchor
with 165 uV of margin.

With an ideal reference and a fixed divider ratio the DC output is exact by construction, so the
deviation that remains measures LOOP GAIN -- which is why this is a real specification once its
threshold is not set so loosely that everything passes.

Closed-loop PEAKING is no longer one of the six. It is a HARD feasibility limit in the evaluator, and
this script applies the same limit when loading the probe, so both score the box identically. The
measurement that forced the change: the threshold can never go below the reference design's own
0.0005 dB, and even there 78 % of the box passes, so reaching the gate's 60 % ceiling would require a
negative threshold.

A point that fails to simulate counts as INFEASIBLE, not as missing data. Dropping such points would
flatter every threshold, because the unmeasurable corners of the box are the violent ones.
"""
import csv
import math
import statistics as S
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "spice"))
import spice_problem_ldo as L                                  # noqa: E402

CSV = Path(__file__).with_name("lhs_probe.csv")
OBJECTIVE = "PSRR_DB"
# every specification is "lower is better" except the two-sided output window
LOWER_IS_BETTER = ("LOADREG_MV_MA", "LINEREG_MV_V", "IQ_UA", "DROPOUT_MV", "AREA_UM2")
SPECS = LOWER_IS_BETTER + ("VOUT_V",)
LABEL = {"LOADREG_MV_MA": "LoadReg_max", "LINEREG_MV_V": "LineReg_max", "IQ_UA": "Iq_max",
         "DROPOUT_MV": "Dropout_max", "AREA_UM2": "Area_max", "VOUT_V": "Vout window"}
UNIT = {"LOADREG_MV_MA": "mV/mA", "LINEREG_MV_V": "mV/V", "IQ_UA": "uA",
        "DROPOUT_MV": "mV", "AREA_UM2": "um^2", "VOUT_V": "V"}
# The window is symmetric about the target; its HALF-WIDTH is calibrated, not declared.
VOUT_TARGET = L.VOUT_TARGET


def load():
    rows = []
    for r in csv.DictReader(CSV.open(encoding="utf-8")):
        if not r.get("i"):
            continue
        d = {"ok": r["sim_ok"] == "1"}
        # The evaluator now rejects a marginally stable loop outright, so the probe must be scored
        # the same way: a point above the hard peaking limit is infeasible, not merely penalised.
        try:
            if float(r["PEAKING_DB"]) > L.PEAKING_HARD_LIMIT_DB:
                d["ok"] = False
        except (ValueError, TypeError, KeyError):
            pass
        for k in SPECS + (OBJECTIVE,):
            try:
                d[k] = float(r[k])
            except (ValueError, TypeError, KeyError):
                d[k] = None
        rows.append(d)
    return rows


def main() -> int:
    if not L.REFERENCE_METRICS:
        print("no frozen reference metrics; run pick_reference.py first")
        return 1
    REF = {"LOADREG_MV_MA": L.REFERENCE_METRICS["LoadReg"],
           "LINEREG_MV_V": L.REFERENCE_METRICS["LineReg"],
           "IQ_UA": L.REFERENCE_METRICS["Iq"],
           "DROPOUT_MV": L.REFERENCE_METRICS["Dropout"],
           "AREA_UM2": L.REFERENCE_METRICS["Area"],
           "VOUT_V": L.REFERENCE_METRICS["Vout"],
           OBJECTIVE: L.REFERENCE_METRICS["PSRR"]}

    rows = load()
    n = len(rows)
    print(f"probe: {n} points, {sum(1 for r in rows if r['ok'])} measurable\n")

    # THE WINDOW HALF-WIDTH IS FIXED AT +/-0.2 %, and that number is not arbitrary.
    #
    # The anchor's measured output is 1.202235 V, a deviation of 2.235 mV. A window calibrated by
    # quantile came out at exactly 2.2 mV -- 35 uV TIGHTER than the anchor's own error, because the
    # quantile was computed from REFERENCE_METRICS rounded to four decimals while the evaluator
    # measures full precision. Gate 1 then failed on the reference design itself, which is the
    # subtlest way this calibration could have gone wrong: a benchmark excluding its own anchor by
    # 35 microvolts.
    #
    # 2.4 mV is the next round figure above it: +/-0.2 % of 1.2 V, an ordinary LDO accuracy spec, and
    # it admits the anchor with 165 uV of margin. A window whose EDGE coincides with the anchor would
    # make feasibility a floating-point coin toss, so the margin is deliberate rather than incidental.
    VOUT_HALF = 0.0024
    ref_dev = abs(REF["VOUT_V"] - VOUT_TARGET)
    if ref_dev >= VOUT_HALF:
        print(f"FATAL: the reference design's deviation {ref_dev*1e3:.3f} mV is not inside the "
              f"{VOUT_HALF*1e3:.1f} mV window; gate 1 cannot pass")
        return 3

    def thresholds(T):
        want = max(1, int(T / 100.0 * n))
        th = {}
        for k in LOWER_IS_BETTER:
            v = sorted(r[k] for r in rows if r["ok"] and r[k] is not None)
            # never TIGHTER than the reference design: gate 1 forbids excluding the anchor
            th[k] = max(v[min(len(v) - 1, want - 1)], REF[k]) if v else REF[k]
        # the output window, by the same rule, on |Vout - target|
        th["_vout_half"] = VOUT_HALF
        return th

    def viol(r, th):
        if not r["ok"]:
            return {k: 1 for k in SPECS}
        d = {k: (1 if r[k] is None else int(r[k] > th[k])) for k in LOWER_IS_BETTER}
        d["VOUT_V"] = (1 if r["VOUT_V"] is None
                       else int(abs(r["VOUT_V"] - VOUT_TARGET) > th["_vout_half"]))
        return d

    def report(T):
        th = thresholds(T)
        vs = [viol(r, th) for r in rows]
        joint = sum(1 for v in vs if sum(v.values()) == 0)
        marg = {k: 100 * sum(1 for v in vs if v[k] == 0) / n for k in SPECS}
        return th, 100 * joint / n, joint, marg

    print("scanning the one knob: target marginal pass rate T")
    print(f"  {'T%':>4s} {'joint%':>7s}  " + " ".join(f"{LABEL[k].split('_')[0]:>11s}"
                                                     for k in SPECS))
    cands = []
    for T in range(20, 81, 5):
        th, pct, cnt, marg = report(T)
        inband = 0.5 <= pct <= 5.0
        bal = all(10 <= marg[k] <= 60 for k in SPECS)
        if inband:
            cands.append(T)
        print(f"  {T:4d} {pct:7.2f}  " + " ".join(f"{marg[k]:11.1f}" for k in SPECS)
              + ("   in band" if inband else "") + ("  balanced" if bal else ""))

    if not cands:
        print("\nNO T PUTS JOINT FEASIBILITY IN [0.5 %, 5 %] -- reported, not worked around.")
        print("The box or the reference design has to change, and that change must be logged.")
        return 2

    centre = math.sqrt(0.5 * 5.0)
    bal_c = [T for T in cands if all(10 <= report(T)[3][k] <= 60 for k in SPECS)]
    pool = bal_c or cands
    T = min(pool, key=lambda t: abs(math.log(report(t)[1] / centre)))
    th, pct, cnt, marg = report(T)
    print(f"\nCHOSEN T = {T} %"
          f"{' (from the balanced candidates)' if bal_c else ' (no candidate was fully balanced)'}"
          f"  ->  {cnt}/{n} jointly feasible = {pct:.3f} %   band centre {centre:.2f} %\n")

    print("FROZEN THRESHOLDS")
    for k in LOWER_IS_BETTER:
        print(f"  {LABEL[k]:16s} {th[k]:11.4f} {UNIT[k]:6s} reference {REF[k]:10.4f}"
              f"   slack {(th[k]/REF[k]-1)*100 if REF[k] else float('nan'):7.2f} %")
    half = th["_vout_half"]
    print(f"  {'Vout window':16s} [{VOUT_TARGET-half:.4f}, {VOUT_TARGET+half:.4f}] V"
          f"   +/-{100*half/VOUT_TARGET:.3f} %   reference {REF['VOUT_V']:.6f}"
          f" (dev {1e3*ref_dev:.3f} mV, margin {1e3*(half-ref_dev):.3f} mV)")

    print("\nACCEPTANCE GATES")
    rv = viol(dict(ok=True, **REF), th)
    g1 = sum(rv.values()) == 0
    print(f"  1 reference design feasible                 : {'PASS' if g1 else 'FAIL'}"
          + ("" if g1 else "   violates " + ", ".join(k for k, v in rv.items() if v)))
    print(f"  2 random joint feasibility in [0.5 %, 5 %]  : "
          f"{'PASS' if 0.5 <= pct <= 5.0 else 'FAIL'}   ({pct:.3f} %)")
    bad = [k for k in SPECS if not 10 <= marg[k] <= 60]
    print(f"  3 marginal pass rates in [10 %, 60 %]       : {'PASS' if not bad else 'FAIL'}")
    for k in sorted(SPECS, key=lambda z: -marg[z]):
        print(f"      {LABEL[k]:16s} {marg[k]:6.1f} %{'   <-- outside' if k in bad else ''}")
    feas = [r for r in rows if r["ok"] and sum(viol(r, th).values()) == 0]
    fo = [r[OBJECTIVE] for r in feas if r[OBJECTIVE] is not None]
    print(f"  4 best random-feasible objective            : "
          + (f"{max(fo):.2f} dB   (of {len(fo)} feasible, median {S.median(fo):.2f} dB)"
             if fo else "none"))
    print(f"      reference design objective              : {REF[OBJECTIVE]:.2f} dB")
    allo = [r[OBJECTIVE] for r in rows if r[OBJECTIVE] is not None]
    print(f"      best anywhere in the box                : {max(allo):.2f} dB")
    if fo and len(fo) < 5:
        print(f"      NOTE: only {len(fo)} feasible probe points, so the gate-4 figure is an "
              f"estimate from {len(fo)} samples")

    print("\nPASTE-READY")
    print("CONSTRAINTS = {")
    print(f'    "Vout_min": {VOUT_TARGET-half:.4f},'.ljust(40)
          + f"# V   +/-{100*half/VOUT_TARGET:.3f} % about {VOUT_TARGET:g} V")
    print(f'    "Vout_max": {VOUT_TARGET+half:.4f},'.ljust(40) + "# V")
    for k in LOWER_IS_BETTER:
        print(f'    "{LABEL[k]}": {th[k]:.4f},'.ljust(40)
              + f"# {UNIT[k]}   reference {REF[k]:.4f}")
    print("}")
    print(f"CALIBRATED = True")
    print(f"PROBE_BEST_FEASIBLE_PSRR_DB = {max(fo):.2f}" if fo
          else "PROBE_BEST_FEASIBLE_PSRR_DB = float('nan')  # no feasible probe point")
    print(f"NOMINAL_OBJECTIVE_DB = {REF[OBJECTIVE]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
