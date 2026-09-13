"""Choose the anchor design for the current-mode (Banba) case, from the 512-point probe.

WHY A CHOICE IS NEEDED HERE AND NOT FOR THE KUIJK CASE. The Kuijk deck's reference design is the
published one: its core is trimmed to that design's 1220 mV and its amplifier sizing comes from it, so
the nominal point is given rather than selected. There is no published SKY130 current-mode bandgap, so
this case has to take its anchor from our own probe.

THE RULE: among probe points jointly feasible under the published-faithful threshold set, take the one
with the BEST objective. It serves as both the anchor for threshold calibration and the single bar the
campaign must beat.

A RULE WAS DECLARED, RUN, AND CHANGED -- recorded rather than replaced silently. The first rule was
"take the MEDIAN feasible point", reasoning that an anchor should be an ordinary engineering design
rather than the box optimum, so that beating it means something. Run, it selected a design with
PSRR = 2.203 dB out of a feasible range of -10.665 to 19.166 dB. That is worse than the best point a
512-point random search found, so the "beat the reference" bar would have been EASIER than the
random-search bar and entirely dominated by it -- the opposite of its purpose. By contrast the Kuijk
reference achieves 23.85 dB against a random-feasible bar of 15.50 dB: there the reference IS the
harder bar, which is what makes it worth quoting.

The deeper reason the first rule failed is structural. Calling any probe point "the reference design"
is a fiction when no published design exists. What this case honestly has is ONE bar -- the best
design a 512-point random search reached -- and that is what is used, labelled as such.

    python pick_reference.py           # prints the anchor and the lines to paste
"""
import csv
import statistics as S
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "spice"))
import spice_problem as sp                                      # noqa: E402

CSV = Path(__file__).with_name("lhs_probe.csv")
VAR_NAMES = [v[0] for v in sp.VARS_BANBA]

# The published-faithful set. Five of the six 65-nm thresholds verbatim; TC alone is re-anchored,
# because the published <= 8 ppm/degC is not reached anywhere in the 508 measured points of this
# design box -- the best is 10.56 ppm -- so under it a campaign would return 0 % success for every
# algorithm. A sample cannot establish that the region is empty, and the manuscript claims only that
# this sample did not reach the published value. The Kuijk case re-anchored TC identically, from 8 to
# 26 ppm, and tab:spice_specs marks three entries as untransplantable, TC among them; an earlier
# draft marked two, and that was corrected. See calibrate.py.
PUBLISHED = [
    ("VREF", lambda v: 0.798 <= v <= 0.802, "VREF 798-802 mV", "published"),
    ("TC", lambda v: v <= 26.0, "TC <= 26 ppm/degC", "re-anchored from 8"),
    ("LOOP_GAIN_DB", lambda v: v >= 40.0, "LoopGain >= 40 dB", "published"),
    ("PHASE_MARGIN_DEG", lambda v: v >= 60.0, "PhaseMargin >= 60 deg", "published"),
    ("GAIN_MARGIN_DB", lambda v: v >= 20.0, "GainMargin >= 20 dB", "published"),
    ("POWER_UW", lambda v: v <= 400.0, "Power <= 400 uW", "published"),
]


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def main() -> int:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    ok = [r for r in rows if r.get("sim_ok") == "1"]
    print(f"probe: {len(rows)} points, {len(ok)} simulated\n")

    print("published-faithful threshold set, marginal pass rates over all 512 points:")
    for k, test, label, origin in PUBLISHED:
        n = sum(1 for r in ok if num(r, k) is not None and test(num(r, k)))
        print(f"    {label:24s} {n:4d}/512  ({100 * n / len(rows):5.1f} %)   [{origin}]")

    feas = [r for r in ok
            if all(num(r, k) is not None and t(num(r, k)) for k, t, _l, _o in PUBLISHED)]
    print(f"\n    jointly feasible        {len(feas):4d}/512  "
          f"({100 * len(feas) / len(rows):5.1f} %)")
    if not feas:
        print("\nNO jointly feasible point: the anchor cannot be chosen and gate 1 cannot pass.")
        return 1

    psrr = sorted(num(r, "PSRR_DB") for r in feas)
    print(f"    feasible PSRR range     {psrr[0]:.3f} .. {psrr[-1]:.3f} dB   "
          f"median {S.median(psrr):.3f}")

    ref = max(feas, key=lambda r: num(r, "PSRR_DB"))

    print(f"\nANCHOR AND RANDOM-SEARCH BAR (best of {len(feas)} feasible probe points)")
    for n in VAR_NAMES:
        print(f"    {n:9s} {float(ref[n]):10.4f}")
    print()
    for k, _t, label, _o in PUBLISHED:
        print(f"    {label:24s} {num(ref, k):10.4f}")
    print(f"    {'PSRR (objective)':24s} {num(ref, 'PSRR_DB'):10.4f} dB")

    print("\npaste into spice_problem.py:")
    vals = ", ".join(f"{n}={float(ref[n]):.4f}" for n in VAR_NAMES)
    print(f"    NOMINAL_BANBA = dict({vals})")
    print(f"    NOMINAL_OBJECTIVE_BANBA_DB = {num(ref, 'PSRR_DB'):.4f}")
    print(f"    PROBE_BEST_FEASIBLE_PSRR_BANBA_DB = {num(ref, 'PSRR_DB'):.4f}")
    best = max(num(r, "PSRR_DB") for r in ok)
    print(f"    PROBE_BEST_PSRR_ANY_BANBA_DB = {best:.4f}   "
          f"# best in the box, feasible or NOT; diagnostic only")
    print("\nNOTE: the anchor and the random-search bar are the SAME number here, by construction.")
    print("Unlike the Kuijk case there is no published design to quote as a second, harder bar, so")
    print("only one bar is reported rather than inventing a weaker one out of the same probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
