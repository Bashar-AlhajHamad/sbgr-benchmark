"""
SpiceBGRProblem -- a transistor-level SKY130 bandgap evaluator that duck-types
problems.SBGRProblem, so the unmodified algorithms.py can optimize against real ngspice.

Semantics reproduced exactly from problems.py:
    fitness  = -PSRR_DB + 1000 * penalty
    penalty  = unweighted COUNT of violated specifications, 0..6
    feasible <=> penalty == 0
and the same metric keys, so run.py's CSV row schema and its statistics helpers work
unchanged. algorithms.py and run.py are never modified.

Measured facts this design is built around (all verified against the real PDK, see the
project plan for the evidence):

  * Spawning ngspice per evaluation costs ~85 s because the sky130 tt library is re-parsed
    every time. A persistent NgspiceServer parses once (~70 s) and then costs ~0.6 s per
    candidate for the full analysis set. That is the difference between 76 CPU-hours and
    2350 CPU-hours for this campaign, so the server is not optional.
  * NaN wins np.argmin: np.argmin([1.0, nan, 2.0]) == 1. Every optimizer in algorithms.py
    seeds its incumbent with np.argmin(fit), so a single NaN fitness among the first `pop`
    evaluations becomes the incumbent, every later `<` comparison against it is False, and
    the run is destroyed in silence. `evaluate` therefore asserts a finite fitness.
  * A deck error can poison the interpreter so that every LATER candidate fails too, which
    would quietly turn the rest of a run infeasible. NgspiceServer restarts on deck_error.
  * Out-of-range geometry is a deck error, so parameters are clipped to the model's valid
    range here rather than being allowed to reach ngspice.
  * The loop-gain measurement is a TWO-PHASE exchange (two AC runs plus a cross-plot
    combination that needs the literal plot name), which `evaluate` cannot express, hence
    the explicit use of NgspiceServer.send().
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

import ngspice_bridge as nb

TEMPLATE = "bgr_sky130.cir.tmpl"
TEMPLATE_HIGHDIM = "bgr_sky130_highdim.cir.tmpl"
TEMPLATE_BANBA = "bgr_banba_sky130.cir.tmpl"
TEMPLATE_BROKAW = "bgr_brokaw_sky130.cir.tmpl"

# ---------------------------------------------------------------- design space
# name, lower, upper, unit, quantisation step (0 = continuous)
VARS = (
    ("w_pdiff", 1.0, 40.0, "um", 0.005),
    ("w_ncasc", 1.0, 40.0, "um", 0.005),
    ("w_pmir", 1.0, 40.0, "um", 0.005),
    ("w_pass", 8.0, 100.0, "um", 0.005),   # 100 um is the model's W ceiling
    ("l_all", 0.15, 8.0, "um", 0.005),     # 0.15 um is the minimum L
    ("iss", 1.0, 40.0, "uA", 0.0),
    ("vbnv", 0.9, 1.7, "V", 0.0),
)
VAR_NAMES = tuple(v[0] for v in VARS)

NOMINAL = dict(w_pdiff=8.0, w_ncasc=4.0, w_pmir=8.0, w_pass=64.0,
               l_all=1.0, iss=10.0, vbnv=1.35)

# ---- high-dimensional variant: the matched-pair and common-length assumptions are removed,
# giving every transistor its own width and length. 7 -> 16 variables (2.3x), mirroring the
# surrogate's SBGR-Highdim growth of 12 -> 30 (2.5x). Constraints stay at the Base values, as
# the surrogate also does, so dimensionality is the only difference.
#
# The mismatch of each pair is a bounded RATIO, not a free absolute size. That was forced by
# measurement: with fully independent per-device widths and lengths, 84 of 96 random points had
# no unity-gain crossing at all -- 94 % of the space unmeasurable, 0 % feasible, so a campaign
# there would have been almost entirely floor values. Bounding the mismatch to [0.5, 2] keeps
# the amplifier functional while still forcing the optimizer to rediscover matching, and it
# models a real design more faithfully: matched pairs have tolerances, not arbitrary sizes.
VARS_HIGHDIM = (
    ("w_pdiff", 1.0, 40.0, "um", 0.005), ("r_wdiff", 0.5, 2.0, "-", 0.0),
    ("l_pdiff", 0.15, 8.0, "um", 0.005), ("r_ldiff", 0.5, 2.0, "-", 0.0),
    ("w_ncasc", 1.0, 40.0, "um", 0.005), ("r_wcasc", 0.5, 2.0, "-", 0.0),
    ("l_ncasc", 0.15, 8.0, "um", 0.005), ("r_lcasc", 0.5, 2.0, "-", 0.0),
    ("w_pmir", 1.0, 40.0, "um", 0.005), ("r_wmir", 0.5, 2.0, "-", 0.0),
    ("l_pmir", 0.15, 8.0, "um", 0.005), ("r_lmir", 0.5, 2.0, "-", 0.0),
    ("w_pass", 8.0, 100.0, "um", 0.005), ("l_pass", 0.15, 8.0, "um", 0.005),
    ("iss", 1.0, 40.0, "uA", 0.0),
    ("vbnv", 0.9, 1.7, "V", 0.0),
)
# every ratio 1 and every length equal -> reduces exactly to the base deck
NOMINAL_HIGHDIM = dict(w_pdiff=8.0, r_wdiff=1.0, l_pdiff=1.0, r_ldiff=1.0,
                       w_ncasc=4.0, r_wcasc=1.0, l_ncasc=1.0, r_lcasc=1.0,
                       w_pmir=8.0, r_wmir=1.0, l_pmir=1.0, r_lmir=1.0,
                       w_pass=64.0, l_pass=1.0, iss=10.0, vbnv=1.35)

# ---- current-mode (Banba) topology: the core the REFERENCE DESIGN actually uses.
# Same seven-variable amplifier problem as the Base case, so the two differ only in core topology.
# `w_pass` becomes `w_core`: the Kuijk pass device and the Banba mirror leg occupy the same position
# in the loop, namely the single element the amplifier output drives.
#
# The core is fixed at a DERIVED trim point (see the deck header): R_PTAT/R_CTAT = 0.105 for zero
# first-order TC, and R_ref/R_CTAT = 0.644 to place V_REF at 800 mV. The branch current is set by
# the core alone -- the loop forces nA = nB, so I_Q = V_T*ln(N)/R_PTAT -- hence V_REF does not move
# with any amplifier variable, exactly as in the Kuijk deck.
#
# WHY THIS TOPOLOGY IS WORTH A CASE. Two of the six published 65-nm specifications could not be
# transplanted to the Kuijk deck, and both failures trace to the topology choice rather than to the
# PDK: V_REF 798-802 mV is unreachable when the resistor network pins V_REF at the bandgap voltage,
# and the reference design's own gain margin is 14.86 dB against a published threshold of 20 dB.
# A current-mode core has a free output resistor, so the published V_REF window becomes usable AS
# PUBLISHED. That removes the larger of the two disclosed departures from the reference problem.
VARS_BANBA = (
    ("w_pdiff", 1.0, 40.0, "um", 0.005),
    ("w_ncasc", 1.0, 40.0, "um", 0.005),
    ("w_pmir", 1.0, 40.0, "um", 0.005),
    ("w_core", 2.0, 60.0, "um", 0.005),    # core mirror leg; carries the branch current
    ("l_all", 0.15, 8.0, "um", 0.005),
    ("iss", 1.0, 40.0, "uA", 0.0),
    ("vbnv", 0.9, 1.7, "V", 0.0),
)
# FROZEN 2026-08-31 from the 512-point probe, before any optimization run.
#
# There is no published SKY130 current-mode bandgap, so unlike the Kuijk case the anchor cannot be a
# published design; it is the best point of a 512-point Latin-hypercube probe that satisfies the
# published-faithful threshold set. It is therefore the anchor AND the random-search bar at once, and
# only one bar is reported rather than inventing a weaker second one from the same probe.
#
# The first selection rule tried was "the MEDIAN feasible point", so that the bar would be an ordinary
# design rather than the box optimum. It selected a design at 2.203 dB out of a feasible range of
# -10.665 to 19.166 dB -- worse than random search, so the bar would have been dominated by the
# random-search bar and meant nothing. Recorded in pick_reference.py rather than replaced silently.
#
# The provisional point this replaces (w_core=16 and the Kuijk nominal elsewhere) measured a phase
# margin of 38.4 deg against the published 60 deg, i.e. it failed gate 1 outright.
NOMINAL_BANBA = dict(w_pdiff=4.3601, w_ncasc=1.9239, w_pmir=38.0700, w_core=38.8064,
                     l_all=6.3894, iss=9.1267, vbnv=0.9871)
# measured at the anchor: VREF 801.1 mV, TC 25.33 ppm/degC, LG 47.26 dB, PM 74.20 deg,
#                         GM 26.74 dB, Power 53.40 uW, PSRR 19.1659 dB
NOMINAL_OBJECTIVE_BANBA_DB = 19.1659
PROBE_BEST_FEASIBLE_PSRR_BANBA_DB = 19.1659      # same number, by construction
PROBE_BEST_PSRR_ANY_BANBA_DB = 47.9835           # best in the box, feasible or NOT; diagnostic

# ---- Brokaw topology: a THIRD core, NPN differential pair with emitter degeneration.
# Six variables are the Kuijk Base values verbatim, including w_pass -- Brokaw has a pass device in
# the same position in the loop. The seventh had to change meaning and therefore range: the amplifier
# uses an NMOS input pair (forced by a rail-voltage argument recorded in the deck header, not a style
# choice), so its cascode bias is a PMOS gate referred to vdd rather than an NMOS gate referred to
# ground. vbnv in [0.9, 1.7] becomes vbpv in [0.1, 0.9].
VARS_BROKAW = (
    ("w_ndiff", 1.0, 40.0, "um", 0.005),
    ("w_pcasc", 1.0, 40.0, "um", 0.005),
    ("w_nmir", 1.0, 40.0, "um", 0.005),
    ("w_pass", 8.0, 100.0, "um", 0.005),
    ("l_all", 0.15, 8.0, "um", 0.005),
    ("iss", 1.0, 40.0, "uA", 0.0),
    ("vbpv", 0.1, 0.9, "V", 0.0),
)
# PROVISIONAL: widths copied from the Kuijk nominal, vbpv set to the mirror of vbnv=1.35 about
# mid-rail. Replaced by pick_reference.py once the probe has run; the Kuijk nominal is not expected to
# be feasible here any more than it was on the Banba deck.
NOMINAL_BROKAW = dict(w_ndiff=8.0, w_pcasc=4.0, w_nmir=8.0, w_pass=64.0,
                      l_all=1.0, iss=10.0, vbpv=0.45)

# ---------------------------------------------------------------- specifications
# Re-anchored to the SKY130 nominal design at the same RELATIVE tightness as the 65-nm
# specification set of Hoang et al. (their Eq. 18) and Nguyen et al. (their Table 3).
#
# WHY THEIR ABSOLUTE V_REF WINDOW CANNOT BE USED -- corrected 2026-08-06 after reading the
# reference design's actual Spectre netlist (library `quocthang`, cell `BGR_modified_3`).
#
# The earlier justification recorded here was WRONG in its reasoning, though not in its
# conclusion. It said a bandgap "lands near the silicon bandgap voltage" so 798-802 mV is an
# empty set. That is false in general, and the reference netlist shows exactly why: their core is
# CURRENT-MODE (Banba-style). Three matched PMOS cascode branches drive VX, VY and Vref
# independently, the amplifier forces VX = VY, and the resulting CTAT+PTAT current is delivered
# into a SEPARATE output resistor R_ref:
#
#       V_REF = R_ref * ( V_EB / R_CTAT  +  dV_EB / R_PTAT )
#
# R_ref is therefore a free design choice, and in their netlist R_ref = 36.5 um against
# R_CTAT = 55.3 um, a ratio of 0.66 -- which takes ~1.2 V to ~0.79 V. Their 800 mV is a
# deliberate, entirely achievable design target, not a physical impossibility.
#
# The circuit used HERE is voltage-mode Kuijk: the resistor network sits between vref and the
# PNP branches, so V_REF is pinned near the bandgap voltage (1.2199 V measured) by the topology
# itself and there is no output resistor to scale it with. So 798-802 mV is unreachable IN THIS
# TOPOLOGY, which is a statement about a design choice we made and must disclose, not a claim
# about bandgap references in general.
#
# The thresholds below are UNCHANGED by this correction -- only the stated reason changes, and a
# wrong reason has to be fixed whether or not it altered the numbers.
#
# Their GM >= 20 dB is separately unreachable on this amplifier (nominal 14.86 dB).
#
# Measured nominal: VREF 1219.90 mV, TC 24.49 ppm/C, P 47.27 uW,
#                   PSRR 56.29 dB, LG 53.21 dB, PM 62.99 deg, GM 14.86 dB.
#
# ============================ FROZEN 2026-08-04 -- DO NOT RE-TUNE ============================
# Calibrated against a 512-point Latin-hypercube probe of the design box (seed 20260804,
# all 512 simulated successfully) and then frozen BEFORE any optimization run was executed.
# That is not bookkeeping: "the thresholds were fixed from the nominal operating point before
# any optimization run and were not adjusted afterwards" is the entire answer to the reviewer
# who asks why the published 65-nm numbers were not used, so it has to be literally true.
# Re-tuning these after seeing campaign results would destroy the argument.
#
# All four calibration gates pass:
#   1. the nominal design is feasible (penalty 0) -> the feasible set is non-empty by
#      construction, and the benchmark does not exclude its own reference design
#   2. joint feasibility of uniform-random in-bounds candidates = 2.15 % (11 of 512),
#      inside the [0.5 %, 5 %] band: not trivially reachable, not unreachable
#   3. every specification's marginal pass rate is in [10 %, 60 %] (measured 29-58 %), so no
#      specification is decorative
#   4. no specification accounts for more than 80 % of the infeasibility (max 73 %), so the
#      problem is not a single constraint wearing six labels
#   penalty histogram 0:11 1:34 2:55 3:148 4:156 5:79 6:29 -- graded, not a cliff
#
# Two of the six published 65-nm values cannot be transplanted, for structural reasons rather
# than convenience:
#   * V_REF 798-802 mV: unreachable IN THIS TOPOLOGY. The reference design is current-mode with a
#     free output resistor, so 800 mV is a deliberate choice there; the circuit here is
#     voltage-mode, which pins V_REF near the bandgap voltage (1219.90 mV measured). See the
#     corrected derivation above -- the earlier wording claimed this was a property of bandgap
#     references in general, which is false.
#   * Gain margin >= 20 dB: the reference design itself achieves 14.86 dB, so the published
#     threshold would exclude the very design the benchmark is built from.
# Phase margin >= 60 deg is kept verbatim. Loop gain was RAISED from the published 40 dB,
# which 84 % of random candidates already satisfied. The V_REF window is +/-1.5 mV = +/-0.12 %,
# tighter than the published +/-2 mV = +/-0.25 % in both absolute and relative terms.
CONSTRAINTS = {
    "VREF_min": 1.2184,        # V   nominal 1.219903 -/+ 1.5 mV
    "VREF_max": 1.2214,
    "TC_max": 26.0,            # ppm/degC   nominal 24.49, +6.2 % margin
    "LoopGain_min": 50.0,      # dB         nominal 53.21, +6.0 % margin
    "PhaseMargin_min": 60.0,   # deg        nominal 62.99, published value kept
    "GainMargin_min": 14.0,    # dB         nominal 14.86, +5.8 % margin
    "Power_max": 55.0,         # uW         nominal 47.27, +16 % margin
}
# Random-search baseline: the campaign must beat this, or the optimization added nothing over
# uniform sampling of the box.
#
# Re-derived 2026-08-06 on TRUBA orfoz186 under the current objective, from the SAME 512
# Latin-hypercube points as the 2026-08-04 calibration (scipy.stats.qmc.LatinHypercube, seed
# 20260804), so only the scoring differs. The previous 50.49 dB belonged to the superseded
# single-frequency objective and is kept only to make that supersession legible.
PROBE_BEST_FEASIBLE_PSRR_DB = 15.50
PROBE_BEST_FEASIBLE_PSRR_DB_SINGLE_FREQ_LEGACY = 50.49

# THE HARDER BAR, and the one that actually matters. Random sampling of 512 points reaches only
# 15.50 dB, but the REFERENCE DESIGN ITSELF achieves 23.85 dB on this objective. So a campaign
# that beats random search has proved very little; the meaningful claim is beating the published
# design the benchmark was built from. Both numbers go in §6, and the second is the headline.
NOMINAL_OBJECTIVE_DB = 23.85

# ==================== SPICE-Hard, FROZEN 2026-08-05 before any Hard run ====================
# The surrogate's Base -> Hard step cannot be copied factor-by-factor. Its Base V_REF window is
# +/-5 % of 1.0 V while ours is +/-0.12 % of 1.2199 V, so halving ours is a far more violent
# step. Applied literally, the result was verified against data already on disk and rejected:
# ZERO of the 180 Base solutions and ZERO of the 512 probe points satisfied it, i.e. a campaign
# guaranteed to return 0 % success for every algorithm.
#
# What the surrogate's Hard case does in effect is make the feasible region about an order of
# magnitude harder to reach. So the tightening is parameterised by ONE scalar and chosen by
# that criterion, from measured data:
#       threshold(t) = Base + t * (tightest observed - Base),      t = 0.20
# where "tightest observed" is the 2nd/98th percentile over all 692 simulated points.
#
# Acceptance criteria, all met at t = 0.20 and all checked BEFORE the case was run:
#   * random feasibility 0.391 % (2/512) against Base's 2.15 % -- 5.5x harder, non-empty
#   * only 8/180 Base solutions satisfy it incidentally, so it is not already solved
#   * 57/180 Base solutions are within ONE violation, so an optimizer has somewhere to climb
#   * the reference design has penalty 3 -- Hard is meant to require real optimization beyond
#     the nominal design, exactly as the surrogate's Hard case does
CONSTRAINTS_HARD = {
    "VREF_min": 1.218694,      # +/-1.21 mV about the nominal (Base: +/-1.50)
    "VREF_max": 1.221112,
    "TC_max": 24.2,            # Base 26.0   nominal 24.49 -> violated
    "LoopGain_min": 52.2,      # Base 50.0   nominal 53.21 -> satisfied
    "PhaseMargin_min": 64.2,   # Base 60.0   nominal 62.99 -> violated
    "GainMargin_min": 18.2,    # Base 14.0   nominal 14.86 -> violated
    "Power_max": 48.9,         # Base 55.0   nominal 47.27 -> satisfied
}

# ==================== Banba thresholds, FROZEN 2026-08-31 before any run ====================
# TWO sets, and both are run. They answer different objections, and reporting both turns "did you
# pick the thresholds that suited you?" into a measured question: does the optimizer ranking change
# between them? If not, that is evidence the ranking is a property of the circuit; if so, that is a
# finding about benchmark sensitivity.
#
# SET A -- PUBLISHED-FAITHFUL. Five of the six 65-nm thresholds verbatim. This is the set the Kuijk
# deck could not use: its V_REF window is unreachable when the resistor network pins V_REF at the
# bandgap voltage, and its 20 dB gain-margin floor excludes the Kuijk reference design (14.86 dB).
# Both become usable on a current-mode core, which is the whole reason this topology is here.
#
# TC IS THE ONE RE-ANCHORED ENTRY, and it had to be. The published value is <= 8 ppm/degC; the best
# of 508 simulated points in this box is 10.56 ppm, so no point the probe measured clears the
# published bound and a campaign under it would return 0 % success for every algorithm. That is what
# the sample shows; it does not show that the feasible set is empty, and the manuscript says only
# that this sample did not reach the published value. The Kuijk case re-anchored TC identically,
# also from 8 to 26, and tab:spice_specs now marks three published thresholds as ones that cannot
# be transplanted, and marks all three: V_REF, gain margin and TC. An earlier draft of that
# caption counted two and left TC unmarked; the undercount was corrected in this revision.
CONSTRAINTS_BANBA = {
    "VREF_min": 0.798,          # V     published
    "VREF_max": 0.802,          #       published
    "TC_max": 26.0,             # ppm   RE-ANCHORED; published 8 is unreachable (best 10.56)
    "LoopGain_min": 40.0,       # dB    published
    "PhaseMargin_min": 60.0,    # deg   published
    "GainMargin_min": 20.0,     # dB    published
    "Power_max": 400.0,         # uW    published
}
# measured: anchor feasible; joint random feasibility 1.17 % (6 of 512)
# GATE RESULTS, recorded as they came out:
#   1 anchor feasible                      PASS
#   2 joint feasibility in [0.5, 5] %      PASS (1.17 %)
#   3 marginals in [10, 60] %              FAIL -- LoopGain 94.9 %, Power 99.2 %
#   4 no specification drives > 80 %       FAIL -- GainMargin drives 85.5 % of infeasibility
# Gates 3 and 4 fail BY CONSTRUCTION and the conflict is structural, not a calibration error:
# fidelity to the published set and gate 3 cannot both hold, because the published loop-gain floor of
# 40 dB is not binding on either topology -- which is precisely why the Kuijk case raised it to 50 and
# thereby stopped being faithful to it. Reporting the failure is the honest resolution, and "the
# published loop-gain floor is not binding on this circuit" is itself a result.

# SET B -- CALIBRATED by the single-knob quantile rule used for Circuits B and C: declare one target
# marginal pass rate T, set each threshold to the empirical quantile that lets T % of the box pass,
# clamp so the anchor still passes, then require joint feasibility in [0.5 %, 5 %]. Declared
# preference among T values: marginals in band first, then joint nearest the geometric centre of the
# band (1.58 %). T = 35 was selected.
CONSTRAINTS_BANBA_CAL = {
    "VREF_min": 0.798247,
    "VREF_max": 0.801753,
    "TC_max": 30.3807,
    "LoopGain_min": 47.2615,
    "PhaseMargin_min": 49.5341,
    "GainMargin_min": 13.4226,
    "Power_max": 69.5084,
}
# GATE RESULTS:
#   1 anchor feasible                      PASS
#   2 joint feasibility in [0.5, 5] %      PASS (1.95 %)
#   3 marginals in [10, 60] %              FAIL -- LoopGain 83.6 %
#   4 no specification drives > 80 %       PASS (VREF window 66.3 %)
# Gate 3 fails here for a different and narrower reason: the loop-gain threshold is CLAMPED to the
# anchor's own 47.26 dB so that gate 1 holds, and the anchor -- chosen for best PSRR -- happens to have
# a modest loop gain against a box median of 55.3 dB. Gates 1 and 3 therefore conflict on this
# specification. Loosening the clamp would make the anchor infeasible, which is worse.

CASES = {
    # case -> (template, variable spec, nominal point, thresholds)
    "base":    (TEMPLATE, VARS, NOMINAL, CONSTRAINTS),
    "hard":    (TEMPLATE, VARS, NOMINAL, CONSTRAINTS_HARD),
    "highdim": (TEMPLATE_HIGHDIM, VARS_HIGHDIM, NOMINAL_HIGHDIM, CONSTRAINTS),
    # Same deck, same variables, same anchor; only the threshold set differs. That is the point:
    # it isolates threshold sensitivity from topology and from dimensionality.
    "banba":     (TEMPLATE_BANBA, VARS_BANBA, NOMINAL_BANBA, CONSTRAINTS_BANBA),
    "banba_cal": (TEMPLATE_BANBA, VARS_BANBA, NOMINAL_BANBA, CONSTRAINTS_BANBA_CAL),
    # Brokaw: a THIRD core, and the cleanest comparison of the three because it reuses the Kuijk
    # variable set unchanged -- it has a pass device in the same position. Thresholds are None until
    # calibrated, so __init__ refuses to build it for a campaign.
    "brokaw":    (TEMPLATE_BROKAW, VARS_BROKAW, NOMINAL_BROKAW, None),
}

# Which cases have thresholds frozen from a calibration probe. A campaign on a case that is False
# here would complete and return a full ranking that means nothing, because with no thresholds every
# candidate is feasible. That is the worst failure available, so it is refused rather than warned
# about. Circuits B and C carry an equivalent module-level CALIBRATED flag.
CALIBRATED = {"base": True, "hard": True, "highdim": True,
              "banba": True, "banba_cal": True, "brokaw": False}

LAMBDA = 1000.0
N_SPECS = 6

# ======================= THE OBJECTIVE, AND WHY IT WAS RESTATED =======================
# The objective is the WORST-CASE rail rejection over the measured band, not the value at a
# single frequency. This is a correction, and the reason is measured, not stylistic.
#
# The first formulation used PSRR at 100 Hz alone (`meas ac m_lf FIND vm AT=100`). Nothing in
# the problem constrained the rest of the band, so the search found a real and legitimate
# engineering trade: buy low-frequency rejection by destroying high-frequency rejection. The
# rail transfer of the design that objective rewards, measured:
#
#       f          10 Hz   1.8 kHz   56 kHz   10 MHz
#       nominal    56.3     56.3      56.1     23.8   dB
#       "optimal"  92.3     77.8      48.0      2.8   dB   <-- 2.8 dB at 10 MHz
#
# It is not a numerical notch -- the curve is smooth and monotonic, and 92 dB at 100 Hz is a
# true simulated value. It is an INCOMPLETE SPECIFICATION. Across the first campaign, 174 of
# 178 feasible solutions exceeded 60 dB and 93 exceeded 100 dB, against a 56.29 dB nominal,
# while the correlation with loop gain was -0.12: 129 dB would require ~126 dB of loop gain and
# the measured value was 53 dB. So the ranking that campaign produced (GWO first) ranked the
# algorithms by how hard they pushed that one trade, not by design quality.
#
# The worst case over the band cannot be won that way, and "maximise the worst-case PSRR over
# the specified band" is the more standard analog specification. The six CONSTRAINTS are
# untouched, so the pre-registered threshold freeze below is unaffected -- the objective is not
# one of the thresholds.
#
# Both quantities are still measured and recorded, and every row records WHICH definition was
# in force, because a column whose meaning changed silently between campaigns is exactly the
# defect that let a protocol deviation go unnoticed for a whole campaign.
OBJECTIVE = "PSRR_worst_case_10Hz_10MHz"

# Recorded in every row so a merge can tell, from the artefact alone, whether that row's timeouts
# were survivable. Before 2026-08-07 a timeout left the ngspice pipe desynchronised and silently
# corrupted everything after it; since the restart in metrics() a timeout costs one PDK reload and
# nothing else. Without this marker the merge guard would have to either reject every row that
# ever timed out -- which is all of them, because a timeout here is an unavoidable hung solve --
# or trust the operator to remember which evaluator produced which file.
EVALUATOR_RECOVERS_TIMEOUTS = 1

# ---------------------------------------------------------------- PVT verification
# Optimization runs at tt / 1.8 V / 27 degC, exactly as the published problem does. The final
# designs are then RE-SIMULATED, without re-optimization, across process and supply. Verified
# present in the PDK on 2026-08-06: 49 corners exist, including the mismatch variants tt_mm /
# ss_mm / ff_mm and Monte Carlo `mc`.
#
# Why this matters more than it costs. The paper's headline word is robustness, and optimization
# at a single corner does not test robustness in the sense an analog designer means. The
# measured fragility of this objective makes it concrete: the designs it rewards buy
# low-frequency rejection by giving up high-frequency rejection, and there is no reason to expect
# such a design to hold up at ss or ff. Cost is ~17 core-hours against the campaign's ~17,460 --
# one part in a thousand -- so leaving it out would be an oversight, not a trade-off.
CORNERS = ("tt", "ss", "ff", "sf", "fs", "tt_mm", "ss_mm", "ff_mm")
PVT_CORNERS = ("tt", "ss", "ff", "sf", "fs")
PVT_SUPPLIES = (1.62, 1.80, 1.98)          # 1.8 V +/- 10 %
# A failed candidate is credited with this PSRR. Must be finite and below anything a working
# candidate can reach, so a failure can never outrank a feasible design.
PSRR_FLOOR_DB = -200.0
# Sanity clamp on a parsed PSRR: a rail-pinned node can parse cleanly and still be nonsense.
PSRR_CEIL_DB = 200.0

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SWEEP_AC = "ac dec 8 10 1e7"          # PSRR band
_SWEEP_LG = "ac dec 30 1e-1 1e10"      # loop gain: needs the low-frequency floor and the
#                                        -180 deg crossing, and enough density that the
#                                        interpolation inside `meas ... WHEN` is accurate
_ALL_PROBES_OFF = ("alter @Vz[acmag] = 0", "alter @Iz[acmag] = 0",
                   "alter @Vz2[acmag] = 0", "alter @Iz2[acmag] = 0")


def _grab(text: str, *names) -> dict:
    out = {}
    for n in names:
        m = re.search(rf"(?im)^\s*{re.escape(n)}\s*=\s*({_NUM})", text or "")
        out[n] = float(m.group(1)) if m else None
    return out


@dataclass
class Counters:
    """Failure bookkeeping. A systematic harness bug shows up here as a high deck_error or
    launch_error rate, which the circuit breaker turns into an exception instead of 450,000
    silently infeasible candidates."""
    evaluations: int = 0
    ok: int = 0
    by_failure: dict = field(default_factory=dict)

    def record(self, failure: str | None) -> None:
        self.evaluations += 1
        if failure is None:
            self.ok += 1
        else:
            self.by_failure[failure] = self.by_failure.get(failure, 0) + 1

    def summary(self) -> str:
        parts = [f"evals={self.evaluations}", f"ok={self.ok}"]
        parts += [f"{k}={v}" for k, v in sorted(self.by_failure.items())]
        return "  ".join(parts)


class SpiceBGRProblem:
    """
    Drop-in replacement for SBGRProblem backed by ngspice.

        p = SpiceBGRProblem(sky130_lib=<path to sky130.lib.spice>)
        f, m = p.evaluate_with_metrics(x)      # x is a length-7 vector in the box
        p.close()

    Not thread-safe: one instance (and therefore one ngspice process) per worker.
    """

    def __init__(self, sky130_lib: str, exe: str | None = None, case: str = "base",
                 constraints: dict | None = None, timeout: float = 30.0,
                 # 20k, not 200k. Measured: 4,072 bytes per entry and a 0 % hit rate over 100
                 # random points, because the variables are continuous on a 0.005 um grid so
                 # repeats are rare. At the published 150,000-evaluation budget a 200k cache is
                 # 583 MB per worker of pure waste, and TRUBA's orfoz gives 2,341 MB per core.
                 # Lowering it cannot change any result: the evaluator is deterministic, so a
                 # cache miss re-simulates to the identical value a hit would have returned.
                 cache_size: int = 20_000, deck_error_budget: float = 0.02,
                 breaker_after: int = 200,
                 corner: str = "tt", vdd: float = 1.8, allow_off_nominal: bool = False,
                 require_calibrated: bool = True):
        if case not in CASES:
            raise ValueError(f"unknown case {case!r}; available: {sorted(CASES)}")
        # Refuse an uncalibrated case unless thresholds are supplied explicitly. A calibration probe
        # passes wide-open thresholds and require_calibrated=False on purpose; a campaign must not.
        if require_calibrated and not CALIBRATED.get(case, False) and constraints is None:
            raise ValueError(
                f"case {case!r} has no frozen thresholds. With no thresholds every candidate is "
                f"feasible, so a campaign would complete and return a full six-way ranking that "
                f"means nothing -- and would look exactly like a result. Calibrate and freeze "
                f"first, or pass constraints=... with require_calibrated=False for a probe.")
        # An OPTIMIZATION run must be at the nominal corner and supply, because that is the
        # published problem and because the frozen thresholds were calibrated there. The PVT
        # verification pass sets allow_off_nominal=True deliberately. Without this guard a corner
        # could leak into a campaign through a stray argument and every threshold comparison
        # would silently be against the wrong physics.
        if (corner != "tt" or vdd != 1.8) and not allow_off_nominal:
            raise ValueError(
                f"corner={corner!r}, vdd={vdd!r} is off-nominal. Optimization runs at tt/1.8 V "
                f"only; pass allow_off_nominal=True if this is the PVT verification pass.")
        if corner not in CORNERS:
            raise ValueError(f"unknown corner {corner!r}; known: {sorted(CORNERS)}")
        self.corner, self.vdd = corner, float(vdd)
        template, self._vars, self._nominal, default_constraints = CASES[case]
        self.case = case
        self.name = "SKY130-BGR" if case == "base" else f"SKY130-BGR-{case}"
        self.dim = len(self._vars)
        self.lb = np.array([v[1] for v in self._vars], dtype=float)
        self.ub = np.array([v[2] for v in self._vars], dtype=float)
        self._step = np.array([v[4] for v in self._vars], dtype=float)
        if constraints is None and default_constraints is None:
            raise ValueError(f"case {case!r} has no default thresholds; pass constraints=...")
        self._constraints = dict(constraints if constraints is not None else default_constraints)
        self.counters = Counters()
        self._cache: dict = {}
        self._cache_size = cache_size
        self._deck_error_budget = deck_error_budget
        self._breaker_after = breaker_after
        self._consecutive_launch_errors = 0

        # Resolve the binary and the library ONCE. find_ngspice() globs the portable install
        # and costs ~50 ms; at 450k evaluations that alone would be 6 CPU-hours.
        self._exe = exe or nb.find_ngspice()
        lib = str(sky130_lib).replace("\\", "/")
        if " " in lib:
            raise ValueError(f"the .lib path must not contain spaces (SPICE tokenises on "
                             f"whitespace): {lib!r}")
        # The rendered values only set the deck's initial .param defaults; every candidate is
        # applied at run time with alterparam. The placeholder names are UPPERCASE and the
        # .param names are lowercase -- getting that mapping wrong is silent, because ngspice
        # accepts alterparam on an unknown name and does nothing, so every candidate would
        # return identical metrics with no error anywhere.
        subs = {"SKY130_LIB": lib, "CORNER": corner, "VDD_DC": repr(float(vdd))}
        subs.update({k.upper(): v for k, v in self._nominal.items()})
        deck = nb.render_netlist(template, subs)
        if "${" in deck:
            raise ValueError(f"template placeholder left unsubstituted in {template}")
        self._srv = nb.NgspiceServer(deck, exe=self._exe, eval_timeout=timeout,
                                     load_timeout=400.0)
        self._srv.load()

    # ------------------------------------------------------------ interface
    @property
    def constraints(self) -> dict:
        return dict(self._constraints)

    def close(self) -> None:
        self._srv.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def evaluate(self, x) -> float:
        f, _ = self.evaluate_with_metrics(x)
        return f

    def evaluate_with_metrics(self, x):
        m = self.metrics(x)
        f = self.fitness_from_metrics(m)
        # The one assertion that must never fire: a non-finite fitness silently destroys the
        # whole run through np.argmin.
        if not math.isfinite(f):
            raise AssertionError(f"non-finite fitness {f!r} for metrics {m!r}")
        return f, m

    def fitness_from_metrics(self, m: dict) -> float:
        return float(-m["PSRR_DB"] + LAMBDA * m["penalty"])

    def metrics(self, x) -> dict:
        params = self.quantise(x)
        # NINE SIGNIFICANT DIGITS IS COARSER THAN THE DECK. quantise() snaps the six gridded
        # variables to their step, but `iss` and `vbnv` carry step = 0.0 and are written into the
        # netlist as full doubles, so this key is the only thing that rounds them. Two candidates
        # differing in the ninth significant digit of `iss` therefore share one cache entry while
        # producing different netlists, and the second one evaluated is served the first one's
        # metrics. Measured consequence in the published campaigns: ten rows across the two
        # current-mode cases -- and none in the three voltage-mode cases -- report a best design and
        # a best-feasible design that differ by 4e-11 to 8e-9 relative in that one variable and
        # collide here, with measured temperature coefficients straddling the threshold by 0.0005 to
        # 0.035 ppm/degC. Which measurement belongs to which vector is not recoverable.
        #
        # THIS IS LEFT AS IT RAN. Keying on the exact doubles, `tuple(sorted(params.items()))`,
        # fixes it in one line and is the right change for new work -- but this archive exists to
        # reproduce the published campaigns, and changing the memoisation would make it reproduce
        # something else. The defect is reported in the README rather than silently repaired.
        key = tuple(sorted((k, float(f"{v:.9g}")) for k, v in params.items()))
        hit = self._cache.get(key)
        if hit is not None:
            self.counters.record(hit.get("sim_failure"))
            return dict(hit)

        raw, failure, detail = self._simulate(params)
        m = (self._metrics_from(raw, params) if failure is None
             else self._failed_metrics(params, failure, detail))
        self.counters.record(failure)

        # A TIMEOUT MUST RESTART THE SERVER. Diagnosed on TRUBA 2026-08-07 from a highdim chunk
        # that reported `evals=200 ok=0 deck_error=107 timeout=93` -- zero successes from the very
        # first evaluation, while the identical deck and the identical random candidates gave 6
        # successes out of 12 on a quiet single-process machine.
        #
        # The mechanism: on timeout the ngspice process is NOT killed, it is still computing the
        # command we gave up waiting for. Its output arrives later and sits in the reader queue,
        # so the NEXT exchange reads the PREVIOUS command's text. From that moment the pipe is
        # desynchronised and every later response is misread -- typically as deck_error, because
        # stray simulator text matches the deck-error patterns. One timeout therefore destroys the
        # whole worker session, which is exactly the 107/93/0 signature.
        #
        # It never appeared locally because nothing here approaches the 30 s limit. It appears on
        # a cluster node running 20 concurrent ngspice processes, where a slow-converging candidate
        # can exceed it. Restarting resynchronises the pipe at the cost of one PDK reload, which is
        # trivial against losing the run.
        # Resync first, restart only if that fails. Measured on TRUBA: 35 timeouts per
        # 2,000-evaluation run, i.e. 1.75 %, against 0.08 % on an idle single-process machine --
        # so these are starvation on a contended node, not unbounded solves, and the interpreter
        # usually finishes moments after we give up on it. Draining the pipe with a fresh unique
        # marker then costs a few seconds instead of a 13-42 s PDK re-parse. At 150,000
        # evaluations that difference is the difference between a campaign that fits the queue's
        # walltime and one that does not.
        if failure == nb.TIMEOUT:
            try:
                if not self._srv.resync():
                    self._srv.restart()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"ngspice could not be recovered after a timeout: {type(e).__name__}: {e}"
                ) from e

        self._check_breaker()
        # Never cache environment-dependent failures: a timeout or a missing binary says
        # something about the machine, not about this design point.
        if failure not in (nb.TIMEOUT, nb.LAUNCH_ERROR):
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = dict(m)
        return m

    # ------------------------------------------------------------ internals
    # Model validity ranges for the SKY130 01v8 FETs, in micrometres. The DERIVED geometries of the
    # highdim deck have to respect these just as the optimized variables do.
    MODEL_W = (0.42, 100.0)
    MODEL_L = (0.15, 16.0)
    # (primary, ratio, range) -- the deck computes primary*ratio and hands it to ngspice
    DERIVED_HIGHDIM = (
        ("w_pdiff", "r_wdiff", MODEL_W), ("l_pdiff", "r_ldiff", MODEL_L),
        ("w_ncasc", "r_wcasc", MODEL_W), ("l_ncasc", "r_lcasc", MODEL_L),
        ("w_pmir", "r_wmir", MODEL_W), ("l_pmir", "r_lmir", MODEL_L),
    )

    def quantise(self, x) -> dict:
        """Clip into the box, snap to the device grid, and keep the deck's DERIVED geometries
        inside the model's validity range too. Never written back into the population -- that
        would be repair, which changes the search dynamics."""
        v = np.minimum(np.maximum(np.asarray(x, dtype=float), self.lb), self.ub)
        out = {}
        for i, (name, lo, hi, _u, step) in enumerate(self._vars):
            val = float(v[i])
            if step > 0:
                # step from lb so that lb and ub are exact grid points
                val = lo + round((val - lo) / step) * step
                val = min(max(val, lo), hi)
            out[name] = val

        # The highdim deck sizes each matched partner as primary*ratio. Clipping the sixteen
        # optimized variables does NOT keep those products in range: l in [0.15, 8] times r in
        # [0.5, 2] spans [0.075, 16], so a derived length can fall below the 0.15 um model minimum
        # while every optimized variable is perfectly legal. ngspice then reports a deck error.
        #
        # That is exactly the case the module docstring already promises to handle -- "out-of-range
        # geometry is a deck error, so parameters are clipped to the model's valid range here
        # rather than being allowed to reach ngspice" -- and the promise was only kept for the
        # primary variables. Measured consequence: an ABC search on highdim accumulated 7 deck
        # errors in 317 evaluations, 2.2 %, which tripped the circuit breaker and killed the run
        # even though 99 of those evaluations had succeeded. The breaker was right about its own
        # rule and wrong about this case: in seven dimensions a deck error really does mean the
        # harness is broken, but in sixteen it can mean one legal ratio landed on an illegal
        # product.
        #
        # Clamping the RATIO rather than the product keeps the correction inside the evaluator and
        # leaves the recorded design vector self-consistent: primary*ratio is then always what the
        # deck received.
        for prim, ratio, (lo, hi) in self.DERIVED_HIGHDIM:
            if prim in out and ratio in out:
                a = out[prim]
                if a > 0:
                    out[ratio] = float(min(max(out[ratio], lo / a), hi / a))
        return out

    def _simulate(self, params: dict):
        """One candidate: PSRR + VREF + TC + Power in one exchange, then the two-phase loop
        gain. Returns (values, failure_kind, detail)."""
        srv = self._srv
        set_params = [f"alterparam {k}={v!r}" for k, v in params.items()]

        # ---- phase 1: rail-referred PSRR, and VREF/Power/TC from one temperature sweep ----
        out = srv.send(set_params + ["reset", "alter @Vdd[acmag] = 1", *_ALL_PROBES_OFF,
                                     "option temp=27",
                                     _SWEEP_AC,
                                     "let vm = mag(v(vref))",
                                     "meas ac m_lf FIND vm AT=100",
                                     "meas ac m_wc MAX vm",
                                     "let psrr_db = -20*log10(m_lf + 1e-18)",
                                     "let psrrwc_db = -20*log10(m_wc + 1e-18)",
                                     "print psrr_db", "print psrrwc_db",
                                     "dc temp -40 125 5",
                                     "meas dc m27 FIND v(vref) AT=27",
                                     "meas dc i27 FIND i(vdd) AT=27",
                                     "meas dc mx MAX v(vref)",
                                     "meas dc mn MIN v(vref)",
                                     "let vref_v = m27",
                                     # THE SUPPLY IS A PARAMETER, SO THE POWER MUST BE TOO.
                                     # This line read -1.8e6*i27 and so reported power at the
                                     # nominal supply even when the deck was built at 1.62 or
                                     # 1.98 V, understating it by 9.1 % at the high corner and
                                     # overstating it by 11.1 % at the low one.
                                     f"let power_uw = {-1e6 * self.vdd:.10g}*i27",
                                     "let tc_ppm = 1e6*(mx-mn)/(m27*165)",
                                     "print vref_v", "print power_uw", "print tc_ppm"])
        if out is None:
            return None, nb.TIMEOUT, "no response during the dc/ac phase"
        kind, detail = nb.classify_output(out, 0)
        if kind is not None:
            return None, kind, detail
        vals = _grab(out, "psrr_db", "psrrwc_db", "vref_v", "power_uw", "tc_ppm")
        missing = [k for k, v in vals.items() if v is None]
        if missing:
            return None, nb.MEASURE_FAILED, "no value for " + ", ".join(missing)

        # ---- phase 2+3: Tian loop gain at the pass gate ----
        lg = self._loop_gain()
        if lg[1] is not None:
            return None, lg[1], lg[2]
        vals.update(lg[0])
        return vals, None, ""

    def _loop_gain(self):
        """Two AC runs combined across plots. The plot name MUST be captured at runtime:
        `$p1.u` is expanded by ngspice as one variable named "p1.u", silently dropped from
        the expression, and "ac1" cannot be hardcoded because ac plots are numbered across
        the whole session."""
        srv = self._srv
        o1 = srv.send(["alter @Vdd[acmag] = 0",       # else the PSRR stimulus superposes
                       *_ALL_PROBES_OFF,
                       "alter @Vz[acmag] = 1", _SWEEP_LG,
                       "let gv = mag(v(nout)-v(nout_b))",
                       "meas ac g_v FIND gv AT=1", "print g_v",
                       "let u = -v(nout_b)", "echo PLOT1=$curplot"])
        if o1 is None:
            return {}, nb.TIMEOUT, "no response during loop-gain run 1"
        m = re.search(r"PLOT1=(\S+)", o1)
        if not m:
            return {}, nb.MEASURE_FAILED, "could not capture the plot name"
        g1 = _grab(o1, "g_v")

        o2 = srv.send([*_ALL_PROBES_OFF, "alter @Iz[acmag] = 1", _SWEEP_LG,
                       "let gi = mag(v(nout)-v(nout_b))",
                       "meas ac g_i FIND gi AT=1", "print g_i",
                       "let w = i(vz)",
                       f"let lgt = 1/({m.group(1)}.u + w) - 1",
                       "let lgdb = db(lgt)", "let lgmg = mag(lgt)",
                       # cph(), not ph(): ph() wraps at +/-180 deg and every WHEN search on
                       # phase then finds the wrong root or none at all.
                       "let lgph = 180*cph(lgt)/pi",
                       "meas ac lg_db FIND lgdb AT=1",
                       "meas ac ph_fc FIND lgph WHEN lgmg=1 FALL=1",
                       "meas ac g180  FIND lgdb WHEN lgph=-180 FALL=1",
                       "print lg_db", "print ph_fc", "print g180"])
        if o2 is None:
            return {}, nb.TIMEOUT, "no response during loop-gain run 2"
        g2 = _grab(o2, "g_i", "lg_db", "ph_fc", "g180")

        # Guards: these catch a probe that was not armed, which would otherwise yield 0/0
        # or a plausible wrong number. reset() discards every alter, so this can happen.
        if g1["g_v"] is None or abs(g1["g_v"] - 1.0) > 1e-6:
            return {}, nb.MEASURE_FAILED, f"voltage injection not armed (|dV|={g1['g_v']})"
        if g2["g_i"] is None or abs(g2["g_i"]) > 1e-6:
            return {}, nb.MEASURE_FAILED, f"current injection not armed (|dV|={g2['g_i']})"

        missing = [k for k in ("lg_db", "ph_fc", "g180") if g2[k] is None]
        if missing:
            # A candidate whose loop has no unity-gain crossing or never reaches -180 deg is
            # a genuine (bad) candidate, not a harness fault: fail it soft.
            return {}, nb.MEASURE_FAILED, "no loop-gain crossing: " + ", ".join(missing)
        out = {"lg_db": g2["lg_db"], "pm_deg": 180.0 + g2["ph_fc"], "gm_db": -g2["g180"]}
        # Every `ac`/`dc` leaves a plot behind in the process. Measured over a 512-candidate
        # probe, per-evaluation cost grew monotonically from 700 ms to 1104 ms (+58 %) as
        # ~1500 plots piled up. Over a 450,000-evaluation campaign that is a slow-motion
        # memory leak that would eventually dominate the runtime. Discard them now that the
        # cross-plot combination is done -- doing it earlier would destroy run 1's vectors
        # before run 2 can reference them.
        self._srv.send(["destroy all"], timeout=10.0)
        return out, None, ""

    def _metrics_from(self, v: dict, params: dict) -> dict:
        c = self._constraints

        def clamp(z):
            return float(min(max(float(z), PSRR_FLOOR_DB), PSRR_CEIL_DB))

        psrr_100 = clamp(v["psrr_db"])          # diagnostic: the single-frequency value
        psrr_wc = clamp(v["psrrwc_db"])         # THE OBJECTIVE: worst case over the band
        psrr = psrr_wc
        vref, tc, power = float(v["vref_v"]), float(v["tc_ppm"]), float(v["power_uw"])
        lg, pm, gm = float(v["lg_db"]), float(v["pm_deg"]), float(v["gm_db"])

        viol_vref = int(vref < c["VREF_min"] or vref > c["VREF_max"])
        viol_tc = int(tc > c["TC_max"])
        viol_lg = int(lg < c["LoopGain_min"])
        viol_pm = int(pm < c["PhaseMargin_min"])
        viol_gm = int(gm < c["GainMargin_min"])
        viol_pw = int(power > c["Power_max"])
        penalty = viol_vref + viol_tc + viol_lg + viol_pm + viol_gm + viol_pw

        m = {
            "PSRR_DB": psrr, "penalty": penalty, "is_feasible": int(penalty == 0),
            "VREF": vref, "TC": tc, "LOOP_GAIN_DB": lg,
            "PHASE_MARGIN_DEG": pm, "GAIN_MARGIN_DB": gm, "POWER_UW": power,
            "viol_vref": viol_vref, "viol_tc": viol_tc, "viol_loop_gain": viol_lg,
            "viol_phase_margin": viol_pm, "viol_gain_margin": viol_gm,
            "viol_power": viol_pw,
            "PSRR_WC_DB": psrr_wc, "PSRR_100HZ_DB": psrr_100, "psrr_def": OBJECTIVE,
            "sim_ok": 1, "sim_failure": None,
        }
        m.update({"c_" + k: val for k, val in c.items()})
        m.update({"x_" + k: val for k, val in params.items()})
        return m

    def _failed_metrics(self, params: dict, failure: str, detail: str) -> dict:
        c = self._constraints
        m = {
            "PSRR_DB": PSRR_FLOOR_DB, "penalty": N_SPECS, "is_feasible": 0,
            "VREF": float("nan"), "TC": float("nan"), "LOOP_GAIN_DB": float("nan"),
            "PHASE_MARGIN_DEG": float("nan"), "GAIN_MARGIN_DB": float("nan"),
            "POWER_UW": float("nan"),
            "viol_vref": 1, "viol_tc": 1, "viol_loop_gain": 1,
            "viol_phase_margin": 1, "viol_gain_margin": 1, "viol_power": 1,
            "PSRR_WC_DB": PSRR_FLOOR_DB, "PSRR_100HZ_DB": float("nan"),
            "psrr_def": OBJECTIVE,
            "sim_ok": 0, "sim_failure": failure, "sim_detail": detail[:120],
        }
        m.update({"c_" + k: val for k, val in c.items()})
        m.update({"x_" + k: val for k, val in params.items()})
        return m

    def _check_breaker(self) -> None:
        """A deck error is OUR bug: every candidate fails the same way. Silently optimizing
        noise for 19 hours is the worst available outcome, so make it loud."""
        n = self.counters.evaluations
        if self.counters.by_failure.get(nb.LAUNCH_ERROR, 0) and self._srv.restarts > 5:
            raise RuntimeError("ngspice keeps failing to launch; aborting rather than "
                               "reporting 450,000 infeasible candidates")
        if n >= self._breaker_after:
            rate = self.counters.by_failure.get(nb.DECK_ERROR, 0) / n
            if rate > self._deck_error_budget:
                raise RuntimeError(
                    f"deck_error rate {rate:.1%} over {n} evaluations exceeds the "
                    f"{self._deck_error_budget:.0%} budget -- this is a harness bug, not a "
                    f"population of bad candidates. {self.counters.summary()}")
