# SKY130 current-mode (Banba) bandgap -- calibration record

Everything here was produced BEFORE any optimization run, and every number in
`spice/spice_problem.py` for the `banba` and `banba_cal` cases traces to a file in this directory.

## Why this topology exists

The reference design the whole SBGR benchmark is formulated from -- Hoang et al. / Nguyen et al.,
Spectre library `quocthang`, cell `BGR_modified_3` -- is **current-mode**:

    V_REF = R_ref * ( V_EB / R_CTAT + dV_EB / R_PTAT )

Our first SKY130 deck is **voltage-mode Kuijk**, which pins V_REF near the bandgap voltage. That
choice cost two of the six published 65-nm specifications: the 798-802 mV V_REF window was
unreachable, and the 20 dB gain-margin floor excludes the Kuijk reference design, which reaches only
14.86 dB. A current-mode core has a free output resistor, so **both become usable as published**.

## Files, in the order they were run

| file | what it did |
|---|---|
| `lhs_probe.py` -> `lhs_probe.csv` | 512-point Latin hypercube, seed 20260831, through `SpiceBGRProblem` itself. 508 of 512 simulated. |
| `pick_reference.py` | chose the anchor: the best probe point feasible under the published-faithful set |
| `calibrate.py` | froze both threshold sets and checked the four gates on each |

## The core trim, done by sweep before the probe

The analytic estimate for zero first-order TC gave `r_ptat/r_ctat = 0.105`, i.e. `r_ptat = 26.2 k`.
Simulated, that is CTAT-heavy and gives TC = 78 ppm/degC. Sweeping locates the turning point at
**24.8 k** (18.5 ppm). The estimate was 5.6 % high because it assumed `dV_EB/dT = -1.7 mV/K`, a
textbook figure rather than this PNP at this current density. `r_ref = 159.90 k` then places V_REF at
**800.47 mV**, inside the published window.

A first trim sweep returned byte-identical metrics for seven different `r_ptat` values. Cause: the
core parameters were declared UPPERCASE on one shared `.param` line, and `alterparam` is silently
accepted on a name ngspice does not have. The tell was that the sweep was too consistent, not that
anything failed. They are now lowercase, one per line.

## The anchor

There is no published SKY130 current-mode bandgap, so the anchor cannot be a published design. It is
the best of the 6 probe points feasible under the published-faithful set, and therefore serves as
both the calibration anchor and the single random-search bar: **19.1659 dB**.

A first rule -- "take the MEDIAN feasible point", so the bar would be an ordinary design rather than
the box optimum -- was declared, run, and changed. It selected a design at 2.203 dB out of a feasible
range of -10.665 to 19.166 dB, i.e. worse than random search, so the bar would have been dominated by
the random-search bar and meant nothing. Recorded in `pick_reference.py` rather than replaced quietly.

## Two threshold sets, both run

**Set A, published-faithful.** Five of six thresholds verbatim. TC alone re-anchored from 8 to
26 ppm/degC, because the best of 508 measured points is 10.56 ppm -- no point the probe measured
clears the published value, so a campaign under it would return 0 % success for every algorithm. The
sample does not establish that the region is empty, and the manuscript claims only that it did not
reach the published value. Joint
feasibility 1.17 %.

**Set B, calibrated.** The single-knob quantile rule used for Circuits B and C, T = 35 %. Joint
feasibility 1.95 %.

Running both makes "did you pick the thresholds that suited you?" a measured question rather than an
accusation: does the optimizer ranking change between A and B?

## Gate results, as they came out

|  | gate 1 anchor feasible | gate 2 joint in [0.5, 5] % | gate 3 marginals in [10, 60] % | gate 4 no spec > 80 % |
|---|---|---|---|---|
| Set A | PASS | PASS (1.17 %) | **FAIL** -- LoopGain 94.9 %, Power 99.2 % | **FAIL** -- GainMargin 85.5 % |
| Set B | PASS | PASS (1.95 %) | **FAIL** -- LoopGain 83.6 % | PASS (66.3 %) |

The gate-3 failures are structural, not calibration errors. **Fidelity to the published set and gate 3
cannot both hold**, because the published loop-gain floor of 40 dB is not binding on either topology --
which is exactly why the Kuijk case raised it to 50 dB and thereby stopped being faithful to it. In
set B the loop-gain threshold is clamped to the anchor's own 47.26 dB so that gate 1 holds, and the
anchor -- chosen for best PSRR -- happens to have a modest loop gain against a box median of 55.3 dB;
gates 1 and 3 conflict on that specification. "The published loop-gain floor is not binding on this
circuit" is itself a reportable result.

## A defect this exposed in the manuscript

`tab:spice_specs` states that **two** of the six published thresholds cannot be transplanted and marks
V_REF and gain margin. TC was also not transplanted -- 8 ppm/degC became 26 -- and is unmarked. The
count should be three. This is independent of anything decided about the new topologies.
