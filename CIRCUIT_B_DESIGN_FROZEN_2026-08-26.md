# Circuit B — design decisions frozen before calibration

**Companion to `PREREGISTRATION_2026-08-26_circuits_2_and_3.md`. Written 2026-08-26, after the deck
was built and its reference design measured, and BEFORE the threshold-calibration probe was run and
before any optimization run existed.**

> **Revised twice on 2026-08-26, both times before any campaign was run, and both times because a
> pre-declared acceptance gate failed.** The parameterisation of the two stage-2 widths and the
> choice of reference design changed; the topology, the objective, the specification set and the
> variable count did not. Section 8 is the revision log, kept so that the sequence is checkable
> rather than merely asserted. Nothing in this file may move now that thresholds are frozen.

The pre-registration fixes the protocol. This file fixes the *problem*: topology, design variables,
box, objective and specification set. It is separate so that the pre-registration stays the untouched
document it was timestamped as, and it exists because these choices are exactly the ones that could
otherwise be quietly revised after seeing a ranking.

Thresholds are **not** in this file. They are set by the rule in section 4 of the pre-registration —
one slack scalar, moved only until random joint feasibility lands in [0.5 %, 5 %] — and are recorded
with their own timestamp once the 512-point probe has run and the four acceptance gates are checked.

---

## 1. Topology

Two-stage Miller-compensated operational amplifier: NMOS input pair with a PMOS mirror load, second
stage a PMOS common-source driver with an NMOS current-sink load, Miller capacitor across the second
stage with no nulling resistor. Bias is a fixed ideal 10 µA source mirrored to the tail and to the
stage-2 sink. Load is a fixed 1 pF.

Deck: `claude/code/spice/templates/opamp_sky130.cir.tmpl`.
Evaluator: `claude/code/spice/spice_problem_opamp.py`, which subclasses the evaluator that produced
Circuit A's results rather than copying it, so the cache, the deck-error breaker, the box clipping and
the ngspice timeout recovery are inherited unmodified and there is one copy of each to be right.

**The input pair is NMOS, and that was forced by the process, not chosen for effect.** Measured on
this PDK at L = 1 µm, tt, 27 °C: |Vth| = 1.033 V for `pfet_01v8` against 0.615 V for `nfet_01v8`. A
PMOS pair therefore requires an input common mode below Vdd − |Vth| − Vdsat ≈ 0.57 V. The first
version of this deck used a PMOS pair at the mid-rail common mode of 0.9 V, which put the pair below
threshold: 0.157 µA in the input devices, 2.2 mV across the tail, and an amplifier that did not
amplify. That version is kept, with its measured operating point, at
`claude/code/spice/templates/rejected/` rather than deleted.

**No nulling resistor is deliberate.** The right-half-plane zero that Miller compensation leaves
behind is what the phase-margin specification has to pay for, and that is the trade-off the optimizer
is being asked to make. Adding the resistor would make the problem easier and less interesting.

## 2. Design variables — seven, matching Circuit A's base case

Seven, so that the two rankings are compared at equal D and dimensionality is not a confound.
Bounds follow Circuit A's wherever the quantity is of the same kind.

| variable | role | lower | upper | unit | grid |
|---|---|---|---|---|---|
| `w1` | input pair M1/M2 | 1.0 | 40.0 | µm | 0.005 |
| `w3` | PMOS mirror load M3/M4 | 1.0 | 20.0 | µm | 0.005 |
| `w5` | NMOS tail sink and bias mirror Mb/M5 | 1.0 | 20.0 | µm | 0.005 |
| `m2` | stage-2 mirror ratio → `w6 = m2·w3` | 1.0 | 5.0 | — | continuous |
| `k` | balance deviation → `w7 = k·w5·m2/2` | 0.5 | 2.0 | — | continuous |
| `l_all` | common channel length | 0.15 | 2.0 | µm | 0.005 |
| `cc` | Miller capacitance | 0.1 | 10.0 | pF | continuous |

**Why the two stage-2 widths are bounded ratios and not free absolute sizes.** With all five widths
free, a 512-point probe found only **3.3 %** of the box had a quiescent output anywhere off a rail.
The cause is structural: the output centres only when the stage-2 driver and sink currents match,
i.e. when `(w6/w3)·(w5/2w7) ≈ 1`, and a product of four independent uniforms lands near 1 rarely. The
result was a benchmark where the output window alone decided feasibility (3.3 % marginal) and the
other five specifications were free (>90 % each) — two constraints wearing six labels.

Circuit A met exactly this on its `highdim` deck and recorded the same fix: with fully independent
per-device sizes, "84 of 96 random points had no unity-gain crossing at all — 94 % of the space
unmeasurable, 0 % feasible", so the *mismatch* was bounded as a ratio instead. Its stated
justification applies verbatim here: matched pairs and mirror ratios have tolerances, not arbitrary
sizes. Under this parameterisation the balance ratio is exactly `1/k` and bounded by construction.

Measured effect: points measurable 95 % → **100 %**, output window 3.3 % → **12.7 %**, and no derived
width hit the model's limits on any of 512 points. The bounds are chosen so that `w6 = m2·w3 ≤ 5×20 =
100 µm`, exactly the model ceiling, so the clamp is a guard rather than a participant.

**Why `l_all` is capped at 2 µm and not Circuit A's 8 µm.** Area is dominated by `l_all` — median
68 µm² for L < 0.5 against 1297 µm² for L > 4. With 8 µm in the box, no area threshold that admits
the reference design can exclude anything at all. Channel lengths beyond ~2 µm are not an analog
design choice at this node in any case; they only add area and cost bandwidth. Circuit A could afford
the wider range because area was not one of its specifications.

## 3. Objective

**DC open-loop differential gain, in dB, maximized.** Scored as a fitness to be minimised,

    f(x) = −Gain_dB + 1000 · penalty

with `penalty` the count of violated specifications. This is identical in form to Circuit A, so the
two circuits are scored the same way and nothing in the comparison depends on the scoring rule.

## 4. Specifications — six, one of them two-sided

| specification | sense | why |
|---|---|---|
| UGB | ≥ | unity-gain bandwidth; the direct antagonist of gain and of the Miller capacitor |
| Phase margin | ≥ | what the uncompensated right-half-plane zero costs |
| Power | ≤ | supply power at the operating point |
| Area | ≤ | Σ W·L over the eight transistors |
| PSRR | ≥ | low-frequency supply rejection |
| Vout | window | quiescent output level |

Six, the same count as Circuit A's, with one two-sided window as Circuit A has for V_REF.

Two of these deserve their reasons on the record:

**PSRR is a constraint here and was the objective in Circuit A.** That is the point of the pair. The
same physical quantity carrying a different role is what makes the two circuits a test of whether an
optimizer ranking transfers, instead of the same problem posed twice.

**The output window is a functional requirement, not a performance target,** which is why it is the
one specification not scaled by the slack scalar: it is set from the rail. An amplifier whose
quiescent output sits at a rail has no swing whatever its gain, and the box contains a great many such
points — the second-stage driver and sink currents must match, and they do so only in a narrow band of
`w7`. At `w7` = 14 the output sits at 1.700 V and at `w7` = 20 at 0.128 V, with the usable region
between. That sharpness is a real property of a two-stage amplifier without output common-mode
feedback, and it is left in rather than engineered away.

## 5. Reference design

Chosen by a criterion **declared before the search that found it**: minimise supply power subject to

    gain ≥ 60 dB,  UGB ≥ 5 MHz,  phase margin ≥ 60°,  Vout ∈ [0.6, 1.2] V

The output bound is deliberately tighter than the specification window, so the anchor does not sit on
the edge of its own specification. One of 720 grid candidates met the criterion.

    w1 = 10   w3 = 10   w5 = 4   m2 = 3   k = 0.85   l_all = 1   cc = 3
    → derived  w6 = 30.0 µm,  w7 = 5.1 µm

Measured at tt, 27 °C, 1.8 V, with every transistor confirmed in saturation:

| | value | position in the design box |
|---|---|---|
| Gain | 83.4923 dB | near the top (box max 84.98) |
| UGB | 5.5498 MHz | above median (2.33) |
| Phase margin | 62.4100 ° | at the median (63.03) |
| Power | 58.5027 µW | **at the median (57.40)** |
| Vout | 1.1066 V | centred |
| Area | 83.1000 µm² | below median (125.6) |
| PSRR | 89.9284 dB | above median (84.42) |

**Why it replaced the hand-derived point.** The first reference — `w5 = 10, w6 = 40, w7 = 17, cc = 2`
— burned 66.9 µW, roughly the **75th percentile** of the box. Because gate 1 forbids a threshold that
excludes the reference design, that anchor made `Power_max` untightenable below 66.9 µW, so 81 % of
the box passed it and Power stopped being a specification at all. A power-hungry anchor is a defect
in the problem setup, not a cosmetic matter. Re-anchoring changes only where the anchor sits: every
algorithm faces identical thresholds whatever the anchor, so it moves difficulty and cannot move
fairness.

**On PSRR at this operating point.** ~90 dB is high, and it is high *because* the bias is an ideal
current source, which pins the mirror gate against supply movement. That is the same modelling choice
declared for Circuit A, and it was disclosed here in advance as a risk: PSRR might turn out to be
nearly free across the box, and gate 3 would be what caught it.

**Outcome, for the record: it was free under the original setup and it is not now.** On the
free-width box PSRR passed at 95.1 % — vacuous. Under the bounded-ratio parameterisation the box
spans 27.7 dB to 99.6 dB with a median of 84.4 dB, and PSRR binds at 60.0 %. The commitment made
above — report a gate failure rather than swap the specification — was not needed in the end, but it
was not needed because the *box* was fixed, not because the specification was.

## 6. Measurement definitions

Phase margin is **180° minus the phase accumulated between 100 Hz and the unity-gain crossing.** An
NMOS pair with a PMOS mirror load inverts, so the absolute phase at the crossing sits near −180° and
the usual "180 + phase" form returns nonsense — on the first draft of this deck it reported 276–325°.
Taking the accumulated phase makes the measure independent of that sign.

A design with **no unity-gain crossing** has no bandwidth and no phase margin to report. It is a real
and informative outcome, not a broken measurement, and is scored as infeasible at the objective floor
— the same as a design violating every specification. It carries its own status string so the campaign
can report how much of the box behaves this way, as Circuit A reports its simulation failures rather
than hiding them.

A design that **fails to simulate counts as infeasible, not as missing data.** Dropping such points
would flatter every threshold, because the unmeasurable corners of the box are precisely the violent
ones.

## 7. Frozen thresholds and the calibration record

**Frozen 2026-08-26 from a 512-point Latin-hypercube probe** (`scipy.stats.qmc`, seed 20260826 — the
pre-registration date, distinct from Circuit A's 20260804, so the two probes are independent samples
rather than the same points re-scored). All 512 simulated successfully. The probe was run *through*
`SpiceOpampProblem` itself, so the quantiser, the derived-geometry clamp, the result cache, the
deck-error breaker and the ngspice timeout recovery were exercised rather than assumed.

Calibration rule: the corrected §4 of the pre-registration — one declared target marginal pass rate
*T*, mapped mechanically to a per-specification quantile, clamped so the reference design still
passes. Candidate selection was declared before the numbers were seen: prefer a *T* whose marginal
rates all lie in [10 %, 60 %], and among those take the one whose joint feasibility is nearest the
geometric centre of [0.5 %, 5 %]. **T = 60 was the only value both in band and balanced.**

| specification | threshold | reference | slack | marginal pass rate |
|---|---|---|---|---|
| `UGB_min` | 1.8602 MHz | 5.5498 | 66.48 % | 60.0 % |
| `PhaseMargin_min` | 56.7470 ° | 62.4100 | 9.07 % | 60.0 % |
| `PSRR_min` | 82.1639 dB | 89.9284 | 8.63 % | 60.0 % |
| `Power_max` | 60.0640 µW | 58.5027 | 2.67 % | 60.0 % |
| `Area_max` | 148.1022 µm² | 83.1000 | 78.22 % | 60.0 % |
| `Vout` window | [0.35, 1.45] V | 1.1066 | *not scaled* | 12.7 % |

### Acceptance gates — all four pass, all checked before any optimization run existed

1. **Reference design feasible** — PASS. Verified through the full evaluator, not just the metric
   mapping: `penalty = 0`, `fitness = −83.4923`. The benchmark does not exclude its own anchor.
2. **Random joint feasibility in [0.5 %, 5 %]** — PASS at **0.586 %** (3 of 512). Not trivially
   reachable, not unreachable. Circuit A's base case was 2.15 % and its hard case 0.391 %.
3. **Every marginal pass rate in [10 %, 60 %]** — PASS. Five at 60.0 % by construction, the output
   window at 12.7 %. *This is the gate that failed twice and forced both revisions in §8.*
4. **The two bars a campaign must clear** — random search reaches **81.18 dB** while feasible; the
   reference design itself reaches **83.49 dB**; the best gain anywhere in the box, feasible or not,
   is 84.98 dB. So beating random search is the weak claim and beating the reference design is the
   meaningful one — the same structure as Circuit A, where the two bars were 15.50 dB and 23.85 dB.

**Honest qualification on gate 4.** Only 3 of 512 probe points are feasible, so 81.18 dB is an
estimate from three samples and should be read as such. Circuit A had 11 of 512 for the same purpose.

## 8. Revision log

Both revisions were made **before any campaign was run**, and each was forced by a pre-declared gate
rather than by a result. Recorded so the sequence is checkable rather than asserted.

| # | what changed | what forced it |
|---|---|---|
| R1 | Input pair PMOS → **NMOS** | The deck did not amplify. On this PDK \|Vth\| = 1.033 V for `pfet_01v8`, so a PMOS pair needs Vcm < 0.57 V; at mid-rail it sat below threshold — 0.157 µA in the pair, 2.2 mV across the tail, output pinned at the rail. The rejected deck is kept at `claude/code/spice/templates/rejected/`. |
| R2 | `w6`, `w7` free → **derived from bounded ratios** `m2`, `k`; `l_all` capped 8 → 2 µm | **Gate 3.** With free widths, the output window passed at 3.3 % and the other five specifications at >90 %: a two-constraint problem wearing six labels. Also gate 2 could only be met at a uniform slack of 1.04, which makes four thresholds negative. |
| R3 | Reference design re-anchored to the lowest-power point meeting a declared criterion | **Gate 3, again.** The hand-derived anchor burned 66.9 µW — the 75th percentile of the box — and gate 1 forbids a threshold excluding the anchor, so `Power_max` could not go below it and 81 % of the box passed. |

Alongside these, the **calibration rule itself** was corrected in the pre-registration: "uniform
relative slack" both mis-described the bandgap precedent (whose slacks are 0.12 %–16.4 %, per
specification) and produced negative thresholds here. The correction is recorded in place in
`PREREGISTRATION_2026-08-26_circuits_2_and_3.md` §4 rather than silently applied.

**What did not change at any point:** the topology family, the objective (DC open-loop gain, dB,
maximised), the six specifications and their senses, the variable count (7, matching Circuit A's base
case), the process and PDK commit, the fitness form, and the protocol in the pre-registration.

## 9. What would invalidate this file

* changing the topology, the variable set, the box, the objective, the specification set, the
  reference design or any frozen threshold now that calibration is complete — §8 is closed;
* replacing a specification that fails calibration gate 3 with one that passes, instead of reporting
  the failure;
* reporting the reference design's figures of merit as anything other than measured values;
* reporting the campaign without the two bars of gate 4 beside it, or without the 0.586 % joint
  feasibility that says how hard the feasible set is to reach.
