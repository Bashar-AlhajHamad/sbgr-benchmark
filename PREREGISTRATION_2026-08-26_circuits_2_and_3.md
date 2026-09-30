# Pre-registration — two additional circuits for the ASOC-D-26-02845 case study

**Written 2026-08-26, before either circuit was built, simulated, or calibrated.**
To be committed to `github.com/Bashar-AlhajHamad/sbgr-abc-benchmark` on the date above so that its
timestamp is externally checkable, and to be approved in writing by all authors before any campaign
is launched.

---

## 1. Why this document exists

The first transistor-level cross-check — a Kuijk bandgap reference in SkyWater SKY130 — has been
completed. Across three pre-registered campaigns and 540 runs, **GWO attained the best average rank
in all three cases and ABC ranked third, fifth and second.** ABC beats GWO in 2 of the 90 paired
runs; the paired Wilcoxon gives p = 1.7 × 10⁻⁶ per case.

That result is inconvenient for a paper whose contribution is framed around ABC, and the decision to
run further circuits was taken **after** it was known. That sequence is stated plainly here because
it is the fact a reader needs in order to weigh what follows. A second circuit chosen after an
unfavourable first result is worth exactly as much as the discipline placed around it, and nothing
more.

Two things make the exercise a test rather than a search:

1. the circuits, the protocol and the reading of every outcome are fixed **here, in advance**;
2. **all three topologies will be reported**, whatever each one shows.

There is also a substantive reason to run them, independent of the first result: **one netlist cannot
establish whether an optimizer ranking transfers.** Section 6 currently draws a conclusion about
surrogate-to-circuit transfer from a single topology, which is the weakest form that conclusion could
take. Two more topologies test it properly.

## 2. The circuits, and why these

Both are chosen because they are already cited in the manuscript's bibliography — the same standard
that justified the SKY130 bandgap — and not for any expected outcome.

| | Reference | Why it is a fair second test |
|---|---|---|
| **Circuit B — two-stage CMOS operational amplifier** | Hershenson, Boyd & Lee, *IEEE TCAD* 20(1):1–21, 2001 (`dokuz` in `ref.bib`) | The canonical analog-sizing benchmark in this literature. Structurally independent of a bandgap: the objective is gain–bandwidth rather than supply rejection, there is no PTAT core, no start-up branch and no temperature-coefficient specification. |
| **Circuit C — low-dropout regulator** | Li & Carusone, *ICCAD* 2023 (`li2023ldosky130`), cited by the survey that motivates this study | An LDO in the same open SKY130 process, and the one open-PDK analog-optimization precedent inside our own citation chain. Shares an error amplifier with the bandgap, which makes it the *less* independent of the two, and it is included for that reason: if the ranking is topology-dependent, a partially similar topology should sit between the two extremes. |

Both are implemented in SKY130 at the same pinned `open_pdks` commit
`c6d73a35f524070e85faff4a6a9eef49553ebc2b`, `sky130A`, model library
`libs.tech/combined/sky130.lib.spice`, MD5 `365ab743568de364c2214767735a89c6`, with `ngspice` 46.

## 3. Protocol — identical to the bandgap campaign, with no exceptions

* the same six optimizers: ABC, GWO, FA, PSO, GA, ACO
* **the same control parameters for every optimizer**, as tabulated in `tab:baseline_params`. No
  optimizer is tuned for these circuits. Tuning one method and not the others would contradict the
  fairness claim already made in Section 5 and already defended in the response to Reviewer 3,
  Weakness 3. If a tuning study is undertaken later it will tune **all six** with equal effort and be
  reported as a separate experiment.
* 30 independent runs per algorithm, one problem instance per run shared by all six (paired design)
* population 40, **150,000 objective evaluations per run**, matching Circuit A's transistor-level
  campaign exactly

> **Correction, 2026-08-29, after the first campaigns had already run.** This line was changed to
> "2,500 objective evaluations per run" on 2026-08-26 and that change was **wrong**. Both campaigns
> ran at 2,500 as a result, and both must be repeated at 150,000.
>
> The passage relied on describes running the **surrogate** at a reduced protocol as a control —
> "the surrogate was additionally run *at the circuit-level protocol* … 2500 evaluations … cutting
> the budget sixty-fold". The sixty-fold cut is of the *surrogate's* budget. The authoritative
> statement is the caption of `tab:spice_results`, reporting Circuit A's actual campaign:
> **"150,000 objective evaluations per run for SKY130-Base and SKY130-Hard and 220,000 for
> SKY130-Highdim"**.
>
> **Why it matters.** The rankings within each new circuit are internally valid — all six optimizers
> faced identical budgets inside a circuit. But the *cross-circuit* claim is confounded: "no optimizer
> wins on more than one topology" could be a budget effect rather than a topology effect, which is
> the first thing a referee would say. And it is not budget-neutral for ABC: the manuscript measures
> ABC's average rank improving monotonically with budget on the surrogate at $D=7$, from 2.300 at
> 2,500 evaluations to 1.750 at 80,000. A 2,500-evaluation budget therefore systematically
> disadvantages ABC, and the first results likely understate it.
>
> **The 2,500-evaluation results are retained, not discarded.** Held at `results/opamp` and
> `results/ldo` while the corrected campaigns write to `results/opamp_150k` and `results/ldo_150k`,
> they become a measured budget-sensitivity control on real circuits — something the manuscript
> currently reports only for the surrogate. That is a by-product of the mistake, not a justification
> for it.
* uniform initialization, clipping-based bound handling, no repair operator
* fitness `f(x) = −objective + 1000 · penalty`, penalty = the count of violated specifications
* Friedman across the six, then paired Wilcoxon with Holm correction, **exact test**
* one nominal case per circuit. Three cases were used on the bandgap to study difficulty scaling,
  which is already answered; the question here is transfer across topology, and one case per
  topology answers it at a third of the cost.

**Seeding.** Each circuit gets its own offset, so the three topologies are independent replicates —
correcting the single shared `CASE_OFFSET` disclosed as a defect in Section 6.5.

## 4. Thresholds — the calibration rule, fixed before any threshold is chosen

For each circuit, six specifications are set by the same rule used for the bandgap:

1. anchor each specification at the value the reference design achieves at the nominal corner;
2. declare **one** target marginal pass rate *T*. Set each specification's threshold to the empirical
   quantile of a 512-point Latin-hypercube probe of the design box that lets *T* % of the box pass
   it, clamped so that the reference design still passes. Then require **random joint feasibility to
   fall between 0.5 % and 5 %**. *T* is the only degree of freedom and it moves all six thresholds
   together;
3. freeze, and record the frozen values with a timestamp before the campaign starts.

Every specification's **marginal** pass rate must also land in [10 %, 60 %], so that no single
specification is either free or the sole binding constraint. A specification that fails this is
reported as failing it, never swapped for one that passes.

> **Correction, 2026-08-26, made before any campaign was run.** This step originally read "apply
> uniform relative slack, adjusted only so that random joint feasibility falls between 0.5 % and
> 5 %". That was wrong on two counts and is corrected here rather than quietly left in place.
>
> First, it mis-describes the bandgap precedent it claims to reuse. The bandgap's frozen slacks are
> 0.12 %, 6.2 %, 16.4 %, 6.0 %, 4.8 % and 5.8 % — **per specification, not uniform**.
>
> Second, applied literally to Circuit B it is degenerate. The uniform scalar has to reach 1.04
> before joint feasibility enters the band, and at that point `UGB_min` = −0.31 MHz,
> `PhaseMargin_min` = −2.44° and `PSRR_min` = −3.66 dB: thresholds that no design can fail. Four of
> the six specifications become vacuous and the benchmark is two constraints wearing six labels.
>
> The replacement is **more** constraining, not less: it makes balance a property of the
> construction instead of something hoped for and checked afterwards. It also cannot favour any
> optimizer — all six algorithms face identical thresholds, so threshold choice sets the problem's
> difficulty and never its fairness.

**Thresholds will not be re-tuned after any campaign is run**, for any reason, including an
unfavourable ranking. This is the rule that was honoured for the bandgap and it is restated because
it is the rule most likely to come under pressure this time.

## 5. The reading of every outcome, fixed now

Let *k* be the number of the three topologies on which ABC attains the best average rank.

| | Conclusion to be reported |
|---|---|
| **k = 3** | The surrogate ranking transfers, and the bandgap result was the exception. Section 6's transfer conclusion is withdrawn and the bandgap case is reported as the outlier it would then be. |
| **k = 1 or 2** | **Optimizer ranking is topology-dependent.** This is the strongest available finding and it is what makes a single-circuit conclusion untenable — including our own. Reported as the principal result, with the per-topology rankings side by side and no aggregation that hides the split. |
| **k = 0** | ABC does not lead on any transistor-level circuit tested. The transfer conclusion of Section 6 is confirmed on three topologies rather than one, the paper's contribution rests on the benchmark and the protocol, and ABC's surviving claims are feasibility attainment and the absence of collapsed designs. |

In every case: **all three topologies appear in the paper, with their per-circuit rankings.** No
topology is dropped, and no aggregate is presented without the per-circuit numbers beside it.

## 6. What would invalidate this pre-registration

Stated so that a reader can check:

* reporting fewer than three topologies;
* tuning any optimizer for one circuit and not the others;
* selecting individual runs rather than reporting all 30;
* adjusting a frozen threshold after a campaign;
* revising the calibration rule, the design box, the reference design or the specification set after
  any campaign has been run. All four were revised for Circuit B **before** its first campaign, each
  time because a pre-declared acceptance gate failed, and each revision is recorded with its reason
  in `CIRCUIT_B_DESIGN_FROZEN_2026-08-26.md`. After a campaign, none of them may move;
* adding a fourth circuit after seeing these results and reporting it in place of one of these three
  — a fourth circuit may be added, but then four are reported.

## 7. Data

Every per-run record from all three topologies will be added to
`https://doi.org/10.5281/zenodo.22071453` and to the GitHub repository, as the bandgap records
already have been.

---

**Approved before launch by:**

Bashar Aqel Younis Alhaj Hamad · Doğan Yıldız · Durmuş Özkan Şahin · Sercan Demirci · Selçuk Aslan
