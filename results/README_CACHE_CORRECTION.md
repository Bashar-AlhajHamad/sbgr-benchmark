# Cache-key collision in the two current-mode campaigns, and the correction applied

`spice/spice_problem.py` memoises the evaluator on the design vector rounded to nine significant
digits (the key is built at the line that reads `float(f"{v:.9g}")`). Two vectors that differ only
below that resolution are one cache entry, so the second is scored with the first one's metrics.

Ten runs are affected, all in the two current-mode campaigns:

| campaign | case | runs |
|---|---|---|
| `banba_150k` | CM-Pub | `PSO` 9, 28 |
| `banba_cal_150k` | CM-Cal | `ACO` 2, 5, 9, 15, 29 and `PSO` 11, 13, 27 |

## What was wrong, precisely

Each of the ten carries a `best_fitness` consistent with **zero penalty** while its final design
vector violates one or two specifications. The recorded PSRR is right — re-measurement reproduces it
to five decimals, so the record does identify the correct vector — but the penalty came from the
colliding cache entry. `is_feasible` was **not** affected: the harness verifies the final vector
separately, and all ten are correctly recorded as infeasible. **No reported success rate changes.**

## What was measured

`recheck_cache_collisions.py` re-measures the ten final vectors with the cache effectively disabled
(`cache_size=1`, so each vector evicts the previous and no two can share an entry), on the machine
that ran the campaign, with the code released here. Output: `cache_recheck.csv` and
`cache_recheck.log`.

All ten reproduce their recorded infeasibility — 10 of 10 — with violations in the temperature
coefficient (eight), reference voltage and temperature coefficient together (one), and loop gain
(one).

## What the released records now contain

The campaign files keep `best_fitness` unchanged: it is the faithful record of what the evaluator
returned during the run, and it is the value the optimizer actually searched against. Two columns
are added beside it rather than overwriting it, in the same spirit as
`README_POWER_CORRECTION.md`:

| column | meaning |
|---|---|
| `cache_collision` | 1 for the ten affected runs, 0 otherwise |
| `best_fitness_remeasured` | the value the final vector actually measures, `-PSRR + 1000 * penalty`; empty for every unaffected run |

## What the paper reports

The paper reports the **verified** value: `best_fitness` with those ten rows replaced by
`best_fitness_remeasured`. That is the quality of the design produced, which is what the tables
claim to describe. The distinction the paper now draws, and which this file records, is between

* **feasibility and objective value reached by the optimizer** — what the run recorded, cache
  included; and
* **feasibility and objective value verified on the final vector** — what re-measurement gives.

## What the correction changes, and what it does not

Nothing that carries a conclusion. Recomputed over the two campaigns:

* **unchanged**: every success rate, every median PSRR, ABC's average rank on
  both cases (2.067 and 2.400), GWO's average rank on both (2.167 and 1.733), every ABC-versus-all
  signed-rank p-value on CM-Pub, and the ABC-GWO comparison on CM-Cal (p = 0.700);
* **changed**: the two SKY130-CM-Cal median-fitness figures, which both change the value printed at three decimals (PSO in the third decimal, ACO in the fourth), because the
  affected runs move from below the median to the top of the ordering; three mean-fitness figures (PSO on CM-Pub, PSO and ACO on CM-Cal, each moving from a
  negative mean to a positive one once the true penalty is included), six average ranks by at most
  0.167, the Friedman statistic on both cases, and ten pairwise exact signed-rank
  p-values — three on CM-Pub and seven on CM-Cal. None of the ten is quoted in the paper and
  none is in the released signed-rank tables, which hold only the ABC-versus-each comparisons;
  every one of those is unchanged to full precision. One of the ten does cross a threshold: FA
  against PSO on CM-Pub, `0.0106` to `0.0667`.

No ordering reverses on either case. The aggregate table is untouched: it pools the three SBGR cases
with the three **voltage-mode** SKY130 cases, and no voltage-mode run is affected.

## What still ships uncorrected

Each of the two current-mode campaign directories carries eight derived CSVs plus its figures,
written by `run_spice.py` when the campaign ran. **None of them is regenerated here.** Six of the
eight are computed from `best_fitness` and are therefore pre-correction:

| file | what is stale in it |
|---|---|
| `pivot_best_fitness.csv` | the run-by-algorithm fitness matrix itself; ten cells |
| `average_ranks.csv` | ranks derived from that pivot |
| `rank_summary.csv` | the same ranks, per algorithm |
| `friedman.csv` | the statistic derived from that pivot |
| `core_stats.csv` | mean, median and standard deviation of fitness |
| `extended_stats.csv` | `best_fitness_*`, `best_feasible_fitness_mean`, `convergence_auc_*` |

`pivot_best_fitness.csv` is the one to watch: it is the file `friedman.csv` and `average_ranks.csv`
were generated from, so a reader who takes it for the recomputation described below will silently
reproduce the pre-correction numbers. The fitness boxplots
(`boxplot_fitness_sky130-bgr.png`) are drawn from the same frame and are stale in the same way --
the ten corrected runs move from about $-21$ to about $+978$, from inside the box to well outside it.

Concretely: `average_ranks.csv` gives PSO `3.8667` and FA `4.9667` on CM-Pub where the paper gives
`3.900` and `4.933`; `friedman.csv` gives `56.190` where the paper gives `55.58`; and
`core_stats.csv` and `extended_stats.csv` both give PSO a mean fitness of `-34.8818` where the paper
gives `65.118`.

**Two of the eight are not stale, and this was checked rather than assumed.** `violations.csv` is
correct: the `penalty` column of `per_run_records.csv` was never corrupted by the collision -- the
ten affected rows carry penalties of 1 and 2, the true violation counts of their final vectors, and
PSO's `avg_penalty` of `0.1` over thirty runs on CM-Pub reconciles exactly with them.
`wilcoxon_holm.csv` and `wilcoxon_holm_exact.csv` are correct because they hold only the
ABC-versus-each comparisons, and none of those p-values moves. Success rates and every PSRR figure
are unaffected throughout, so the success-rate, violation and PSRR figures are sound.

They ship uncorrected for the same reason `best_fitness` does: this release adds corrections beside
the recorded values and never over them, and a regenerated summary would be the one artifact mixing
the two conventions. Anything computed from the six should be recomputed from
`per_run_records.csv`, substituting `best_fitness_remeasured` wherever `cache_collision` is 1 --
which reproduces the paper's ranks, means, medians and Friedman statistics.

## A limitation this does not remove

The `best_feasible_*` block records the metrics of a feasible point encountered during a run, but the
**design vector of that point is not stored**. It therefore cannot be re-measured, and the
distinction above cannot be drawn for it. The ten affected runs all carry such a block. This is a
property of what the harness wrote, not something the re-measurement can repair.
