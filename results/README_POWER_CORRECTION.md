# Power at off-nominal supplies in the corner sweeps

## What was wrong

The evaluator computed supply power as

```
let power_uw = -1.8e6 * i27
```

with the supply hard-coded at the nominal 1.8 V. The corner sweep runs at **1.62, 1.80 and 1.98 V**
(`PVT_SUPPLIES` in `spice/spice_problem.py`) and substitutes the real value into the deck as
`VDD_DC`, so the **current** was measured at the correct supply while the **power** was always
reported as if the supply were nominal.

| supply | reported power | true power | error |
|---|---|---|---|
| 1.62 V | 1.8 × I | 1.62 × I | **overstated 11.1 %** |
| 1.80 V | 1.8 × I | 1.8 × I | correct |
| 1.98 V | 1.8 × I | 1.98 × I | **understated 9.1 %** |

The optimization campaigns are unaffected: they run at `tt` / 1.8 V only, and the problem class
refuses to build at any other corner or supply unless `allow_off_nominal=True` is passed, which only
the verification pass does. The defect is confined to the fifteen-condition sweep.

The source is fixed. `spice_problem.py` now writes `f"let power_uw = {-1e6 * self.vdd:.10g}*i27"`.

## What was done to the shipped data

**Nothing was overwritten.** `pvt_all_conditions.csv` in each campaign directory keeps its original
`POWER_UW`, `feasible` and `penalty` columns — those are what the published results were computed
from, and replacing them would make that unauditable. Three columns were **added**:

| column | meaning |
|---|---|
| `POWER_UW_AT_VDD` | `POWER_UW × vdd / 1.8` — the power at the supply the row was actually simulated at. Exact, because power is current times supply and the current was measured correctly. |
| `viol_power_at_vdd` | whether that value exceeds the campaign's frozen power cap |
| `feasible_at_vdd` | feasibility re-derived from all six specifications using the corrected power, with unmeasured rows left infeasible |

`.bak` copies of the pre-correction files sit beside them.

## How the correction was validated

Three checks were fixed before the correction was computed, and the script refuses to write if any
fails.

1. **The re-derivation reproduces the harness.** At 1.80 V, where the old and new power are
   identical by construction, `feasible_at_vdd` must equal the shipped `feasible` for every row that
   returned a measurement. It does, across **1,899 rows** (638 + 666 + 595). Same data, same frozen
   thresholds, same answer — which is what establishes that the re-derivation itself is right.
2. **The published corner result is unchanged.** Designs passing all fifteen conditions under the
   corrected column: **0 of 180 in each campaign, 0 of 540 in total**, exactly as Section 6 reports.
3. **Only the power verdict moves.** Not one of the other five specifications changes verdict on any
   row, which is the expected behaviour of a correction that touches only power.

## What changes, in numbers

**247 power verdicts move** across the three campaigns: **226 to fail** and **21 to pass**.

| campaign | to fail | to pass |
|---|---|---|
| `base_150k` | 64 | 0 |
| `hard_150k` | 78 | 0 |
| `highdim_220k` | 84 | 21 |

The asymmetry is expected: at 1.98 V the true power is higher than reported, so designs that looked
compliant were not; at 1.62 V it is lower, so a few that looked non-compliant were fine. Only
`highdim_220k` had any design close enough to the cap at the low supply for the second direction to
matter.

## Which column to use

For any new analysis, use `POWER_UW_AT_VDD` and `feasible_at_vdd`. The original columns are retained
so that the numbers in the paper can be traced to the file they came from, in the same way that
`pvt_summary_corrected.csv` is retained beside `pvt_summary.csv` for the separate scoring-convention
defect documented in the main README.

No figure reported in the manuscript changes.
