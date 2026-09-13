# `per_run.csv` contains 540 duplicate rows — read this before computing anything from it

`per_run.csv` holds 3,960 rows, of which **540 are exact full-row duplicates**. They are confined to
two rehearsal cells at `budget = 2,500`:

| cell | rows | of which duplicated |
|---|---|---|
| `dim 7, pop 20, budget 2,500` | 900 | 180 |
| `dim 7, pop 40, budget 2,500` | 1,080 | 360 |
| every other cell | 180 | 0 |

**The published protocol cell is clean.** `dim 12, pop 40, budget 150,000` holds exactly 180 rows,
zero duplicates, `seed_base = 42` only, and reproduces the Section 5 table (`tab:core_quant_results`)
bit-for-bit against `results_penalty/per_run_records.csv` at λ=1000 — 180/180 cells, max difference
0.0. **No published number is affected by the duplicates.**

## Do not blindly deduplicate

Most of the extra rows in those two cells are **not** duplicates. They are a legitimate four-seed
sweep: `seed_base ∈ {42, 777, 999983, 20260806}`, 180 rows each. Dropping rows by
`(dim, pop, budget)` would destroy it.

The correct filter, if you need those cells:

```python
p = pd.read_csv("per_run.csv").drop_duplicates()      # removes exactly the 540
```

Deduplicating on `(dim, pop, budget, seed_base, run, algo)` gives the same 540, which confirms the
repeats carry no distinguishing information.

## What the duplicates would corrupt

Any unweighted statistic over the whole file double- or triple-weights those two rehearsal cells.
Nothing in the manuscript is computed that way — Section 5's numbers come from `results_penalty`, and
this file is used only for the protocol-choice sweep and as the independent replication of the base
case. Both read a single cell at a time.
