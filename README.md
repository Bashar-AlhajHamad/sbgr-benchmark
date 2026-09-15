# SBGR: A Surrogate Bandgap-Reference Benchmark for Constrained Metaheuristic Optimization

This repository contains the benchmark, optimizers, and experiment scripts for the study:

> **Artificial Bee Colony for Constrained Optimization in 6G-Motivated Analog Integrated Circuit Design: A Surrogate Benchmark and a SKY130 Cross-Check**
> (under review, *Applied Soft Computing*).

It reproduces all experiments in the paper: the main algorithm comparison and the penalty-scaling sensitivity analysis.

---

## What is SBGR?

**SBGR (Surrogate Bandgap-Reference)** is a deterministic, closed-form benchmark that mirrors the *optimization structure* of bandgap-reference (BGR) analog transistor sizing: maximizing power-supply rejection ratio (PSRR) subject to several coupled specifications (reference voltage, temperature coefficient, loop gain, phase margin, gain margin, and power). It is designed for fair, reproducible, simulator-independent comparison of constrained metaheuristics.

Given a design vector, SBGR computes six BGR-inspired metrics and PSRR from fixed closed-form expressions, then a penalty-based fitness

```
f(x) = -PSRR(x) + lambda * penalty(x)      # lambda = 1000 by default
```

where `penalty(x)` counts violated specifications (a solution is feasible when `penalty = 0`). Three cases of increasing difficulty are provided:

| Case      | Dimension | Description                                  |
|-----------|:---------:|----------------------------------------------|
| `base`    | 12        | Nominal specifications                       |
| `hard`    | 12        | Tightened specifications                     |
| `highdim` | 30        | Higher-dimensional variant (scalability)     |

> **Important — scope.** SBGR metrics are **dimensionless surrogate quantities on internally consistent scales**, calibrated to reproduce the optimization difficulty of BGR sizing (coupled metrics, a narrow feasible region, multimodality). They are **not** physical units (volts, ppm/°C, dB, µW) and SBGR is **not** a circuit simulator. It targets *relative* algorithm comparison, not absolute circuit-performance prediction. See the paper for the full metric definitions and the fidelity/scope discussion.

---

## Repository contents

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `problems.py`             | The SBGR benchmark: metric model, constraints, penalty, and fitness.        |
| `algorithms.py`           | Six optimizers (ABC, GWO, FA, PSO, GA, ACO/ACOR) + a shared evaluation wrapper. |
| `run.py`                  | Main comparison campaign; writes per-case and combined results and figures. |
| `penalty_sensitivity.py`  | Penalty-scaling sensitivity study (sweeps `lambda`).                        |
| `protocol_sweep.py`       | Protocol-choice sweep (dimension, population, budget, seed families), and the independent replication of the published `base` cell. |
| `sbgr_controls.py`        | Controls testing alternative explanations for the surrogate-to-circuit disagreement: dimension, constraint pressure, unmeasurable regions. |
| `requirements.txt`        | Python dependencies.                                                        |

**Section 6 — transistor-level cross-check on SKY130** (see the dedicated section below):

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `run_spice.py`            | Circuit-level campaign driver — same protocol as `run.py`, ngspice evaluator. |
| `spice/spice_problem.py`  | The ngspice-backed problem: netlist rendering, `.measure` parsing, six specifications, penalized fitness. |
| `spice/ngspice_bridge.py` | Persistent ngspice server (`alterparam`/`reset`), so the PDK is parsed once per worker. |
| `spice/templates/`        | The SKY130 bandgap decks: the voltage-mode Kuijk core (`bgr_sky130.cir.tmpl`) and its higher-dimensional variant, and the current-mode Banba core (`bgr_banba_sky130.cir.tmpl`) used for the two reference-topology cases. `bgr_brokaw_sky130.cir.tmpl` is a third core included for completeness only — no reported result depends on it. |
| `truba/`                  | Cluster scripts: ngspice built from source, PDK fetched at a pinned commit, SLURM job arrays. |
| `pvt_reference_control.py`| Runs the untouched SKY130 voltage-mode reference design through all fifteen process/supply conditions. It prints to stdout; the captured run released here is `results/base_150k/pvt/reference_control_k1.log` (8 of 15 passed), with `reference_control.log` the same run under the pre-K1 power formula. Submit with `truba/05_reference_control.slurm`. |
| `pvt_summary_fix.py`      | Writes a corrected PVT summary beside the shipped one — see *Known issues*.  |
| `wilcoxon_exact.py`       | Regenerates the exact signed-rank tables from the shipped per-run records and checks them against the p-values printed in Section 6. `--check` diffs without writing. |
| `results_spice_banba/`    | Calibration record of the two current-mode cases: the 512-point design-box probe, the anchor selection, and the threshold derivation with its four acceptance gates. |
| `results_spice_brokaw/`   | The Brokaw probe — 512 points, of which 305 did not converge. Kept as the evidence for why that core is not run as a campaign; no reported number uses it. |
| `results/controls_dim/`, `results/controls_pressure/` | The two surrogate controls behind Section 6.5: SBGR re-run at the circuit's dimensionalities, and with its thresholds relaxed until random in-box feasibility matches the circuit's. Each carries the published `D = 12` cell as a control on the control; both reproduce it exactly. Regenerate with `python sbgr_controls.py --experiment {dim,pressure} --job-index N --outdir DIR`, then `--merge`. |
| `results_spice/`          | The voltage-mode design-box probe: 512 Latin-hypercube points, all 512 measured, best temperature coefficient 10.73 ppm/°C. This is the sample the Table 9 caption cites; `verify_thresholds.py` regenerates it to this path. |
| `corner_verify.py`, `verify_*.py`, `preflight_spice.py`, `audit_before_launch.py` | Pre-launch and post-hoc checks: grid independence, threshold freeze, anchor reproduction, population and budget audits. |

---

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `scipy`, `matplotlib` (and `Jinja2` for optional LaTeX table export)

Install:

```bash
pip install -r requirements.txt
```

---

## Reproducing the paper's experiments

All runs are deterministic given the seed. Each `(case, run)` uses a fixed seed shared by every algorithm (paired design), so comparisons are fair.

### 1. Main comparison campaign (540 runs)

Six algorithms × three cases × 30 runs, population 40, evaluation budgets 150,000 (base/hard) and 220,000 (highdim):

```bash
python run.py \
  --cases base hard highdim \
  --pop 40 --runs 30 --seed 42 \
  --max-evals 150000 --highdim-max-evals 220000 \
  --outdir results_main
```

> Note: `run.py`'s built-in defaults are smaller (pop 30, 30,000 evaluations, 20 runs, `base` only) for quick tests. The command above reproduces the published configuration.

### 2. Penalty-scaling sensitivity (2,160 runs)

The same protocol repeated for `lambda ∈ {10, 100, 1000, 10000}`:

```bash
python penalty_sensitivity.py \
  --runs 30 --pop 40 --seed 42 \
  --max-evals 150000 --highdim-max-evals 220000 \
  --lambdas 10,100,1000,10000 \
  --outdir results_penalty
```

Both scripts print progress and, when finished, write CSV summaries (and, for `run.py`, figures) into the chosen `--outdir`. The full campaigns take on the order of hours on a desktop CPU; reduce `--runs` and `--max-evals` for a faster smoke test.

---

## Outputs

`run.py` writes, per case, a `case_<name>/` folder containing:

- `summary_<case>.csv` — one row per run per algorithm (final fitness, feasibility, metrics, runtime);
- statistics CSVs (descriptive stats, violation counts, ranks, Friedman, pairwise Wilcoxon + Holm);
- figures: `convergence_sbgr-<case>.png/.pdf`, `boxplot_fitness_sbgr-<case>.png/.pdf`, `violations_bar_sbgr-<case>.png/.pdf`.

Combined files (`summary.csv`, overall ranks, overall Friedman) are written at the top level of `--outdir`.

`penalty_sensitivity.py` writes `per_run_records.csv`, `summary_by_lambda.csv`, and a plain-text `verdict.txt`.

---

## Section 6 — transistor-level cross-check on a SKY130 bandgap reference

The surrogate buys statistical power, and that leaves one question open: **does an optimizer ranking obtained on SBGR say anything about the same ranking on a real netlist?** Section 6 of the paper answers it by re-posing the identical optimization problem — same objective, same penalized fitness, same protocol — against transistor-level bandgap references in the open-source SkyWater SKY130 process, evaluated with `ngspice`. Five campaigns of 180 runs each are reported. Two use the **current-mode (Banba) core that the motivating reference design actually uses**, which is what makes five of its six published specifications applicable verbatim; three use a **voltage-mode (Kuijk) core**, retained because the surrogate-transfer comparison rests on them.

**The answer depends on what is held fixed, and each factor was varied on its own.** On the reference topology under its published specification set, ABC attains the best average rank (2.067) and the exact signed-rank test with Holm correction separates it from all five baselines. Recalibrating the thresholds alone — same netlist, same design box, same thirty seeds — reverses it: GWO leads at 1.733 and the ABC–GWO difference is no longer significant (`p = 0.700`). The two regimes differ in which specification is active at the optimum, gain margin in the first and power in the second. On the voltage-mode cases GWO attains the best average rank in all three and ABC ranks third, fifth and second — so on those, the ordering measured on the surrogate does not predict the ordering measured on the circuit.

What transfers is reliability rather than peak objective value: **ABC reaches a feasible design in every run of all five cases**, and on the reference topology its worst of thirty runs (49.92 dB) exceeds the median of the next-best method (34.62 dB). Separately, of the designs put through the fifteen-condition process and supply sweep, **none survives it — 0 of 540 — and neither does the untouched reference design, which passes 8 of 15 under the Base threshold set.** That control was re-measured on the cluster on 2026-09-15 with the code released here and captured to `results/base_150k/pvt/reference_control_k1.log`; the best design of each of the six optimizers passes between 2 and 4. That sweep covers the three voltage-mode campaigns only; see *Known issues*.

### Additional requirements

- **`ngspice` 46**, built from source (see `truba/00_setup_native.sh`).
- **SKY130 PDK**, `sky130A` variant, fetched with [`volare`](https://github.com/efabless/volare) at a pinned commit. The PDK is **not vendored here** — it is 216 MB and the setup script retrieves exactly the right build.

### PDK provenance — read this before reproducing

The kit ships **more than one model library** under the same commit, and the results depend on which one is used. Every number in Section 6 comes from:

| | |
|---|---|
| library | `sky130A/libs.tech/combined/sky130.lib.spice` — the standard **binned** set, **not** `combined/continuous/` |
| MD5 | `365ab743568de364c2214767735a89c6` |
| `open_pdks` commit | `c6d73a35f524070e85faff4a6a9eef49553ebc2b` (via `volare`) |
| simulator | `ngspice` 46, built with `gcc (GCC) 11.3.1` |

`env.sh`, generated by the setup script on the cluster before the campaign, records all of this and is sourced by every job. **Check its MD5 line matches the value above before comparing your numbers to ours.**

### Running it

```bash
python run_spice.py --lib /path/to/sky130A/libs.tech/combined/sky130.lib.spice --case base --evals 150000 --pop 40 --runs 30 --outdir results/base_150k
```

`--case hard` uses the tightened thresholds and `--case highdim` the 16-dimensional variant at 220,000 evaluations; both use the voltage-mode core. `--case banba` and `--case banba_cal` are the two current-mode cases — the same deck, the same design box and the same anchor, differing in nothing but the threshold set, which is what isolates threshold sensitivity from topology and from dimensionality. Their thresholds are frozen constants in `spice/spice_problem.py`, derived by `results_spice_banba/calibrate.py`; a case whose thresholds have not been through that procedure is refused at construction unless explicit `constraints=` are passed with `require_calibrated=False`, which is how the probes are run. `--case brokaw` is registered but uncalibrated by design and cannot be launched as a campaign. Note that `run_spice.py`'s own `--evals` default is 2,500, for a quick smoke test; the published protocol is the 150,000 given above (220,000 for `highdim`), which is what the `truba/*.slurm` scripts pass. On a cluster, use `truba/runcase.sh`, which chunks the 180 `(algorithm, run)` tasks into SLURM arrays. Budget accordingly: the median wall clock of a *single* `(algorithm, run)` task, from the shipped `runtime_sec` column, is 5.03 h on `base`, 4.50 h on `hard`, 7.13 h on `highdim`, 12.78 h on `banba` and 15.94 h on `banba_cal`, so one 180-run case is 777, 749, 1,160, 2,159 and 2,677 run-hours respectively. The evaluator is memory-bandwidth-bound, so more workers per node makes it *slower*, not faster; the campaigns were chunked across array tasks rather than run on one node.

> **Before submitting jobs**, edit two placeholders in `truba/*.slurm`: `YOUR_ACCOUNT` (the SLURM allocation) and `YOUR_USER` in the log paths. SLURM parses `#SBATCH` directives before shell expansion, so these cannot be variables. Everywhere else the cluster path is built from `$USER`. Note that `TRUBA_USER` is honoured only by `truba/00_setup_native.sh`; the job scripts and the generated `env.sh` use `$USER` directly, so if your login name is not your cluster account name, set `USER` for the whole session rather than `TRUBA_USER`.

### Data layout

```
results/base_150k/      per_run_records.csv   180 rows = 6 algorithms x 30 runs  <- the authoritative file
results/hard_150k/      rows.zip              the 360 per-task shards it was merged from
results/highdim_220k/   pvt/                  the 15-condition sweep (2,700 rows per campaign)
results/banba_150k/     <case>/               the two current-mode campaigns, CM-Pub and CM-Cal.
results/banba_cal_150k/                       Same layout without pvt/ -- the corner sweep was not
                                              run on these two; see Known issues.
results/grid_density_check_banba.csv          the best design of every algorithm in every case,
results/grid_density_check_kuijk.csv          re-measured at 8 and at 256 frequency points per decade
results/penalty_sensitivity/                  Section 5, lambda sweep; the lambda=1000 slice is the published campaign
results/protocol_sweep/                       protocol-choice sweep; see the note in that folder
results_spice_banba/                          calibration record of the current-mode thresholds
results_spice_brokaw/                         probe of the third core; not used by any result
env.sh                                        the cluster provenance record: PDK build, model library, simulator
```

The three voltage-mode campaign directories also carry `pvt/pvt_summary_corrected.csv` — use that
rather than `pvt_summary.csv`, for the reason given below. The two current-mode directories have no
`pvt/` at all; the corner sweep was not run on them. `rows.zip` is provided so the merge can be audited;
`per_run_records.csv` is what every table in the paper is computed from, and the two agree by
construction (`run_spice.py --merge` refuses a wrong row count or a duplicated `(algo, run)`).

Each circuit case also carries `wilcoxon_holm_exact.csv` beside the `wilcoxon_holm.csv` that the
plotting pipeline emits. The latter uses a normal approximation, which at `n = 30` saturates at
about `1.7344e-6`: five comparisons across these cases have a signed-rank statistic of exactly zero,
meaning all thirty pairs fall on one side, and the approximation returns that same value for each of
them however decisive the win. The exact test returns `1.8626e-9` there (`2 / 2**30`), and that is
what Section 6 quotes throughout. Both tables ship, and `python wilcoxon_exact.py --check` rebuilds
the exact one from `per_run_records.csv` and diffs it against what is shipped.

### Known issues in the released data

Everything below was found by our own checks, not by a reviewer, and each is discussed in the paper.

1. **`pvt/pvt_summary.csv` is unusable as shipped.** Its `nominal_feasible` column contradicts `per_run_records.csv` for 173 of 540 designs at the very condition the optimizer ran at, because simulation failures are scored identically to specification violations. Use **`pvt_summary_corrected.csv`** in the same folder, regenerated by `pvt_summary_fix.py`, which carries both scoring conventions plus the failure counts. The `pvt_robust` column (0 everywhere) is unaffected.
2. **Per-algorithm PVT figures cannot rank optimizers.** Between 24.8 % and 29.7 % of the corner re-simulations returned no measurement within the solver time limit, and the failure rate tracks the algorithm (95.3 % for ABC on `hard` against 0.0 % for PSO). The only PVT figure invariant to the scoring convention is the 0-of-540 result.
3. **`highdim` is not an equal-effective-budget comparison.** 21.68 % of its evaluations returned no measurement, with usable fractions from 57.8 % (GWO) to 99.6 % (FA). It does not manufacture the ordering — the winner holds the *smallest* effective budget — but it is unequal.
4. **All five Section 6 cases share one seed offset.** `run_spice.py` applies a single `CASE_OFFSET`, so every case draws the same thirty run seeds. Two pairs go further and have byte-identical initial populations, because the members differ in nothing but their thresholds: `base`/`hard`, and `banba`/`banba_cal`. Within-case pairing — which the Friedman and Wilcoxon tests require — is intact, but the cases are variants rather than independent replicates. The resulting run-level coupling is not the same for the two pairs: correlating per-run best fitness within each algorithm gives a mean `r` of `-0.02` for `base`/`hard` (max `0.19`) and `+0.27` for `banba`/`banba_cal` (max `+0.47`, GA). Only the pooled sensitivity analysis crosses cases.
5. **`results/protocol_sweep/per_run.csv` contains 540 duplicate rows**, confined to two rehearsal cells at `budget = 2500`. See `README_duplicate_rows.md` in that folder before computing anything from it. The published cell is clean.
6. **The fifteen-condition sweep covers three of the five circuit campaigns.** It was run on the voltage-mode cases (`base`, `hard`, `highdim`) and not on the two current-mode ones, so the 0-of-540 figure is 0 of 180 in each of three campaigns and says nothing either way about the current-mode designs. The paper scopes it the same way.
7. **The evaluator's memoisation key is coarser than the netlist it stands for.** `quantise()` snaps the six gridded variables to their step, but `iss` and `vbnv` have `step = 0.0` and reach the deck as full doubles, while the cache key rounds every variable to nine significant digits. Candidates differing below that resolution share one entry. In the published data this is visible in exactly ten rows, all in the two current-mode cases: the best design and the best-feasible design differ by 4e-11 to 8e-9 relative in one variable, collide, and carry measured temperature coefficients on opposite sides of the bound by 0.0005 to 0.035 ppm/degC. Those ten rows are what makes three cells of the success-rate table fall below 100 %. Fitness is unaffected — it is the value the optimizer recorded during the run — so no rank or test changes. The code is shipped **as it ran**; keying on the exact doubles (`tuple(sorted(params.items()))`) fixes it for new work.
8. **Power was computed at the nominal supply in the corner sweeps.** The evaluator hard-coded 1.8 V while the sweep runs at 1.62, 1.80 and 1.98 V, so the current was measured at the right supply but the power was not — overstated 11.1 % at the low supply and understated 9.1 % at the high one. The optimization campaigns are unaffected (they run at `tt`/1.8 V only). The source is fixed, and each `pvt_all_conditions.csv` now carries `POWER_UW_AT_VDD`, `viol_power_at_vdd` and `feasible_at_vdd` **beside** the original columns rather than replacing them. 247 power verdicts move, 226 to fail and 21 to pass, and the published 0-of-540 corner result is unchanged. See `results/README_POWER_CORRECTION.md` for the validation.
9. **The Section 5 artifact does not record its own protocol.** `results/penalty_sensitivity/per_run_records.csv` has nine columns — `lambda, case, run, algo, best_fitness, PSRR_DB, penalty, is_feasible, runtime_sec` — and none of them is dimension, seed, budget or evaluation count, because `penalty_sensitivity.py` never writes them. The protocol is the one documented above and in the paper, but it cannot be confirmed from that file alone. The five circuit campaigns do carry `seed`, `pop`, `eval_budget` and `actual_evals`.
10. **The campaign CSVs do not record their own provenance** — no PDK commit, no library path, no simulator version. That is what `env.sh` is for.

---

## Citation

If you use this benchmark or code, please cite the paper (details to be updated on publication):

```bibtex
@article{sbgr2026,
  title   = {Artificial Bee Colony for Constrained Optimization in 6G-Motivated
             Analog Integrated Circuit Design: A Surrogate Benchmark and a SKY130 Cross-Check},
  author  = {Alhaj Hamad, Bashar Aqel Younis and Y{\i}ld{\i}z, Do{\u{g}}an and
             {\c{S}}ahin, Durmu{\c{s}} {\"O}zkan and Demirci, Sercan and Aslan, Sel{\c{c}}uk},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Under review}
}
```

> Please confirm the final author order and update the entry (volume, pages, DOI) once the paper is published.

---

## License

This code is released under the MIT License; see `LICENSE`.

Two third-party components are used but **not** redistributed here. The SkyWater SKY130 PDK is Apache-2.0 and is fetched at a pinned commit by `truba/00_setup_native.sh`. `ngspice` is distributed under the BSD 3-Clause licence and is built from source by the same script.

---

## Contact

For questions about the benchmark or code, please contact the corresponding author (see the paper).