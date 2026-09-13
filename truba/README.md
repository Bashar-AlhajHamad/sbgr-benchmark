# TRUBA runbook — SKY130 BGR case study (ASOC-D-26-02845)

Everything is gated. Each step's failure is cheap; skipping a step is not.

`ROOT=/arf/scratch/${USER}/sky130-bgr`

> **A container build was attempted first and abandoned.** `ngspice` is built natively, which is
> why every script here carries a `_native` or `_orfoz` suffix and why `00_setup_native.sh` opens by
> deleting the leftover `.sif`. The container scripts are not included, because publishing a path
> that produced none of the reported numbers would only create ambiguity about which one did.

> **Two placeholders to edit before submitting**: `YOUR_ACCOUNT` and `YOUR_USER` in the `#SBATCH`
> lines. SLURM parses those directives before shell expansion, so they cannot be variables. Override
> on the command line instead if you prefer: `sbatch -A <account> --output=<path> …`. Everything
> outside `#SBATCH` is built from `$USER`. `TRUBA_USER` is read only by `00_setup_native.sh`; if your login name differs from your cluster account, set `USER` for the session instead.

---

## 0. Copy the code and build — login node, needs internet

```bash
scp -r code <user>@<truba-login>:/arf/scratch/${USER}/sky130-bgr/
# exclude pdk/ tools/ results_* __pycache__ — the PDK is fetched on the cluster
ssh <user>@<truba-login>
bash /arf/scratch/${USER}/sky130-bgr/code/truba/00_setup_native.sh
```

Builds `ngspice`-46 from source, creates the venv, and fetches the PDK at commit `c6d73a35…` with
`volare`. Compute nodes generally have no route to the internet, so this must happen on the login
node.

It writes **`env.sh`**, which every job sources and which is the only record of what produced the
numbers. Check it:

```bash
source $ROOT/env.sh && md5sum "$SKY130_LIB"
```

| | expected |
|---|---|
| `SKY130_LIB` | `…/sky130A/libs.tech/combined/sky130.lib.spice` — the **binned** set, not `combined/continuous/` |
| MD5 | `365ab743568de364c2214767735a89c6` |
| `SKY130_PDK_COMMIT` | `c6d73a35f524070e85faff4a6a9eef49553ebc2b` |
| `ngspice` / compiler | 46 / `gcc (GCC) 11.3.1` |

The kit ships two model libraries under one commit and the results depend on which is used, so a
mismatched MD5 here means your numbers are not comparable to ours.

## 1. The gate — `debug` queue, ~1 h of its 4 h limit

```bash
sbatch code/truba/01_gate_native.slurm
```

Five questions the workstation cannot answer:

| # | Check | Pass criterion |
|---|---|---|
| 1 | nominal operating point | 1219.90 mV · 24.49 ppm/°C · 47.27 µW |
| 2 | loop gain | 53.21 dB · PM 62.99° · GM 14.86 dB |
| 3 | frozen thresholds | nominal penalty **0** on Base, **3** on Hard |
| 4 | speed | 220,000 evals × s/eval **< 72 h** |
| 5 | 512-point probe | joint feasibility 0.5–5 %, new random baseline |

A different compiler, libm and AVX-512 can move where Newton iteration converges. Usually invisible
— but the frozen thresholds sit a few percent from the nominal point, so "usually" is not enough.

**If it fails: the thresholds are NOT re-tuned.** The 2026-08-04 freeze predates every optimization
run and is the entire answer to the goalpost objection. A failure is an explicit decision to be
taken and disclosed.

**Then, before step 2:** paste the printed `PROBE_BEST_FEASIBLE_PSRR_DB` into
`spice/spice_problem.py`, and set `--time` from the measured hours-per-task with margin.

## 2. The campaign — `orfoz`, ~30–45 h wall clock

Launch through `runcase.sh`, which splits the 180 `(algorithm, run)` tasks of a case into SLURM
array chunks, skips tasks whose row file already exists, and refuses to double-submit:

```bash
TAG=_150k EVALS=150000 CHUNKS=12 WORKERS=5 bash code/truba/runcase.sh base
TAG=_150k EVALS=150000 CHUNKS=12 WORKERS=5 bash code/truba/runcase.sh hard
TAG=_220k EVALS=220000 CHUNKS=12 WORKERS=5 WALL=3-00:00:00 bash code/truba/runcase.sh highdim
```

`TAG` is not cosmetic. Row files are named `{case}_{algo}_run{NNN}.csv` with no budget in the name
and `--job-index` skips a job whose row file exists, so a 150,000-evaluation run written into
`results/base` would skip all 180 jobs and exit 0 having simulated nothing. `TAG=_150k` sends it to
`results/base_150k` while `--case` stays `base`, leaving the frozen thresholds untouched.

`WORKERS=5`, not 15: the evaluator is memory-bandwidth-bound, and 0.151 s/eval at 5 workers per node
becomes 1.1–1.6 s/eval at 15.

540 tasks = the published campaign's 540 runs, ~17,460 core-hours ≈ 4.8 `orfoz` nodes.

**One queue only.** Spreading tasks over `orfoz` and `hamsi` would put part of the difference
between two algorithms inside the difference between two CPUs, and no statistic downstream could
separate them.

Merge each case when its array finishes:

```bash
source $ROOT/env.sh
$SKY130_PY run_spice.py --lib $SKY130_LIB --outdir $ROOT/results/base_150k \
    --case base --evals 150000 --pop 40 --merge
```

`--merge` refuses to proceed on a wrong row count, a duplicated `(algo, run)`, or rows that mix
cases, populations, budgets or objective definitions. 540 tasks is too many to check by eye, and a
merge that silently drops one yields a plausible wrong result.

## 3. PVT verification — `orfoz`, ~20 min

```bash
CASE=base sbatch code/truba/03_pvt_orfoz.slurm      # after step 2 has been merged
# then
source $ROOT/env.sh
$SKY130_PY corner_verify.py --csv $ROOT/results/base_150k/per_run_records.csv \
    --case base --outdir $ROOT/results/base_150k/pvt --merge
```

5 corners (`tt ss ff sf fs`) × 3 supplies (1.62 / 1.80 / 1.98 V), no re-optimization. ~17 core-hours
against the campaign's 17,460.

**Rule fixed before the data:** feasible at tt/1.8 V and infeasible at *any* condition counts as
**infeasible**.

> **Then run `pvt_summary_fix.py`.** The summary this step writes scores a simulation that returned
> nothing identically to a design that violates all six specifications, which makes its
> `nominal_feasible` column contradict `per_run_records.csv` for 173 of 540 designs. The fix script
> writes `pvt_summary_corrected.csv` beside it, carrying both scoring conventions and the failure
> counts. Do not use the uncorrected file.

## 4. Controls — `barbun`, minutes

Pure surrogate, no ngspice, so these are cheap:

```bash
EXP=dim      sbatch --array=0-2 code/truba/04_controls.slurm   # D = 12, 7, 16
EXP=pressure sbatch --array=0-1 code/truba/04_controls.slurm   # frozen vs relaxed thresholds
```

Each includes the published `D=12` setting as a control on the control: it must reproduce
ABC 2.300 / PSO 3.000 / GA 3.033 / ACO 3.100 / FA 4.633 / GWO 4.933. If it does not, the harness is
wrong and neither experiment can be read.

## 5. Bring back

| File | Per case |
|---|---|
| `results/<case>/per_run_records.csv` | the campaign |
| `results/<case>/{core_stats,friedman,average_ranks,wilcoxon_holm}.csv` | statistics |
| `results/<case>/pvt/pvt_all_conditions.csv` | the raw 2,700-row sweep |
| `results/<case>/pvt/pvt_summary_corrected.csv` | robustness — **not** `pvt_summary.csv` |
| `env.sh` | the provenance record for §6 |

`chain_protocol.sh` runs the remaining two cases of the protocol back to back, unattended, once the
first has been launched — each case needs 1,440 cores, so they cannot share the queue. `status.sh`
and `pvt_retry.sh` are monitoring and requeue helpers for the recurring
*"launch failed requeued held"* state.

---

## Facts this setup depends on

- **`orfoz`**: 504 nodes × 112 cores, 256 GB/node = 2,341 MB per core. Measured need is ~743 MB per
  worker (ngspice 40 MB + cache 583 MB + interpreter ~120 MB), so ~3× headroom. The cache was cut
  from 200,000 to 20,000 entries after measuring 4,072 B/entry and a **0 %** hit rate — 583 MB of
  pure waste at a 150,000-evaluation budget.
- **3-day walltime** is the binding constraint, not core count. That is what step 1's timing
  measurement exists to check.
- **Objective**: worst-case PSRR over 10 Hz – 10 MHz, restated 2026-08-06. The previous
  single-frequency (100 Hz) form is retained as a recorded diagnostic. Every CSV row carries
  `psrr_def`, so no column's meaning can change silently between campaigns.
- **Two declared protocol deviations**, both disclosed in the paper:
  1. **dimension** — 7 vs 12 and 16 vs 30, fixed by the circuit's topology and not chooseable;
  2. **seeding** — `run_spice.py` applies a single `CASE_OFFSET` where `run.py` offsets per case, so
     all three cases draw the same thirty run seeds, and `base`/`hard` additionally share dimension
     and bounds and therefore have identical initial populations. Within-case pairing is intact, so
     every per-case statistic is sound; the three cases are variants, not independent replicates.

  SN=40, the 150,000/220,000 budgets, 30 runs, the six algorithms, λ=1000, the six-spec penalty
  count, clipping and the shared-seed-per-run design all match the manuscript.
