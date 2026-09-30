#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J ldo_camp
#SBATCH -p orfoz
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/arf/scratch/YOUR_USER/sky130-bgr/logs/ldo_%A_%a.out
#SBATCH --error=/arf/scratch/YOUR_USER/sky130-bgr/logs/ldo_%A_%a.err

# Circuit C -- the SKY130 low-dropout regulator.
#
#   RECOMMENDED -- six nodes, ~40 minutes:
#     CHUNKS=6 sbatch --array=0-5 06_ldo_orfoz.slurm
#     # then, once all six have finished:
#     $SKY130_PY run_spice_ldo.py --lib $SKY130_LIB --outdir <OUT> --merge
#
#   Single node, ~4.1 hours (what --array-less submission does):
#     sbatch --time=12:00:00 06_ldo_orfoz.slurm
#
#   Other overrides:
#     WORKERS=60 CHUNKS=6 sbatch --array=0-5 06_ldo_orfoz.slurm
#     OUTNAME=ldo_rerun CHUNKS=6 sbatch --array=0-5 06_ldo_orfoz.slurm
#
# WALL TIME BY NODE COUNT, from the measured 0.0122 runs/s per node:
#     1 node 4.1 h | 2 nodes 2.0 h | 4 nodes 1.0 h | 6 nodes 41 min | 13 nodes 19 min
# A 1536-core allocation is about 13 orfoz nodes, so six chunks per circuit lets BOTH campaigns run
# at once inside 87 % of it.
#
# CHUNKS must match the --array range. The script cannot see what range Slurm was given, so it
# compares the two and refuses to start if they disagree -- a mismatch would silently drop or
# duplicate runs, which is far worse than a failed submission.
#
# WALL TIME AND BUDGET. 150,000 evaluations per run, matching Circuit A's transistor-level campaign.
# MEASURED on TRUBA at this budget: 897-6740 s per run, so a shard of 30 concurrent runs finishes in
# under 2 h -- BUT THAT MEASUREMENT IS SELECTION-BIASED. It was taken from shards that had
# COMPLETED; shards still running contributed nothing, so its upper end is truncated by construction.
# The live campaign shows the cost varies enormously BY ALGORITHM. At 2.5 h in, with one run per
# worker: FA and GA had finished all 30 runs, ABC was at 20/30 and PSO at 15/30, while GWO and ACO
# had not finished a single run -- their row files did not yet exist.
#
# The mechanism is the evaluator's result cache. FA and GA converge, their candidates snap to the
# same 0.005 um quantisation grid and hit the cache, so the effective cost per evaluation collapses.
# GWO and ACO keep producing distinct candidates, so far more of their 150,000 evaluations actually
# reach ngspice. Mostly-uncached at ~0.5 s per evaluation is about 20 h for one run.
#
# --time=24:00:00 is therefore set against the SLOWEST algorithm, not the fastest. orfoz allows
# MaxTime=3-00:00:00, so this is well inside the limit, and a shard that finishes early releases its
# node immediately -- an unused wall-time request costs nothing but scheduling priority.
#
# An earlier --time=48:00:00 came from estimating 22 h per run by scaling a 2,500-evaluation
# measurement (0.53 s/evaluation) linearly to 150,000. That extrapolation is wrong, because the
# evaluator caches results: as the population converges, candidates snap to the same 0.005 um
# quantisation grid and repeat, so the effective cost per evaluation collapses late in a run. It is
# The 48 h request also coincided with 7 of 12 shards ending "launch failed requeued held", and
# that was attributed to a wall-time limit. THAT ATTRIBUTION WAS WRONG: orfoz reports
# MaxTime=3-00:00:00 and every QOS an empty MaxWall, so 48 h was inside the limit. The hold has
# no established cause. It is recorded here as unexplained rather than left with a tidy but
# false explanation, because the same false explanation was later printed as a diagnosis by the
# campaign driver and cost a launch.
#
# More shards does NOT shorten this. The 180 runs already run at once; what changes is only the
# per-run speed, which improves slightly with fewer workers per node. Six shards per circuit lets both
# campaigns run together inside a 1536-core allocation.
#
# THE BUDGET WAS 2,500 IN THE FIRST CAMPAIGN, AND THAT WAS AN ERROR. The manuscript's 2,500 figure
# describes the SURROGATE run at a reduced protocol as a control; the caption of tab:spice_results
# states Circuit A's real campaign as 150,000 per run. At 2,500 the new circuits were not comparable
# to Circuit A, which is the only reason for running them -- and since ABC's rank improves with budget
# (2.300 at 2,500 to 1.750 at 80,000 on the surrogate at D=7), the short budget disadvantaged ABC
# specifically. The 2,500 results are kept as a budget control under results/<circuit>/.
#
# MEMORY. --mem=64G, and WORKERS defaults to 30 rather than 90: a shard holds only 30 runs, so 90
# workers left 60 threads idle while the active ones competed for memory. With the plot leak fixed
# each ngspice sits flat at ~45 MB, so this is headroom rather than necessity. Do NOT drop --mem: orfoz otherwise applies a partition
# default per CPU, which is the trap recorded in 02_campaign_orfoz.slurm.

set -euo pipefail

# --- array chunking -----------------------------------------------------------------------------
# Each array task takes a strided slice of the 180 runs and writes its own row file; --merge
# combines them afterwards. Without --array this stays a single job doing everything, as before.
CHUNKS="${CHUNKS:-1}"
CHUNK="${SLURM_ARRAY_TASK_ID:-}"

# A PARTIAL resubmission is the normal repair path -- one shard dies, you resend just that index --
# so fewer array tasks than CHUNKS is legitimate and only worth noting. MORE tasks than CHUNKS is
# always wrong: task indices at or above CHUNKS would be rejected by the driver anyway, and the
# striding would not cover the run list.
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_COUNT}" -gt "${CHUNKS}" ]; then
        echo "FATAL: --array has ${SLURM_ARRAY_TASK_COUNT} tasks but CHUNKS=${CHUNKS}." >&2
        echo "       CHUNKS must be at least the number of array tasks, or the striding does" >&2
        echo "       not cover every run. e.g. CHUNKS=6 sbatch --array=0-5 ..." >&2
        exit 1
    elif [ "${SLURM_ARRAY_TASK_COUNT}" -lt "${CHUNKS}" ]; then
        echo "NOTE: ${SLURM_ARRAY_TASK_COUNT} array task(s) against CHUNKS=${CHUNKS} -- this is a"
        echo "      PARTIAL resubmission. Correct if you are repairing specific shards; wrong if"
        echo "      you meant to run the whole campaign."
    fi
fi
if [ "${CHUNKS}" -gt 1 ] && [ -z "${CHUNK}" ]; then
    echo "FATAL: CHUNKS=${CHUNKS} but this is not an array job. Submit with --array=0-$((CHUNKS-1))." >&2
    exit 1
fi

ROOT=/arf/scratch/YOUR_USER/sky130-bgr
source "${ROOT}/env.sh"          # provides SKY130_CODE, SKY130_LIB, SKY130_PY

OUTNAME="${OUTNAME:-ldo_150k}"
OUT="${ROOT}/results/${OUTNAME}"
WORKERS="${WORKERS:-30}"
EVALS="${EVALS:-150000}"
POP="${POP:-40}"
RUNS="${RUNS:-30}"
ALGOS="${ALGOS:-ABC GWO FA PSO GA ACO}"
TIMEOUT="${TIMEOUT:-60}"

mkdir -p "${OUT}/rows" "${ROOT}/logs"
SHARD_ARGS=()
if [ -n "${CHUNK}" ]; then
    SHARD_ARGS=(--chunks "${CHUNKS}" --chunk "${CHUNK}" --rows-dir "${OUT}/rows")
fi

cd "${SKY130_CODE}"

echo "=== Circuit C campaign: SKY130 low-dropout regulator ==="
echo "  node        : $(hostname)"
echo "  cpu         : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
echo "  cpus        : ${SLURM_CPUS_PER_TASK:-?}    workers ${WORKERS}"
echo "  protocol    : ${RUNS} runs x ${EVALS} evaluations, pop ${POP}"
echo "  algorithms  : ${ALGOS}"
echo "  library     : ${SKY130_LIB}"
echo "  outdir      : ${OUT}"
echo "  shard       : ${CHUNK:-none} of ${CHUNKS}"
echo "  started     : $(date)"
echo

# Fail once, loudly, rather than 90 times in parallel.
if [ ! -f "${SKY130_LIB}" ]; then
    echo "FATAL: model library not found: ${SKY130_LIB}" >&2
    exit 1
fi

# Refuse to launch against unfrozen thresholds. This guards the worst failure available: with no
# thresholds every candidate is feasible, so the campaign completes and returns a full ranking that
# means nothing at all -- and looks exactly like a result.
"${SKY130_PY}" - <<'PY'
import sys
sys.path.insert(0, "spice")
import spice_problem_ldo as sp
if not sp.CALIBRATED or not sp.CONSTRAINTS:
    sys.exit("FATAL: Circuit C thresholds are not frozen; refusing to launch")
print("thresholds frozen:")
for k, v in sp.CONSTRAINTS.items():
    print(f"    {k:18s} {v}")
print(f"  bars to clear: random-feasible {sp.PROBE_BEST_FEASIBLE_PSRR_DB:.2f} dB, "
      f"reference design {sp.NOMINAL_OBJECTIVE_DB:.2f} dB")
PY
echo

srun --cpu-bind=none "${SKY130_PY}" run_spice_ldo.py \
    --lib     "${SKY130_LIB}" \
    --outdir  "${OUT}" \
    --algos   ${ALGOS} \
    --runs    "${RUNS}" \
    --evals   "${EVALS}" \
    --pop     "${POP}" \
    --workers "${WORKERS}" \
    --timeout "${TIMEOUT}" \
    "${SHARD_ARGS[@]}"

echo
echo "  finished : $(date)"
echo
if [ -n "${CHUNK}" ]; then
    echo "This was shard ${CHUNK} of ${CHUNKS}. When ALL shards have finished, merge:"
    echo "  ${SKY130_PY} run_spice_ldo.py --lib ${SKY130_LIB} \\"
    echo "      --outdir ${OUT} --merge --rows-dir ${OUT}/rows"
    echo "  The merge reports if any runs are missing, and resubmitting the array fills only"
    echo "  the gaps -- shards resume rather than repeat."
    echo
fi
echo "Read these before believing anything:"
echo "  ${OUT}/ldo/average_ranks.csv    -- the ranking"
echo "  ${OUT}/ldo/friedman.csv         -- whether the spread is significant at all"
echo "  ${OUT}/ldo/wilcoxon_holm.csv    -- ABC against each baseline, Holm-corrected"
echo "  ${OUT}/ldo/violations.csv       -- WHICH specification each method fails"
echo "  ${OUT}/ldo/extended_stats.csv   -- success rate and feasible-only gain"
echo "  ${OUT}/per_run_records.csv        -- all 180 rows, both parameterisations recorded"
