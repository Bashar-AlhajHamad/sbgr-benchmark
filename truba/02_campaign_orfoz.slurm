#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J bgr_camp
#SBATCH -p orfoz
#SBATCH --array=0-1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=32G
#SBATCH --time=2-18:00:00
#SBATCH --output=/arf/scratch/YOUR_USER/sky130-bgr/logs/camp_%A_%a.out
#SBATCH --error=/arf/scratch/YOUR_USER/sky130-bgr/logs/camp_%A_%a.err

# The campaign, restructured for the orfoz queue's whole-node policy.
#
#   CASE=base    EVALS=150000 sbatch 02_campaign_orfoz.slurm
#   CASE=hard    EVALS=150000 sbatch 02_campaign_orfoz.slurm
#   CASE=highdim EVALS=220000 sbatch --time=3-00:00:00 02_campaign_orfoz.slurm
#
# WALLTIME, revised from measurement rather than from the idle-node estimate. The gate measured
# 128 ms per evaluation on a quiet node, which predicted 5.3 h for a 150,000-evaluation task. The
# first campaign then ran 27 h without a single task finishing, i.e. at least 0.65 s per
# evaluation with 20 concurrent workers on a shared 40-core node -- five times the idle figure,
# because ngspice is memory-bandwidth bound and twenty of them saturate a socket. The completed
# highdim rows bracket the real range at 0.13-0.39 s. 66 h therefore covers 150,000 evaluations up
# to 1.6 s each. This margin is not caution for its own sake: every job writes its row only when
# it finishes, so a task killed one minute short loses its entire run.
#
# WHY THIS SHAPE. orfoz refuses anything but 56 or 112 cores per node:
#   "Orfoz kuyruguna gonderilen isleride node basina 56/112 cekirdek talep ediniz."
# So the one-array-task-per-core design is not submittable there. Instead each array task takes a
# WHOLE node (112 cores) and farms out its own share of the 180 (algo, run) jobs with xargs. Two
# array tasks x 90 jobs = the 180 runs of one case, all resident at once, so wall-clock is still
# ONE job's duration -- 17.2 h at the measured 413 ms/evaluation for base/hard, 25.2 h for highdim.
#
# MEMORY IS REQUESTED EXPLICITLY, and that matters more than it looks. Without --mem, SLURM
# applies the partition default of 8500 MB per CPU, so --cpus-per-task=40 silently requests
# 40 x 8500 = 340 GB -- 89 % of a barbun node's 384 GB. Measured need is ~8-10 GB (MaxRSS 8.4-9.4
# GB for a 20-worker chunk; ~240 MB per worker after the result cache was cut to 20,000 entries).
# A 34x over-request forces the scheduler to find a node with 340 GB free, which is a strong
# candidate for the repeated "launch failed requeued held" -- and a held job has Priority=0 and
# never starts again on its own. 32 GB is roughly 4x the measured peak.
#
# Each worker still writes its OWN row file, so 90 concurrent processes never contend for a file
# and a worker that dies takes nothing else with it. Re-submitting the same array skips whatever
# already has a row file, so a walltime kill is resumable rather than fatal.
#
# BEFORE SUBMITTING:
#   1. the gate has PASSED on an orfoz node (it did: orfoz318, all 8 anchor values within
#      0.05 %, thresholds gating 0/3, bit-identical re-simulation, 413 ms per evaluation)
#   2. --time carries real margin over the measured hours-per-task. A task killed at the limit
#      wastes the whole node-day it consumed.
#   3. cd into /arf/scratch/... first -- TRUBA rejects submissions from $HOME.
#
# ONE QUEUE ONLY. Do not run some cases here and others on hamsi or barbun: different
# microarchitectures would put part of the difference between two algorithms inside the difference
# between two CPUs, and no statistic downstream could separate them.

set -uo pipefail

CASE="${CASE:-base}"
EVALS="${EVALS:-150000}"
POP="${POP:-40}"
RUNS="${RUNS:-30}"
NALGO="${NALGO:-6}"
WORKERS="${WORKERS:-90}"

TOTAL=$((NALGO * RUNS))                  # 180 (algo, run) jobs per case
# CHUNKS must be overridable, because a PARTIAL resubmission is the normal repair path. If a
# couple of array tasks fail to launch, they are re-sent as e.g. --array=2-3, and
# SLURM_ARRAY_TASK_COUNT is then 2 -- so deriving the split from it would compute PER=90 and map
# chunk 2 to jobs 180..269, i.e. outside the range, and the resubmission would silently do
# NOTHING while reporting success. Pass CHUNKS=9 explicitly whenever --array is not the full set.
CHUNKS="${CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-2}}"
CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
PER=$(( (TOTAL + CHUNKS - 1) / CHUNKS ))
FIRST=$(( CHUNK * PER ))
LAST=$(( FIRST + PER - 1 ))
[ "${LAST}" -ge "${TOTAL}" ] && LAST=$(( TOTAL - 1 ))

ROOT=/arf/scratch/${USER}/sky130-bgr
source "${ROOT}/env.sh"
# OUTNAME decouples WHERE the rows go from WHICH case's thresholds are used. Without it a second
# run of the same case at a different budget is a silent no-op: --job-index mode skips a job when
# its row file already exists (run_spice.py:514), and the filename is {case}_{algo}_run{NNN}.csv,
# which carries no budget. So base@150000 pointed at results/base would find 180 finished row
# files, skip all 180, and exit 0 having simulated nothing -- while printing a plausible log.
# --case stays ${CASE} because the thresholds must not change.
OUTNAME="${OUTNAME:-${CASE}}"
OUT="${ROOT}/results/${OUTNAME}"
mkdir -p "${OUT}/rows" "${ROOT}/logs/${OUTNAME}"

echo "=== node chunk ${CHUNK}/$((CHUNKS-1))  of array ${SLURM_ARRAY_JOB_ID:-manual} ==="
echo "  node        : $(hostname)"
echo "  cpu         : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
echo "  cores alloc : ${SLURM_CPUS_PER_TASK:-?}"
echo "  case        : ${CASE}   evals ${EVALS}   SN ${POP}   runs ${RUNS}"
echo "  chunking    : CHUNKS=${CHUNKS}  PER=${PER}  of ${TOTAL} total jobs"
echo "  job indices : ${FIRST}..${LAST}   ($(( LAST - FIRST + 1 )) jobs, ${WORKERS} at a time)"
# A partial resubmission that computed the split wrongly would sit here doing nothing and exit 0.
# Fail loudly instead.
if [ "${FIRST}" -ge "${TOTAL}" ] || [ "${LAST}" -lt "${FIRST}" ]; then
    echo "FATAL: chunk ${CHUNK} maps to an empty range with CHUNKS=${CHUNKS}."
    echo "       Pass CHUNKS explicitly when --array is not the full set, e.g. CHUNKS=9."
    exit 1
fi
echo "  started     : $(date -Is)"
echo

cd "${SKY130_CODE}"

# One process per (algo, run). Each parses the PDK once (~13 s) and then costs ~413 ms per
# evaluation. Output goes to a per-job log so 90 concurrent workers do not interleave into one
# unreadable stream.
run_one() {
    local i="$1"
    "${SKY130_PY}" -u run_spice.py \
        --lib "${SKY130_LIB}" \
        --outdir "${OUT}" \
        --rows-dir "${OUT}/rows" \
        --case "${CASE}" \
        --evals "${EVALS}" \
        --pop "${POP}" \
        --runs "${RUNS}" \
        --job-index "${i}" \
        > "${ROOT}/logs/${OUTNAME}/job_${i}.log" 2>&1
    echo "  [$(date +%H:%M:%S)] job ${i} exit $?  rows=$(ls "${OUT}/rows" | wc -l)/${TOTAL}"
}
export -f run_one
export SKY130_PY SKY130_LIB OUT CASE EVALS POP RUNS ROOT TOTAL

seq "${FIRST}" "${LAST}" | xargs -P "${WORKERS}" -I{} bash -c 'run_one {}'

echo
echo "=== chunk ${CHUNK} finished $(date -Is) ==="
echo "  row files now: $(ls "${OUT}/rows" | wc -l) of ${TOTAL}"
echo
echo "When BOTH chunks are done, merge and run the statistics:"
echo "  cd \$SKY130_CODE"
echo "  \$SKY130_PY run_spice.py --lib \$SKY130_LIB --outdir ${OUT} \\"
echo "      --case ${CASE} --evals ${EVALS} --pop ${POP} --merge"
echo
echo "--merge refuses to proceed on a wrong row count, a duplicated (algo, run), or rows that mix"
echo "cases, populations, budgets or objective definitions. 180 jobs is too many to check by eye,"
echo "and a merge that silently drops one produces a plausible wrong result."
