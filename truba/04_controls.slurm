#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J bgr_ctl
#SBATCH -p barbun
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/arf/scratch/YOUR_USER/sky130-bgr/logs/ctl_%A_%a.out
#SBATCH --error=/arf/scratch/YOUR_USER/sky130-bgr/logs/ctl_%A_%a.err
# ---------------------------------------------------------------------------------------
# BEFORE SUBMITTING: SLURM parses #SBATCH directives before shell expansion, so the account
# and the log paths below cannot use $USER. Replace YOUR_ACCOUNT and YOUR_USER with your own,
# or override them on the command line:
#     sbatch -A <account> --output=<path> --error=<path> <this script>
# ---------------------------------------------------------------------------------------
#
# The two surrogate controls for Section 6. Pure SBGR -- no ngspice, no PDK -- so a run costs
# microseconds per evaluation and the whole thing is minutes rather than hours.
#
#   EXP=dim      sbatch --array=0-2 truba/04_controls.slurm     # D = 12, 7, 16
#   EXP=pressure sbatch --array=0-1 truba/04_controls.slurm     # frozen vs relaxed thresholds
#
# One array task per CONDITION; the 30 runs of that condition are farmed across the node's 40
# cores with xargs. 3 conditions x 30 runs x 6 algorithms x 150,000 evaluations is 81 M surrogate
# evaluations, which is why this is worth 40 cores for a few minutes and not 1,440 for hours.
#
# NOTHING HERE TOUCHES SECTION 5. Both experiments add new conditions on the same surrogate with
# the same code, protocol and seeds; the published D = 12 setting is included as the control on
# the control and must reproduce ABC 2.300 / PSO 3.000 / GA 3.033 / ACO 3.100 / FA 4.633 /
# GWO 4.933. If it does not, the harness is wrong and neither experiment can be read.
#
# problems.py and algorithms.py are read-only and md5-pinned. The relaxed thresholds in the
# pressure experiment are set on the problem INSTANCE, not in the module.

set -uo pipefail

EXP="${EXP:?set EXP=dim or EXP=pressure}"
WORKERS="${WORKERS:-30}"
RUNS="${RUNS:-30}"
COND="${SLURM_ARRAY_TASK_ID:-0}"

ROOT=/arf/scratch/${USER}/sky130-bgr
source "${ROOT}/env.sh"
OUT="${ROOT}/results/controls_${EXP}"
mkdir -p "${OUT}/rows" "${ROOT}/logs/ctl_${EXP}"
cd "${SKY130_CODE}"

FIRST=$(( COND * RUNS ))
LAST=$(( FIRST + RUNS - 1 ))

echo "=== control '${EXP}', condition ${COND}, jobs ${FIRST}..${LAST} ==="
echo "  node   : $(hostname)"
echo "  started: $(date -Is)"
echo

run_one() {
    "${SKY130_PY}" -u sbgr_controls.py --experiment "${EXP}" --outdir "${OUT}" \
        --job-index "$1" > "${ROOT}/logs/ctl_${EXP}/job_$1.log" 2>&1
    echo "  [$(date +%H:%M:%S)] job $1 exit $?  rows=$(ls "${OUT}/rows" | wc -l)"
}
export -f run_one
export SKY130_PY EXP OUT ROOT

seq "${FIRST}" "${LAST}" | xargs -P "${WORKERS}" -I{} bash -c 'run_one {}'

echo
echo "=== condition ${COND} finished $(date -Is), rows now $(ls "${OUT}/rows" | wc -l) ==="
echo
echo "When every condition is done:"
echo "  \$SKY130_PY sbgr_controls.py --experiment ${EXP} --outdir ${OUT} --merge"
