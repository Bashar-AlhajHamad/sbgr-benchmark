#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J bgr_refctl
#SBATCH -p orfoz
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=00:30:00
#SBATCH --output=/arf/scratch/YOUR_USER/sky130-bgr/logs/refctl_%j.out
#SBATCH --error=/arf/scratch/YOUR_USER/sky130-bgr/logs/refctl_%j.err
# ---------------------------------------------------------------------------------------
# BEFORE SUBMITTING: SLURM parses #SBATCH directives before shell expansion, so the account
# and the log paths above cannot use $USER. Replace YOUR_ACCOUNT and YOUR_USER with your own,
# or override them on the command line:
#     sbatch -A <account> --output=<path> --error=<path> <this script>
# ---------------------------------------------------------------------------------------

# The reference-design corner control, captured to a file so it can be released.
#
# WHY THIS SCRIPT EXISTS, AND WHY IT DID NOT BEFORE. Section 6.6 of the manuscript reports that the
# untouched reference design "passes only 5 of the 15 conditions", and the abstract ends on that
# clause. It is the measurement that upgrades "0 of 540 designs survive the sweep" from a statement
# about the optimizers into a statement about single-corner formulation -- which is the paper's most
# transferable engineering conclusion.
#
# `pvt_reference_control.py` produces it, but it PRINTS TO STDOUT AND WRITES NO FILE, and no job
# script ever invoked it: `status.sh` only checks that the file is PRESENT. So the number came from
# an interactive run whose output was never saved, and nothing in the released archive lets a reader
# check it -- every `pvt` artifact contains the six optimizers and no reference-design row, while
# Data availability promises "the complete per-run records of every campaign reported here".
#
# This job closes that gap: it runs the control on the machine that ran the campaign and keeps the
# output inside the results tree, where the release picks it up.
#
#   CASE=base OUTNAME=base_150k sbatch 05_reference_control.slurm
#
# COST. Seven designs -- the untouched reference plus each algorithm's best -- across 5 process
# corners x 3 supplies = 105 simulations at ~0.41 s, so under a minute of simulation. The half-hour
# wall clock is entirely margin for the PDK load. This is three orders of magnitude cheaper than the
# campaign it qualifies.
#
# WHAT COUNTS AS A RESULT. The rule is the one pre-registered on 2026-08-06 and already applied to
# the campaign designs: feasible at tt/1.8 V and infeasible at ANY of the 15 conditions counts as
# INFEASIBLE. It is not re-declared here, it is inherited -- the control is only meaningful if it is
# scored exactly as the 540 were.
#
# The control cuts both ways and both outcomes are reportable:
#   reference survives 15/15 -> the optimizers traded robustness for objective, and 0/540 is
#                               a statement about them
#   reference fails too      -> single-corner optimization cannot produce corner-robust designs on
#                               this circuit, and the campaign inherited that from the problem

set -uo pipefail

CASE="${CASE:-base}"
ROOT=/arf/scratch/${USER}/sky130-bgr
source "${ROOT}/env.sh"

# Same split as 03_pvt_orfoz.slurm: OUTNAME says WHERE the campaign's rows live, --case says WHICH
# frozen threshold set scores them. The thresholds must not move between the campaign and its
# control, or the control measures the thresholds instead of the design.
OUTNAME="${OUTNAME:-${CASE}}"
OUT="${ROOT}/results/${OUTNAME}"
CSV="${OUT}/per_run_records.csv"
LOG="${OUT}/pvt/reference_control.log"

if [ ! -f "${CSV}" ]; then
    echo "FATAL: ${CSV} not found. Run the campaign and its --merge step first."
    exit 1
fi

mkdir -p "${OUT}/pvt"
cd "${SKY130_CODE}"

{
    echo "=== reference-design corner control, case ${CASE}, rows from ${OUTNAME} ==="
    echo "  node    : $(hostname)"
    echo "  lib     : ${SKY130_LIB}"
    echo "  csv     : ${CSV}  ($(( $(wc -l < "${CSV}") - 1 )) designs)"
    echo "  started : $(date -Is)"
    echo
} | tee "${LOG}"

# -u so the per-condition lines reach the log as they are produced rather than at exit: if the job
# is killed on time, a partial control is still a readable one.
"${SKY130_PY}" -u pvt_reference_control.py \
    --lib "${SKY130_LIB}" \
    --case "${CASE}" \
    --csv "${CSV}" 2>&1 | tee -a "${LOG}"
STATUS="${PIPESTATUS[0]}"

{
    echo
    echo "=== finished $(date -Is), exit ${STATUS} ==="
} | tee -a "${LOG}"

echo
echo "The releasable artifact is ${LOG}."
echo "Copy it back with the rest of results/${OUTNAME}/pvt/ and it ships with the archive."
exit "${STATUS}"
