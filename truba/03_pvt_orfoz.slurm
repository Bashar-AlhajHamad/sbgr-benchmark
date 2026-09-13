#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J bgr_pvt
#SBATCH -p orfoz
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/YOUR_USER/sky130-bgr/logs/pvt_%j.out
#SBATCH --error=/arf/scratch/YOUR_USER/sky130-bgr/logs/pvt_%j.err
# ---------------------------------------------------------------------------------------
# BEFORE SUBMITTING: SLURM parses #SBATCH directives before shell expansion, so the account
# and the log paths below cannot use $USER. Replace YOUR_ACCOUNT and YOUR_USER with your own,
# or override them on the command line:
#     sbatch -A <account> --output=<path> --error=<path> <this script>
# ---------------------------------------------------------------------------------------

# PVT verification: re-simulate every final design at 5 process corners x 3 supplies, WITHOUT
# re-optimizing. All 15 conditions farmed onto one whole node, because orfoz will not allocate
# fewer than 56 cores.
#
#   CASE=base sbatch 03_pvt_orfoz.slurm      # after 02_campaign has been merged
#
# WHY THIS RUNS AT ALL. The paper's headline word is robustness. The campaign optimizes at a single
# corner because that is the published problem -- but optimization quality at one corner is not
# robustness in the sense an analog designer means, so a case study that stops there does not test
# the paper's own thesis.
#
# The need is measured, not hypothetical: the objective rewards designs that buy low-frequency rail
# rejection by giving up high-frequency rejection (92 dB at 100 Hz against 2.8 dB at 10 MHz on one
# measured design). There is no reason such a design holds at ss or ff.
#
# Cost: 180 designs x 15 conditions x ~0.41 s = ~19 min of CPU per case, run 15-wide so a few
# minutes of wall-clock, against the campaign's ~10,728 core-hours. Three orders of magnitude
# cheaper, so leaving it out would have been an oversight rather than a trade-off.
#
# THE RULE, FIXED BEFORE THE DATA (plan file, PRE-REGISTRATION, 2026-08-06): a design feasible at
# tt/1.8 V and infeasible at ANY of the 15 conditions counts as INFEASIBLE. Declared in advance
# because it decides the third pre-registered hypothesis, and choosing it afterwards would decide
# the result rather than measure it.

set -uo pipefail

CASE="${CASE:-base}"
ROOT=/arf/scratch/${USER}/sky130-bgr
source "${ROOT}/env.sh"
# OUTNAME separates WHERE the campaign's rows live from WHICH case's thresholds score them, so a
# second campaign on the same case at a different budget (results/base_150k) can be verified
# without overwriting the first. --case stays ${CASE} because the frozen thresholds must not move.
OUTNAME="${OUTNAME:-${CASE}}"
OUT="${ROOT}/results/${OUTNAME}"
CSV="${OUT}/per_run_records.csv"

if [ ! -f "${CSV}" ]; then
    echo "FATAL: ${CSV} not found. Run the campaign and its --merge step first."
    exit 1
fi

mkdir -p "${ROOT}/logs/pvt_${CASE}"
cd "${SKY130_CODE}"

echo "=== PVT verification, case ${CASE} ==="
echo "  node   : $(hostname)"
echo "  designs: $(( $(wc -l < "${CSV}") - 1 ))"
echo "  started: $(date -Is)"
echo

pvt_one() {
    local i="$1"
    "${SKY130_PY}" -u corner_verify.py \
        --lib "${SKY130_LIB}" \
        --csv "${CSV}" \
        --case "${CASE}" \
        --outdir "${OUT}/pvt" \
        --job-index "${i}" \
        > "${ROOT}/logs/pvt_${CASE}/cond_${i}.log" 2>&1
    echo "  [$(date +%H:%M:%S)] condition ${i} exit $?"
}
export -f pvt_one
export SKY130_PY SKY130_LIB CSV CASE OUT ROOT

seq 0 14 | xargs -P 15 -I{} bash -c 'pvt_one {}'

echo
echo "=== merging the 15 conditions ==="
"${SKY130_PY}" -u corner_verify.py \
    --csv "${CSV}" --case "${CASE}" --outdir "${OUT}/pvt" --merge

echo
echo "=== finished $(date -Is) ==="
