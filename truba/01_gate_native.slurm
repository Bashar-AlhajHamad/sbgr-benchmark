#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -J bgr_gate
#SBATCH -p debug
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --output=/arf/scratch/YOUR_USER/sky130-bgr/logs/gate.out
#SBATCH --error=/arf/scratch/YOUR_USER/sky130-bgr/logs/gate.err
# ---------------------------------------------------------------------------------------
# BEFORE SUBMITTING: SLURM parses #SBATCH directives before shell expansion, so the account
# and the log paths below cannot use $USER. Replace YOUR_ACCOUNT and YOUR_USER with your own,
# or override them on the command line:
#     sbatch -A <account> --output=<path> --error=<path> <this script>
# ---------------------------------------------------------------------------------------

# THE GATE. Nothing else gets submitted until this passes.
#
# Four questions that cannot be answered from the workstation:
#   1. does THIS ngspice build reproduce the validated circuit measurements?
#      1219.90 mV / 24.49 ppm/degC / 47.27 uW, loop gain 53.21 dB, PM 62.99 deg, GM 14.86 dB
#   2. do the frozen thresholds still gate as declared -- nominal penalty 0 on Base, 3 on Hard?
#   3. how fast is one core here, i.e. does a 220,000-evaluation task fit inside 72 hours?
#   4. what is the random-search baseline under the corrected worst-case-over-band objective?
#
# It matters because this binary was compiled by a different compiler against a different libm on
# a different microarchitecture than the one every reference number was measured on. ngspice
# solves a nonlinear circuit by Newton iteration to a tolerance, so a last-digit difference can
# move which point it converges to. Usually invisible -- but the frozen thresholds sit only a few
# percent from the nominal operating point, so "usually" is not sufficient.
#
# IF IT FAILS: the thresholds are NOT re-tuned. The 2026-08-04 freeze predates every optimization
# run and is the entire answer to the reviewer who asks why the published 65-nm numbers were not
# used. A failure is an explicit decision to be taken and disclosed, not a calibration step.

set -euo pipefail

ROOT=/arf/scratch/${USER}/sky130-bgr
mkdir -p "${ROOT}/logs" "${ROOT}/results"
source "${ROOT}/env.sh"

echo "=== job ==="
echo "  job id      : ${SLURM_JOB_ID}"
echo "  node        : $(hostname)"
echo "  partition   : ${SLURM_JOB_PARTITION}"
echo "  cpu model   : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
echo "  os          : $(cat /etc/redhat-release 2>/dev/null || uname -sr)"
echo "  ngspice     : ${NGSPICE_EXE}  (built from source ${SKY130_NGSPICE_VERSION})"
echo "  compiler    : ${SKY130_BUILD_CC}"
echo "  pdk commit  : ${SKY130_PDK_COMMIT}"
echo "  started     : $(date -Is)"
echo

cd "${SKY130_CODE}"

echo "=== 1. code-level pre-flight (no simulator needed) ==="
"${SKY130_PY}" preflight_spice.py
echo

echo "=== 2. anchor + thresholds + determinism + timing + 512-point probe ==="
# --probe 512 with the ORIGINAL calibration seed 20260804, so the design points are identical to
# the 2026-08-04 probe and ONLY the objective's scoring differs. That is what makes the new
# random-search baseline comparable to the old one rather than a fresh unrelated number.
"${SKY130_PY}" verify_anchor.py \
    --lib "${SKY130_LIB}" \
    --probe 512 \
    --probe-seed 20260804 \
    --timing 200 \
    2>&1 | tee "${ROOT}/results/anchor_gate.txt"

echo
echo "=== finished $(date -Is) ==="
echo
echo "Read ${ROOT}/results/anchor_gate.txt before submitting anything else."
echo "If the gate PASSED:"
echo "  1. paste the printed PROBE_BEST_FEASIBLE_PSRR_DB into spice/spice_problem.py"
echo "  2. set --time in 02_campaign_native.slurm from the measured hours-per-task, with margin"
echo "  3. then, and only then, submit the campaign"
