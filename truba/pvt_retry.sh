#!/bin/bash
# Keep resubmitting the PVT job until its 15 condition files exist.
#
#   CASE=highdim nohup bash truba/pvt_retry.sh > ~/pvt_highdim.log 2>&1 &
#
# WHY. PVT is a single job, not an array, so runcase.sh's repair loop does not cover it -- and the
# "launch failed requeued held" fault hits single jobs too. A held job has Priority=0 and never
# starts again on its own; `scontrol release` is refused because a requeued task is classified
# interactive. Cancel-and-resubmit is the only repair that works, and on highdim the campaign
# needed roughly eight attempts on two of its chunks before they landed, so one retry is not
# enough to assume.
#
# Safe to run twice: it submits only when nothing is queued or what is queued is held, and it
# stops the moment the 15 condition files are on disk. corner_verify.py itself skips any
# condition already computed, so a resubmission never repeats finished work.

set -uo pipefail

CASE="${CASE:?set CASE=base|hard|highdim}"
ROOT=/arf/scratch/${USER}/sky130-bgr
COND="${ROOT}/results/${CASE}/pvt/conditions"
TRIES="${TRIES:-30}"

cd "${ROOT}/code" || exit 1

submit() {
    CASE="${CASE}" sbatch -p barbun --cpus-per-task=20 --mem=32G --time=02:00:00 \
        truba/03_pvt_orfoz.slurm
}

for i in $(seq 1 "${TRIES}"); do
    n=$(ls "${COND}" 2>/dev/null | wc -l)
    if [ "${n}" -ge 15 ]; then
        echo "$(date +%H:%M:%S)  ${CASE} PVT COMPLETE ${n}/15"
        exit 0
    fi

    q=$(squeue -u "$USER" -n bgr_pvt -h -o "%i|%T|%R" 2>/dev/null)
    if [ -z "${q}" ]; then
        echo "$(date +%H:%M:%S)  ${n}/15, nothing queued -> submitting"
        submit
    elif echo "${q}" | grep -qi held; then
        echo "$(date +%H:%M:%S)  ${n}/15, HELD -> cancel and resubmit  [${q}]"
        echo "${q}" | grep -i held | cut -d'|' -f1 | xargs -r scancel
        sleep 5
        submit
    else
        echo "$(date +%H:%M:%S)  ${n}/15, in queue: ${q}"
    fi
    sleep 60
done

n=$(ls "${COND}" 2>/dev/null | wc -l)
echo "$(date +%H:%M:%S)  gave up after ${TRIES} passes, ${n}/15 conditions present"
exit 1
