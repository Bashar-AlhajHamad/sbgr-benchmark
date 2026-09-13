#!/bin/bash
# Run the remaining two cases of the PRE-REGISTERED protocol, back to back, unattended.
#
#   nohup bash truba/chain_protocol.sh > ~/chain.log 2>&1 &
#
# WHY A CHAIN AND NOT TWO SUBMISSIONS. Each case takes 36 chunks x 40 cores = 1440 of the
# account's 1536-core GrpTRES, so two cases cannot be resident at once: the second would sit on
# AssocGrpCpuLimit, and if it did partially start it would share nodes with the first and change
# the per-evaluation cost mid-campaign -- which would put a throughput difference inside the
# comparison between two algorithms. Sequential is not a convenience here, it is a control.
#
# runcase.sh already blocks until its case reaches 180 rows, so sequencing is just calling it
# twice. What this adds is the wait for base_150k (launched separately and still running), the
# merge after each case, and a log that survives a dropped ssh session.
#
# BUDGETS ARE THE PRE-REGISTERED ONES: 150,000 for base and hard, 220,000 for highdim
# (plan file, PRE-REGISTRATION). The finished 20,000-evaluation set is retained as the
# budget-sensitivity arm, which is why these write to *_150k / *_220k and not over it.
#
# MEASURED BASIS FOR THE WALLTIME: the canary measured 0.137-0.171 s per evaluation at 5 workers
# on 40 cores -- 9x faster than the 1.1-1.6 s seen at 15 workers, and close to the 0.128 s quiet
# node figure, i.e. memory-bandwidth contention is essentially gone. That puts 150,000 at ~7.1 h
# and 220,000 at ~11 h. WALL=48:00:00 is between 4x and 6.8x that, which also covers a full
# timeout load if the canary happened to sample fewer hangs than a long run will.

set -uo pipefail

ROOT=/arf/scratch/${USER}/sky130-bgr
cd "${ROOT}/code" || exit 1
source ../env.sh

rows_of() { ls "../results/$1/rows" 2>/dev/null | grep -v curve | wc -l; }

wait_for() {                       # $1 = outname to wait on
    local n
    n=$(rows_of "$1")
    if [ "${n}" -ge 180 ]; then
        echo "$(date +%H:%M)  $1 already complete (${n}/180)"
        return 0
    fi
    echo "$(date +%H:%M)  waiting for $1 (${n}/180) before starting the next case"
    while [ "$(rows_of "$1")" -lt 180 ]; do
        sleep 300
        echo "$(date +%H:%M)  $1 $(rows_of "$1")/180"
    done
    echo "$(date +%H:%M)  $1 COMPLETE"
}

merge_case() {                     # $1 = case, $2 = outname, $3 = evals
    echo
    echo "=================================================================="
    echo "$(date +%H:%M)  merging $2   (--case $1, ${3} evals)"
    echo "=================================================================="
    # --case carries the THRESHOLDS and must be the real case, not the tagged directory name.
    "${SKY130_PY}" run_spice.py --lib "${SKY130_LIB}" \
        --outdir "../results/$2" --rows-dir "../results/$2/rows" \
        --case "$1" --evals "$3" --pop 40 --merge 2>&1 | tail -32
}

echo "=================================================================="
echo "PRE-REGISTERED PROTOCOL CHAIN   started $(date -Is)"
echo "  1. wait for base_150k   (running separately)"
echo "  2. hard    @ 150,000 -> results/hard_150k"
echo "  3. highdim @ 220,000 -> results/highdim_220k"
echo "  36 chunks x 5 workers x 40 cores = 1440 of 1536 cores, one case at a time"
echo "=================================================================="

wait_for base_150k
merge_case base base_150k 150000

echo
echo "$(date +%H:%M)  launching hard @ 150,000"
TAG=_150k CHUNKS=36 WORKERS=5 EVALS=150000 WALL=48:00:00 ./runcase.sh hard
merge_case hard hard_150k 150000

echo
echo "$(date +%H:%M)  launching highdim @ 220,000"
TAG=_220k CHUNKS=36 WORKERS=5 EVALS=220000 WALL=48:00:00 ./runcase.sh highdim
merge_case highdim highdim_220k 220000

echo
echo "=================================================================="
echo "$(date +%H:%M)  CHAIN FINISHED $(date -Is)"
for o in base_150k hard_150k highdim_220k; do
    printf "  %-14s rows %3s/180   merged %s\n" "$o" "$(rows_of "$o")" \
        "$([ -f "../results/$o/per_run_records.csv" ] && echo yes || echo NO)"
done
echo
echo "PVT is NOT run by this chain -- it needs the merged CSV of each case:"
echo "  CASE=base    OUTNAME=base_150k    sbatch -p barbun --cpus-per-task=20 --mem=32G \\"
echo "      --time=02:00:00 truba/03_pvt_orfoz.slurm"
echo "  (repeat for hard_150k and highdim_220k; 03_pvt_orfoz.slurm reads results/\${OUTNAME})"
echo "=================================================================="
