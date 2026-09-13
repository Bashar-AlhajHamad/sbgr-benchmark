#!/bin/bash
# One command that answers "what state is this campaign actually in?" -- written because after a
# long session of partial launches, moves and resubmissions, memory is not evidence. Read-only:
# it inspects and prints, it never creates, moves or deletes anything.
#
#   bash /arf/scratch/${USER}/sky130-bgr/code/truba/status.sh

R=/arf/scratch/${USER}/sky130-bgr
cd "$R" 2>/dev/null || { echo "FATAL: $R not found"; exit 1; }

echo "=================================================================="
echo " 1. CAMPAIGN ROWS AND MERGED ARTEFACTS"
echo "=================================================================="
for c in base hard highdim; do
    n=$(ls "results/$c/rows" 2>/dev/null | grep -vc curve)
    printf "  %-8s rows %3s/180" "$c" "${n:-0}"
    f="results/$c/per_run_records.csv"
    if [ -f "$f" ]; then
        printf "   MERGED: %s data rows, %s\n" \
            "$(( $(wc -l < "$f") - 1 ))" "$(stat -c %y "$f" 2>/dev/null | cut -d. -f1)"
        # the budget and pop actually recorded, so a wrong-protocol merge cannot hide
        awk -F, 'NR==1{for(i=1;i<=NF;i++){h[$i]=i}} NR==2{printf "           case=%s pop=%s budget=%s objective=%s\n", $h["case"], $h["pop"], $h["eval_budget"], $h["psrr_def"]}' "$f"
    else
        printf "   not merged\n"
    fi
done

echo
echo "=================================================================="
echo " 2. PVT -- every directory, with condition counts and timestamps"
echo "=================================================================="
found=0
for d in results/*/pvt results/*/pvt_* ; do
    [ -d "$d" ] || continue
    found=1
    n=$(ls "$d/conditions" 2>/dev/null | wc -l)
    newest=$(ls -t "$d/conditions" 2>/dev/null | head -1)
    when=$(stat -c %y "$d/conditions/$newest" 2>/dev/null | cut -d. -f1)
    printf "  %-46s %2s/15 conditions" "$d" "$n"
    [ -n "$newest" ] && printf "  newest: %s (%s)" "$newest" "$when"
    [ -f "$d/pvt_summary.csv" ] && printf "  [summary present]"
    echo
done
[ "$found" = 0 ] && echo "  no pvt directory anywhere under results/"
echo
echo "  -> the directory holding 15 condition files is the valid one. If it is NOT named"
echo "     exactly results/<case>/pvt, the merge cannot see it and it must be renamed back."

echo
echo "=================================================================="
echo " 3. QUEUE NOW"
echo "=================================================================="
squeue -u "$USER" 2>/dev/null
echo "  running cores: $(squeue -u "$USER" -h -t R -o '%C' 2>/dev/null | awk '{s+=$1} END {print s+0}')/512"

echo
echo "=================================================================="
echo " 4. JOB HISTORY, last 25"
echo "=================================================================="
sacct -u "$USER" --starttime=now-2days \
      --format=JobID%18,JobName%14,State%12,Elapsed,ExitCode -n 2>/dev/null \
    | grep -v '\.' | tail -25

echo
echo "=================================================================="
echo " 5. WRAPPER LOGS"
echo "=================================================================="
for l in ~/base.log ~/hard.log ~/highdim.log; do
    [ -f "$l" ] || continue
    echo "  --- $l  (last 3 lines) ---"
    tail -3 "$l" | sed 's/^/      /'
done
echo "  runcase.sh loops still alive: $(pgrep -fc runcase.sh 2>/dev/null || echo 0)"

echo
echo "=================================================================="
echo " 6. WHICH CODE VERSION IS INSTALLED"
echo "=================================================================="
printf "  runcase.sh            busy_chunks fix : %s\n" \
    "$(grep -qc busy_chunks code/runcase.sh 2>/dev/null && echo PRESENT || echo MISSING)"
printf "  corner_verify.py      fillna fix      : %s\n" \
    "$(grep -q 'sim_failure.fillna' code/corner_verify.py 2>/dev/null && echo PRESENT || echo MISSING)"
printf "  pvt_reference_control.py              : %s\n" \
    "$([ -f code/pvt_reference_control.py ] && echo PRESENT || echo MISSING)"
printf "  verify_grid_and_confounds.py          : %s\n" \
    "$([ -f code/verify_grid_and_confounds.py ] && echo PRESENT || echo MISSING)"
echo
echo "  read-only files that must never change:"
for f in algorithms.py problems.py run.py; do
    printf "    %-16s %s\n" "$f" "$(md5sum "code/$f" 2>/dev/null | cut -d' ' -f1)"
done
echo "    expected: algorithms 9d7eb27c3ec266c822c5ea174af57259"
echo "              problems   d8bc236111d05afe797b8aaad125741e"
echo "              run        35301bddf1605f3bff950bc6d786ece8"
