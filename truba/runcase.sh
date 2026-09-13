#!/bin/bash
C=$1
CHUNKS=${CHUNKS:-12}
WORKERS=${WORKERS:-15}
EVALS=${EVALS:-20000}
WALL=${WALL:-48:00:00}
cd /arf/scratch/${USER}/sky130-bgr/code
source ../env.sh
# TAG separates a second campaign on the SAME case from the first. Row files are named
# {case}_{algo}_run{NNN}.csv with no budget in the name, and --job-index skips a job whose row
# file exists, so base at 150,000 written into results/base would skip all 180 jobs and exit 0
# having simulated nothing. TAG=_150k sends it to results/base_150k instead, while --case stays
# "base" so the frozen thresholds are unchanged.
TAG=${TAG:-}
OUTNAME="${C}${TAG}"
R=../results/$OUTNAME/rows
n() { ls $R 2>/dev/null | grep -v curve | wc -l; }

# A chunk is BUSY if it is running OR pending-and-not-held. Counting only the running ones was
# a real defect: five minutes after submission nothing has started yet, every chunk is PENDING,
# so `busy` was empty, the "nothing is running" branch fired, and all twelve chunks were
# resubmitted on top of themselves -- 24 array tasks, 960 cores against a 512-core cap, and two
# processes writing the same row file. Held tasks are deliberately NOT busy: they never start.
#
# squeue collapses pending array tasks into ranges ("3-11", sometimes "3-11%2"), so %K cannot be
# read as a plain integer. expand() handles range, comma-list and single forms.
busy_chunks() {
  { squeue -u "$USER" -n "bgr_${OUTNAME}" -h -t R  -o "%K"
    squeue -u "$USER" -n "bgr_${OUTNAME}" -h -t PD -o "%K|%R" | grep -vi held | cut -d'|' -f1
  } 2>/dev/null | tr '\n' ' '
}

todo_chunks() {
  $SKY130_PY - "$R" "$CHUNKS" "$(busy_chunks)" <<'PY'
import sys, os, re
R, CH = sys.argv[1], int(sys.argv[2])

def expand(s):
    out = set()
    for part in s.replace(",", " ").split():
        part = part.split("%")[0]
        if "-" in part:
            a, _, b = part.partition("-")
            if a.isdigit() and b.isdigit():
                out.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out

busy = expand(sys.argv[3] if len(sys.argv) > 3 else "")
A = ["ABC", "GWO", "FA", "PSO", "GA", "ACO"]
PER = (180 + CH - 1) // CH
have = set()
if os.path.isdir(R):
    for f in os.listdir(R):
        if "curve" in f:
            continue
        m = re.match(r".+_(%s)_run(\d{3})\.csv$" % "|".join(A), f)
        if m:                      # run_spice.py:510  run_idx, algo = divmod(index, 6)
            have.add(int(m.group(2)) * 6 + A.index(m.group(1)))
need = sorted({j // PER for j in range(180) if j not in have} - busy)
print(",".join(map(str, need)))
PY
}

submit() {
  # CHUNKS must ride along on EVERY submission. A resubmission of a single index has
  # SLURM_ARRAY_TASK_COUNT=1, so without it the slurm script computes PER=180, maps the chunk to
  # jobs far past 180, finds an empty range and exits 0 having done nothing at all.
  CASE=$C OUTNAME=$OUTNAME EVALS=$EVALS WORKERS=$WORKERS CHUNKS=$CHUNKS sbatch \
    -J "bgr_${OUTNAME}" -p barbun --array="$1" \
    --cpus-per-task=40 --mem=32G --exclusive --time=$WALL \
    truba/02_campaign_orfoz.slurm >/dev/null
}

# base@20,000 measured 7.2 h median and ~10 h for the slowest single run. A run writes its row
# only when it finishes, so a task killed at the wall loses everything it computed.
H=${WALL%%:*}
if [ "$EVALS" -gt 20000 ] && [ "$H" -le 48 ]; then
  echo "WARNING: EVALS=$EVALS with WALL=$WALL. base@20000 took ~10 h for its slowest run, so"
  echo "         $EVALS needs roughly $(( EVALS / 2000 )) h. A task killed at the wall loses its"
  echo "         entire run. Re-run with WALL=72:00:00 if that is too tight."
fi

echo "$(date +%H:%M)  $C  START  $(n)/180   evals=$EVALS chunks=$CHUNKS workers=$WORKERS wall=$WALL"
submit "0-$((CHUNKS-1))"

while [ $(n) -lt 180 ]; do
  sleep 300
  HELD=$(squeue -u "$USER" -n "bgr_${OUTNAME}" -h -t PD -o "%K %R" 2>/dev/null | grep -ci held)
  RUN=$(squeue -u "$USER" -n "bgr_${OUTNAME}" -h -t R 2>/dev/null | wc -l)
  PEND=$(squeue -u "$USER" -n "bgr_${OUTNAME}" -h -t PD 2>/dev/null | wc -l)
  echo "$(date +%H:%M)  $C  $(n)/180   running=$RUN pending=$PEND held=$HELD"
  # Act on held tasks IMMEDIATELY -- a chunk held at minute one must not wait out the eleven
  # hours the others take. But only repair when something is actually wrong: a queue that has
  # not started yet is not a fault, and treating it as one is what caused the double submission.
  if [ "$HELD" -gt 0 ] || { [ "$RUN" -eq 0 ] && [ "$PEND" -eq 0 ]; }; then
    for j in $(squeue -u "$USER" -n "bgr_${OUTNAME}" -h -t PD -o "%i %R" 2>/dev/null \
               | grep -i held | awk '{print $1}'); do
      scancel "$j" 2>/dev/null
    done
    sleep 10
    T=$(todo_chunks)
    [ -n "$T" ] && { submit "$T"; echo "  resubmit chunks: $T"; }
  fi
done
echo "$(date +%H:%M)  $C COMPLETE $(n)/180"
