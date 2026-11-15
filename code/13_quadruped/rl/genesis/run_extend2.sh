#!/bin/sh
# Continues an already-extended run once more. Job = "mode:seed:from_run:new_run"
# (from_run / new_run are run names, e.g. baseline_s2_ext -> baseline_s2_ext2). Evaluates like run_extend.sh.
#   PY="uv run python" sh run_extend2.sh 8192 4600 baseline:2:baseline_s2_ext:baseline_s2_ext2
cd "$(dirname "$0")"
PY=${PY:-uv run python}
ENVS=$1; IT=$2; shift 2
CMDS="0.06,0;0.12,0;0,0.3;0.06,-0.3"
mkdir -p results
for job in "$@"; do
  IFS=: read mode seed from new <<EOJ
$job
EOJ
  if [ -f results/$new.txt ]; then echo "[extend2] $new done, skip"; continue; fi
  D0=$(ls -d logs/rsl_rl/arduino_quad_velocity/*_$from | tail -1)
  CK0=$(ls $D0/model_*.pt | sort -V | tail -1)
  echo "[extend2] $(date +%T) train $new from $CK0"
  $PY train.py --mode $mode --seed $seed --iterations $IT --num-envs $ENVS --run-name $new --resume $CK0 > results/$new.train.log 2>&1
  D=$(ls -d logs/rsl_rl/arduino_quad_velocity/*_$new | tail -1)
  CK=$(ls $D/model_*.pt | sort -V | tail -1)
  { echo "checkpoint: $CK"
    for dr in "" "--dr"; do $PY sim2sim.py --checkpoint $CK --sim genesis --commands "$CMDS" $dr; done
  } > results/$new.txt.tmp 2>&1 && mv results/$new.txt.tmp results/$new.txt
  echo "[extend2] $(date +%T) done $new"
done
