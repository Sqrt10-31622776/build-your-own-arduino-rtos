#!/bin/sh
# Adds a zero-initialised stage B to a finished baseline run (stage A frozen, its std / critic / curriculum
# position inherited) and trains only B, then evaluates it like run_queue.sh.
#   PY="uv run python" sh run_frozen_on_baseline.sh 8192 4600 1:<A checkpoint> 4:<A checkpoint> 7:<A checkpoint>
# Results: results/frozenA_s<seed>.txt / .train.log ; the run: logs/.../*_frozenA_s<seed>
cd "$(dirname "$0")"
PY=${PY:-uv run python}
ENVS=$1; IT=$2; shift 2
CMDS="0.06,0;0.12,0;0,0.3;0.06,-0.3"
mkdir -p results
for job in "$@"; do
  seed=${job%%:*}; A=${job#*:}
  name=frozenA_s$seed
  if [ -f results/$name.txt ]; then echo "[frozenA] $name done, skip"; continue; fi
  echo "[frozenA] $(date +%T) train $name, stage A = $A"
  $PY train.py --mode frozen --inherit-a --stage-a $A --seed $seed --iterations $IT --num-envs $ENVS --run-name $name > results/$name.train.log 2>&1
  D=$(ls -d logs/rsl_rl/arduino_quad_velocity/*_$name | tail -1)
  CK=$(ls $D/model_*.pt | sort -V | tail -1)
  { echo "checkpoint: $CK"
    for dr in "" "--dr"; do $PY sim2sim.py --checkpoint $CK --sim genesis --commands "$CMDS" $dr; done
  } > results/$name.txt.tmp 2>&1 && mv results/$name.txt.tmp results/$name.txt
  echo "[frozenA] $(date +%T) done $name"
done
