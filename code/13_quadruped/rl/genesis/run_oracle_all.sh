#!/bin/sh
# oracle_all: the plain MLP on the whole 4-frame history (s_{t-3..t}, a_{t-4..t-1}, every joint), no servo
# delay, trained from scratch in one run. An upper bound for what stage B could add. Evaluated at 4600 and at the
# last checkpoint, like run_queue.sh.
#   PY="uv run python" sh run_oracle_all.sh 8192 9200 1 2 4
# Results: results/oracle_all_s<seed>.txt / _4600.txt / .train.log ; the run: logs/.../*_oracle_all_s<seed>
cd "$(dirname "$0")"
PY=${PY:-uv run python}
ENVS=$1; IT=$2; shift 2
CMDS="0.06,0;0.12,0;0,0.3;0.06,-0.3"
mkdir -p results
evaluate() {   # checkpoint, output file
  { echo "checkpoint: $1"
    for dr in "" "--dr"; do $PY sim2sim.py --checkpoint $1 --sim genesis --commands "$CMDS" $dr; done
  } > $2.tmp 2>&1 && mv $2.tmp $2
}
for seed in "$@"; do
  name=oracle_all_s$seed
  if [ -f results/$name.txt ]; then echo "[oracle_all] $name done, skip"; continue; fi
  echo "[oracle_all] $(date +%T) train $name"
  $PY train.py --mode oracle_all --seed $seed --iterations $IT --num-envs $ENVS --run-name $name > results/$name.train.log 2>&1
  D=$(ls -d logs/rsl_rl/arduino_quad_velocity/*_$name | tail -1)
  [ -f $D/model_4600.pt ] && evaluate $D/model_4600.pt results/${name}_4600.txt
  evaluate $(ls $D/model_*.pt | sort -V | tail -1) results/$name.txt
  echo "[oracle_all] $(date +%T) done $name"
done
