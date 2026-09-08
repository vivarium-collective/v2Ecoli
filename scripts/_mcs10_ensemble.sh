#!/bin/bash
# mcs-10 multi-seed ensemble: basal + with_aa, seeds, 3 gens each. Continue-on-error.
set -u
PY=/Users/eranagmon/code/v2ecoli/.venv/bin/python
R=scripts/run_condition_multigen_parquet.py
run() { # cond cache seed
  local cond=$1 cache=$2 seed=$3
  local exp="mcs10_${cond}_seed${seed}"
  echo "[$(date +%H:%M:%S)] START $exp"
  $PY -u "$R" --cache-dir "$cache" --out-dir "out/$exp" --experiment-id "$exp" \
      --generations 3 --max-min 75 --seed "$seed" \
      > "logs_mcs/$exp.log" 2>&1 && echo "[$(date +%H:%M:%S)] DONE $exp" || echo "[$(date +%H:%M:%S)] FAIL $exp (continuing)"
}
# basal (seed20 already ran as calibration -> add 5 more)
for s in 21 22 23 24 25; do run basal out/cache $s; done
# with_aa
for s in 20 21 22 23 24 25; do run with_aa out/cache-with_aa $s; done
echo "[$(date +%H:%M:%S)] ENSEMBLE COMPLETE"
