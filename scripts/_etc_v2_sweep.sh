#!/bin/bash
cd /Users/eranagmon/code/v2ecoli--multiscale-showcase
PY=/Users/eranagmon/code/v2ecoli/.venv/bin/python
A=workspace/investigations/multiscale-complexity-showcase/analysis
for fm in 6 10 14 18 22; do
  echo "[$(date +%H:%M:%S)] forward-min=$fm"
  $PY -u $A/etc_fix_v2.py --cache-dir out/cache --minutes 18 --burn-in 6 --forward-min $fm \
      --out etc_v2_fmin${fm}.json > logs_mcs/etc_v2_fmin${fm}.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] done fmin=$fm" || echo "[$(date +%H:%M:%S)] FAIL fmin=$fm"
done
echo "SWEEP COMPLETE"
