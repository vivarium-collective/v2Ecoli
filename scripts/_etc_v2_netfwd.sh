#!/bin/bash
cd /Users/eranagmon/code/v2ecoli--multiscale-showcase
PY=/Users/eranagmon/code/v2ecoli/.venv/bin/python
A=workspace/investigations/multiscale-complexity-showcase/analysis
for fm in 4 8 12; do
  $PY -u $A/etc_fix_v2.py --cache-dir out/cache --minutes 18 --burn-in 6 --forward-min $fm --reverse-max 0 \
      --out etc_v2_fwd${fm}_rev0.json > logs_mcs/etc_v2_fwd${fm}_rev0.log 2>&1 && echo "done fwd=$fm" || echo "FAIL fwd=$fm"
done
echo COMPLETE
