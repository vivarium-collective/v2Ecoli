#!/bin/bash
cd /Users/eranagmon/code/v2ecoli--multiscale-showcase
PY=/Users/eranagmon/code/v2ecoli/.venv/bin/python
A=workspace/investigations/multiscale-complexity-showcase/analysis
for h in 4 3 6; do
  $PY -u $A/etc_stoich_fix.py --cache-dir out/cache --minutes 16 --burn-in 6 --hplusper $h \
      --out etc_stoich_h${h}.json > logs_mcs/etc_stoich_h${h}.log 2>&1 && echo "done h=$h" || echo "FAIL h=$h"
done
echo COMPLETE
