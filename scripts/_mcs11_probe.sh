#!/bin/bash
cd /Users/eranagmon/code/v2ecoli--multiscale-showcase
PY=/Users/eranagmon/code/v2ecoli/.venv/bin/python
A=workspace/investigations/multiscale-complexity-showcase/analysis
for pair in "glucose:out/cache" "succinate:out/cache-succinate" "withaa:out/cache-with_aa" "acetate:out/cache-acetate"; do
  cond=${pair%%:*}; cache=${pair##*:}
  echo "[$(date +%H:%M:%S)] probe $cond"
  $PY -u $A/metabolism_probe.py --cache-dir "$cache" --minutes 30 --burn-in 8 \
      --label "$cond" --out "$A/mcs11_probe_${cond}.json" > "logs_mcs/mcs11_probe_${cond}.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE $cond" || echo "[$(date +%H:%M:%S)] FAIL $cond"
done
echo "MCS11 PROBES COMPLETE"
