#!/bin/bash
# Phase 3b follow-up sims. Waits for the mcs-10 ensemble to finish (avoid CPU
# contention), then runs mcs-06 paired noise-off/noise-on ensembles, and
# best-effort mcs-09 (ppGpp/RelA knockdown) + mcs-11 (acetate-onset sweep).
# Continue-on-error throughout so one failure never aborts the batch.
set -u
cd /Users/eranagmon/code/v2ecoli--multiscale-showcase
PY=/Users/eranagmon/code/v2ecoli/.venv/bin/python
R=scripts/run_condition_multigen_parquet.py
mkdir -p logs_mcs

echo "[$(date +%H:%M:%S)] phase3b: waiting for mcs-10 ensemble to complete..."
for i in $(seq 1 240); do
  grep -q "ENSEMBLE COMPLETE" logs_mcs/_ensemble_driver.log 2>/dev/null && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] phase3b: starting."

run() { # exp cache seed gens extra_env...
  local exp=$1 cache=$2 seed=$3 gens=$4; shift 4
  echo "[$(date +%H:%M:%S)] START $exp (env: $*)"
  env "$@" $PY -u "$R" --cache-dir "$cache" --out-dir "out/$exp" --experiment-id "$exp" \
      --generations "$gens" --max-min 75 --seed "$seed" \
      > "logs_mcs/$exp.log" 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE $exp" || echo "[$(date +%H:%M:%S)] FAIL $exp (continuing)"
}

# --- mcs-06: paired noise-OFF vs noise-ON ensembles (same seeds), 4 gens ---
for s in 30 31 32 33 34 35; do run "mcs06_noiseoff_seed${s}" out/cache "$s" 4; done
for s in 30 31 32 33 34 35; do run "mcs06_noiseon_seed${s}"  out/cache "$s" 4 V2E_D_PERIOD_CV=0.15; done

# --- mcs-09: ppGpp/RelA knockdown vs baseline (resolve relA RNA id from sim_data) ---
echo "[$(date +%H:%M:%S)] resolving relA transcription-unit id..."
RELA_TU=$($PY - <<'PYEOF' 2>/dev/null
import pickle, sys
try:
    import dill
    with open("out/cache/sim_data_cache.dill","rb") as f: sd=dill.load(f)
    # find the RNA(s) whose gene symbol is relA; return the cistron/RNA id used for synth prob
    rna=sd.process.transcription.rna_data
    ids=[str(x) for x in rna["id"]]
    # gene symbol mapping
    hit=[i for i in ids if "relA" in i or "RELA" in i or "EG10835" in i]  # relA b-number/frameID heuristics
    # fall back: search common names table
    try:
        names=sd.common_names if hasattr(sd,"common_names") else None
    except Exception:
        names=None
    print(hit[0] if hit else "")
except Exception as e:
    print("")
PYEOF
)
if [ -n "$RELA_TU" ]; then
  echo "[$(date +%H:%M:%S)] relA id = $RELA_TU"
  run "mcs09_baseline_basal" out/cache 20 3
  echo "[$(date +%H:%M:%S)] START mcs09_relaKD_basal (perturbation $RELA_TU=1e-6)"
  $PY -u "$R" --cache-dir out/cache --out-dir out/mcs09_relaKD_basal \
      --experiment-id mcs09_relaKD_basal --generations 3 --max-min 75 --seed 20 \
      --perturbation "${RELA_TU}=1e-6" > logs_mcs/mcs09_relaKD_basal.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] DONE mcs09_relaKD_basal" || echo "[$(date +%H:%M:%S)] FAIL mcs09_relaKD_basal"
else
  echo "[$(date +%H:%M:%S)] mcs-09 SKIP: could not resolve relA id (leave study as designed/planned)"
fi

# --- mcs-11: acetate-overflow-onset sweep with the ETC fix on, across carbon ladder ---
for pair in "acetate:out/cache-acetate" "succinate:out/cache-succinate" "glucose:out/cache" "withaa:out/cache-with_aa"; do
  cond=${pair%%:*}; cache=${pair##*:}
  run "mcs11_overflow_${cond}" "$cache" 20 2
done

echo "[$(date +%H:%M:%S)] PHASE3B COMPLETE"
