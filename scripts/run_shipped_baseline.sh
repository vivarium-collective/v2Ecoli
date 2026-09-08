#!/usr/bin/env bash
# model-vs-experimental-proteome baseline: the SHIPPED model, DEFAULT config.
#
# No --config-override. mechanistic_replisome stays at the composite's default
# (False), so this is what v2ecoli actually ships — not the subunit-gated regime
# the rest of this investigation studies. Schmidt 2016 measured balanced
# exponential growth, and this is the arm that provides it.
set -uo pipefail
cd "$(dirname "$0")/.."
STUDY=model-vs-experimental-proteome
GENS=12
LOG=out/shipped_baseline_sweep.log
: > "$LOG"
for seed in 0 1 2; do
  eid="${STUDY}__shipped-default__seed${seed}"; out="out/${eid}"
  if grep -q "^DONE" "${out}/run.log" 2>/dev/null; then
    echo "skip  ${eid} (already complete)" >> "$LOG"; continue
  fi
  mkdir -p "$out"
  .venv/bin/python3 scripts/simlock.py run --label "${eid}" -- \
      .venv/bin/python3 -u scripts/run_condition_multigen_parquet.py \
      --cache-dir out/cache --out-dir "$out" --experiment-id "$eid" \
      --generations "$GENS" --seed "$seed" \
      --study-dir "workspace/studies/${STUDY}" --spec-id "$STUDY" \
      > "${out}/run.log" 2>&1
  echo "$( [ $? -eq 0 ] && echo ok || echo FAIL )  ${eid}" >> "$LOG" &
done
wait
echo "sweep complete" >> "$LOG"
