#!/usr/bin/env bash
# tu-expression-identifiability axes 5-6: vertex vs minimum-norm tie-break.
# Arms differ ONLY in the ParCa cache; both built from the same code with
# V2ECOLI_MIN_NORM_TU_FIT as the only difference. Serialised by simlock.
set -uo pipefail
cd "$(dirname "$0")/.."
# Re-runnable: skips arms that already completed, so a partial sweep resumes
# rather than redoing finished work.
STUDY=promoter-specific-regulation
GENS=12
LOG=out/both_sweep.log
: > "$LOG"
run_one () {
  local arm="$1" cache="$2" seed="$3"
  local eid="${STUDY}__${arm}__seed${seed}" out
  out="out/${eid}"
  if grep -q "^DONE" "${out}/run.log" 2>/dev/null; then
    echo "skip  ${eid} (already complete)" >> "$LOG"; return 0
  fi
  mkdir -p "$out"
  # -u: unbuffered, so a killed run still leaves its progress in the log.
  # Without it Python buffers stdout to a file and a SIGTERM loses everything,
  # which is exactly what made the first attempt undiagnosable.
  .venv/bin/python3 scripts/simlock.py run --label "${eid}" -- \
      .venv/bin/python3 -u scripts/run_condition_multigen_parquet.py \
      --cache-dir "$cache" --out-dir "$out" --experiment-id "$eid" \
      --generations "$GENS" --seed "$seed" \
      --config-override 'ecoli-chromosome-replication.mechanistic_replisome=true' \
      --study-dir "workspace/studies/${STUDY}" --spec-id "$STUDY" \
      > "${out}/run.log" 2>&1
  echo "$( [ $? -eq 0 ] && echo ok || echo FAIL )  ${eid}" >> "$LOG"
}
for seed in 0 1 2; do
  for pair in "promspec-plus-minnorm:out/cache_both"; do
    run_one "${pair%%:*}" "${pair##*:}" "$seed" &
  done
done
wait
echo "sweep complete" >> "$LOG"
