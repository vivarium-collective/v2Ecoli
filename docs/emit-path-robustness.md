# Emit-path robustness: root cause and design (CD2 "no emitted output")

Branch: `fix/robust-emit-path-from-generator`. Written for review before any PR.

## 0. Summary

The failure the CD2 Runs 1–3 hit is viva-api's `run_pbg.py` output gate
(`PBG_REQUIRE_OUTPUT is set but the run produced no emitted output ...`, sms-api
#475), not `LineageStep._has_output` (that gate only exists on the Nextflow
path, which Runs 1–3 do not use). The gate is correct. Behind it there are **two
distinct mechanisms**, both reproduced locally, and **neither is an empty
`emit_paths` allow-list**:

| dispatch shape | what the gate saw | mechanism (confirmed) |
|---|---|---|
| pbg-native `lineage_ray_batch` (Run 1 K4 cell-only canary, Run 2 dispatch 438, 13 cell-only dispatches) | `final_state.json` with `global_time: 1.0`, every lineage `summary: {}`, `complete: false`, zero bytes from every worker | viva-api sends `-n 1`; each `ray:LineageProcess` node has `interval = max_duration_per_gen` (3600); process-bigraph only invokes a process whose next event lies within the run window (`future <= end_time`, `composite.py:2887`), so **nothing was ever invoked**. |
| chain dispatch (Run 3 `antibiotic_cocktail_native_run.json`, Jim's #375 "real division, zero history" finding) | only a 500-byte `default/history/1.pq` (1 row, 1 column) | `BatchBaselineRunner` is a `V2Step`, and `V2Step.invoke` **swallows every exception** raised by `update()` (`v2ecoli/steps/base.py`). Any failure inside the batch (a stale cache, an injection-seam error, an S3 write failure) leaves `Composite.run(1)` returning normally with an empty `batch` store; the only artifact is the outer document's global_time-only emitter row. |

**Empty `emit_paths` is not emit-nothing.** On the lineage path `emit_paths` are
*extra* store paths merged on top of the baseline set; an empty list leaves the
baseline set (`global_time` / `bulk` / `listeners`) intact
(`lineage.py:412-414` only sets the key when non-empty; `_helpers._get_special_step`
parquet branch always emits the three roots). The single-cell `filter_listener_paths`
allow-list likewise treats empty as "keep everything" (`tests/test_ecoli_baseline_emit_paths.py`).

## 1. Evidence

### 1a. pbg-native under-run (`-n 1`)

* viva-api `simulation_service_ray.py::_submit_multi_node_composite`:
  `steps = int(mnp_dispatch.get("steps") or 1)` → `_multi_node_composite_command`
  emits `python /tmp/run_pbg.py --composite-id lineage_ray_batch ... -n {steps}`.
  Still on `main` at the time of writing.
* cplong90 read the `RAY_JOB_CMD` of all thirteen cell-only dispatches:
  thirteen of thirteen carry `-n 1` (sms-ecoli #166, §2 of the 2026-09-07 comment).
  Dispatch 438's `final_state.json`: `global_time = 1.0`,
  `lineage_0000.interval = 3600.0`, `complete: false`, empty summary.
* Local reproduction (no biology, `scratchpad/repro_underrun.py`): a process node
  with `interval = 3600` under `Composite.run(n)`:

  ```
  run(1):     invoked=0  global_time=1.0     complete=False
  run(3599):  invoked=0  global_time=3599.0  complete=False
  run(3600):  invoked=1  global_time=3600.0  complete=True
  run(14400): invoked=4  global_time=14400.0 complete=True
  ```

  `process_bigraph/composite.py:2871-2887`: `future = process_time + interval`;
  `if future <= end_time: invoke`. Not a bug in process-bigraph; a contract the
  document builder documents in words but the caller did not honour.

### 1b. chain-dispatch swallow (`V2Step.invoke`)

Local reproduction of the exact chain-dispatch document viva-api builds
(`_seed_generation_command` → `run_pbg.py --composite-id
v2ecoli.composites.ecoli_baseline.ecoli_baseline --overrides {n_seeds:1,
n_generations:1, stop_at_division:true, ...} -n 1`), run the way `run_pbg` runs it
(`scratchpad/repro_chain_probe2.py`, with a spy that prints what the batch raised):

```
[probe2] dispatch_batch RAISED after 0s:
  ... LineageProcess._build_generation -> baseline -> load_cache_bundle
v2ecoli.library.cache_version.StaleCacheError: Cache at '.../out/cache' is stale ...
[probe2] composite.run(1) returned in 0s        <-- returned NORMALLY
[probe2] batch: {}
        500  default/history/1.pq                 <-- the only artifact
```

`scratchpad/repro_chain_probe3.py` isolates the layer: `runner.update()` raises,
`runner.invoke()` returns, `composite.run(1)` returns. process-bigraph itself
propagates Step exceptions (verified with a plain `Step`, single- and multi-step
layers, `parallel_steps` on and off); the swallow is `V2Step.invoke`'s
`except Exception: update = {}` from the initial commit ("so missing data
doesn't crash the Composite's step cascade").

The `500-byte default/history/1.pq` is the batch document's own top-level emitter
(`_build_batch_document` installs a `paths: ["global_time"]` ParquetEmitter so the
declared default is not installed with an empty config, #496), which under
`run_pbg`'s parquet override writes a flat, one-column file. It is the artifact
viva-api #475 special-cases and v2ecoli #661 had to exclude from the hive glob.

What raised inside Run 3's batch on GovCloud is not determinable from here; the
class is. The CloudWatch log of a failing Run 3 dispatch shows nothing because
the exception never reached a logger. With this branch it will: the traceback
propagates out of `Composite.run()` and `run_pbg` exits non-zero before the gate.

### 1c. What is NOT the cause

* Empty `emit_paths` → see §0. Verified in code and, on real biology, by
  validation A below (a `LineageStep` with `emit_paths=[]` writes full history).
* The lineage's parquet override reaching the inner composite: `_build_generation`
  sets it around `baseline()`; the `emitter` special step honours it before the
  generator-declared default. Validation A/B confirm rows land.

## 2. What this branch changes (v2ecoli only)

All four changes sit on the path every lineage shape passes through.

1. **`v2ecoli/steps/base.py` — `V2Step.raise_update_errors`** (default `False`,
   unchanged behaviour for per-tick listeners) and a **once-per-class
   `RuntimeWarning`** when an update is swallowed, so a silently no-op step is
   at least visible in the run log.
2. **`v2ecoli/steps/batch_baseline_runner.py`** — `BatchBaselineRunner.raise_update_errors = True`:
   the orchestrator's one `update()` *is* the run; its failure now propagates
   out of `Composite.run()` with the real traceback. `dispatch_batch` also refuses
   to record a batch in which **no** seed reported back (`completed: True` with
   every seed `error: run produced no result`); a *partial* batch keeps the
   existing visible-but-not-fatal per-seed `error` entries.
3. **`v2ecoli/workflow/lineage.py` — `LineageProcess.require_output`** (default
   `True`): at the end of every generation, after the emitter flush and before the
   summary/checkpoint, `_assert_generation_emitted` refuses a generation whose
   inner composite was built without the lineage's parquet sink, whose sink
   received zero rows, or whose history partition holds no non-empty `*.pq`
   (listed through the emitter's own fsspec filesystem, so `s3://` out_dirs are
   verified, not assumed; a listing *error* warns "UNVERIFIED" rather than
   failing — "could not look" is not "no output"). xarray-only lineages require
   at least one populated emit (the Run 4 metadata-only-zarr case).
   `emitter="null"` lineages are exempt (they emit nothing by design).
   The emitter is captured at build time because `Division` may pop it from the
   registry before the generation ends (#687).
4. **`v2ecoli/composites/_helpers.py` — the parquet emit ROOT SET comes from the
   generator's declaration.** `_parquet_emit_set` derives the roots from
   `_DEFAULT_EMITTER_DECL["paths"]` (what `baseline()` sets from its own
   `@composite_generator(emitters=[...])` entry around every build, overrides
   included), attaches v2ecoli's typed schema per root (`bulk: array[integer]`,
   `listeners: <listener schema>`), then merges the config-declared `emit_paths`
   extras. Both former literal sites (`_get_special_step` parquet-override
   branch, `_build_declared_emitter`) now call it. A declaration with an emitter
   but **no paths is refused at build time** (it would capture only
   `global_time` — the 1-column parquet #475 special-cases). `_merge_emit_paths`
   no longer downgrades an already-typed root to `node`. Explicit `emit_paths`
   still win on top; with no declaration in scope the literal baseline set is
   the fallback, so bare callers are unchanged.
5. **`v2ecoli/workflow/batch_lineage_ray.py`** — the `lineage_ray_batch` document
   now carries a top-level `required_run_interval` (= `n_generations *
   max_duration_per_gen`; `Composite` ignores unknown top-level keys, verified)
   plus a `required_run_interval()` helper, so a runner can honour the contract
   without knowing the lineage internals. This does not by itself fix the
   under-run — see §4.

Tests: `tests/test_emit_path_robustness.py` (25 tests). The ones that fail
without the fix: `test_batch_runner_reraises_a_failed_dispatch`,
`test_chain_dispatch_shaped_composite_fails_loud_when_the_batch_fails` (the exact
viva-api chain document under `Composite.run(1)`),
`test_generation_whose_sink_received_no_rows_fails`,
`test_generation_built_without_a_parquet_sink_fails`,
`test_generation_whose_rows_never_landed_fails`, `test_dispatch_batch_refuses_a_batch_in_which_no_seed_reported`,
`test_a_declared_emitter_with_no_paths_is_refused`,
`test_an_extra_path_naming_a_declared_root_does_not_downgrade_its_schema`.
Stub-based lineage tests opt out with `require_output: False`;
`test_batch_injected_threading.py`'s stub workflow now reports a branch per seed.

## 3. Validation on real biology

Against a cache rebuilt with `scripts/build_cache.py` on this tree
(`schema_version 3`; the shared `sms-ecoli/out/cache` is schema 2 and is refused
by `verify_cache_version` — which is, incidentally, exactly the exception the
chain path used to swallow), one short generation each (`max_duration_per_gen=12`),
**no `emit_paths` anywhere**:

* **A — `LineageStep` (the Nextflow task shape), `emitter="parquet"`:**
  `history/.../13.pq` = 13 rows × 206 columns, roots `bulk`/`global_time`/`listeners`
  (`listeners__mass__*`, `bulk` counts, ...), plus `configuration/config.pq` and the
  `success/s.pq` sentinel. The new end-of-generation guard passed on the real
  emitter (no false positive); `LineageStep._has_output` passed.
* **B — the chain-dispatch document under `Composite.run(1)`, exactly as
  `run_pbg` runs it:** same 13 × 206 history + sentinel under
  `validateB/`, `batch.complete=True`, one seed `generations_reached: 1`, the
  post-sim flush ran (analysis.json, ptools, viz), and — as described in §1b — the
  outer document's own `default/history/1.pq` (1 row × 1 column,
  `global_time` only) alongside.

So on both shapes an empty `emit_paths` emits the full declared set; the empty
emits seen on GovCloud came from the two mechanisms in §0, not from the emit
set.

Existing suites run on this branch: `test_declared_emit_paths`,
`test_ecoli_baseline_emit_paths`, `test_lineage_step`, `test_workflow_lineage`,
`test_lineage_time_offset`, `test_lineage_emitter_finalize`, `test_batch_baseline`,
`test_batch_injected_threading`, `test_perform_update_gate`,
`test_workflow_batch_lineage_ray`, `test_workflow_xarray`,
`test_lineage_checkpoint_observability`, `test_lineage_bookkeeper`,
`test_lineage_injected_threading`, `test_population_doubling_runner`,
`test_inject_swap`, `test_inject_native_seam`, `test_store_groups`,
`test_baseline_single_cell_division_warning`, `test_agent_id_generation_threading`,
`test_output_metadata_parquet`, `test_workflow_nf` — all green except
`test_workflow_nf`'s two `-stub-run` compile tests, which fail inside the local
Nextflow 24.10.4 stub of `parca_v0` ("Cannot invoke method clone() on null
object") on code this branch does not touch.

## 4. Layer 2: routing, and the cross-repo boundary

### 4a. The `-n 1` under-run is a viva-api fix

`_submit_multi_node_composite` must not default `steps` to `1` for a lineage
document. Options, in order of preference:

1. Read the document's `required_run_interval` in `run_pbg.run()` and run
   `max(steps, required_run_interval)` (or refuse a shorter request with a
   message naming both numbers). Generic: any composite can declare it.
2. In `_submit_multi_node_composite`, when `composite_id` is a lineage batch and
   `steps` is unset, compute `n_generations * max_duration_per_gen` from `params`.

A v2ecoli-side alternative was considered and **not** built: giving
`LineageProcess` a `run_to_completion` mode (whole lineage per invocation, node
`interval = time_step`) would make `-n 1` run everything, but it changes the
deployed pbg-native contract that dispatches 313/517 relied on, and the
`ray:` proxy does not forward `calculate_timestep`, so every extra tick of a long
`-n` becomes a remote no-op call. That is a design decision for the humans.

### 4b. Study / auto-flush routing

None of the three CD2 shapes goes through the workbench's composite-auto-results
/ study wrapper (vwb #1022/#1025, sms-api #464): they enter through viva-api's
`run_pbg.py --composite-id ...` (pbg-native and chain dispatch) or the Nextflow
renderer. Analyses on the chain path run as a separate DAG node
(`_analysis_command`), on pbg-native via `run_analyses` over the S3 sweep, and on
Nextflow via `AnalysisTaskStep` — three ad-hoc flushes, none of which can
catch an empty emit uniformly. The uniform catch now lives one layer down
(§2.3, inside `LineageProcess`), which every one of those routes passes through,
so routing them through the study wrapper is no longer required to *catch* empty
emits; it remains the right direction to make analysis flush automatic and
declared, and is scoped as a viva-api/workbench change (out of this branch).

### 4c. Also observed, not changed here

* `LineageStep._has_output` walks `out_dir` with `os.walk`; an `s3://` out_dir
  always reads as empty. Task-local out_dirs are what workflow_nf sets today,
  so this is latent, not live.
* `V2Step`'s blanket swallow also covers `Division`, `MarkDPeriod`,
  `exchange_data`, `media_update`, `environment_driver`,
  `population_aggregator`, the reactor bridges and `lineage_bookkeeper`. The
  once-per-class warning added here will show whether any of them is silently
  failing on real runs; flipping the default to raise is a follow-up that needs
  a full baseline run to validate.
* xarray emission on the pbg-native path is one sample per *generation*
  (`_run_until_division` emits once per outer tick, and the outer tick is
  `max_duration_per_gen`), versus one per second on the meta-composite path
  (`interval = time_step`). Worth knowing when reading a `lineage_ray_batch` zarr.
* The batch document's top-level global_time-only emitter (§1b) exists only to
  satisfy `CompositeSpec._with_emitters`; under `run_pbg`'s override it produces
  the flat one-column file every reader has to skip.

## 5. Open questions for review

1. Flip `V2Step.raise_update_errors` to `True` globally after validating on a
   full baseline run? (This branch: opt-out only for the batch orchestrator.)
2. Which of §4a's two viva-api options; and should `run_pbg` also refuse
   `steps < required_run_interval` outright rather than clamp?
3. Should `lineage_ray_batch` *declare* an emitter (`emitters=[...]`) so
   `_redirect_emitters` has something to redirect, or is the internal per-generation
   override (what works today) the intended contract? Declaring one over the
   `lineages/*` summary map would install a top-level step over paths that
   resolve, but it would not carry the science.
4. The chain path's stray `default/history/1.pq`: keep (and keep excluding it),
   or make `_build_batch_document`'s placeholder emitter a `RAMEmitter`?
