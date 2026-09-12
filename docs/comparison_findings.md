# v2ecoli ↔ vEcoli comparison — findings & lessons

What we learned standing up the standardized v2ecoli↔vEcoli comparison harness on
AWS GovCloud (5 media conditions × 4 seeds × 2 generations; v2ecoli = process-bigraph
port on Ray/zarr, vEcoli = original on Nextflow/parquet; both from the same ParCa).

## Scientific result

**The port is faithful in basal, but diverges in alternative media.**
Matched-timepoint median |Δ| over generation 1 (v2ecoli vs vEcoli):

| condition | cell_mass | protein | rna | growth_rate |
|---|---:|---:|---:|---:|
| basal      | **3.2%** | 2.0% | 2.8% | 10% |
| with_aa    | 29%  | 24% | 22% | 79% |
| succinate  | 33%  | 11% | 38% | 28% |
| no_oxygen  | 3.9% | 3.4% | 31% | 123% |
| acetate    | 26%  | 11% | 44% | 55% |

Overall report-card verdict: **drift**.

Key interpretation — the non-basal divergence is **real**, not an artifact:
- It is **matched-timepoint** (trajectories aligned on simulation time), not end-of-run
  snapshots. (Snapshots gave spurious ±100% swings — see Methodology below.)
- It is on **correct per-condition initial states**. Notably **acetate's initial mass
  matches vEcoli to ~1% yet its trajectory drifts to 26%** → the divergence is
  *dynamic*, in v2ecoli's condition-specific behavior under alternative nutrients, not
  initialization. (with_aa additionally starts ~11% off: 2687 vs 3013 fg.)

So: **v2ecoli reproduces vEcoli in basal (~3%) and has genuine alternative-media
divergences** (the `condition_to_doubling_time`/media selection and how alternative-media
fluxes propagate through metabolism/elongation) — the concrete thing to investigate next.

## Methodology lessons

1. **Compare at MATCHED TIMEPOINTS, never end-of-run snapshots.** Two whole-cell models
   that divide at slightly different times have their last-emitted tick at different
   cell-cycle phases (one just-divided ≈ half-mass, the other near-division ≈ double),
   producing ±100% mass swings that *sign-flip by condition* from two near-identical
   models. This single mistake made an early report look catastrophic; matched-timepoint
   alignment dissolved it. Both engines store full timeseries, so this is a reader choice.

2. **Same ParCa is the control.** v2ecoli embeds vEcoli's ParCa; basal initial states
   match to <0.7%. Equal initial state is the precondition for a meaningful comparison —
   verify it explicitly (the report's ParCa/initial-state + config panels do this).

3. **Condition is applied at CACHE-BUILD, not composite-build.** v2ecoli selects the
   condition-specific initial state via `save_sim_input(condition=…)` →
   `LoadSimData(condition=…)` (reads `condition_to_doubling_time`/saved media — no refit).
   It is NOT a `build_composite("ecoli_baseline")` parameter (that raises "unknown parameter").
   The viva-api ParCa builds a basal cache, so non-basal v2ecoli runs must regenerate a
   condition-specific bundle from the raw `simData.cPickle`.

4. **Configs live in disjoint namespaces.** Where the two engines' configs share a key
   they agree (condition/seed/time_step); but ~58 vEcoli knobs (max_duration, generations,
   parca_options, mar_regulon, process_configs, swap_processes, …) have NO v2ecoli config
   counterpart — they're set inside v2ecoli's `baseline()`/ParCa, not its config. A
   straight `translate_vecoli_config` maps almost nothing. True equivalence-by-construction
   needs a **name-mapping layer** (vEcoli key → v2ecoli baseline param OR ParCa/cache knob).

## Engineering lessons (the bugs, in order)

Running it on real infra surfaced a chain of issues — each invisible until executed:
- **vEcoli-as-native-Ray-composite is dead** — its process API migration to `interface()`
  is ~1/3 done (36/103); `build_composite_native` yields an incomplete composite that
  `KeyError`s at the first tick. Use vEcoli's **native Nextflow** path instead.
- **Nextflow submit bugs (viva-api 0.9.10/0.9.11):** a leaked basal `sim_data_path` default
  made `generate_code` hash a nonexistent file (set it `None` to signal run-parca; popping
  it lets the config.template default win); plus config-name-keyed stale K8s
  job/configmap collisions on re-submit.
- **v2ecoli compact emitter** dropped `active_RNAP`/`active_ribosome` (scalar counts share
  a leaf name with coordinate vectors) → `include_vectors=True`.
- **Condition not threaded** → every non-basal v2ecoli run silently ran basal (the worst
  bug: it produced plausible-but-wrong ±100% numbers).
- **Per-condition cache regen raced** — `generate_initial_state` writes a default emitter
  to a fixed relative path; concurrent seeds collide (`FileExistsError`). Isolate in a
  per-(condition,seed) cwd.
- **Swallowed errors hide everything** — the multigen runner caught the first-chunk
  exception and emitted an empty-but-valid zarr with a success exit; the real error was on
  Ray-worker stdout, not the driver log. Fail loud on first-chunk failure.

Meta-lesson: **a faithful-looking number can be a harness bug.** Every alarming result here
(±100% snapshots, the condition-not-applied basal-everywhere) was an artifact caught only
by checking initial states, matched timepoints, and per-condition configs — which is
exactly why the standardized report surfaces all three.

## How to reproduce

```
aws sso login --profile stanford-sso        # SSO expires periodically; re-auth as needed
nohup bash ~/code/sms-cdk/scripts/ptools-proxy.sh -s smsvpctest &   # tunnel → localhost:8080
bash scripts/comparison_harness.sh all       # register → launch 5×2×2 engines → report
# or, against existing S3 runs:
eval "$(aws configure export-credentials --format env)"
.venv/bin/python scripts/comparison_report_card.py --only all
```
Artifacts: `out/full_compare/standardized_comparison_report.html`, `verdict.json`
(`report_card_verdict/v1`), `report_card.html`. Experiment ids are pinned in
`out/full_compare/experiments.json` (`{cond: [v2_dir, ve_dir]}`).

## Open follow-ups
- Investigate the **alternative-media dynamic divergence** (acetate/succinate/with_aa) —
  localize which process (metabolism FBA fluxes, elongation supply) drives it.
- Wire **RNAP/ribosome** matched-timepoint grading (currently `ungraded`).
- Look at the **with_aa ~11% initial-state gap** (condition/media selection).
- Build the **vEcoli→v2ecoli config name-mapping layer** (`--translate-vecoli-config` seam
  is in place) for equivalence-by-construction.
- Scale to **16 seeds × 16 generations** (same harness, larger inputs) once the above land.
