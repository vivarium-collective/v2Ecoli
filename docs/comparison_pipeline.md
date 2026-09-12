# v2ecoli ↔ vEcoli comparison pipeline

The canonical, reproducible way to compare a v2ecoli build against genuine
`CovertLab/vEcoli`. **Both engines run as process-bigraph composites on the same
runtime**, emitting the same v2ecoli-format zarr, so one report card reads both.

## The ONE vEcoli loader (no ambiguity)

The genuine-vEcoli side is **always** loaded as a SINGLE process-bigraph node
with vivarium-core's own Engine inside — `vecoli_source = "vivarium-process"`
(`v2ecoli.library.vivarium_ecoli_engine`). It is faithful by construction
(vivarium handles partition / update-reconciliation / division natively) with
**zero edits** to the upstream checkout.

This is the only supported loader. The old colony-wrapper (`upstream`) and
`composite-softfloor` paths — and their standalone runners — were **removed**;
`--vecoli-source` accepts only `vivarium-process` (and defaults to it). Don't
reintroduce alternate vEcoli loaders: it was the colony-wrapper path that had the
mass-explosion / division-handoff bugs the vivarium-process pivot eliminated.

## Reproducible local run (canonical)

```bash
# 1. Build BOTH ParCa caches (idempotent). v2ecoli runs on its own ParCa
#    (out/cache_full); genuine vEcoli runs on the UPSTREAM ParCa
#    (out/compare_harness/vecoli_parca). They are SEPARATE — feeding vEcoli v2's
#    simData makes its FBA go negative (InvalidBoundaryError).
bash scripts/build_comparison_caches.sh        # --mode full (all conditions)

# 2. Run the comparison: both engines × 5 conditions, matched-initial-state.
#    Args: SEEDS GENS PER_GEN_STEPS  (4 seeds × 4 generations × 5 conditions).
bash scripts/run_local_4x4x5.sh 4 4 15000
```

Output: per-condition zarr stores under `out/local_4x4x5/<condition>/`
(`v2ecoli_seed<NN>.zarr` / `vecoli_seed<NN>.zarr`) + a multi-seed report at
`out/local_4x4x5/report/`. Env overrides: `V2E_CACHE`, `V2E_VECOLI_CACHE`,
`V2E_VECOLI_DIR`, `V2E_CONDITIONS` (subset for a partial run).

### Three reproducibility fixes baked into the run (don't remove)

Single-seed v2-vs-vEcoli comparisons are dominated by stochastic INITIAL-STATE
sampling (fixes 1–2) and by which MEDIA each condition actually runs on (fix 3).
Three complementary mechanisms make rows reflect dynamics on the correct media,
not sampling luck or a stale cache (validated: deviations <0.6% on all 5
conditions, down from 17–44%):

1. **Matched-initial-state seeding** (`--match-initial-state`, on in the driver).
   Overlays genuine vEcoli's initial `bulk` onto v2 by molecule name, so both
   engines start from identical molecule counts. This removes the low-copy
   regulator divergence — e.g. SpoT (ppGpp hydrolase, ~1–12 molecules): at seed 0
   v2 drew 12 vs vEcoli's 1, which drained v2's ppGpp, released the elongation
   throttle, and produced a ~35% acetate FBA divergence that vanished once the
   counts matched. (Expected SpoT is the SAME in both — it was sampling tails.)
2. **Basal regen-per-seed** (`_build_v2ecoli`). Non-basal conditions already
   regenerate the initial state per (condition, seed) from `simData`; basal used
   a FIXED cache snapshot, so every basal seed started identical (e.g.
   active_ribosome 12441) while vEcoli resamples (seed2 = 9947). Basal now regens
   per seed too, reproducing vEcoli's draw to <0.1%.
3. **Per-condition media guard** (`_build_v2ecoli`). A v2ecoli per-condition
   bundle (`out/cache_full/cond_<name>_seed<NN>/`) bakes its `media_id` into the
   config at generation time. A bundle built before a condition→media fix could
   bake the WRONG media (e.g. `with_aa` baked `minimal`), and the old
   marker-exists-only reuse silently ran it — no amino-acid uptake → methionine
   starvation → RelA/ppGpp runaway (5–13×) → RNAP active fraction halved → a
   ~14% RNA / ~7% growth-rate deficit that read as a bogus "port divergence".
   The guard derives the REQUIRED media from `sim_data.conditions[cond].nutrients`
   (single source of truth), compares it to the bundle's recorded `media_id`
   (`metadata.json`), and regenerates any mismatched bundle — so every condition
   self-corrects. A fail-loud post-build assertion confirms the composite runs on
   the required media. (The cache-version fingerprint that should have caught this
   was inert — it hashed source files against a `chdir`'d dir, so every file read
   as MISSING; now anchored to the package, it busts stale caches for real.)

Unique molecules (ribosomes/RNAP/chromosome) are NOT overlaid (v2's unique arrays
carry a `pool_label` field vEcoli's lack, so a direct copy raises) — the basal
regen fix covers the dominant unique-count case instead.

### Generations / `max_steps` semantics (important)

`--max-steps` means different things to the two runners: v2ecoli's
`run_multigen_xarray` treats it as a TOTAL across all generations; the vEcoli
`run_vivarium_ecoli_pbg_multigen` treats it as PER-generation. The driver hides
this: it passes v2ecoli `GENS × PER_GEN` (total) and vEcoli `PER_GEN` (per-gen),
so both get the same per-generation budget and divide naturally for `GENS`
generations (division is mass-driven; the cap is a non-binding safety net).
Verified: both engines reach 4 real divisions on basal.

### Known limitation: `no_oxygen` vEcoli runs only 1 generation

On `no_oxygen` (anaerobic) media the **genuine vEcoli** engine dies at the
gen-1→gen-2 boundary (~900 s): its FBA homeostatic objective computes a tiny
negative target (`BIOTIN[c] = -1e-5`) and, once that's clamped, the GLPK basis
goes singular (`GLP_ESING`) — the solver is chronically ill-conditioned
(`cond ≈ 1e12`, warned every step) and tips fully singular on anaerobic media.
vEcoli's `solve()` only retries the *identical* solve, so it can't recover. This
is a pre-existing numerical fragility in the **reference** FBA, not the port and
not the media fix: **v2ecoli runs `no_oxygen` for the full 4 generations.** So the
`no_oxygen` row is a valid *1-generation* comparison (within_tol on gen 1) against
v2ecoli's full run. A real fix means basis refactorization/perturbation in the
shared GLPK solver, which would shift the reference baseline for *every*
condition — out of scope; the reference is kept pristine.

## Cloud run (viva-api / Ray on GovCloud)

The cloud orchestration (`scripts/comparison_harness.sh all --spec ...`:
register → launch → wait → report) submits `composite=v2ecoli` and
`composite=vecoli` Ray jobs that run the SAME `run_comparison_ensemble.py`.
Because `vivarium-process` is now the default `vecoli_source`, the vEcoli jobs
use it automatically. `--match-initial-state` is threaded through viva-api (router
query param → config → `_sim_command`), staging the upstream ParCa as a 2nd
input (`stage_inputs_2`) so the v2 job can build vEcoli's reference initial state;
it requires the upstream cache to exist in S3 for the commit.

> NOTE (pending follow-up): `comparison_harness.sh`'s `vecoli_engine`
> terminology (`upstream-wrapper` / `nextflow`) predates the loader
> consolidation and is being cleaned up — the ENGINE is always vivarium-process
> now; that field only selects the orchestration route. The local path above is
> the unambiguous reference until the cloud harness is consolidated.

## Report card

```bash
# multi-seed local
.venv/bin/python scripts/comparison_report_card.py \
    --local-pbg-dir '{"basal":["out/local_4x4x5/basal","out/local_4x4x5/basal"], ...}' \
    --local-pbg-seeds 4 --only all -o out/local_4x4x5/report
# cloud (reads both engines' zarr from S3)
.venv/bin/python scripts/comparison_report_card.py --pbg-vs-pbg --out <exp_dir> --only all
```
