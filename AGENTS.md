# AGENTS.md

Guidance for AI coding assistants working in this repo. Read this before
editing biological process code, composite wiring, or the type system.
Humans should read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

## What this repo is

v2ecoli is a whole-cell *E. coli* model. It ports 55 biological processes
from [CovertLab/vEcoli](https://github.com/CovertLab/vEcoli) onto
[process-bigraph](https://github.com/vivarium-collective/process-bigraph)
and [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema).
The biology must match upstream unless a PR explicitly justifies a change.

## Frameworks you must understand

### process-bigraph

- A **Composite** = processes + stores (state) + wires (edges from ports to
  store paths). Composites are assembled in `v2ecoli/composite*.py` and the
  corresponding `generate*.py` files build them for each architecture.
- A **Process** declares `inputs` and `outputs` schemas and implements an
  `update(state, interval) -> update` method. Updates are applied to the
  shared store after each timestep.
- A **Step** runs to convergence within a timestep rather than stepping
  through time. Departitioned fuses partitioned requester/evolver halves
  into Steps.
- Don't reinvent composition patterns — copy from an existing process in
  `v2ecoli/processes/` or `v2ecoli/steps/` and adapt.

### bigraph-schema

- Types describe the shape of stores and ports. Project-specific types live
  in `v2ecoli/types/`.
- Types register serializers via `@_serialize.dispatch`. A save state that
  cannot round-trip through the serializer is broken. Test this.
- Units are carried as `pint.Quantity` values; schemas enforce dimension.

For deeper questions about either framework, invoke the `pbg-expert` skill.

## Checks you must apply when adding or editing a process

1. **Schema round-trip.** `serialize(state) -> JSON -> deserialize` must
   reproduce the original state. No pickle anywhere in this path.
2. **Port contract.** Everything `update` reads must be declared in `inputs`;
   everything it writes must be declared in `outputs`. Mismatches are silent
   bugs in process-bigraph.
3. **Units.** Every dimensioned quantity at a port is a `pint.Quantity`.
   No bare floats, no `Unum`. The Unum bridge at
   `v2ecoli/library/unit_bridge.py` exists only for upstream vEcoli interop.
4. **Conservation.** Mass, molecule counts, and charge must balance across
   the update unless the process has an explicit source/sink with a stated
   biological justification.
5. **Behavior test.** A new process requires `tests/test_behavior_<name>.py`
   that runs the process inside a composite and asserts on an outcome
   (growth rate, molecule count, concentration, etc.). A unit test of a
   helper function does not substitute.
6. **Parity gate (behavior-preserving refactors).** Any change claiming to
   preserve behavior (deriver/flow consolidation, port-schema edits, renames)
   must pass the committed gate before commit — never claim "byte-identical"
   from memory:

   ```
   PYTHONPATH=$PWD .venv/bin/python scripts/parity_check.py \
       --seconds 120 --compare tests/golden/baseline_parity_signature.json \
       --build-check
   ```

   Two gates: a deep null-emitter signature vs the committed golden, AND a
   real-emitter `build_composite` (the second catches emitter-schema resolve
   failures the null emitter hides). Exit 0 = both pass. Re-capture the golden
   (`--out`) only from a clean `origin/main` worktree when main's behavior
   intentionally changes.

## E. coli domain details

- **EcoCyc** is the reference database for E. coli pathways, genes,
  transcription units, and regulatory sites. Processes representing a known
  biological entity should cite the EcoCyc ID (e.g., `EG10001`, `TU0-42`) in
  the process docstring. Parameters for metabolism, transcription, and
  translation all descend from EcoCyc via the ParCa pipeline.

### EcoCyc ID conventions you'll see everywhere in this repo

| Prefix | Meaning | Example |
|---|---|---|
| `PD` | Polypeptide (apo / unmodified monomer) | `PD03831[c]` — apo DnaA |
| `MONOMER` | Monomer / single-subunit functional form (often nucleotide-bound) | `MONOMER0-160[c]` — DnaA-ATP · `MONOMER0-4565[c]` — DnaA-ADP |
| `CPLX` | Multi-subunit complex | `CPLX0-7710[c]` — MarR · `CPLX0-3953[c]` — 30S ribosome |
| `RXN` | Reaction (FBA / metabolism flat data) | `RXN0-7444` — RIDA (DnaA-ATP → DnaA-ADP) |
| `<NAME>_RXN` | ParCa-fitted mass-action equilibrium reaction | `MONOMER0-160_RXN` — apo + ATP ⇌ DnaA-ATP |
| `EG` | Encoding gene (RegulonDB-derived) | `EG10235` — dnaA |
| `EG<N>_RNA` | The mRNA transcribed from that gene | `EG10235_RNA` — dnaA mRNA |
| `TU` | Transcription unit | `TU0-42` |
| Compartment suffix | `[c]` cytoplasm · `[p]` periplasm · `[i]` inner-membrane · `[o]` outer-membrane |

### How to find an ID when you don't already know it

The IDs are scattered across many places. Search in roughly this order:

1. **Workspace expert docs first.** `workspace/references/expert/*.html` or `.pdf` —
   for the DnaA investigation, the May 2026 prior-art HTML report
   (`v2ecoli_replication_initiation_report`) enumerates every relevant
   bulk ID + reaction ID with biological context. Check
   `workspace.yaml.expert_docs[]` for the registered set, plus
   `workspace/references/notes/*.md` for the per-paper digests.

2. **Existing process modules.** Many processes hardcode the IDs they
   touch as module-level constants — `grep -rn 'PD0\|MONOMER0\|CPLX0\|RXN0' v2ecoli/processes/` finds them fast.
   (Example from the DnaA investigation branch — not on `main`:
   `v2ecoli/processes/dnaa_box_binding.py` has
   `DNAA_ATP_ID = "MONOMER0-160[c]"` + `DNAA_ADP_ID = "MONOMER0-4565[c]"`.)

3. **The dnaa-box catalog** (`v2ecoli/data/dnaa_box_catalog.py`, on the DnaA
   investigation branch, not `main`) defines the 307 consensus chromosomal
   boxes + their region partition + per-box affinity / form-preference.
   Authoritative for any DnaA-binding question.

4. **A real `runs.db` state blob.** Every history row is a JSON snapshot
   of the live cell state; `state.bulk` is a list of `[id, count, ...]`
   tuples covering every molecule in the model (~16 000 entries). One
   `SELECT state FROM history LIMIT 1` + `json.loads` gives you the
   ground-truth list. Useful when grep doesn't find a name.

5. **EcoCyc itself.** When you have a biological name and need the ID
   (or the reverse), the canonical lookup is https://ecocyc.org — but
   for routine work the sources above usually answer it without leaving
   the repo.

Common failure mode (and why this section exists): writing assumptions
or test paths that *paraphrase* the biology ("DnaA-ATP", "RIDA
hydrolysis", "300 chromosomal boxes") instead of pinning the actual
EcoCyc IDs (`MONOMER0-160[c]`, `RXN0-7444`, **307** boxes). The
paraphrased form looks plausible but won't grep against the runs.db
state — listener paths that reference a non-existent key silently
produce empty charts. Always include the EcoCyc ID alongside the
biological name.
- **ParCa** (Parameter Calculator) builds `sim_data` from raw EcoCyc-derived
  knowledge bases. It's expensive (minutes to hours). Never run ParCa in CI —
  CI uses a frozen gzipped cache at `tests/fixtures/cache/`.
  - **Always build ParCa with `--mode full` for any simulation.** `--mode fast`
    (debug) reduces the set of TF conditions and mis-calibrates regulation — it
    over-expresses dnaA ~2× and breaks replication initiation. Fast mode is for
    pipeline debugging only, never for sim data. `scripts/build_cache.py` guards
    against using a fast-built cache.
- **Architectures**:
  - `baseline` — partitioned, 55 processes, upstream-parity (the reference).
  - `colony` — many baseline cells in a shared environment (multi-agent).
  - `millard_pdmp_baseline` — piecewise-deterministic Markov-process variant.
  A change to a process must work across all of them, or the PR must explain
  why it's scoped.

### Adding a new composite architecture

Each architecture is a function decorated with `@composite_generator`
from `pbg_superpowers.composite_generator`. To add one:

1. Create `v2ecoli/composites/<arch>.py` with a
   `<arch>(core, *, seed, cache_dir) -> dict` function decorated
   `@composite_generator(name="<arch>", description=..., parameters={...})`.
2. Append `from v2ecoli.composites import <arch>` to
   `v2ecoli/composites/__init__.py`.
3. Update `v2ecoli/library/cache_version.py:INPUT_FILES` to include the
   new file.

Once registered, callers reach it via
`v2ecoli.build_composite("<arch>", ...)`. The function returns a
process-bigraph state document; `build_composite` wraps it in a `Composite`.

## Running multiseed/multigen jobs in parallel

Independent seeds are embarrassingly parallel, and v2ecoli can run them
concurrently — **one Ray worker process per seed** — reaching parity with
vEcoli's Nextflow fan-out (see [`docs/perf/v2ecoli-vs-vecoli-performance.md`](docs/perf/v2ecoli-vs-vecoli-performance.md)).
Use this instead of a hand-rolled `for seed in seeds:` loop whenever the seeds
don't share state.

**The helper.** `v2ecoli.library.parallel_seeds.run_seeds_parallel(seeds, run_one, *, mode, run_kwargs=...)`:

```python
from v2ecoli.library.parallel_seeds import run_seeds_parallel

def run_one(seed, **kw):             # MUST be a top-level function (Ray pickles it
    set_null_emitter_override(True)  # to a fresh worker) — do per-worker setup here,
    composite = build_composite("ecoli_baseline", cache_dir=CACHE_DIR, seed=seed)  # use ABSOLUTE paths,
    ...                              # and pass everything via run_kwargs (NOT module globals).
    return {"seed": seed, ...}

run = run_seeds_parallel(range(n_seeds), run_one, mode=_parallel_mode(),
                         run_kwargs={"n_steps": n_steps})
per_seed = run.results               # in seed order; run.wall_s = critical-path wall
```

**The flag.** `mode` comes from `workspace.yaml` `runtime.parallel` (`'ray'` |
unset). The helper is always safe: `mode != 'ray'` (or `ray` not installed) →
runs sequentially with **identical results**. It thread-balances each worker to
`cores // n_seeds` so N workers don't oversubscribe BLAS. Requires the `ray`
extra (`pip install 'v2ecoli[ray]'`).

**Rules of thumb:** only for *independent* seeds (no shared environment — for
*coupled* cells like colonies, use one parallel composite instead). Parallelism
changes wall time, never the science. Reference adoption:
`scripts/run_phase0_xarray_ensemble.py` (`--parallel ray|off`).

## Reports

Regenerate the relevant report and inspect it before opening a PR that
touches processes, steps, or composite wiring.

- `reports/workflow_report.py` → `out/reports/workflow_report.html` — full cell lifecycle,
  division at ~42 min.
- `reports/multigeneration_report.py` → `multigeneration_report.html` — N-generation single
  lineage with mass trajectories and fold-change.
- `reports/colony_report.py` → `colony_report.html` — mixed colony with pymunk
  physics, growth, and division.
- `reports/network_report.py` → `network_*.html` — per-architecture Cytoscape
  topology. Click a process to see ports, schemas, config, docstring, math.
- `reports/v1_v2_report.py` → `v1_v2_comparison.html` — vEcoli 1.0 vs 2.0 vs v2ecoli.
- `reports/benchmark_report.py` — v2ecoli vs vEcoli composite subprocess benchmark.

Published at https://vivarium-collective.github.io/v2ecoli/.

### HTML reports with provenance banners — attach to substantial PRs

When a PR makes a substantial change to a Process / Step / composite /
biology behaviour, generate an HTML report that captures the change and
attach it to the PR as committed evidence.

The HTML must include a **provenance banner** at the top so it stays
self-describing once the file is months old. Capture, at minimum:

- ISO-8601 generated timestamp
- Git SHA (full + short), linked to the GitHub commit URL
- Git branch + a `DIRTY TREE` badge if `git status --porcelain` is non-empty
- Last commit message + author + date (so the reader can see what code
  produced the artefact without leaving the page)
- Path to the generator script (relative to repo root)
- Host name + OS + Python version

Pattern: see `scripts/compare_pdmp_vs_baseline.py::collect_provenance`.

**Standard tool — `scripts/pr_session_report.py`.** For a PR/session report,
use this reusable generator rather than hand-rolling one — it produces the
provenance banner *and* before/after parity plots:

```bash
python scripts/pr_session_report.py capture --out /tmp/after.json --steps 60
cp scripts/pr_session_report.py /tmp/prr.py          # so it exists on the base ref
git checkout main && python /tmp/prr.py capture --out /tmp/before.json --steps 60
git checkout -
python scripts/pr_session_report.py render --before /tmp/before.json \
  --after /tmp/after.json --out reports/figures/<study>/report.html \
  --title "..." --summary-file <summary.html>
```

`capture` is self-contained (no branch-only imports) so the same file runs on
any ref; the before/after overlay is how a refactor shows it preserved behavior
(curves coincide, final-step rel diff ~0).

Save TWO copies per run:

1. `reports/figures/<study>/<short_name>.html` — overwritten each render;
   the "latest" entry the PR description / README links to.
2. `reports/figures/<study>/<short_name>_<YYYYMMDDTHHMMSS>_<git_short>.html`
   — archival, never overwritten. Add a `git add -f` for this file
   (`reports/` is gitignored so new artefacts need `-f`), then commit and
   reference it in a PR comment so the artefact is versioned alongside
   the code that produced it.

PR comment template (when posting the archive link):

> Evidence: [pdmp_vs_baseline_20260530T103045_aa7a2de.html](…)
> Generated from `aa7a2de` on `v2ecoli-pdmp`, 600 s sim, seed=0.

This makes every substantive review easier — the reviewer can open the
HTML and see the exact state of the simulation the PR is claiming.

## Tests you must know about

- `tests/test_model_behavior.py` — 7 definitive behavior tests. Gate every
  PR. Do not weaken thresholds without a reviewed justification.
- `tests/test_cell_cycle_regressions.py` — slow full-cycle tests. Run in a
  nightly job, not on every PR. Marked with `@pytest.mark.slow`.
- `@pytest.mark.sim` separates behavior tests from fast tests. CI splits
  them into parallel jobs. Don't remove the marker.
- Behavior fixtures live in `tests/fixtures/`. The pre-division state
  (`pre_division_state.json.gz`) and ParCa cache (`cache/`) are load-bearing.
- `out/cache/` is fingerprinted by `v2ecoli/library/cache_version.py`.
  `v2ecoli.core.load_cache_bundle` — the production loader that
  `build_composite` and every composite architecture route through — calls
  `verify_cache_version` on every call before loading, so a cache that was
  built from a different `models/parca/parca_state.pkl.gz`,
  `v2ecoli/library/sim_data.py`, or unit-bridge raises `StaleCacheError`
  with a one-line rebuild command instead of a 10-frame-deep `AttributeError`.
  Rebuild with `python scripts/build_cache.py` (fast; reuses the committed
  ParCa fixture — no ParCa re-run). See `docs/generate_full_parca.md` for
  the full ParCa path. Set `V2ECOLI_SKIP_CACHE_VERIFY=1` to bypass the check
  for deliberate cross-version work (e.g. comparing a cache built on a
  different commit); this logs a one-time warning and should not be used
  routinely.

## What NOT to do

- Do not modify anything under `tests/fixtures/`. Regenerating a fixture is a
  deliberate act with its own PR.
- Do not edit `.github/workflows/` or bump `pyproject.toml` dependencies
  without explicitly calling it out in the PR description.
- Do not introduce `pickle`, `dill`, or `cloudpickle` in save-state paths.
  Save states round-trip through bigraph-schema JSON. ParCa caches are the
  only exception.
- Do not bypass the Unum/pint boundary in `v2ecoli/library/unit_bridge.py`.
  Unum should appear nowhere else.
- Do not add a process to one architecture without a plan for the other two.
- Do not commit generated reports (`out/`) or ParCa scratch output.

## Before opening a PR

```bash
python scripts/build_cache.py                # regenerate out/cache if stale
pytest -m "not sim"                          # fast tests
pytest -m sim tests/test_model_behavior.py   # behavior tests
```

If you changed a process, also run the relevant report script and attach the
output HTML (or the diff against `main`) to the PR.

## PR conventions

Two distinct PR types live in this repo, and they look different on GitHub:

### Feature / fix PRs (merge candidates)

Standard `gh pr create`. Conventional-commit-style title
(`feat(...)`, `fix(...)`, `chore(...)`). Opens as ready-for-review (not
draft). Target `main`. CI must pass before merge. Live examples: most
PRs on `main`'s log.

### Investigation PRs (living integration branches)

A long-running branch hosting an investigation (a collection of related
studies under a shared research question). NOT a merge target — the
branch lives indefinitely; infrastructure carved out of it gets shipped
via separate feature PRs against `main`.

Two conventions:

1. **Title prefix `investigation:`** — e.g.
   `investigation: DnaA-driven replication initiation — mechanistic vs heuristic`.
   Visually distinguishes from feature PRs in the PR list.

2. **Draft state** — open with `gh pr create --draft ...` (or, when
   creating from the vivarium-workbench GitHub tab, the `draft=True`
   default applies automatically). Draft signals "don't merge me" to
   both reviewers and to GitHub's auto-merge / branch-policy machinery.

Live examples on this repo: #59 (dnaa-replication), #69 (multiscale-
bioprocess), #72 (PDMP whole-cell reformulation).

Body should call out:
- That this branch is NOT a merge target.
- Companion feature PRs (already merged) that ship the infrastructure
  this investigation depends on, vs. content that stays on this branch.
- The investigation's primary research question + headline figures.
