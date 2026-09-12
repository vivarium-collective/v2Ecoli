# Generate full-mode ParCa fixture

Instructions for a Claude session to produce a production-quality
`parca_state.pkl.gz` that supports full simulation (not just comparison
testing).

## Context

> **Corrected 2026-08-18 — this document described the pre-regeneration world.**
> The shipped fixture was replaced with a **full-mode** build by `71442761`
> (2026-06-03), which updated `parca_state.pkl.gz` and `runtimes.json` but not
> the prose here or in `models/parca/README.md`.  Both files therefore still
> called a full-mode artifact `--mode fast`.  Measured on the shipped state:
> **51 `cell_specs`** (a `--mode fast` build has 7).  The runtime figures below
> were also stale by 1–2 orders of magnitude; they are now measurements.

The shipped `models/parca/parca_state.pkl.gz` is a **`--mode full`** state
(all TF conditions fitted; 51 `cell_specs`).  A `--mode fast` build
(debug=True) fits a reduced TF condition set — sufficient for comparison
testing and the ParCa comparison report, but the online simulation's
equilibrium solver crashes on it because some metabolite concentrations are
left unpopulated when only a subset of TF conditions are fitted.

A `--mode full` run produces a complete sim_data supporting end-to-end
simulation (single cell → division → daughters).

## Steps

### 1. Build Cython extensions

```bash
cd /Users/eranagmon/code/v2ecoli   # or wherever the repo lives
bash scripts/parca_cython_build.sh
```

This compiles `_build_sequences`, `_fastsums`, and `mc_complexation` for
the current Python.  Only needed once per Python version.

### 2. Run the full ParCa pipeline

```bash
python scripts/parca_run.py --mode full --cpus 8 -o out/sim_data
```

**Expected runtime: ~5 minutes** — measured 2026-08-18, `--mode full --cpus 8`
on an Apple-silicon laptop: **287.6 s wall / 268.3 s summed over steps** (the
~19 s difference is imports, Cython load and writing the 657 MB state).  Steps
4 and 5 dominate at ~100 s each; every other step is under 35 s.  A KO variant
build measured 183 s.  Runtime scales with CPU count, so treat this as an
order-of-magnitude reference rather than a guarantee — but the previous
"4–8 hours" figure was wrong by ~60×, and the condition count is **51**, not
the "~300" this document claimed twice.

The script emits per-step checkpoints at `out/sim_data/checkpoint_step_N.pkl`
and a final `out/sim_data/parca_state.pkl`.  If step 5 succeeds but a later
step crashes, you can resume:

```bash
python scripts/parca_run.py --mode full --cpus 8 \
  --resume-from-step 6 --resume-pickle out/sim_data/checkpoint_step_5.pkl
```

### 3. Package the fixture

```bash
gzip -c out/sim_data/parca_state.pkl > models/parca/parca_state.pkl.gz
cp out/sim_data/runtimes.json models/parca/runtimes.json
```

### 4. Generate the simulation cache

The online simulation needs `initial_state.json` + `sim_data_cache.dill`,
produced by `save_cache` from the `simData.cPickle`:

```python
import sys, os, dill
sys.path.insert(0, '.')
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)
from v2ecoli.core import save_cache

# Hydrate the fixture → simData.cPickle.  ``hydrate_sim_data_from_state``
# copies sibling composite stores (expected_dry_mass_increase_dict,
# translation_supply_rate, …) onto sim_data_root so downstream code
# reaching ``sim_data.expectedDryMassIncreaseDict`` etc. finds them.
state = load_parca_state('models/parca/parca_state.pkl.gz')
sim_data = hydrate_sim_data_from_state(state)
os.makedirs('out/workflow', exist_ok=True)
sd_path = 'out/workflow/simData.cPickle'
with open(sd_path, 'wb') as f:
    dill.dump(sim_data, f)

# Generate the cache
save_cache(sd_path, 'out/cache')
```

### 5. Verify with a short simulation

```python
from v2ecoli import build_composite
composite = build_composite("ecoli_baseline", cache_dir='out/cache')
composite.run(10)  # 10 timesteps, ~30s
print('OK')
```

If this succeeds without `RuntimeError: Could not solve ODEs in equilibrium
to SS`, the fixture is production-ready.

### 6. Commit and push

```bash
git add models/parca/parca_state.pkl.gz models/parca/runtimes.json
git commit -m "models/parca: full-mode ParCa fixture (all TF conditions)

Replaces the fast-mode fixture (debug=True, 7 conditions) with a
full-mode run (all 51 conditions, XXXs on N cores).  The
full fixture supports end-to-end simulation — the equilibrium solver
no longer crashes on unpopulated metabolite concentrations.

Runtimes: step_5=XXXXs, total=XXXXs."

git push
```

### 7. (Optional) Update the comparison report

```bash
python scripts/parca_compare.py
open out/compare/report.html
```

## Notes

- The `--mode fast` fixture is still useful for CI (fast ParCa tests,
  comparison report generation) but should NOT be used as the shipped
  `models/parca/parca_state.pkl.gz` unless simulation support isn't needed.
- The full-mode fixture is ~20–25 MB gzipped (vs ~18 MB for fast mode).
- If you need to re-run only steps 6–9 after a successful step 5 (e.g. to
  test a change in promoter fitting), use the resume path:
  `scripts/parca_rerun_from_step5.sh` or pass `--resume-from-step 6`.
- The Cython extensions must be built for the SAME Python version that runs
  the pipeline.  If you switch Python versions, re-run
  `scripts/parca_cython_build.sh`.
