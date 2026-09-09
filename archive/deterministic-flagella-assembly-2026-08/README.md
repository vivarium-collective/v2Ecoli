# Archive: deterministic flagellar assembly Steps (removed 2026-08-21)

This folder preserves the four custom deterministic Steps that modeled
flagellar structural assembly (C-ring/MS-ring through hook-basal-body
completion) in the `v2ecoli` flagella-cascade investigation
(`workspace/investigations/flagella-cascade/`), from 2026-08-06/11 through
2026-08-21. Removed from the active codebase at Maya Abdalla's explicit
instruction, having committed to the NFsim rule-based replacement
(`flagella-04-complexation-nfsim`) as the one path forward. Kept here for
reference, not because the mechanism was wrong or the work was wasted.

## What this was

Four Steps, each converting one stage of flagellar assembly from the
generic Gillespie complexation framework (which couldn't express the real
biology -- see each file's own docstring for the specific structural gap
that forced it out) into a real, incremental/deterministic mechanism:

1. **`flagella_motor_switch_assembly.py`** (`ecoli-flagella-motor-switch-assembly`)
   -- C-ring/MS-ring formation (CPLX0-7450), consuming FliF/FliG/FliM/FliN
   at their real cryo-EM stoichiometry (34/34/34/111).
2. **`flagella_export_apparatus_assembly.py`** (`ecoli-flagella-export-apparatus-assembly`)
   -- export apparatus formation (CPLX0-7451), added 2026-08-11 specifically
   to fix a same-tick SSA/Step race (the export apparatus used to fire via
   the generic Gillespie complexation process while depending on a molecule
   -- CPLX0-7450 -- that only existed transiently inside this deterministic
   Step's own tick).
3. **`flagella_motor_complex_assembly.py`** (`ecoli-flagella-motor-complex-assembly`)
   -- full motor complex formation (FLAGELLAR-MOTOR-COMPLEX), consuming the
   export apparatus plus the rod/L-ring/stator proteins.
4. **`flagella_filament_nucleation.py`** (`ecoli-flagella-filament-nucleation`)
   -- hook completion (120x FlgE) + nucleation of a new `nascent_flagellum`
   unique molecule, Poisson-triggered at a real rate (0.00167/s, Sisti et al.
   2017) on a fixed ~600s interval.

**NOT archived, still active** -- these are shared infrastructure reused
as-is by the NFsim pipeline, not deterministic-assembly-specific:
- `flagella_filament_elongation.py` (`ecoli-flagella-filament-elongation`)
- `flagella_flgm_secretion.py` (`ecoli-flagella-flgm-secretion`)
- `flagella_transcription_regulation.py` (`ecoli-flagella-transcription-regulation`, the Kalir & Alon SUM-gate)

## Why removed

Not because it was wrong -- it was real, carefully cross-checked structural
biology (each stoichiometry fix cited against cryo-EM literature, see the
files' own docstrings). But it was always meant to be a bridge to the
NFsim rule-based network (`flagella-04-complexation-nfsim`), which:
- represents each assembling structure as an individually tracked graph
  object (not a bulk count), enforcing assembly ORDER per-instance via rule
  pattern-matching rather than by careful Step-execution ordering (the
  same-tick races these deterministic Steps had to work around, e.g. the
  export-apparatus/C-ring race above, don't exist in a single unified rule
  engine);
- has now had its own real bugs found and fixed this session (scaffold/
  internal-observable division-state bug, FliD double-consumption); and
- is the path Maya has committed to going forward (2026-08-21) -- keeping
  two parallel, mutually-exclusive assembly pipelines alive was adding
  maintenance surface with no plan to actually run the deterministic one
  again.

See `composite_wiring_snapshot.md` for the exact removed `ecoli_baseline.py`
feature-module block (including its own internal history of prior
iterations, already preserved in-line via the standing preserve-old-code
rule) and the removed cache-config / sim_data registrations.

## Supporting infrastructure also removed (see snapshot file for exact text)

- `ecoli_baseline.py`: the `'flagella_regulation'` feature-module entry
  (`FEATURE_MODULES` dict) and its four imports + step-registry entries.
- `core.py`: the four deterministic step names removed from
  `_CACHE_CONFIG_NAMES`.
- `library/sim_data.py`: the four `"ecoli-flagella-*-assembly"` /
  `"ecoli-flagella-filament-nucleation"` entries removed from the
  step-name -> config-getter mapping. The `get_flagella_*_config` methods
  themselves are left in place in `sim_data.py` (inert, unreferenced) --
  not worth surgically deleting out of a large shared file for what's
  otherwise a clean, reversible disconnection.

## Reference material (not touched, historical record)

Numerous docs/diagnostic scripts elsewhere in
`workspace/investigations/flagella-cascade/` still reference
`flagella_regulation` by name (CHANGES_*.md, study.yaml files,
session_notes, and several study-02/03 diagnostic scripts with a
`--feature flagella_regulation` CLI choice). These are historical record of
already-completed comparison work, not live/default-running code -- left
as-is rather than edited, since editing them wouldn't change any runtime
behavior (calling `enable_features('flagella_regulation')` now, after this
removal, no-ops silently rather than erroring -- `build_execution_layers`
looks up the feature name and skips unrecognized ones).
