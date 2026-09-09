# Removed composite wiring (exact text, as of 2026-08-21)

## `v2ecoli/composites/ecoli_baseline.py` -- `FEATURE_MODULES['flagella_regulation']`

The entire feature-module entry (including its own already-preserved
in-line history of prior iterations) was removed:

```python
    # flagella-cascade investigation (ported from Maya Abdalla's vEcoli `biofilm`
    # branch): the Kalir & Alon SUM-gate + FlgM secretion gate. An earlier
    # hard-coded flagella-count ceiling ("ecoli-flagella-nucleation-cap") was
    # tried first to fix the flagella-count-unbounded-runaway finding, and was
    # removed 2026-08-06 per Maya's explicit instruction not to keep an
    # artificial cap (see feedback_biology_first_no_quick_fixes). Its
    # ordering-bug fix (the after_steps/before_steps split on
    # build_execution_layers) is kept as general infrastructure even though no
    # feature currently needs insert_after for this one.
    #
    # REMOVED 2026-08-10 (Maya's explicit instruction): FlhD4C2 (ClpXP-mediated)
    # basal degradation (ecoli-flhdc-degradation, added 2026-08-05) and the
    # FliT-mediated negative-feedback checkpoint on FlhD4C2
    # (ecoli-flit-flhdc-checkpoint, added 2026-08-06, Utsey & Keener 2020
    # fast-equilibrium reduction). [...] Full code + reaction-network entries
    # archived at archive/flit-flhdc-regulation-2026-08/ for reference.
    #
    # ALSO added 2026-08-06: ecoli-flagella-filament-nucleation and
    # ecoli-flagella-filament-elongation REPLACE CPLX0-7452_RXN as the actual
    # creator of complete-flagellum (CPLX0-7452) counts. [...]
    # OFF by default (opt-in via enable_features('flagella_regulation')).
    'flagella_regulation': {
        'insert_before': 'ecoli-transcript-initiation',
        # Old list (kept per standing preserve-old-code rule):
        # 'steps': [
        #     'ecoli-flagella-flgm-secretion',
        #     'ecoli-flagella-transcription-regulation',
        # ],
        # Previous list (also kept):
        # 'steps': [
        #     'ecoli-flhdc-degradation',
        #     'ecoli-flagella-flgm-secretion',
        #     'ecoli-flagella-transcription-regulation',
        # ],
        # Previous list (also kept, before filament nucleation/elongation):
        # 'steps': [
        #     'ecoli-flhdc-degradation',
        #     'ecoli-flit-flhdc-checkpoint',
        #     'ecoli-flagella-flgm-secretion',
        #     'ecoli-flagella-transcription-regulation',
        # ],
        # Previous list (also kept, before FliT checkpoint removal 2026-08-10):
        # 'before_steps': [
        #     'ecoli-flagella-motor-switch-assembly',
        #     'ecoli-flagella-motor-complex-assembly',
        #     'ecoli-flagella-filament-nucleation',
        #     'ecoli-flagella-filament-elongation',
        #     'ecoli-flhdc-degradation',
        #     'ecoli-flit-flhdc-checkpoint',
        #     'ecoli-flagella-flgm-secretion',
        #     'ecoli-flagella-transcription-regulation',
        # ],
        # Previous list (also kept, before export-apparatus Step conversion
        # 2026-08-11): CPLX0-7451 still fired via ecoli-complexation (SSA),
        # racing against the deterministic C-ring Step for the same-tick
        # CPLX0-7450 it now depends on:
        # 'before_steps': [
        #     'ecoli-flagella-motor-switch-assembly',
        #     'ecoli-flagella-motor-complex-assembly',
        #     'ecoli-flagella-filament-nucleation',
        #     'ecoli-flagella-filament-elongation',
        #     'ecoli-flagella-flgm-secretion',
        #     'ecoli-flagella-transcription-regulation',
        # ],
        'before_steps': [
            'ecoli-flagella-motor-switch-assembly',
            'ecoli-flagella-export-apparatus-assembly',
            'ecoli-flagella-motor-complex-assembly',
            'ecoli-flagella-filament-nucleation',
            'ecoli-flagella-filament-elongation',
            'ecoli-flagella-flgm-secretion',
            'ecoli-flagella-transcription-regulation',
        ],
    },
```

## Imports removed

```python
    from v2ecoli.processes.flagella_filament_nucleation import FlagellaFilamentNucleation
    from v2ecoli.processes.flagella_motor_switch_assembly import FlagellaMotorSwitchAssembly
    from v2ecoli.processes.flagella_export_apparatus_assembly import FlagellaExportApparatusAssembly
    from v2ecoli.processes.flagella_motor_complex_assembly import FlagellaMotorComplexAssembly
```

## Step-registry entries removed

```python
        'ecoli-flagella-filament-nucleation': FlagellaFilamentNucleation,
        'ecoli-flagella-motor-switch-assembly': FlagellaMotorSwitchAssembly,
        'ecoli-flagella-export-apparatus-assembly': FlagellaExportApparatusAssembly,
        'ecoli-flagella-motor-complex-assembly': FlagellaMotorComplexAssembly,
```

## `v2ecoli/core.py` -- `_CACHE_CONFIG_NAMES`

Removed entries (kept `'ecoli-flagella-filament-elongation'`, shared):

```python
    'ecoli-flagella-motor-switch-assembly', 'ecoli-flagella-export-apparatus-assembly',
    'ecoli-flagella-motor-complex-assembly',
    'ecoli-flagella-filament-nucleation',
```

## `v2ecoli/library/sim_data.py` -- step-name -> config-getter mapping

Removed entries (kept `"ecoli-flagella-filament-elongation"`, shared):

```python
            "ecoli-flagella-motor-switch-assembly": self.get_flagella_motor_switch_assembly_config,
            "ecoli-flagella-export-apparatus-assembly": self.get_flagella_export_apparatus_assembly_config,
            "ecoli-flagella-motor-complex-assembly": self.get_flagella_motor_complex_assembly_config,
            "ecoli-flagella-filament-nucleation": self.get_flagella_filament_nucleation_config,
```

The corresponding `get_flagella_motor_switch_assembly_config`,
`get_flagella_export_apparatus_assembly_config`,
`get_flagella_motor_complex_assembly_config`, and
`get_flagella_filament_nucleation_config` method bodies are left in place
in `sim_data.py`, now unreferenced/inert -- not surgically deleted out of a
large shared file.
