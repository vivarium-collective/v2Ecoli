# Reaction network entries, as they existed before removal (2026-08-10)

Exact text removed from the live ParCa flat_overrides, preserved here for
reference. See README.md for why these were removed and what stays.

## equilibrium_reactions.tsv (removed block)

```
# FLIT-FLID-CPLX_RXN (flagella-cascade investigation, 2026-08-05): the FliT homodimer
# (FLIT-DIMER, see complexation_reactions_added.tsv) chaperones FliD (EG10841-MONOMER)
# at the measured 2:1 FliT:FliD stoichiometry (Yamamoto & Kutsukake 2006, J Bacteriol
# 188:5124), expressed here as 1 FLIT-DIMER + 1 FliD (both order-1 reactants -- see
# complexation_reactions_added.tsv for why the dimer is tracked as its own species
# rather than writing this as "2 FliT + 1 FliD" directly). Modeled as an equilibrium,
# not a complexation reaction -- complexation reactions require exactly one
# positive-coefficient product, which blocked an earlier design where CPLX0-7452_RXN
# both formed CPLX0-7452 and released free FliT directly; see the note in
# complexation_reactions_modified.tsv. This equilibrium and CPLX0-7452_RXN (unmodified,
# still consuming raw free FliD) draw on the same free-FliD pool: as flagellum assembly
# consumes free FliD, this equilibrium shifts toward dissociation to partially
# replenish it, releasing FliT-dimer as a direct consequence of shared-reactant
# mass-action dynamics -- not a hard-coded co-product. This is the real trigger for
# the FliT negative-feedback checkpoint on FlhD4C2 (Utsey & Keener 2020, PLOS Comput
# Biol 16:e1007689; Yamamoto & Kutsukake 2006; Yakhnin et al., PMC4239645). No
# literature Kd exists for FliT:FliD binding specifically, so the rate
# (equilibrium_reaction_rates.tsv) is an assumed value in the same numerically-stable
# range already used for FLGM-FLIA-CPLX_RXN above, not a literature-sourced constant
# -- flagged here rather than presented as measured.
"FLIT-FLID-CPLX_RXN"	{"FLIT-FLID-CPLX": 1, "FLIT-DIMER": null, "EG10841-MONOMER": null}	"FliT:FliD export chaperone complex (2:1 FliT:FliD)"
```

## equilibrium_reaction_rates.tsv (removed row)

```
"FLIT-FLID-CPLX_RXN"	1	2E-7	2E-7	false
```

## complexation_reactions_modified.tsv (trimmed comment, CPLX0-7452_RXN itself unchanged)

The full historical comment block that used to precede `CPLX0-7452_RXN`
(explaining the FliT:FliD complexation attempt that was tried and reverted
back in 2026-08-05) has been trimmed to a one-line pointer to this archive.
Original text:

```
# CPLX0-7452_RXN (flagella-cascade investigation, 2026-08-05): a modification consuming
# FLIT-FLID-CPLX and releasing free FliT directly from this reaction was tried and
# reverted -- complexation reactions require exactly one positive-coefficient product
# (molecule_groups.py `assert len(complex_ids) == 1`), so a reaction cannot both form
# CPLX0-7452 AND release free FliT. CPLX0-7452_RXN is back to its unmodified,
# vendored stoichiometry (still consuming raw FliD, EG10841-MONOMER, directly). The
# FliT-release mechanism now lives entirely in equilibrium_reactions.tsv/
# equilibrium_reaction_rates.tsv (FLIT-FLID-CPLX_RXN): FliT:FliD sequestration and
# CPLX0-7452_RXN's consumption of free FliD compete for the same FliD pool, so
# flagellum assembly progress couples to FliT release through shared-reactant
# equilibrium dynamics rather than a hard-coded co-product.
```

Note: CPLX0-7452_RXN's actual stoichiometry was never permanently modified for
FliT:FliD (the attempt above was tried and reverted the same day) -- it still
consumes raw free FliD (EG10841-MONOMER, -5) directly. No stoichiometry change
was needed when removing FLIT-FLID-CPLX_RXN; only this historical comment was
trimmed.

## complexation_reactions_added.tsv (FLIT-DIMER_RXN -- KEPT, not archived)

FLIT-DIMER_RXN (FliT homodimer formation, `{"FLIT-DIMER": 1, "EG11389-MONOMER": -2}`)
is **still active** -- it was not removed. FliT dimerization is real, independent
biology (Yamamoto & Kutsukake 2006) that doesn't depend on the regulatory checkpoint;
it's simply no longer consumed by anything else in the network now that
FLIT-FLID-CPLX_RXN is gone. Left in place as a building block for the planned
NFsim-based FliT representation.

## sim_data.py config methods (removed)

```python
def get_flhdc_degradation_config(self, time_step=1):
    """Config for the FlhD4C2 (ClpXP-mediated) degradation Step.

    Added 2026-08-05 to address the flagella-count runaway found in Maya's
    flagella-cascade investigation. Rate is a literature-anchored ESTIMATE,
    not a directly-measured E. coli/CPLX0-3930-specific constant -- see
    v2ecoli/processes/flagella_flhdc_degradation.py for full provenance.
    """
    return {
        "bulk_molecule_ids": self.sim_data.internal_state.bulk_molecules.bulk_data["id"],
        "flhdc_id": "CPLX0-3930[c]",
        "degradation_rate": 0.00289,
    }

def get_flit_flhdc_checkpoint_config(self, time_step=1):
    """Config for the FliT-mediated FlhD4C2 checkpoint Step.

    Added 2026-08-06 as the real, literature-grounded mechanism for the
    flagella-count-runaway problem, replacing an earlier hard-coded
    nucleation cap (removed 2026-08-06 per Maya's explicit "no artificial
    cap" instruction) -- see
    v2ecoli/processes/flagella_flit_flhdc_checkpoint.py for full
    provenance (Utsey & Keener 2020 fast-equilibrium reduction; delta2 is
    literature-sourced, k_half is a documented estimate reusing the
    SUM-gate's K_flhDC scale).
    """
    return {
        "bulk_molecule_ids": self.sim_data.internal_state.bulk_molecules.bulk_data["id"],
        "flhdc_id": "CPLX0-3930[c]",
        "flit_dimer_id": "FLIT-DIMER[c]",
        "bound_degradation_rate": 0.05,
        "k_half": 50.0,
    }
```

## ecoli_baseline.py wiring (removed)

- Imports: `from v2ecoli.processes.flagella_flhdc_degradation import FlhDCDegradation`,
  `from v2ecoli.processes.flagella_flit_flhdc_checkpoint import FliTFlhDCCheckpoint`
- `STANDALONE_STEPS` dict entries: `'ecoli-flhdc-degradation': FlhDCDegradation`,
  `'ecoli-flit-flhdc-checkpoint': FliTFlhDCCheckpoint`
- `FEATURE_MODULES['flagella_regulation']['before_steps']` entries:
  `'ecoli-flhdc-degradation'`, `'ecoli-flit-flhdc-checkpoint'`
- `get_special_step_config` dict (sim_data.py:631-632): both config-method
  registrations
