"""
==========
Metabolism
==========

Encodes molecular simulation of microbial metabolism using flux-balance
analysis (FBA).

Mathematical Model
------------------
**Flux Balance Analysis**

The core optimization problem solved each timestep:

    max  c^T v
    s.t. S v = 0        (steady-state mass balance)
         v_lb <= v <= v_ub   (flux bounds)

where:
    - S: stoichiometric matrix (metabolites x reactions)
    - v: reaction flux vector (mmol/g_DCW/h)
    - c: objective coefficients (biomass production)
    - v_lb, v_ub: lower/upper flux bounds from:
      * environment exchange constraints (nutrient availability)
      * enzyme capacity constraints (kinetic limits)
      * maintenance energy requirements (NGAM)

**Unit conversion chain**

Fluxes from FBA (mmol/g_DCW/h) are converted to molecule count changes:

    delta_counts = stochasticRound(
        flux * dry_mass * dt / (molecular_weight * conversion_factor)
    )

The conversion constants defined at module level:
    - COUNTS_UNITS = mmol
    - VOLUME_UNITS = L
    - MASS_UNITS = g
    - TIME_UNITS = s
    - CONC_UNITS = mmol/L  (concentration)
    - GDCW_BASIS = mmol/g/h  (FBA flux basis)

**ppGpp regulation** (optional)

When include_ppgpp=True, ppGpp concentration reduces the growth rate
objective, coupling transcriptional regulation to metabolic capacity.

**Kinetic constraints** (optional)

When USE_KINETICS=True, amino acid supply rates from polypeptide
elongation are used as additional flux constraints, coupling
translation demand to metabolic output.

NOTE:
- Metabolism runs after all other processes have completed and internal
  states have been updated (deriver-like, no partitioning necessary)
"""

from typing import Any, Optional
import warnings

from bigraph_schema.contract import ProcessContract
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from v2ecoli.library.ecoli_step import EcoliStep as Step
# topology_registry removed
from v2ecoli.library.fba_fast import extract_results, precompute_indices
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.library.unit_bridge import unum_magnitude_in, unum_to_pint, pint_to_unum
from wholecell.utils.random import stochasticRound
from wholecell.utils.modular_fba import FluxBalanceAnalysis
import pint
Quantity = pint.Quantity  # type alias for annotations
REVERSE_TAG = " (reverse)"


# Register default topology for this process, associating it with process name
NAME = "ecoli-metabolism"
TOPOLOGY = {
    "bulk": ("bulk",),
    "bulk_total": ("bulk",),
    "listeners": ("listeners",),
    "environment": ("environment",),
    "boundary": ("boundary",),
    "polypeptide_elongation": ("process_state", "polypeptide_elongation"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "metabolism"),
    # FBA-bridge: Millard-ODE-derived hard flux pins written by the coupler at
    # the agent root as {fba_reaction_id: flux_bound}.
    "pinned_flux_targets": ("pinned_flux_targets",),
    # Externally-imposed reaction flux bounds written by an injected subsystem
    # at the agent root as {reaction_id: {"upper_bound"?: v, "lower_bound"?: v}}.
    "imposed_flux_bounds": ("imposed_flux_bounds",),
}

# Unit conversion constants for FBA flux -> molecule count conversion:
#   flux (mmol/gDCW/h) * dry_mass (g) * dt (s) / (3600 s/h) -> mmol
#   mmol * N_A -> molecule count
COUNTS_UNITS = units.mmol          # internal amount basis
VOLUME_UNITS = units.L             # volume basis
MASS_UNITS = units.g               # mass basis (for dry cell weight)
TIME_UNITS = units.s               # simulation time basis
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS   # mmol/L = mM
CONVERSION_UNITS = MASS_UNITS * TIME_UNITS / VOLUME_UNITS  # g*s/L


def _mM_magnitude(value):
    """External concentrations are declared ``float[mM]`` in the ports schema and
    are compared here against the plain-float ``import_constraint_threshold``.
    The amino-acid-supplemented media path (e.g. ``basal_with_trp``) can leave a
    pint ``Quantity`` in ``boundary.external`` instead of a bare float, and pint
    refuses ``Quantity > float`` — the ``metabolism.py:841`` crash that fails the
    ``_with_trp`` arm while plain ``minimal`` (no AA keys) never hits this path.
    Return the mM magnitude whether the stored value is already a float or a
    unit-carrying Quantity, so the availability test is unit-safe either way.
    Mirrors the ``hasattr(q, "magnitude")`` idiom used elsewhere in this module.
    """
    return value.to("mM").magnitude if hasattr(value, "magnitude") else value
GDCW_BASIS = units.mmol / units.g / units.h  # FBA flux units

USE_KINETICS = True


def apply_flux_pins_with_fallback(fba, pins, *, set_hard_bound, open_bounds,
                                  set_soft_target, is_feasible):
    """Hard-pin each reaction; relax to a soft target any whose hard pin makes
    the LP infeasible. Returns the list of relaxed reaction ids.

    The fba-mutating operations are injected as callables so this logic is
    unit-testable without a real FBA solver and API-agnostic at the call site:

        set_hard_bound(rid, v) -> pin reaction `rid` to flux exactly `v`
        open_bounds(rid)       -> un-pin `rid` (restore its default lb/ub)
        set_soft_target(rid, v)-> set a soft kinetic target of `v` for `rid`
        is_feasible()          -> True iff the LP solves with the current bounds

    Relaxation is LIFO: the most-recently-added pin is relaxed first, since the
    pin order reflects the coupler's priority (earlier == higher priority).
    """
    if not pins:
        return []
    for rid, v in pins.items():
        set_hard_bound(rid, v)
    relaxed = []
    while not is_feasible() and len(relaxed) < len(pins):
        candidates = [r for r in pins if r not in relaxed]
        if not candidates:
            break
        rid = candidates[-1]            # LIFO: relax the most-recently-added
        open_bounds(rid)
        set_soft_target(rid, pins[rid])
        relaxed.append(rid)
    return relaxed


def is_carbon_starved(enabled, carbon_source_ids, importable):
    """Substrate-exhaustion detection (#572).

    True when the arrest is enabled AND at least one carbon source is configured
    AND none of those carbon sources is importable this tick (the exchange gate
    has closed on all of them). Pure/​unit-testable.

    Args:
        enabled: the ``carbon_exhaustion_arrest`` opt-in flag.
        carbon_source_ids: media carbon-source ids as they appear in the gate
            (e.g. ``{"GLC[p]"}``). Empty -> never starved (returns False).
        importable: the set of molecule ids importable this tick
            (unconstrained ∪ rate-constrained).
    """
    if not (enabled and carbon_source_ids):
        return False
    return not (set(carbon_source_ids) & set(importable))


def arrest_monomer_supply(delta_metabolites_final, monomer_mask):
    """Zero the NET SUPPLY (positive deltas) of biomass-monomer metabolites so
    the cell cannot polymerize new dry mass at carbon exhaustion (#572).

    Only positive deltas at ``monomer_mask`` positions are zeroed; consumption
    (negative deltas) and every non-monomer metabolite are untouched. Returns
    the same array object when nothing changed (so callers can detect a no-op),
    else a modified copy. Pure/​unit-testable.
    """
    supply = monomer_mask & (delta_metabolites_final > 0)
    if not supply.any():
        return delta_metabolites_final
    out = delta_metabolites_final.copy()
    out[supply] = 0
    return out


class Metabolism(Step):
    """Metabolism Process

    Encodes molecular simulation of microbial metabolism using FBA.
    Runs as a time-driven process (not partitioned).
    """

    # Formal description (standardized via Edge.describe()).
    description = (
        "Metabolism — flux-balance analysis (FBA).\n\n"
        "Each timestep solve for the reaction flux vector v:\n"
        "    max cᵀv   s.t.  S·v = 0,   v_lb ≤ v ≤ v_ub\n"
        "  S: stoichiometry (metabolites×reactions); c: biomass objective;\n"
        "  bounds from nutrient uptake, enzyme kcat capacity, NGAM maintenance.\n"
        "The FBA returns per-metabolite concentration changes Δc (mmol/L, the\n"
        "output_molecule_changes). Flux→counts uses counts_to_molar = 1/(N_A·V_cell)\n"
        "(the mM-per-count factor); its inverse maps Δc back to a count change:\n"
        "    Δn = stochasticRound(n₀ + Δc/counts_to_molar) − n₀   (zero-floored),\n"
        "  i.e. Δc·N_A·V_cell counts added to the initial count n₀. Exchange\n"
        "  fluxes are reported on a gDCW basis via κ = m_dry/m_cell·density·dt.\n"
        "Optional ppGpp growth coupling and kinetic (kcat) flux constraints."
    )

    contract = ProcessContract(
        summary=(
            "Flux-balance analysis (FBA): each step maximize the biomass "
            "objective subject to steady-state mass balance and flux bounds "
            "from nutrient uptake, enzyme kcat capacity, and NGAM maintenance; "
            "convert the resulting fluxes to bulk molecule-count changes. Runs "
            "deriver-like after all other processes (not partitioned)."
        ),
        symbols={
            "c": "FBA objective coefficients — biomass production weights (dimensionless)",
            "v": "reaction flux vector solved by the LP (mmol/g_DCW/h)",
            "S": "stoichiometric matrix, metabolites × reactions (dimensionless)",
            "v_lb": "lower flux bounds (mmol/g_DCW/h): uptake, kcat capacity, NGAM maintenance",
            "v_ub": "upper flux bounds (mmol/g_DCW/h)",
            "Δn": "per-metabolite molecule-count change applied to bulk this step (count)",
            "Δc": "FBA per-metabolite concentration change this step (output_molecule_changes, mmol/L)",
            "n₀": "initial per-metabolite count before the step, from bulk (count)",
            "counts_to_molar": "count→concentration factor = 1/(N_A·V_cell) (mM per count); its inverse (N_A·V_cell) converts Δc to a count change",
            "N_A": "Avogadro constant, from config avogadro (1/mol)",
            "V_cell": "cell volume = cell_mass/cell_density (L)",
            "m_dry": "cell dry mass, from listeners.mass.dry_mass (g); feeds the exchange-flux coefficient κ",
            "dt": "simulation timestep (s)",
            "κ": "exchange flux↔gDCW-basis coefficient = m_dry/m_cell·density·dt (g·s/L); used for exchange-flux reporting and the NGAM bound, NOT the metabolite count conversion",
            "NGAM": "non-growth-associated maintenance ATP flux, pinned as a bound (mmol/g/h)",
            "ppGpp": "guanosine tetraphosphate; when include_ppgpp, an objective target coupling regulation to growth (mM)",
        },
        inputs={
            "bulk": "Reads metabolite, catalyst, and kinetic enzyme/substrate counts → internal concentrations, catalyst reaction upper bounds, and kinetic targets; writes back the FBA-derived Δmetabolite counts.",
            "bulk_total": "Reads total (pre-partition) amino-acid counts to update tRNA-charging concentration targets.",
            "listeners": "Reads mass.cell_mass/dry_mass (cell volume, counts_to_molar, flux↔conc coefficient κ) and rna_mass/protein_mass (RNA:protein ratio for biomass targets); writes fba_results, enzyme_kinetics, and fba_bridge diagnostics.",
            "environment": "Reads media_id and exchange_data (constrained max uptake rates + unconstrained importers) to set external molecule availability for the LP.",
            "boundary": "Reads external[aa] concentrations to decide which amino acids are present in the media (mechanistic AA-transport gating).",
            "polypeptide_elongation": "Reads gtp_to_hydrolyze (translation GTP maintenance flux), aa_count_diff (AA supply/consumption for target updates), and aa_exchange_rates (mechanistic AA uptake rates).",
            "pinned_flux_targets": "Reads {fba_reaction_id: flux} hard pins from the Millard-ODE coupler; pins those bounds before the solve and relaxes any pin that makes the LP infeasible.",
            "global_time": "Clock read used to gate the update and stamp next_update_time.",
            "timestep": "Integration/flux-to-count timestep dt (s).",
            "next_update_time": "Scheduled next-run time gating update_condition.",
        },
        outputs={
            "bulk": "Writes stochastically-rounded, zero-floored Δmetabolite counts from the FBA output_molecule_changes.",
            "environment": "Writes exchange = per-nutrient molecule-count delta exchanged with the environment store.",
            "listeners": "Writes fba_results (objective value, reaction/exchange fluxes, shadow prices, reduced costs, conc/target updates), enzyme_kinetics (actual/target fluxes, counts_to_molar), and fba_bridge.relaxed_reactions.",
            "next_update_time": "Writes global_time + timestep for the next scheduled solve.",
        },
        config={
            "include_ppgpp": "If True, add ppGpp as a homeostatic concentration target coupling regulation to the growth objective.",
            "mechanistic_aa_transport": "If True, apply mechanistic amino-acid uptake constraints from elongation aa_exchange_rates.",
            "use_trna_charging": "If True, update amino-acid concentration targets from tRNA-charging supply/demand each step.",
            "ngam": "Non-growth-associated maintenance ATP flux (mmol/g/h), pinned as the maintenance reaction bound.",
            "dark_atp": "Growth-associated maintenance ATP (mmol/g); GAM = dark_atp × cell_dry_mass_fraction.",
            "cell_dry_mass_fraction": "Dry fraction of cell mass; multiplies dark_atp to give the GAM maintenance cost.",
            "cell_density": "Cell density (g/L); cell volume = cell_mass/density → counts_to_molar and the flux↔conc coefficient.",
            "doubling_time": "Baseline doubling time setting the initial biomass concentration targets.",
            "nutrientToDoublingTime": "Map media_id → doubling time, selecting media-specific biomass concentration targets.",
            "avogadro": "Avogadro number for counts↔moles conversion.",
            "import_constraint_threshold": "External concentration above which a nutrient is treated as available for uptake.",
            "removed_aa_uptake": "Amino acids whose environmental uptake is disabled.",
            "media_id": "Default media id used when the environment media is unmapped.",
            "base_reaction_ids": "Base (lumped) reaction ids; the FBA→base mapping aggregates split/reverse fluxes for reporting.",
            "fba_reaction_ids_to_base_reaction_ids": "Map each FBA reaction (incl. reverse) to its base reaction, with sign, for base_reaction_fluxes.",
        },
        assumptions=[
            "Steady-state mass balance (S·v = 0) is imposed each timestep: metabolite pools are quasi-steady relative to dt.",
            "Runs deriver-like after all other processes update internal state; not partitioned (no request/allocate arbitration).",
            "The FBA returns per-metabolite concentration changes (mmol/L); these are converted to counts via counts_to_molar = 1/(N_A·V_cell), then stochastically rounded and floored at zero. Exchange fluxes use the separate κ = m_dry/m_cell·density·dt gDCW-basis coefficient.",
            "Enzyme capacity is modeled by catalyst presence: a reaction with no catalyst present gets an upper flux bound of zero.",
            "Amino-acid uptake is capped to the homeostatic concentration need to prevent AAs acting as an unbounded carbon/nitrogen source.",
            "Objective is homeostatic, or mixed homeostatic+kinetic when kcat constraints are enabled (kinetic_objective_weight > 0).",
        ],
        references=[
            "wcEcoli whole-cell model FBA (wholecell.utils.modular_fba.FluxBalanceAnalysis)",
            "Homeostatic + kinetic mixed FBA objective, Covert lab E. coli whole-cell model",
        ],
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'aa_exchange_names': {'_type': 'list[string]', '_default': []},
        'aa_names': {'_type': 'list[string]', '_default': []},
        'aa_targets_not_updated': {'_type': 'node', '_default': set()},
        'amino_acid_ids': {'_type': 'map', '_default': {}},
        'avogadro': {'_type': 'quantity[float,1/mol]', '_default': 6.02214076e+23},
        'base_reaction_ids': {'_type': 'list[string]', '_default': []},
        # Substrate-exhaustion growth arrest (#572). OPT-IN (default off) so the
        # validated fed regime is byte-identical. When enabled, if none of
        # ``carbon_source_ids`` is importable this tick (the exchange gate has
        # closed on the carbon source), metabolism stops SUPPLYING net biomass
        # monomers (amino acids + (d)NTPs) so the cell cannot build biomass from
        # phantom internal carbon — it arrests instead of growing at zero carbon.
        # ``carbon_source_ids`` MUST list the media's carbon source(s) as they
        # appear in the exchange gate (e.g. ``["GLC[p]"]`` for M9 glucose); left
        # empty the arrest never triggers.
        'carbon_exhaustion_arrest': {'_type': 'boolean', '_default': False},
        'carbon_source_ids': {'_type': 'list[string]', '_default': []},
        'cell_density': {'_type': 'quantity[g/L]', '_default': 1100.0},
        'cell_dry_mass_fraction': {'_type': 'float', '_default': 0.3},
        'dark_atp': {'_type': 'quantity[float,mmol/g]', '_default': 33.565052868380675},
        'doubling_time': {'_type': 'quantity[float,min]', '_default': 44.0},
        'exchange_data_from_media': {'_type': 'method', '_default': None},
        'exchange_molecules': {'_type': 'list[string]', '_default': []},
        'fba_reaction_ids_to_base_reaction_ids': {'_type': 'list[string]', '_default': []},
        'get_biomass_as_concentrations': {'_type': 'method', '_default': None},
        'get_import_constraints': {'_type': 'method', '_default': None},
        'get_masses': {'_type': 'method', '_default': None},
        'get_ppGpp_conc': {'_type': 'method', '_default': None},
        'import_constraint_threshold': {'_type': 'integer', '_default': 0},
        'imports': {'_type': 'map[node]', '_default': {}},
        'include_ppgpp': {'_type': 'boolean', '_default': False},
        'mechanistic_aa_transport': {'_type': 'boolean', '_default': False},
        'media_id': {'_type': 'string', '_default': 'minimal'},
        'metabolism': {'_type': 'node', '_default': None},
        'ngam': {'_type': 'quantity[float,mmol/g/h]', '_default': 8.39},
        'nutrientToDoublingTime': {'_type': 'map[quantity[float,min]]', '_default': {}},
        'ppgpp_id': {'_type': 'string', '_default': 'ppgpp'},
        'removed_aa_uptake': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer', '_default': 1},
        'use_trna_charging': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'bulk_total': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                    'dry_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                    'rna_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                    'protein_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                },
            },
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
                'exchange_data': {
                    'constrained': 'overwrite[map[float]]',
                    'unconstrained': 'overwrite[list[string]]',
                },
            },
            # `external` leaves are declared PER-LEAF overwrite, not
            # `overwrite[map[...]]`. The distinction is load-bearing and the
            # opposite of the sibling `exchange_data` declaration above:
            #   - exchange_data WANTS whole-map replace. Dropping a key is how
            #     "max flux 0" is expressed (see exchange_constraints), so
            #     `overwrite[map[...]]` is correct there.
            #   - boundary.external must KEEP every molecule. Its only two
            #     writers (EnvironmentMirror, MediaUpdate) emit PARTIAL dicts,
            #     so a whole-map replace would delete every molecule the writer
            #     did not mention -- the cell would silently lose its media.
            # Per-leaf overwrite also makes an `inf` boundary WRITABLE. Bare
            # float leaves accumulate (`state + update`), and no delta can move
            # `inf`: `inf + -inf` is NaN, which is why driver control of
            # dissolved O2 was unreachable (#566).
            'boundary': {'external': 'map[overwrite[float[mM]]]'},
            'polypeptide_elongation': {
                'gtp_to_hydrolyze': {'_type': 'float', '_default': 0.0},  # count, dimensionless
                'aa_count_diff': {'_type': 'array[float]', '_default': []},  # count change per AA species
                'aa_exchange_rates': {'_type': 'array[float[mmol/g/h]]', '_default': []},
            },
            'global_time': {'_type': 'float[s]', '_default': 0.0},
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'next_update_time': {'_type': 'float[s]', '_default': 1.0},
            # FBA-bridge: {fba_reaction_id: flux_bound} hard pins from the
            # Millard ODE coupler. Absent/empty -> no pins -> no-op.
            'pinned_flux_targets': {'_type': 'map[float]', '_default': {}},
            # Externally-imposed reaction flux bounds from an injected subsystem
            # as {reaction_id: {"upper_bound"?: v, "lower_bound"?: v}}. Drug- and
            # mechanism-agnostic; absent/empty -> no-op. See _apply_imposed_bounds.
            'imposed_flux_bounds': {'_type': 'map[map[float]]', '_default': {}},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'environment': {'exchange': 'map[float]'},
            'listeners': {
                'fba_results': {
                    # Coefficient for flux→delta conversion (g*s/L)
                    'coefficient': {'_type': 'overwrite[float[g*s/L]]', '_default': 0.0},
                    # GTP from polypeptide elongation (count, dimensionless)
                    'translation_gtp': {'_type': 'overwrite[float]', '_default': 0.0},
                    # Concentration updates per molecule (mM = mmol/L)
                    'conc_updates': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
                    # Homeostatic target concentrations (mM)
                    'target_concentrations': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
                    # FBA solver outputs (mostly mmol/g/h flux units, but
                    # stored without units in the listener history)
                    'reaction_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'external_exchange_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'base_reaction_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'objective_value': {'_type': 'overwrite[float]', '_default': 0.0},
                    'shadow_prices': {'_type': 'overwrite[array[float]]', '_default': []},
                    'reduced_costs': {'_type': 'overwrite[array[float]]', '_default': []},
                    'homeostatic_objective_values': {'_type': 'overwrite[array[float]]', '_default': []},
                    'kinetic_objective_values': {'_type': 'overwrite[array[float]]', '_default': []},
                    # Counts (dimensionless)
                    'catalyst_counts': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'delta_metabolites': {'_type': 'overwrite[array[integer]]', '_default': []},
                    # Identifiers and constraint sets — flexible
                    'media_id': {'_type': 'overwrite[string]', '_default': ''},
                    'unconstrained_molecules': {'_type': 'overwrite[list[string]]', '_default': []},
                    'constrained_molecules': {'_type': 'overwrite[map[float]]', '_default': {}},
                    'uptake_constraints': {'_type': 'overwrite[array[float]]', '_default': []},
                },
                'enzyme_kinetics': {
                    # Counts→molar conversion (mmol/L = mM)
                    'counts_to_molar': {'_type': 'overwrite[float[mM]]', '_default': 1.0},
                    # Counts (dimensionless)
                    'metabolite_counts_init': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'metabolite_counts_final': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'enzyme_counts_init': {'_type': 'overwrite[array[integer]]', '_default': []},
                    # Fluxes (mmol/L/s) — leave plain for now since
                    # the listener stores per-timestep values, not rates
                    'actual_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'target_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'target_fluxes_upper': {'_type': 'overwrite[array[float]]', '_default': []},
                    'target_fluxes_lower': {'_type': 'overwrite[array[float]]', '_default': []},
                    'target_aa_conc': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
                },
                # FBA-bridge diagnostics: which pinned reactions had their hard
                # pin relaxed to a soft target because the pin made the LP
                # infeasible this tick.
                'fba_bridge': {
                    'relaxed_reactions': {'_type': 'overwrite[list[string]]', '_default': []},
                },
            },
            'next_update_time': 'overwrite[float]',
        }



    def initialize(self, config):

        # Use information from the environment and sim
        self.get_import_constraints = self.parameters["get_import_constraints"]
        self.nutrientToDoublingTime = self.parameters["nutrientToDoublingTime"]
        self.use_trna_charging = self.parameters["use_trna_charging"]
        self.include_ppgpp = self.parameters["include_ppgpp"]
        self.mechanistic_aa_transport = self.parameters["mechanistic_aa_transport"]
        self.current_timeline = self.parameters["current_timeline"]
        self.media_id = self.parameters["media_id"]
        self.exchange_molecules = self.parameters["exchange_molecules"]
        self.environment_molecules = sorted(
            [mol[:-3] for mol in self.exchange_molecules]
        )

        # Create model to use to solve metabolism updates
        self.model = FluxBalanceAnalysisModel(
            self.parameters,
            timeline=self.current_timeline,
            include_ppgpp=self.include_ppgpp,
        )

        self.nAvogadro = self.parameters["avogadro"]
        self.cellDensity = self.parameters["cell_density"]
        # GDCW_BASIS is a module-level constant — convert once at init
        # rather than every tick (called from update() into
        # get_import_constraints, an upstream-Unum-native API).
        self._gdcw_basis_unum = pint_to_unum(GDCW_BASIS)
        # Module-level Unit composites — pint constructs the composite
        # `units.mmol / units.g / units.h` afresh on every multiplication.
        # Build them once at init; the unit instances are immutable and
        # safe to share across ticks.
        self._constraint_unit = units.mmol / units.g / units.h
        self._fg_unit = units.fg
        # Pre-strip the dimensional constants to magnitudes in documented
        # unit bases, used by _do_update for pure-float arithmetic. The
        # pint Quantity copies above remain for the upstream-Unum boundary
        # functions (exchange_constraints, get_import_constraints,
        # get_kinetic_constraints) that demand them — those are the only
        # places we rebuild Quantities.
        #
        # Dimensional bookkeeping (see docs/refactors/metabolism_magnitude_arithmetic.md):
        #   self._n_avogadro_per_mol  = 6.022e23 (1/mol)
        #   self._cell_density_g_per_L = 1100.0  (g/L)
        self._n_avogadro_per_mol = float(self.nAvogadro.magnitude)
        self._cell_density_g_per_L = float(self.cellDensity.magnitude)

        # Track updated AA concentration targets with tRNA charging
        self.aa_targets = {}
        self.aa_targets_not_updated = self.parameters["aa_targets_not_updated"]
        self.aa_names = self.parameters["aa_names"]
        # Store as plain float (boundary.external values are plain floats in mM)
        self.import_constraint_threshold = float(
            self.parameters["import_constraint_threshold"]
        )

        # Molecules with concentration updates for listener
        self.linked_metabolites = self.parameters["linked_metabolites"]
        doubling_time = self.nutrientToDoublingTime.get(
            self.media_id, self.nutrientToDoublingTime["minimal"]
        )
        update_molecules = list(
            self.model.getBiomassAsConcentrations(doubling_time).keys()
        )
        if self.use_trna_charging:
            update_molecules += [
                aa for aa in self.aa_names if aa not in self.aa_targets_not_updated
            ]
            update_molecules += list(self.linked_metabolites.keys())
        if self.include_ppgpp:
            update_molecules += [self.model.ppgpp_id]
        self.conc_update_molecules = sorted(update_molecules)

        self.aa_exchange_names = self.parameters["aa_exchange_names"]
        self.removed_aa_uptake = self.parameters["removed_aa_uptake"]
        self.aa_environment_names = [aa[:-3] for aa in self.aa_exchange_names]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.metabolite_idx = None

        # Get conversion matrix to compile individual fluxes in the FBA
        # solution to the fluxes of base reactions
        self.fba_reaction_ids = self.model.fba.getReactionIDs()
        self.base_reaction_ids = self.parameters["base_reaction_ids"]
        fba_reaction_ids_to_base_reaction_ids = self.parameters[
            "fba_reaction_ids_to_base_reaction_ids"
        ]
        self.externalMoleculeIDs = self.model.fba.getExternalMoleculeIDs()
        self.outputMoleculeIDs = self.model.fba.getOutputMoleculeIDs()
        self.kineticTargetFluxNames = self.model.fba.getKineticTargetFluxNames()
        self.homeostaticTargetMolecules = self.model.fba.getHomeostaticTargetMolecules()

        # --- Substrate-exhaustion growth arrest (#572, opt-in) --------------
        # See config_schema. When enabled and the carbon source is not importable
        # this tick, _do_update suppresses NET SUPPLY of biomass monomers so the
        # cell arrests rather than building biomass from phantom internal carbon.
        self.carbon_exhaustion_arrest = bool(
            self.parameters.get("carbon_exhaustion_arrest", False))
        self.carbon_source_ids = set(
            self.parameters.get("carbon_source_ids", []) or [])
        # Mask over the metabolite-delta array (which is ordered by
        # ``metaboliteNamesFromNutrients``, same as ``metabolite_counts_init``)
        # selecting biomass MONOMERS — amino acids + (deoxy)nucleotides, the
        # polymerization substrates translation / transcription / replication
        # draw on to build dry mass.
        def _strip_loc(mid):
            return str(mid).split("[")[0]
        _monomer_names = {_strip_loc(a) for a in self.aa_names} | {
            "ATP", "GTP", "CTP", "UTP", "DATP", "DGTP", "DCTP", "TTP", "DTTP"}
        self._biomass_monomer_mask = np.array(
            [_strip_loc(n) in _monomer_names
             for n in self.model.metaboliteNamesFromNutrients], dtype=bool)
        fba_reaction_id_to_index = {
            rxn_id: i for (i, rxn_id) in enumerate(self.fba_reaction_ids)
        }
        base_reaction_id_to_index = {
            rxn_id: i for (i, rxn_id) in enumerate(self.base_reaction_ids)
        }
        base_rxn_indexes = []
        fba_rxn_indexes = []
        v = []

        for fba_rxn_id in self.fba_reaction_ids:
            base_rxn_id = fba_reaction_ids_to_base_reaction_ids[fba_rxn_id]
            base_rxn_indexes.append(base_reaction_id_to_index[base_rxn_id])
            fba_rxn_indexes.append(fba_reaction_id_to_index[fba_rxn_id])
            if fba_rxn_id.endswith(REVERSE_TAG):
                v.append(-1)
            else:
                v.append(1)

        base_rxn_indexes = np.array(base_rxn_indexes)
        fba_rxn_indexes = np.array(fba_rxn_indexes)
        v = np.array(v)
        shape = (len(self.base_reaction_ids), len(self.fba_reaction_ids))

        self.reaction_mapping_matrix = csr_matrix(
            (v, (base_rxn_indexes, fba_rxn_indexes)), shape=shape
        )

        # Pre-resolve name → solver-index mappings. Replaces per-tick
        # list-comps over reactionIDs / externalExchangeIDs / shadow IDs
        # in upstream getReducedCosts / getReactionFluxes / getShadowPrices
        # with single numpy slices into the cached col_primals / col_duals
        # / row_duals vectors. See v2ecoli/library/fba_fast.py.
        self.fba_extract_indices = precompute_indices(
            self.model.fba,
            self.model.metaboliteNamesFromNutrients,
            self.model.kinetics_constrained_reactions,
        )

    def __getstate__(self):
        return self.parameters

    def __setstate__(self, state):
        self.__init__(state)
    def update_condition(self, timestep, states):
        """
        See :py:meth:`~ecoli.processes.partition.Requester.update_condition`.
        """
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep for the global clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def update(self, states, interval=None):
        timestep = states.get('timestep', 1)
        return self._do_update(timestep, states)

    def _init_bulk_indices(self, bulk_ids):
        """One-time molecule-name -> bulk-array-index resolution (cold path,
        guarded by ``self.metabolite_idx is None``)."""
        self.metabolite_idx = bulk_name_to_idx(
            self.model.metaboliteNamesFromNutrients, bulk_ids)
        self.catalyst_idx = bulk_name_to_idx(self.model.catalyst_ids, bulk_ids)
        self.kinetics_enzymes_idx = bulk_name_to_idx(
            self.model.kinetic_constraint_enzymes, bulk_ids)
        self.kinetics_substrates_idx = bulk_name_to_idx(
            self.model.kinetic_constraint_substrates, bulk_ids)
        self.aa_idx = bulk_name_to_idx(self.aa_names, bulk_ids)

    def _fba_output_to_deltas(self, fba_out, metabolite_counts_init,
                              counts_to_molar, coefficient, timestep):
        """Convert the FBA solution into cell-state changes.

        The FBA returns concentration changes / exchange fluxes (in CONC_UNITS,
        i.e. mmol/L per timestep). This maps them back to molecule counts:

          * internal metabolites: delta_count = (1/counts_to_molar) * dconc,
            stochastically rounded and floored at zero;
          * environmental exchange: delta_count = (1/counts_to_molar) * flux
            (the cumulative count added to the environment store);
          * exchange fluxes also reported in gDCW basis (flux/coefficient).

        Returns (delta_metabolites_final, metabolite_counts_final,
        delta_nutrients, converted_exchange_fluxes, reaction_fluxes).
        """
        delta_metabolites = (1 / counts_to_molar) * (
            CONC_UNITS * fba_out["output_molecule_changes"]
        )
        metabolite_counts_final = np.fmax(
            stochasticRound(
                self.random_state,
                metabolite_counts_init + delta_metabolites.magnitude,
            ),
            0,
        ).astype(np.int64)
        delta_metabolites_final = metabolite_counts_final - metabolite_counts_init

        exchange_fluxes = CONC_UNITS * fba_out["external_exchange_fluxes"]
        converted_exchange_fluxes = (
            exchange_fluxes / coefficient).to(GDCW_BASIS).magnitude
        delta_nutrients = (
            ((1 / counts_to_molar) * exchange_fluxes).magnitude.astype(int)
        )
        reaction_fluxes = fba_out["reaction_fluxes"] / timestep
        return (delta_metabolites_final, metabolite_counts_final,
                delta_nutrients, converted_exchange_fluxes, reaction_fluxes)

    def _apply_imposed_bounds(self, fba, imposed_flux_bounds):
        """Apply externally-imposed reaction flux bounds supplied by an injected
        subsystem via the ``imposed_flux_bounds`` store, before the LP solve.

        Drug- and mechanism-agnostic: this process only APPLIES the bounds it is
        handed; the imposing subsystem (e.g. an antibiotic layer in a downstream
        package) owns which reaction and what value, and computes them from its
        own state. This keeps ecoli-metabolism free of any drug-specific
        knowledge. An empty/absent store is a no-op, leaving the LP identical.

        Each entry is ``{reaction_id: {"upper_bound"?: float, "lower_bound"?:
        float}}`` (either key optional). Unknown reaction ids are skipped with a
        warning. ``raiseForReversible=False`` matches the pin path's semantics.
        """
        if not imposed_flux_bounds:
            return
        valid_ids = getattr(self, "_pin_valid_reaction_ids", None)
        if valid_ids is None:
            valid_ids = set(fba.getReactionIDs().tolist())
            self._pin_valid_reaction_ids = valid_ids
        for rid, bounds in imposed_flux_bounds.items():
            if rid not in valid_ids:
                print(f"Warning: ignoring imposed flux bound for unknown "
                      f"reaction '{rid}'")
                continue
            kwargs = {}
            if "upper_bound" in bounds:
                kwargs["upperBounds"] = float(bounds["upper_bound"])
            if "lower_bound" in bounds:
                kwargs["lowerBounds"] = float(bounds["lower_bound"])
            if kwargs:
                fba.setReactionFluxBounds(
                    rid, raiseForReversible=False, **kwargs)

    def _apply_flux_pins(self, fba, pinned_flux_targets):
        """Hard-pin each Millard-ODE-derived reaction flux before the LP solve,
        relaxing to a soft target any pin that makes the LP infeasible.

        Returns the list of relaxed reaction ids (written to
        ``listeners.fba_bridge.relaxed_reactions``). A missing/empty
        ``pinned_flux_targets`` is a no-op, leaving the LP identical to today.

        The real fba operations are bound here and injected into the
        unit-tested ``apply_flux_pins_with_fallback`` helper:

          * hard pin  -> ``setReactionFluxBounds(rid, lowerBounds=v, upperBounds=v)``
          * un-pin    -> reset to the WCM default bounds (lb=0, ub=inf)
          * feasibility -> attempt a single solve; infeasible raises (status
            code != GLP_OPT), which we catch
          * soft target -> ``setKineticTarget`` *only* for reactions that were
            initialized as kinetic targets; for other reactions no soft nudge
            is available, so opening the bounds alone restores feasibility.
        """
        if not pinned_flux_targets:
            return []

        # Drop pins for unknown reaction ids and clamp to the non-negative net
        # flux that setReactionFluxBounds requires (reverse flux is a separate
        # reaction id in this FBA formulation).
        valid_ids = getattr(self, "_pin_valid_reaction_ids", None)
        if valid_ids is None:
            valid_ids = set(fba.getReactionIDs().tolist())
            self._pin_valid_reaction_ids = valid_ids
        kinetic_target_ids = getattr(fba, "_kineticTargetFluxes", set())

        pins = {}
        for rid, v in pinned_flux_targets.items():
            if rid not in valid_ids:
                print(f"Warning: ignoring flux pin for unknown reaction '{rid}'")
                continue
            pins[rid] = max(float(v), 0.0)
        if not pins:
            return []

        def set_hard_bound(rid, v):
            fba.setReactionFluxBounds(
                rid, lowerBounds=v, upperBounds=v, raiseForReversible=False
            )

        def open_bounds(rid):
            fba.setReactionFluxBounds(
                rid, lowerBounds=0.0, upperBounds=np.inf, raiseForReversible=False
            )

        def set_soft_target(rid, v):
            # setKineticTarget only accepts reactions pre-initialized as
            # kinetic targets. For others the open-bounds un-pin above is the
            # whole relaxation; record but skip the (unavailable) soft nudge.
            if rid in kinetic_target_ids:
                fba.setKineticTarget(rid, v)
            else:
                print(
                    f"Warning: relaxed flux pin '{rid}' has no kinetic target "
                    "slot; un-pinned without a soft target"
                )

        def is_feasible():
            try:
                fba.solve(0)  # single attempt; raises if status != GLP_OPT
                return True
            except Exception:
                return False

        return apply_flux_pins_with_fallback(
            fba, pins,
            set_hard_bound=set_hard_bound,
            open_bounds=open_bounds,
            set_soft_target=set_soft_target,
            is_feasible=is_feasible,
        )

    def _do_update(self, timestep, states):
        # At t=0, resolve molecule-name -> bulk-index maps (runs once).
        if self.metabolite_idx is None:
            self._init_bulk_indices(states["bulk"]["id"])

        timestep = states["timestep"]

        # Load current state of the sim
        # Get internal state variables
        metabolite_counts_init = counts(states["bulk"], self.metabolite_idx)
        catalyst_counts = counts(states["bulk"], self.catalyst_idx)
        kinetic_enzyme_counts = counts(states["bulk"], self.kinetics_enzymes_idx)
        kinetic_substrate_counts = counts(states["bulk"], self.kinetics_substrates_idx)

        translation_gtp = states["polypeptide_elongation"]["gtp_to_hydrolyze"]
        cell_mass = as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        dry_mass = as_quantity(states["listeners"]["mass"]["dry_mass"], units.fg)

        # Calculate state values
        cellVolume = cell_mass / self.cellDensity
        counts_to_molar = (1 / (self.nAvogadro * cellVolume)).to(CONC_UNITS)

        # Coefficient to convert between flux (mol/g DCW/hr) basis and
        # concentration (M) basis
        coefficient = dry_mass / cell_mass * self.cellDensity * timestep * units.s

        # Get exchange constraints. The producer writes plain float magnitudes
        # and the store does not re-attach units (the resolved schema for this
        # leaf is `overwrite[...]`, whose apply replaces without descending), so
        # re-attach the known
        # mol/mass/time units (mmol/g/h) before crossing back into upstream
        # Unum-native code (exchange_constraints multiplies by an Unum
        # coefficient). Values arriving as plain floats become pint
        # Quantities; anything already unit-bearing is normalized through
        # the bridge.
        unconstrained = set(states["environment"]["exchange_data"]["unconstrained"])
        constraint_unit = self._constraint_unit
        constrained = {}
        for mol, val in states["environment"]["exchange_data"]["constrained"].items():
            q = unum_to_pint(val)
            if not hasattr(q, "magnitude"):
                q = q * constraint_unit
            constrained[mol] = q

        # Substrate-exhaustion arrest (#572, opt-in): carbon is exhausted when
        # none of the configured carbon sources is importable via the gate
        # (unconstrained or rate-constrained). Computed from the RAW gate here,
        # before get_import_constraints reshapes `unconstrained`/`constrained`.
        carbon_starved = is_carbon_starved(
            self.carbon_exhaustion_arrest,
            self.carbon_source_ids,
            unconstrained | set(constrained.keys()),
        )

        # Determine updates to concentrations depending on the current state
        current_media_id = states["environment"]["media_id"]
        doubling_time = self.nutrientToDoublingTime.get(
            current_media_id, self.nutrientToDoublingTime[self.media_id]
        )
        # getBiomassAsConcentrations / getppGppConc are upstream Unum-native;
        # they accept and return Unum quantities. Keep them Unum through the
        # merge below; a single fused conversion at the end goes Unum →
        # magnitude without building 60+ transient pint.Quantity objects.
        if self.include_ppgpp:
            conc_updates = self.model.getBiomassAsConcentrations(doubling_time)
            conc_updates[self.model.ppgpp_id] = self.model.getppGppConc(
                doubling_time)
        else:
            rp_ratio = (
                states["listeners"]["mass"]["rna_mass"]
                / states["listeners"]["mass"]["protein_mass"]
            )
            conc_updates = self.model.getBiomassAsConcentrations(
                doubling_time, rp_ratio=rp_ratio
            )

        if self.use_trna_charging:
            conc_updates.update(
                self.update_amino_acid_targets(
                    counts_to_molar,
                    dict(
                        zip(
                            self.aa_names,
                            states["polypeptide_elongation"]["aa_count_diff"],
                        )
                    ),
                    dict(zip(self.aa_names, counts(states["bulk_total"], self.aa_idx))),
                )
            )

        # Unum → magnitude in CONC_UNITS, direct (no intermediate pint
        # Quantity). Conversion factors are memoized by (src_unit,
        # CONC_UNITS) in ``_conversion_factor``, so every met with the
        # same source unit reuses one scalar multiplication.
        conc_updates = {
            met: unum_magnitude_in(conc, CONC_UNITS)
            for met, conc in conc_updates.items()
        }

        aa_uptake_package = None
        if self.mechanistic_aa_transport:
            aa_in_media = np.array(
                [
                    _mM_magnitude(states["boundary"]["external"][aa_name])
                    > self.import_constraint_threshold
                    for aa_name in self.aa_environment_names
                ]
            )
            aa_in_media[self.removed_aa_uptake] = False
            exchange_rates = (
                states["polypeptide_elongation"]["aa_exchange_rates"] * timestep
            )
            aa_uptake_package = (
                exchange_rates[aa_in_media],
                self.aa_exchange_names[aa_in_media],
                True,
            )

        # Update FBA problem based on current state
        # Set molecule availability (internal and external)
        self.model.set_molecule_levels(
            metabolite_counts_init,
            counts_to_molar,
            coefficient,
            current_media_id,
            unconstrained,
            constrained,
            conc_updates,
            aa_uptake_package,
        )

        # Set reaction limits for maintenance and catalysts present
        self.model.set_reaction_bounds(
            catalyst_counts, counts_to_molar, coefficient, translation_gtp
        )

        # Constrain reactions based on targets
        targets, upper_targets, lower_targets = self.model.set_reaction_targets(
            kinetic_enzyme_counts,
            kinetic_substrate_counts,
            counts_to_molar,
            timestep * units.s,
        )

        # Solve FBA problem and update states
        n_retries = 3
        fba = self.model.fba

        # Apply any externally-imposed reaction flux bounds supplied by an
        # injected subsystem via the imposed_flux_bounds store (drug- and
        # mechanism-agnostic; empty/absent store is a no-op).
        self._apply_imposed_bounds(fba, states.get("imposed_flux_bounds", {}))

        # FBA-bridge: hard-pin Millard-ODE-derived reaction fluxes (if any)
        # before solving; relax any pin that makes the LP infeasible.
        relaxed_reactions = self._apply_flux_pins(
            fba, states.get("pinned_flux_targets", {})
        )

        fba.solve(n_retries)

        # Pull all per-tick LP result arrays in one pass — batched numpy
        # slices into the solver's cached col_primals / col_duals / row_duals
        # vectors, plus a tight Python loop for the per-output-molecule sum
        # over the live _outputMoleculeCoeffs (mutated each tick).
        # See v2ecoli/library/fba_fast.py.
        fba_out = extract_results(fba, self.fba_extract_indices)

        # Map the FBA solution back to molecule-count changes
        # (see _fba_output_to_deltas for the unit conversions).
        (delta_metabolites_final, metabolite_counts_final, delta_nutrients,
         converted_exchange_fluxes, reaction_fluxes) = self._fba_output_to_deltas(
            fba_out, metabolite_counts_init, counts_to_molar, coefficient,
            timestep)

        # Substrate-exhaustion arrest (#572): with no importable carbon source,
        # suppress NET SUPPLY of biomass monomers (positive deltas of amino acids
        # + (d)NTPs) written to bulk, so translation/transcription/replication
        # cannot polymerize new dry mass from phantom internal carbon — the cell
        # arrests instead of growing at zero carbon. Only positive monomer deltas
        # are zeroed: consumption and all non-monomer (maintenance / energy)
        # metabolism are untouched. Inert unless opt-in AND carbon-starved, so
        # the validated fed regime is byte-identical.
        if carbon_starved:
            _clamped = arrest_monomer_supply(
                delta_metabolites_final, self._biomass_monomer_mask)
            if _clamped is not delta_metabolites_final:
                delta_metabolites_final = _clamped
                metabolite_counts_final = (
                    metabolite_counts_init + delta_metabolites_final)

        # get_import_constraints is upstream Unum-native; convert constrained
        # values back to Unum at the boundary. GDCW_BASIS is a constant —
        # use the converted instance cached in initialize().
        constrained_unum = {k: pint_to_unum(v) for k, v in constrained.items()}
        unconstrained, constrained, uptake_constraints = self.get_import_constraints(
            unconstrained, constrained_unum, self._gdcw_basis_unum
        )

        update = {
            "bulk": [(self.metabolite_idx, delta_metabolites_final)],
            "environment": {
                "exchange": {
                    str(molecule[:-3]): delta_nutrients[index]
                    for index, molecule in enumerate(self.externalMoleculeIDs)
                }
            },
            "listeners": {
                "fba_results": {
                    "media_id": current_media_id,
                    "conc_updates": [
                        conc_updates.get(m, 0) for m in self.conc_update_molecules
                    ],
                    "catalyst_counts": catalyst_counts,
                    "translation_gtp": translation_gtp,
                    "coefficient": coefficient.to(CONVERSION_UNITS).magnitude,
                    "unconstrained_molecules": unconstrained,
                    "constrained_molecules": constrained,
                    "uptake_constraints": uptake_constraints,
                    "delta_metabolites": delta_metabolites_final,
                    "reaction_fluxes": reaction_fluxes,
                    "external_exchange_fluxes": converted_exchange_fluxes,
                    "objective_value": fba.getObjectiveValue(),
                    "shadow_prices": fba_out["shadow_prices"],
                    "reduced_costs": fba_out["reduced_costs"],
                    "target_concentrations": [
                        self.model.homeostatic_objective[mol]
                        for mol in fba.getHomeostaticTargetMolecules()
                    ],
                    "homeostatic_objective_values": fba.getHomeostaticObjectiveValues(),
                    "kinetic_objective_values": fba.getKineticObjectiveValues(),
                    "base_reaction_fluxes": self.reaction_mapping_matrix.dot(
                        reaction_fluxes
                    ),
                },
                "enzyme_kinetics": {
                    "metabolite_counts_init": metabolite_counts_init,
                    "metabolite_counts_final": metabolite_counts_final,
                    "enzyme_counts_init": kinetic_enzyme_counts,
                    "counts_to_molar": counts_to_molar.to(CONC_UNITS).magnitude,
                    "actual_fluxes": fba_out["kinetics_constrained_fluxes"] / timestep,
                    "target_fluxes": targets / timestep,
                    "target_fluxes_upper": upper_targets / timestep,
                    "target_fluxes_lower": lower_targets / timestep,
                    "target_aa_conc": [
                        self.aa_targets.get(id_, 0.0) for id_ in self.aa_names
                    ],
                },
                "fba_bridge": {
                    "relaxed_reactions": relaxed_reactions,
                },
            },
            "next_update_time": states["global_time"] + states["timestep"],
        }

        return update

    def update_amino_acid_targets(
        self,
        counts_to_molar: Quantity,
        count_diff: dict[str, float],
        amino_acid_counts: dict[str, float],
    ) -> dict[str, Quantity]:
        """
        Finds new amino acid concentration targets based on difference in
        supply and number of amino acids used in polypeptide_elongation.
        Skips updates to molecules defined in self.aa_targets_not_updated:
        - L-SELENOCYSTEINE: rare AA that led to high variability when updated

        Args:
            counts_to_molar: conversion from counts to molar

        Returns:
            ``{AA name (str): new target AA conc (float with mol/volume units)}``
        """

        if len(self.aa_targets):
            for aa, diff in count_diff.items():
                if aa in self.aa_targets_not_updated:
                    continue
                self.aa_targets[aa] += diff
                # TODO (Santiago): Improve targets update
                if self.aa_targets[aa] < 0:
                    print(
                        "Warning: updated amino acid target for "
                        f"{aa} was negative - adjusted to be positive."
                    )
                    self.aa_targets[aa] = 1.0

        # First time step of a simulation so set target to current counts to
        # prevent concentration jumps between generations
        else:
            for aa, counts in amino_acid_counts.items():
                if aa in self.aa_targets_not_updated:
                    continue
                self.aa_targets[aa] = float(counts)

        conc_updates = {
            aa: counts * counts_to_molar for aa, counts in self.aa_targets.items()
        }

        # Update linked metabolites that will follow an amino acid
        for met, link in self.linked_metabolites.items():
            conc_updates[met] = (
                conc_updates.get(link["lead"], 0 * counts_to_molar) * link["ratio"]
            )

        return conc_updates


class FluxBalanceAnalysisModel(object):
    """
    Metabolism model that solves an FBA problem with modular_fba.
    """

    def __init__(
        self,
        parameters: dict[str, Any],
        timeline: tuple[tuple[int, str], ...],
        include_ppgpp: bool = True,
    ):
        """
        Args:
            parameters: parameters from simulation data
            timeline: timeline for nutrient changes during simulation
                (time of change, media ID), by default [(0.0, 'minimal')]
            include_ppgpp: if True, ppGpp is included as a concentration target
        """
        nutrients = timeline[0][1]

        # Local sim_data references
        metabolism = parameters["metabolism"]
        self.stoichiometry = metabolism.reaction_stoich
        self.maintenance_reaction = metabolism.maintenance_reaction

        # Load constants
        self.ngam = parameters["ngam"]
        gam = parameters["dark_atp"] * parameters["cell_dry_mass_fraction"]

        self.exchange_constraints = metabolism.exchange_constraints

        self._biomass_concentrations = {}  # type: dict
        self.getBiomassAsConcentrations = parameters["get_biomass_as_concentrations"]

        # Include ppGpp concentration target in objective if not handled
        # kinetically in other processes
        self.ppgpp_id = parameters["ppgpp_id"]
        self.getppGppConc = parameters["get_ppGpp_conc"]

        # go through all media in the timeline and add to metaboliteNames
        metaboliteNamesFromNutrients = set()
        conc_from_nutrients = (
            metabolism.concentration_updates.concentrations_based_on_nutrients
        )
        if include_ppgpp:
            metaboliteNamesFromNutrients.add(self.ppgpp_id)
        for time, media_id in timeline:
            exchanges = parameters["exchange_data_from_media"](media_id)
            metaboliteNamesFromNutrients.update(
                conc_from_nutrients(imports=exchanges["importExchangeMolecules"])
            )
        self.metaboliteNamesFromNutrients = list(sorted(metaboliteNamesFromNutrients))
        exchange_molecules = sorted(parameters["exchange_molecules"])
        molecule_masses = dict(
            zip(
                exchange_molecules,
                unum_to_pint(parameters["get_masses"](exchange_molecules)).to(
                    MASS_UNITS / COUNTS_UNITS
                ).magnitude,
            )
        )

        # Setup homeostatic objective concentration targets
        # Determine concentrations based on starting environment
        conc_dict = conc_from_nutrients(
            media_id=nutrients, imports=parameters["imports"]
        )
        doubling_time = parameters["doubling_time"]
        conc_dict.update(self.getBiomassAsConcentrations(doubling_time))
        if include_ppgpp:
            conc_dict[self.ppgpp_id] = self.getppGppConc(doubling_time)
        self.homeostatic_objective = dict(
            (key, unum_to_pint(conc_dict[key]).to(CONC_UNITS).magnitude)
            for key in conc_dict
        )

        # Include all concentrations that will be present in a sim for constant
        # length listeners
        for met in self.metaboliteNamesFromNutrients:
            if met not in self.homeostatic_objective:
                self.homeostatic_objective[met] = 0.0

        # Data structures to compute reaction bounds based on enzyme
        # presence/absence
        self.catalyst_ids = metabolism.catalyst_ids
        self.reactions_with_catalyst = metabolism.reactions_with_catalyst

        i = metabolism.catalysis_matrix_I
        j = metabolism.catalysis_matrix_J
        v = metabolism.catalysis_matrix_V
        shape = (i.max() + 1, j.max() + 1)
        self.catalysis_matrix = csr_matrix((v, (i, j)), shape=shape)

        # Function to compute reaction targets based on kinetic parameters and
        # molecule concentrations
        self.get_kinetic_constraints = metabolism.get_kinetic_constraints

        # Remove disabled reactions so they don't get included in the FBA
        # problem setup
        kinetic_constraint_reactions = metabolism.kinetic_constraint_reactions
        constraintsToDisable = metabolism.constraints_to_disable
        self.active_constraints_mask = np.array(
            [(rxn not in constraintsToDisable) for rxn in kinetic_constraint_reactions]
        )
        self.kinetics_constrained_reactions = list(
            np.array(kinetic_constraint_reactions)[self.active_constraints_mask]
        )

        self.kinetic_constraint_enzymes = metabolism.kinetic_constraint_enzymes
        self.kinetic_constraint_substrates = metabolism.kinetic_constraint_substrates

        # Set solver and kinetic objective weight (lambda)
        solver = metabolism.solver
        kinetic_objective_weight = metabolism.kinetic_objective_weight
        kinetic_objective_weight_in_range = metabolism.kinetic_objective_weight_in_range

        # Disable kinetics completely if weight is 0 or specified in file above
        if not USE_KINETICS or kinetic_objective_weight == 0:
            objective_type = "homeostatic"
            self.use_kinetics = False
            kinetic_objective_weight = 0
        else:
            objective_type = "homeostatic_kinetics_mixed"
            self.use_kinetics = True

        # Set up FBA solver
        # reactionRateTargets value is just for initialization, it gets reset
        # each timestep during evolveState
        fba_options = {
            "reactionStoich": metabolism.reaction_stoich,
            "externalExchangedMolecules": exchange_molecules,
            "objective": self.homeostatic_objective,
            "objectiveType": objective_type,
            "objectiveParameters": {
                "kineticObjectiveWeight": kinetic_objective_weight,
                "kinetic_objective_weight_in_range": kinetic_objective_weight_in_range,
                "reactionRateTargets": {
                    reaction: 1 for reaction in self.kinetics_constrained_reactions
                },
                "oneSidedReactionTargets": [],
            },
            "moleculeMasses": molecule_masses,
            # The "inconvenient constant"--limit secretion (e.g., of CO2)
            "secretionPenaltyCoeff": metabolism.secretion_penalty_coeff,
            "solver": solver,
            "maintenanceCostGAM": gam.to(COUNTS_UNITS / MASS_UNITS).magnitude,
            "maintenanceReaction": metabolism.maintenance_reaction,
        }
        self.fba = FluxBalanceAnalysis(**fba_options)

        self.metabolite_names = {
            met: i for i, met in enumerate(self.fba.getOutputMoleculeIDs())
        }
        self.aa_names_no_location = [x[:-3] for x in parameters["amino_acid_ids"]]

    def update_external_molecule_levels(
        self,
        objective: dict[str, Quantity],
        metabolite_concentrations: Quantity,
        external_molecule_levels: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Limit amino acid uptake to what is needed to meet concentration
        objective to prevent use as carbon source, otherwise could be used
        as an infinite nutrient source.

        Args:
            objective: homeostatic objective for internal
                molecules (molecule ID: concentration in counts/volume units)
            metabolite_concentrations: concentration for each
                molecule in metabolite_names
            external_molecule_levels: current limits on
                external molecule availability

        Returns:
            Updated limits on external molecule availability

        TODO(wcEcoli):
            determine rate of uptake so that some amino acid uptake can
            be used as a carbon/nitrogen source
        """

        external_exchange_molecule_ids = self.fba.getExternalMoleculeIDs()
        for aa in self.aa_names_no_location:
            if aa + "[p]" in external_exchange_molecule_ids:
                idx = external_exchange_molecule_ids.index(aa + "[p]")
            elif aa + "[c]" in external_exchange_molecule_ids:
                idx = external_exchange_molecule_ids.index(aa + "[c]")
            else:
                continue

            conc_diff = objective[aa + "[c]"] - metabolite_concentrations[
                self.metabolite_names[aa + "[c]"]
            ].to(CONC_UNITS).magnitude
            if conc_diff < 0:
                conc_diff = 0

            if external_molecule_levels[idx] > conc_diff:
                external_molecule_levels[idx] = conc_diff

        return external_molecule_levels

    def set_molecule_levels(
        self,
        metabolite_counts: npt.NDArray[np.int64],
        counts_to_molar: Quantity,
        coefficient: Quantity,
        current_media_id: str,
        unconstrained: set[str],
        constrained: set[str],
        conc_updates: dict[str, Quantity],
        aa_uptake_package: Optional[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.str_], bool]
        ] = None,
    ):
        """
        Set internal and external molecule levels available to the FBA solver.

        Args:
            metabolite_counts: counts for each metabolite with a concentration target
            counts_to_molar: conversion from counts to molar (counts/volume)
            coefficient: coefficient to convert from mmol/g DCW/hr to mM basis
                (mass*time/volume)
            current_media_id: ID of current media
            unconstrained: molecules that have unconstrained import
            constrained: molecules (keys) and their limited max uptake rates
                (mol / mass / time)
            conc_updates: updates to concentrations targets for molecules (mmol/L)
            aa_uptake_package: (uptake rates, amino acid names, force levels),
                determines whether to set hard uptake rates
        """

        # Update objective from media exchanges. exchange_constraints is
        # upstream Unum-native — convert the pint args at the boundary.
        external_molecule_levels, objective = self.exchange_constraints(
            self.fba.getExternalMoleculeIDs(),
            pint_to_unum(coefficient),
            pint_to_unum(CONC_UNITS),
            current_media_id,
            unconstrained,
            {k: pint_to_unum(v) for k, v in constrained.items()},
            conc_updates,
        )
        self.fba.update_homeostatic_targets(objective)
        self.homeostatic_objective = {**self.homeostatic_objective, **objective}

        # Internal concentrations
        metabolite_conc = counts_to_molar * metabolite_counts
        self.fba.setInternalMoleculeLevels(metabolite_conc.to(CONC_UNITS).magnitude)

        # External concentrations
        external_molecule_levels = self.update_external_molecule_levels(
            objective, metabolite_conc, external_molecule_levels
        )
        self.fba.setExternalMoleculeLevels(external_molecule_levels)

        if aa_uptake_package:
            levels, molecules, force = aa_uptake_package
            self.fba.setExternalMoleculeLevels(
                levels, molecules=molecules, force=force, allow_export=True
            )

    def set_reaction_bounds(
        self,
        catalyst_counts: npt.NDArray[np.int64],
        counts_to_molar: Quantity,
        coefficient: Quantity,
        gtp_to_hydrolyze: float,
    ):
        """
        Set reaction bounds for constrained reactions in the FBA object.

        Args:
            catalyst_counts: counts of enzyme catalysts
            counts_to_molar: conversion from counts to molar (counts/volume)
            coefficient: coefficient to convert from mmol/g DCW/hr to mM basis
                (mass*time/volume)
            gtp_to_hydrolyze: number of GTP molecules to hydrolyze to
                account for consumption in translation
        """

        # Maintenance reactions
        # Calculate new NGAM
        flux = (self.ngam * coefficient).to(CONC_UNITS).magnitude
        self.fba.setReactionFluxBounds(
            self.fba._reactionID_NGAM,
            lowerBounds=flux,
            upperBounds=flux,
        )

        # Calculate GTP usage based on how much was needed in polypeptide
        # elongation in previous step.
        flux = (counts_to_molar * gtp_to_hydrolyze).to(CONC_UNITS).magnitude
        self.fba.setReactionFluxBounds(
            self.fba._reactionID_polypeptideElongationEnergy,
            lowerBounds=flux,
            upperBounds=flux,
        )

        # Set hard upper bounds constraints based on enzyme presence
        # (infinite upper bound) or absence (upper bound of zero).
        # Catalyst presence flips for <0.1% of reactions per second of sim
        # time; the previous implementation pushed all ~9k bounds through
        # the GLPK Python wrapper every step (5.7M setFlowBounds calls per
        # 600s sim). Cache the prior mask and forward only the diff —
        # setFlowBounds is already a value-keyed no-op when value is
        # unchanged, so this just removes redundant Python dispatch.
        no_rxn_mask = self.catalysis_matrix.dot(catalyst_counts) == 0
        prev_mask = getattr(self, "_prev_no_rxn_mask", None)
        if prev_mask is None or prev_mask.shape != no_rxn_mask.shape:
            reaction_bounds = np.where(no_rxn_mask, 0.0, np.inf)
            self.fba.setReactionFluxBounds(
                self.reactions_with_catalyst,
                upperBounds=reaction_bounds,
                raiseForReversible=False,
            )
        else:
            changed = no_rxn_mask != prev_mask
            if changed.any():
                rxn_ids = [
                    self.reactions_with_catalyst[i]
                    for i in np.flatnonzero(changed)
                ]
                upper = np.where(no_rxn_mask[changed], 0.0, np.inf)
                self.fba.setReactionFluxBounds(
                    rxn_ids,
                    upperBounds=upper,
                    raiseForReversible=False,
                )
        self._prev_no_rxn_mask = no_rxn_mask

    def set_reaction_targets(
        self,
        kinetic_enzyme_counts: npt.NDArray[np.int64],
        kinetic_substrate_counts: npt.NDArray[np.int64],
        counts_to_molar: Quantity,
        time_step: Quantity,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Set reaction targets for constrained reactions in the FBA object.

        Args:
            kinetic_enzyme_counts: counts of enzymes used in kinetic constraints
            kinetic_substrate_counts: counts of substrates used in kinetic
                constraints
            counts_to_molar: conversion from counts to molar (counts/volume)
            time_step: current time step (time)

        Returns:
            3-element tuple containing

            - **mean_targets**: mean target for each constrained reaction
            - **upper_targets**: upper target limit for each constrained reaction
            - **lower_targets**: lower target limit for each constrained reaction
        """

        if self.use_kinetics:
            enzyme_conc = counts_to_molar * kinetic_enzyme_counts
            substrate_conc = counts_to_molar * kinetic_substrate_counts

            # Set target fluxes for reactions based on their most relaxed
            # constraint
            # get_kinetic_constraints is upstream Unum-native
            reaction_targets = unum_to_pint(self.get_kinetic_constraints(
                pint_to_unum(enzyme_conc), pint_to_unum(substrate_conc)
            ))

            # Calculate reaction flux target for current time step
            targets = (time_step * reaction_targets).to(CONC_UNITS).magnitude[
                self.active_constraints_mask, :
            ]
            lower_targets = targets[:, 0]
            mean_targets = targets[:, 1]
            upper_targets = targets[:, 2]

            # Set kinetic targets only if kinetics is enabled
            self.fba.set_scaled_kinetic_objective(time_step.to(units.s).magnitude)
            self.fba.setKineticTarget(
                self.kinetics_constrained_reactions,
                mean_targets,
                lower_targets=lower_targets,
                upper_targets=upper_targets,
            )
        else:
            lower_targets = np.zeros(len(self.kinetics_constrained_reactions))
            mean_targets = np.zeros(len(self.kinetics_constrained_reactions))
            upper_targets = np.zeros(len(self.kinetics_constrained_reactions))

        return mean_targets, upper_targets, lower_targets


def test_metabolism_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    data = sim.query()
    reaction_fluxes = data["agents"]["0"]["listeners"]["fba_results"]["reaction_fluxes"]
    assert isinstance(reaction_fluxes[0], list)
    assert isinstance(reaction_fluxes[1], list)


if __name__ == "__main__":
    test_metabolism_listener()
