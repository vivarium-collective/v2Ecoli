"""
=============
Mass Listener
=============

Represents the total cellular mass.
"""

import numpy as np
from numpy.lib import recfunctions as rfn

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import counts, bulk_name_to_idx
# topology_registry removed — topology defined as class attribute
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import fg_magnitude

# Register default topology for this process, associating it with process name
NAME = "ecoli-mass-listener"
TOPOLOGY = {
    "bulk": ("bulk",),
    "unique": ("unique",),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


class MassDeriver(Step):
    """MassDeriver"""

    description = (
        "Aggregates total cell mass and submasses (protein, RNA by class, "
        "DNA, small molecules, water) from bulk and unique molecule counts. "
        "Derives cell volume, plus growth rate and mass fold-changes over "
        "the cell cycle."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'cellDensity': {'_type': 'float', '_default': 1100.0},
        'bulk_ids': 'list[string]',
        'bulk_masses': {'_type': 'overwrite[array[float]]', '_default': None},
        'unique_ids': 'list[string]',
        'unique_masses': {'_type': 'overwrite[array[float]]', '_default': None},
        'submass_to_idx': {'_type': 'map[integer]', '_default': {
            "rRNA": 0,
            "tRNA": 1,
            "mRNA": 2,
            "miscRNA": 3,
            "nonspecific_RNA": 4,
            "protein": 5,
            "metabolite": 6,
            "water": 7,
            "DNA": 8,
        }},
        'compartment_indices': {'_type': 'map[list[integer]]', '_default': {
            "projection": [],
            "cytosol": [],
            "extracellular": [],
            "flagellum": [],
            "membrane": [],
            "outer_membrane": [],
            "periplasm": [],
            "pilus": [],
            "inner_membrane": [],
        }},
        'compartment_id_to_index': 'map[integer]',
        'compartment_abbrev_to_index': 'map[integer]',
        'n_avogadro': {'_type': 'quantity[float,1/mol]', '_default': 6.0221409e23},
        'time_step': {'_type': 'float', '_default': 1.0},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'match_wcecoli': {'_type': 'boolean', '_default': False},
        'condition': {'_type': 'string', '_default': ''},
        'condition_to_doubling_time': 'map[quantity[float,min]]',
    }


    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'unique': 'map[node]',
            'listeners': {
                'mass': {
                    'dry_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                },
            },
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            'listeners': {
                'mass': {
                    # Cell + submass compartments — femtograms
                    'cell_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'water_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'dry_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'rna_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'rRna_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'tRna_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'mRna_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'dna_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'protein_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'smallMolecule_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    # Compartment masses — also femtograms
                    'projection_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'cytosol_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'extracellular_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'flagellum_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'membrane_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'outer_membrane_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'periplasm_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'pilus_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'inner_membrane_mass': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    # Volume — femtoliters
                    'volume': {'_type': 'overwrite[quantity[float,fL]]', '_default': 0.0},
                    # Growth — fg per second
                    'growth': {'_type': 'overwrite[quantity[float,fg]]', '_default': 0.0},
                    'instantaneous_growth_rate': {'_type': 'overwrite[quantity[float,1/s]]', '_default': 0.0},
                    # Dimensionless ratios and fractions
                    'protein_mass_fraction': {'_type': 'overwrite[float]', '_default': 0.0},
                    'rna_mass_fraction': {'_type': 'overwrite[float]', '_default': 0.0},
                    'dry_mass_fold_change': {'_type': 'overwrite[float]', '_default': 0.0},
                    'protein_mass_fold_change': {'_type': 'overwrite[float]', '_default': 0.0},
                    'rna_mass_fold_change': {'_type': 'overwrite[float]', '_default': 0.0},
                    'small_molecule_fold_change': {'_type': 'overwrite[float]', '_default': 0.0},
                    'expected_mass_fold_change': {'_type': 'overwrite[float]', '_default': 0.0},
                },
            },
        }


    def initialize(self, config):

        # molecule indexes and masses
        self.bulk_ids = self.parameters["bulk_ids"]
        self.bulk_masses = self.parameters["bulk_masses"]
        if self.bulk_masses is None:
            self.bulk_masses = np.zeros([1, 9])
        self.unique_ids = self.parameters["unique_ids"]
        self.unique_masses = self.parameters["unique_masses"]
        if self.unique_masses is None:
            self.unique_masses = np.zeros([1, 9])

        self.submass_listener_indices = {
            "rna": np.array(
                [
                    self.parameters["submass_to_idx"][name]
                    for name in ["rRNA", "tRNA", "mRNA", "miscRNA", "nonspecific_RNA"]
                ]
            ),
            "rRna": self.parameters["submass_to_idx"]["rRNA"],
            "tRna": self.parameters["submass_to_idx"]["tRNA"],
            "mRna": self.parameters["submass_to_idx"]["mRNA"],
            "dna": self.parameters["submass_to_idx"]["DNA"],
            "protein": self.parameters["submass_to_idx"]["protein"],
            "smallMolecule": self.parameters["submass_to_idx"]["metabolite"],
            "water": self.parameters["submass_to_idx"]["water"],
        }
        self.ordered_submasses = [0] * len(self.parameters["submass_to_idx"])
        for submass, idx in self.parameters["submass_to_idx"].items():
            self.ordered_submasses[idx] = f"{submass}_submass"

        # compartment indexes
        self.compartment_id_to_index = self.parameters["compartment_id_to_index"]
        self.projection_index = self.parameters["compartment_indices"]["projection"]
        self.cytosol_index = self.parameters["compartment_indices"]["cytosol"]
        self.extracellular_index = self.parameters["compartment_indices"][
            "extracellular"
        ]
        self.flagellum_index = self.parameters["compartment_indices"]["flagellum"]
        self.membrane_index = self.parameters["compartment_indices"]["membrane"]
        self.outer_membrane_index = self.parameters["compartment_indices"][
            "outer_membrane"
        ]
        self.periplasm_index = self.parameters["compartment_indices"]["periplasm"]
        self.pilus_index = self.parameters["compartment_indices"]["pilus"]
        self.inner_membrane_index = self.parameters["compartment_indices"][
            "inner_membrane"
        ]

        # Set up matrix for compartment mass calculation
        self.compartment_abbrev_to_index = self.parameters[
            "compartment_abbrev_to_index"
        ]
        if self.compartment_abbrev_to_index:
            self._bulk_molecule_by_compartment = np.stack(
                [
                    np.core.defchararray.chararray.endswith(self.bulk_ids, abbrev + "]")
                    for abbrev in self.compartment_abbrev_to_index
                ]
            )

        # units and constants
        self.cellDensity = self.parameters["cellDensity"]
        self.n_avogadro = self.parameters["n_avogadro"]

        self.time_step = self.parameters["time_step"]
        self.first_time_step = True

        self.massDiff_names = [
            "massDiff_" + submass for submass in self.parameters["submass_to_idx"]
        ]

        self.cell_cycle_len = self.parameters["condition_to_doubling_time"][
            self.parameters["condition"]
        ].to(units.s).magnitude

        # Helper indices for Numpy indexing
        self.bulk_idx = None

        # Enable flag for perfect recapitulation of wcEcoli mass calculations
        self.match_wcecoli = self.parameters["match_wcecoli"]

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        if self.bulk_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.bulk_idx = bulk_name_to_idx(self.bulk_ids, bulk_ids)
            if self.match_wcecoli:
                self.bulk_addon = np.zeros((len(self.bulk_idx), 16))
            # Per-molecule mass fields are constant for the simulation —
            # cache the unstructured (n_molecules, n_submasses) array on
            # first tick rather than re-slicing + re-converting the
            # structured array via rfn.structured_to_unstructured every
            # tick. ParCa-derived masses do not change at runtime.
            bulk_masses_struct = states["bulk"][self.ordered_submasses][
                self.bulk_idx]
            self._bulk_masses_cached = rfn.structured_to_unstructured(
                bulk_masses_struct)

        mass_update = {}

        # Get previous dry mass, for calculating growth later. Read tolerantly
        # (the store now holds a Quantity[fg]); internal math below stays in
        # bare fg magnitudes and the emitted fields are wrapped at the end.
        old_dry_mass = fg_magnitude(states["listeners"]["mass"]["dry_mass"])

        # get submasses from bulk and unique
        bulk_counts = counts(states["bulk"], self.bulk_idx)
        bulk_masses = self._bulk_masses_cached
        bulk_submasses = np.dot(bulk_counts, bulk_masses)
        bulk_compartment_masses = np.dot(
            bulk_counts * self._bulk_molecule_by_compartment, bulk_masses
        )
        if self.match_wcecoli:
            bulk_counts = np.hstack(
                [self.bulk_addon, counts(states["bulk"], self.bulk_idx)[:, np.newaxis]]
            )
            bulk_submasses = np.dot(bulk_counts.T, bulk_masses).sum(axis=0)
            bulk_compartment_masses = np.dot(
                bulk_counts.sum(axis=1) * self._bulk_molecule_by_compartment,
                bulk_masses,
            )

        unique_submasses = np.zeros(len(self.massDiff_names))
        unique_compartment_masses = np.zeros_like(bulk_compartment_masses)
        for unique_id, unique_mass in zip(self.unique_ids, self.unique_masses):
            molecules = states["unique"].get(unique_id)
            # self.unique_ids comes from the cache, which lists every
            # unique-molecule type the ParCa fixture defines (incl. plasmid
            # molecules baked in by --mode full). State from a non-plasmid
            # sim (e.g. the pre-division checkpoint, or any daughter built
            # from it) omits those keys — treat absent == zero count.
            if molecules is None:
                continue
            n_molecules = molecules["_entryState"].sum()

            if n_molecules == 0:
                continue

            unique_submasses += unique_mass * n_molecules
            unique_compartment_masses[self.compartment_abbrev_to_index["c"], :] += (
                unique_mass * n_molecules
            )

            # Extract mass diffs efficiently: stack field arrays into (n_mols, n_fields)
            mol_mask = molecules["_entryState"].view(np.bool_)
            massDiffs = np.column_stack(
                [molecules[name][mol_mask] for name in self.massDiff_names])
            if self.match_wcecoli:
                massDiffs = np.core.records.fromarrays(
                    [molecules[name][mol_mask] for name in self.massDiff_names]
                ).view((np.float64, len(self.massDiff_names)))
            unique_submasses += massDiffs.sum(axis=0)
            unique_compartment_masses[self.compartment_abbrev_to_index["c"], :] += (
                massDiffs.sum(axis=0)
            )

        # all of the submasses
        all_submasses = bulk_submasses + unique_submasses

        # save cell mass, water mass, dry mass
        mass_update["cell_mass"] = all_submasses.sum()
        mass_update["water_mass"] = all_submasses[
            self.submass_listener_indices["water"]
        ]
        mass_update["dry_mass"] = mass_update["cell_mass"] - mass_update["water_mass"]

        # Store submasses
        for submass, indices in self.submass_listener_indices.items():
            mass_update[submass + "_mass"] = all_submasses[indices].sum()

        mass_update["volume"] = mass_update["cell_mass"] / self.cellDensity

        if self.first_time_step:
            mass_update["growth"] = 0.0
            self.dryMassInitial = mass_update["dry_mass"]
            self.proteinMassInitial = mass_update["protein_mass"]
            self.rnaMassInitial = mass_update["rna_mass"]
            self.smallMoleculeMassInitial = mass_update["smallMolecule_mass"]
            self.timeInitial = states["global_time"]
        else:
            mass_update["growth"] = mass_update["dry_mass"] - old_dry_mass

        # Compartment submasses
        compartment_submasses = bulk_compartment_masses + unique_compartment_masses
        mass_update["projection_mass"] = compartment_submasses[
            self.projection_index, :
        ].sum()
        mass_update["cytosol_mass"] = compartment_submasses[self.cytosol_index, :].sum()
        mass_update["extracellular_mass"] = compartment_submasses[
            self.extracellular_index, :
        ].sum()
        mass_update["flagellum_mass"] = compartment_submasses[
            self.flagellum_index, :
        ].sum()
        mass_update["membrane_mass"] = compartment_submasses[
            self.membrane_index, :
        ].sum()
        mass_update["outer_membrane_mass"] = compartment_submasses[
            self.outer_membrane_index, :
        ].sum()
        mass_update["periplasm_mass"] = compartment_submasses[
            self.periplasm_index, :
        ].sum()
        mass_update["pilus_mass"] = compartment_submasses[self.pilus_index, :].sum()
        mass_update["inner_membrane_mass"] = compartment_submasses[
            self.inner_membrane_index, :
        ].sum()

        # This listener tracks the mass changes caused by each process
        # TODO: Blame processes?
        # mass_update['processMassDifferences'] = sum(
        #     state.process_mass_diffs() for state in self.internal_states.values()
        # ).sum(axis=1)

        if mass_update["dry_mass"] != 0:
            mass_update["protein_mass_fraction"] = (
                mass_update["protein_mass"] / mass_update["dry_mass"]
            )
            mass_update["rna_mass_fraction"] = (
                mass_update["rna_mass"] / mass_update["dry_mass"]
            )
            mass_update["instantaneous_growth_rate"] = (
                mass_update["growth"] / self.time_step / mass_update["dry_mass"]
            )
            mass_update["dry_mass_fold_change"] = (
                mass_update["dry_mass"] / self.dryMassInitial
            )
            mass_update["protein_mass_fold_change"] = (
                mass_update["protein_mass"] / self.proteinMassInitial
            )
            mass_update["rna_mass_fold_change"] = (
                mass_update["rna_mass"] / self.rnaMassInitial
            )
            mass_update["small_molecule_fold_change"] = (
                mass_update["smallMolecule_mass"] / self.smallMoleculeMassInitial
            )
            mass_update["expected_mass_fold_change"] = np.exp(
                np.log(2)
                * (states["global_time"] - self.timeInitial)
                / self.cell_cycle_len
            )

        self.first_time_step = False

        # --- Units-on-ports: emit dimensioned fields as pint Quantities so the
        # unit travels through the store + serialization. Internal arithmetic
        # above stays in bare fg magnitudes; wrap only here, at the emit
        # boundary. Dimensionless ratios (*_fraction, *_fold_change) stay float.
        for _field in list(mass_update):
            if _field.endswith("_mass"):
                mass_update[_field] = mass_update[_field] * units.fg
        mass_update["volume"] = mass_update["volume"] * units.fL
        mass_update["growth"] = mass_update["growth"] * units.fg
        if "instantaneous_growth_rate" in mass_update:
            # bare value is already the 1/s magnitude; `/ units.s` tags it 1/s
            mass_update["instantaneous_growth_rate"] = (
                mass_update["instantaneous_growth_rate"] / units.s
            )

        update = {"listeners": {"mass": mass_update}}
        return update




class PostDivisionMassDeriver(MassDeriver):
    """
    Normally, the mass listener updates after all other processes and steps
    have run. However, after division, the mass must be updated immediately
    so other processes have access to the accurate mass of their daughter
    cell. This process ensures that the mass seen by other processes following
    division is accurate.
    """

    name = "post-division-mass-listener"

    def update_condition(self, timestep, states):
        return self.first_time_step
