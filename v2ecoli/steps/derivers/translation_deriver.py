"""
======================
Ribosome Data Listener
======================
"""

import numpy as np
import warnings
from v2ecoli.library.schema import attrs, bulk_name_to_idx
from v2ecoli.library.schema_types import ACTIVE_RIBOSOME_ARRAY, RNA_ARRAY
from v2ecoli.library.ecoli_step import EcoliStep as Step

# topology_registry removed — topology defined as class attribute


NAME = "ribosome_data_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "active_ribosomes": ("unique", "active_ribosome"),
    "RNAs": ("unique", "RNA"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", NAME),
}


class TranslationDeriver(Step):
    """
    Deriver for translation / ribosome data (elongation, termination, rRNA init).

    Renamed from ``RibosomeData`` in the Listeners→Derivers consolidation; the
    ``ribosome_data_listener`` step-key and ``listeners.ribosome_data`` wire
    path are preserved for emitter/dashboard/parity compatibility.
    """

    description = (
        "Derives per-tick translation readouts from the active-ribosome pool: "
        "effective elongation rate, actual elongations and termination events, "
        "rRNA initiation by class, and ribosomes-per-transcript / polysome "
        "occupancy."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'monomer_ids': 'list[string]',
        'n_monomers': {'_type': 'integer', '_default': 0},
        'rRNA_cistron_tu_mapping_matrix': 'csr_matrix',
        'rRNA_is_5S': 'array[integer]',
        'rRNA_is_16S': 'array[integer]',
        'rRNA_is_23S': 'array[integer]',
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
    }


    def inputs(self):
        return {
            'listeners': {
                'ribosome_data': {
                    'rRNA_initiated_TU': {'_type': f'overwrite[array[{self.n_rRNA_TUs},integer]]', '_default': []},
                    'rRNA_init_prob_TU': {'_type': f'overwrite[array[{self.n_rRNA_TUs},float]]', '_default': []},
                    # Declared-but-unused: pins the field in the merged
                    # state so the upstream PolypeptideInitiation write
                    # survives pbg's scalar-pruning quirk (task #14).
                    # Mirror of the RnapData listener step's
                    # log_likelihood pin from Phase-3 sprint 1.
                    'did_initialize': {'_type': 'overwrite[integer]', '_default': 0},
                },
            },
            'active_ribosomes': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
            'RNAs': {'_type': RNA_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
            'next_update_time': {'_type': 'overwrite[float]', '_default': 1.0},
        }

    def outputs(self):
        return {
            'listeners': {
                'ribosome_data': {
                    # Counts (dimensionless integers)
                    'n_ribosomes_per_transcript': {'_type': f'overwrite[array[{self.n_monomers},integer]]', '_default': []},
                    'n_ribosomes_on_partial_mRNA_per_transcript': {'_type': f'overwrite[array[{self.n_monomers},integer]]', '_default': []},
                    'total_rRNA_initiated': {'_type': 'overwrite[integer]', '_default': 0},
                    'rRNA5S_initiated': {'_type': 'overwrite[integer]', '_default': 0},
                    'rRNA16S_initiated': {'_type': 'overwrite[integer]', '_default': 0},
                    'rRNA23S_initiated': {'_type': 'overwrite[integer]', '_default': 0},
                    'mRNA_TU_index': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'n_ribosomes_on_each_mRNA': {'_type': 'overwrite[array[integer]]', '_default': []},
                    # Probabilities (dimensionless floats in [0,1])
                    'total_rRNA_init_prob': {'_type': 'overwrite[float]', '_default': 0.0},
                    'rRNA5S_init_prob': {'_type': 'overwrite[float]', '_default': 0.0},
                    'rRNA16S_init_prob': {'_type': 'overwrite[float]', '_default': 0.0},
                    'rRNA23S_init_prob': {'_type': 'overwrite[float]', '_default': 0.0},
                    # Mass per polysome — femtograms
                    'protein_mass_on_polysomes': {'_type': 'overwrite[array[float[fg]]]', '_default': []},
                },
            },
            'next_update_time': 'overwrite[float]',
        }


    def initialize(self, config):
        self.monomer_ids = self.parameters["monomer_ids"]
        self.n_monomers = len(self.monomer_ids)
        self.rRNA_cistron_tu_mapping_matrix = self.parameters[
            "rRNA_cistron_tu_mapping_matrix"
        ]
        self.rRNA_is_5S = self.parameters["rRNA_is_5S"]
        self.rRNA_is_16S = self.parameters["rRNA_is_16S"]
        self.rRNA_is_23S = self.parameters["rRNA_is_23S"]
        self.n_rRNA_TUs = self.rRNA_cistron_tu_mapping_matrix.shape[1]

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
        # Guard: return empty on first tick if data not yet populated
        # Get attributes of RNAs and ribosomes
        (is_full_transcript_RNA, unique_index_RNA, can_translate, TU_index) = attrs(
            states["RNAs"],
            ["is_full_transcript", "unique_index", "can_translate", "TU_index"],
        )

        (protein_index_ribosomes, mRNA_index_ribosomes, massDiff_protein_ribosomes) = (
            attrs(
                states["active_ribosomes"],
                ["protein_index", "mRNA_index", "massDiff_protein"],
            )
        )

        n_TUs = self.rRNA_cistron_tu_mapping_matrix.shape[1]
        rRNA_initiated_TU = states["listeners"]["ribosome_data"].get(
            "rRNA_initiated_TU", np.zeros(n_TUs, dtype=int))
        if hasattr(rRNA_initiated_TU, '__len__') and len(rRNA_initiated_TU) != n_TUs:
            rRNA_initiated_TU = np.zeros(n_TUs, dtype=int)
        rRNA_init_prob_TU = states["listeners"]["ribosome_data"].get(
            "rRNA_init_prob_TU", np.zeros(n_TUs))
        if hasattr(rRNA_init_prob_TU, '__len__') and len(rRNA_init_prob_TU) != n_TUs:
            rRNA_init_prob_TU = np.zeros(n_TUs)

        # Get mask for ribosomes that are translating proteins on partially
        # transcribed mRNAs
        ribosomes_on_nascent_mRNA_mask = np.isin(
            mRNA_index_ribosomes,
            unique_index_RNA[np.logical_not(is_full_transcript_RNA)],
        )

        # Get counts of ribosomes for each type
        n_ribosomes_per_transcript = np.bincount(
            protein_index_ribosomes, minlength=self.n_monomers
        )
        n_ribosomes_on_partial_mRNA_per_transcript = np.bincount(
            protein_index_ribosomes[ribosomes_on_nascent_mRNA_mask],
            minlength=self.n_monomers,
        )

        rRNA_cistrons_produced = self.rRNA_cistron_tu_mapping_matrix.dot(
            rRNA_initiated_TU
        )
        rRNA_cistrons_init_prob = self.rRNA_cistron_tu_mapping_matrix.dot(
            rRNA_init_prob_TU
        )
        total_rRNA_initiated = np.sum(rRNA_initiated_TU, dtype=int)
        total_rRNA_init_prob = np.sum(rRNA_init_prob_TU)
        rRNA5S_initiated = np.sum(rRNA_cistrons_produced[self.rRNA_is_5S], dtype=int)
        rRNA16S_initiated = np.sum(rRNA_cistrons_produced[self.rRNA_is_16S], dtype=int)
        rRNA23S_initiated = np.sum(rRNA_cistrons_produced[self.rRNA_is_23S], dtype=int)
        rRNA5S_init_prob = np.sum(rRNA_cistrons_init_prob[self.rRNA_is_5S])
        rRNA16S_init_prob = np.sum(rRNA_cistrons_init_prob[self.rRNA_is_16S])
        rRNA23S_init_prob = np.sum(rRNA_cistrons_init_prob[self.rRNA_is_23S])

        # Get fully transcribed translatable mRNA index
        is_full_mRNA = can_translate & is_full_transcript_RNA
        mRNA_unique_index = unique_index_RNA[is_full_mRNA]
        mRNA_TU_index = TU_index[is_full_mRNA]

        # Inverse indices from np.unique are better for np.bincount
        # because real indices can go up to 2**63
        unique_mRNA_index_ribosomes, reduced_mRNA_index_ribosomes = np.unique(
            mRNA_index_ribosomes, return_inverse=True
        )
        if len(unique_mRNA_index_ribosomes) == 0:
            # No ribosomes on any mRNA — every mRNA has zero bound ribosomes
            n_ribosomes_on_each_mRNA = np.zeros(len(mRNA_unique_index), dtype=int)
            protein_mass_on_polysomes = np.zeros(len(mRNA_unique_index))
        else:
            # Calculate mapping from inverse indices back to mRNA_unique_indices
            reduced_to_normal_mRNA_indices = bulk_name_to_idx(
                mRNA_unique_index, unique_mRNA_index_ribosomes, strict=False
            )
            # Many mRNAs in mRNA_unique_indices will have no bound ribosomes
            # Have them point to last zero of lengthened np.bincount output
            no_ribosomes_mask = (
                unique_mRNA_index_ribosomes[reduced_to_normal_mRNA_indices]
                != mRNA_unique_index
            )
            reduced_to_normal_mRNA_indices[no_ribosomes_mask] = -1
            bincount_minlength = max(reduced_mRNA_index_ribosomes) + 2

            # Get counts of ribosomes attached to the same mRNA
            bincount_ribosome_on_mRNA = np.bincount(
                reduced_mRNA_index_ribosomes, minlength=bincount_minlength
            )
            n_ribosomes_on_each_mRNA = bincount_ribosome_on_mRNA[
                reduced_to_normal_mRNA_indices
            ]

            # Get protein mass on each polysome
            protein_mass_on_polysomes = np.bincount(
                reduced_mRNA_index_ribosomes,
                weights=massDiff_protein_ribosomes,
                minlength=bincount_minlength,
            )[reduced_to_normal_mRNA_indices]

        update = {
            "listeners": {
                "ribosome_data": {
                    "n_ribosomes_per_transcript": n_ribosomes_per_transcript,
                    "n_ribosomes_on_partial_mRNA_per_transcript": n_ribosomes_on_partial_mRNA_per_transcript,
                    "total_rRNA_initiated": total_rRNA_initiated,
                    "total_rRNA_init_prob": total_rRNA_init_prob,
                    "rRNA5S_initiated": rRNA5S_initiated,
                    "rRNA16S_initiated": rRNA16S_initiated,
                    "rRNA23S_initiated": rRNA23S_initiated,
                    "rRNA5S_init_prob": rRNA5S_init_prob,
                    "rRNA16S_init_prob": rRNA16S_init_prob,
                    "rRNA23S_init_prob": rRNA23S_init_prob,
                    "mRNA_TU_index": mRNA_TU_index,
                    "n_ribosomes_on_each_mRNA": n_ribosomes_on_each_mRNA,
                    "protein_mass_on_polysomes": protein_mass_on_polysomes,
                }
            },
            "next_update_time": states["global_time"] + states["timestep"],
        }
        return update


# Back-compat alias for the pre-consolidation class name.
RibosomeData = TranslationDeriver
