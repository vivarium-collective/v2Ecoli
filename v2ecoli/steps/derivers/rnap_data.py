"""
====================
RNAP Data Listener
====================
"""

import numpy as np
import warnings
from v2ecoli.library.schema import attrs
from v2ecoli.library.schema_types import ACTIVE_RNAP_ARRAY, RNA_ARRAY, ACTIVE_RIBOSOME_ARRAY
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.types.labeled_array import register_labeled_array

# topology_registry removed — topology defined as class attribute
from v2ecoli.processes.transcript_elongation import get_mapping_arrays


NAME = "rnap_data_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "active_RNAPs": ("unique", "active_RNAP"),
    "RNAs": ("unique", "RNA"),
    "active_ribosomes": ("unique", "active_ribosome"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", NAME),
}


class RnapData(Step):
    """
    Listener for RNAP data.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'stable_RNA_indexes': 'array[integer]',
        'cistron_ids': 'list[string]',
        'cistron_tu_mapping_matrix': 'csr_matrix',
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
    }


    def inputs(self):
        return {
            'listeners': {
                'rnap_data': {
                    'rna_init_event': {'_type': f'overwrite[array[{self.n_TUs},integer]]', '_default': []},
                },
            },
            'active_RNAPs': {'_type': ACTIVE_RNAP_ARRAY, '_default': []},
            'RNAs': {'_type': RNA_ARRAY, '_default': []},
            'active_ribosomes': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
            'next_update_time': {'_type': 'overwrite[float]', '_default': 1.0},
        }

    def outputs(self):
        return {
            'listeners': {
                'rnap_data': {
                    'rna_init_event_per_cistron': 'rna_init_event_per_cistron_vec',
                    'active_rnap_coordinates': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'active_rnap_domain_indexes': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'active_rnap_unique_indexes': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'active_rnap_on_stable_RNA_indexes': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'active_rnap_n_bound_ribosomes': {'_type': 'overwrite[array[integer]]', '_default': []},
                },
            },
            'next_update_time': 'overwrite[float]',
        }


    def initialize(self, config):
        self.stable_RNA_indexes = self.parameters["stable_RNA_indexes"]
        self.cistron_ids = self.parameters["cistron_ids"]
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]
        self.n_TUs = self.cistron_tu_mapping_matrix.shape[1]
        self.n_cistrons = len(self.cistron_ids)

        # Register named labeled-array type for rna_init_event_per_cistron so
        # the output_metadata walker can recover cistron IDs without sim_data.
        if self.core is not None:
            register_labeled_array(
                self.core, 'rna_init_event_per_cistron_vec',
                shape=(self.n_cistrons,),
                data=np.dtype('int64'),
                labels=tuple(self.cistron_ids),
            )

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
        # Read coordinates of all active RNAPs
        coordinates, domain_indexes, RNAP_unique_indexes = attrs(
            states["active_RNAPs"], ["coordinates", "domain_index", "unique_index"]
        )

        (RNA_RNAP_index, is_full_transcript, RNA_unique_indexes, TU_indexes) = attrs(
            states["RNAs"],
            ["RNAP_index", "is_full_transcript", "unique_index", "TU_index"],
        )

        is_partial_transcript = np.logical_not(is_full_transcript)
        is_stable_RNA = np.isin(TU_indexes, self.stable_RNA_indexes)
        partial_RNA_RNAP_indexes = RNA_RNAP_index[is_partial_transcript]
        partial_RNA_unique_indexes = RNA_unique_indexes[is_partial_transcript]

        (ribosome_RNA_index,) = attrs(states["active_ribosomes"], ["mRNA_index"])

        RNA_index_counts = dict(zip(*np.unique(ribosome_RNA_index, return_counts=True)))

        try:
            partial_RNA_to_RNAP_mapping, _ = get_mapping_arrays(
                partial_RNA_RNAP_indexes, RNAP_unique_indexes
            )
        except IndexError:
            # State inconsistency — RNAs and RNAPs may be out of sync
            # when updates are applied in different order than v1.
            # Use identity mapping as fallback.
            partial_RNA_to_RNAP_mapping = np.arange(len(partial_RNA_RNAP_indexes))

        update = {
            "listeners": {
                "rnap_data": {
                    "active_rnap_coordinates": coordinates,
                    "active_rnap_domain_indexes": domain_indexes,
                    "active_rnap_unique_indexes": RNAP_unique_indexes,
                    "active_rnap_on_stable_RNA_indexes": RNA_RNAP_index[
                        np.logical_and(is_stable_RNA, is_partial_transcript)
                    ],
                    "active_rnap_n_bound_ribosomes": np.array(
                        [
                            RNA_index_counts.get(partial_RNA_unique_indexes[i], 0)
                            for i in partial_RNA_to_RNAP_mapping
                        ]
                    ),
                    # Calculate hypothetical RNA initiation events per cistron
                    "rna_init_event_per_cistron": (
                        self.cistron_tu_mapping_matrix.dot(
                            states["listeners"]["rnap_data"]["rna_init_event"]
                        )
                        if np.asarray(
                            states["listeners"]["rnap_data"]["rna_init_event"]
                        ).shape == (self.n_TUs,)
                        else np.zeros(self.n_cistrons, dtype=int)
                    ),
                }
            },
            "next_update_time": states["global_time"] + states["timestep"],
        }
        return update
