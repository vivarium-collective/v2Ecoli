"""
=========================
DNA Supercoiling Listener
=========================
"""

import numpy as np
from v2ecoli.library.schema import attrs
from v2ecoli.library.schema_types import CHROMOSOMAL_SEGMENT_ARRAY
from v2ecoli.library.ecoli_step import EcoliStep as Step

# topology_registry removed — topology defined as class attribute


NAME = "dna_supercoiling_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "chromosomal_segments": ("unique", "chromosomal_segment"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


class DnaSupercoiling(Step):
    """
    Listener for DNA supercoiling data.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'relaxed_DNA_base_pairs_per_turn': 'float{0.0}',
        'emit_unique': 'boolean{false}',
        'time_step': 'float{1.0}',
    }


    def inputs(self):
        return {
            'chromosomal_segments': {'_type': CHROMOSOMAL_SEGMENT_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            'listeners': {
                'dna_supercoiling': {
                    'segment_left_boundary_coordinates': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'segment_right_boundary_coordinates': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'segment_domain_indexes': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'segment_superhelical_densities': {'_type': 'overwrite[array[float]]', '_default': []},
                },
            },
        }


    def initialize(self, config):
        self.relaxed_DNA_base_pairs_per_turn = self.parameters[
            "relaxed_DNA_base_pairs_per_turn"
        ]

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        # Guard: return empty on first tick if data not yet populated
        boundary_coordinates, domain_indexes, linking_numbers = attrs(
            states["chromosomal_segments"],
            ["boundary_coordinates", "domain_index", "linking_number"],
        )

        if len(boundary_coordinates) == 0:
            return {
                "listeners": {
                    "dna_supercoiling": {
                        "segment_left_boundary_coordinates": [],
                        "segment_right_boundary_coordinates": [],
                        "segment_domain_indexes": [],
                        "segment_superhelical_densities": [],
                    }
                }
            }

        # Get mask for segments with nonzero lengths
        segment_lengths = boundary_coordinates[:, 1] - boundary_coordinates[:, 0]

        assert np.all(segment_lengths >= 0)
        nonzero_length_mask = segment_lengths > 0

        # Calculate superhelical densities
        linking_numbers_relaxed_DNA = (
            segment_lengths[nonzero_length_mask] / self.relaxed_DNA_base_pairs_per_turn
        )

        update = {
            "listeners": {
                "dna_supercoiling": {
                    "segment_left_boundary_coordinates": boundary_coordinates[
                        nonzero_length_mask, 0
                    ],
                    "segment_right_boundary_coordinates": boundary_coordinates[
                        nonzero_length_mask, 1
                    ],
                    "segment_domain_indexes": domain_indexes[nonzero_length_mask],
                    "segment_superhelical_densities": np.divide(
                        linking_numbers[nonzero_length_mask]
                        - linking_numbers_relaxed_DNA,
                        linking_numbers_relaxed_DNA,
                    ),
                }
            }
        }
        return update
