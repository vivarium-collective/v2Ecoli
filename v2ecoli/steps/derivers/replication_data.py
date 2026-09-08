"""
=========================
Replication Data Listener
=========================
"""

import numpy as np
from v2ecoli.library.schema import attrs
from v2ecoli.library.schema_types import ORIC_ARRAY, DNAA_BOX_ARRAY, ACTIVE_REPLISOME_ARRAY
from v2ecoli.library.ecoli_step import EcoliStep as Step

# topology_registry removed — topology defined as class attribute


NAME = "replication_data_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "oriCs": ("unique", "oriC"),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "active_replisomes": ("unique", "active_replisome"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


class ReplicationData(Step):
    """
    Listener for replication data.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
    }


    def inputs(self):
        return {
            'oriCs': {'_type': ORIC_ARRAY, '_default': []},
            'DnaA_boxes': {'_type': DNAA_BOX_ARRAY, '_default': []},
            'active_replisomes': {'_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1},
        }

    def outputs(self):
        return {
            'listeners': {
                'replication_data': {
                    'fork_coordinates': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'fork_domains': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'fork_unique_index': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'number_of_oric': {'_type': 'overwrite[integer]', '_default': []},
                    'free_DnaA_boxes': {'_type': 'overwrite[integer]', '_default': []},
                    'total_DnaA_boxes': {'_type': 'overwrite[integer]', '_default': []},
                    # dnaa-3 Phase 2: per-pool occupancy by nucleotide form.
                    # Pool labels: 0 chromosomal_high · 1 oriC_high
                    #             · 2 oriC_low · 3 promoter_high
                    'chromosomal_high_free': {'_type': 'overwrite[integer]', '_default': 0},
                    'chromosomal_high_bound_atp': {'_type': 'overwrite[integer]', '_default': 0},
                    'chromosomal_high_bound_adp': {'_type': 'overwrite[integer]', '_default': 0},
                    'oriC_high_free': {'_type': 'overwrite[integer]', '_default': 0},
                    'oriC_high_bound_atp': {'_type': 'overwrite[integer]', '_default': 0},
                    'oriC_high_bound_adp': {'_type': 'overwrite[integer]', '_default': 0},
                    'oriC_low_free': {'_type': 'overwrite[integer]', '_default': 0},
                    'oriC_low_bound_atp': {'_type': 'overwrite[integer]', '_default': 0},
                    'promoter_high_free': {'_type': 'overwrite[integer]', '_default': 0},
                    'promoter_high_bound_atp': {'_type': 'overwrite[integer]', '_default': 0},
                    'promoter_high_bound_adp': {'_type': 'overwrite[integer]', '_default': 0},
                    # dnaa-3 Phase 2 (per-box): raw arrays of active DnaA_box
                    # attributes so plots can reconstruct per-chromosome state.
                    # Length equals the number of active boxes at the tick.
                    'dnaa_box_domain_index': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'dnaa_box_pool_label': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'dnaa_box_bound_form': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'dnaa_box_coordinates': {'_type': 'overwrite[array[integer]]', '_default': []},
                },
            },
        }


    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        # Guard: return empty on first tick if data not yet populated
        fork_coordinates, fork_domains, fork_unique_index = attrs(
            states["active_replisomes"], ["coordinates", "domain_index", "unique_index"]
        )

        DnaA_box_bound, pool_label, bound_form, domain_index, coordinates = attrs(
            states["DnaA_boxes"],
            ["DnaA_bound", "pool_label", "DnaA_bound_form",
             "domain_index", "coordinates"],
        )

        # Per-pool occupancy (pool_label values match dnaa_box_binding.py).
        def _pool_counts(pool_id):
            mask = (pool_label == pool_id)
            if not mask.any():
                return 0, 0, 0
            forms = bound_form[mask]
            n_atp = int(np.count_nonzero(forms == 1))
            n_adp = int(np.count_nonzero(forms == 2))
            n_free = int(mask.sum()) - n_atp - n_adp
            return n_free, n_atp, n_adp

        ch_free, ch_atp, ch_adp = _pool_counts(0)
        oh_free, oh_atp, oh_adp = _pool_counts(1)
        ol_free, ol_atp, _ol_adp = _pool_counts(2)  # ATP-only pool
        pr_free, pr_atp, pr_adp = _pool_counts(3)

        update = {
            "listeners": {
                "replication_data": {
                    "fork_coordinates": fork_coordinates,
                    "fork_domains": fork_domains,
                    "fork_unique_index": fork_unique_index,
                    "number_of_oric": states["oriCs"]["_entryState"].sum(),
                    "total_DnaA_boxes": len(DnaA_box_bound),
                    "free_DnaA_boxes": np.count_nonzero(np.logical_not(DnaA_box_bound)),
                    "chromosomal_high_free": ch_free,
                    "chromosomal_high_bound_atp": ch_atp,
                    "chromosomal_high_bound_adp": ch_adp,
                    "oriC_high_free": oh_free,
                    "oriC_high_bound_atp": oh_atp,
                    "oriC_high_bound_adp": oh_adp,
                    "oriC_low_free": ol_free,
                    "oriC_low_bound_atp": ol_atp,
                    "promoter_high_free": pr_free,
                    "promoter_high_bound_atp": pr_atp,
                    "promoter_high_bound_adp": pr_adp,
                    "dnaa_box_domain_index": np.asarray(
                        domain_index, dtype=np.int64),
                    "dnaa_box_pool_label": np.asarray(
                        pool_label, dtype=np.int64),
                    "dnaa_box_bound_form": np.asarray(
                        bound_form, dtype=np.int64),
                    "dnaa_box_coordinates": np.asarray(
                        coordinates, dtype=np.int64),
                }
            }
        }
        return update
