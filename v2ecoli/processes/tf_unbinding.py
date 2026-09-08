"""
=============
TF Unbinding
=============

Unbind transcription factors from DNA to allow signaling processes
before binding back to DNA.

Mathematical Model
------------------
Deterministic release of all DNA-bound transcription factors (TFs)
back into the free active pool. For each TF species j:

    n_released_j = sum_i(bound_TF[i, j])    over all promoters i

The bound_TF matrix (promoters x TF species) is reset to zeros, and
the associated mass is transferred from the promoter submass stores:

    delta_mass_i = -sum_j(bound_TF[i, j] * m_j)

where m_j is the mass (fg) of one molecule of active TF species j.
"""

import numpy as np
import warnings

from v2ecoli.library.ecoli_step import EcoliStep as Step

# topology_registry removed
from v2ecoli.library.schema import bulk_name_to_idx, attrs
from v2ecoli.library.schema_types import PROMOTER_ARRAY
from bigraph_schema.contract import ProcessContract

# Register default topology for this process, associating it with process name
NAME = "ecoli-tf-unbinding"
TOPOLOGY = {
    "bulk": ("bulk",),
    "promoters": (
        "unique",
        "promoter",
    ),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "tf_unbinding"),
}


class TfUnbinding(Step):
    """TfUnbinding"""

    description = (
        "TfUnbinding — deterministic release of all DNA-bound TFs to the free active pool.\n\n"
        "    n_released_j = ∑_i bound_TF[i,j]     (sum over promoters i)\n"
        "    bound_TF ← 0     (reset promoter occupancy matrix)\n"
        "    Δmass_i = -∑_j bound_TF[i,j] · m_j   (mass moved promoter submass → bulk)\n"
        "  bound_TF: promoters × TF species;  m_j: mass (fg) of active TF j."
    )

    contract = ProcessContract(
        summary=(
            "Deterministically releases every DNA-bound TF back to the free active "
            "bulk pool, zeroes the promoter occupancy matrix, and removes the "
            "corresponding TF mass from promoter submass — run before signalling so "
            "TFs can re-equilibrate before TfBinding re-binds them."
        ),
        inputs={
            'promoters': (
                "reads the bound_TF occupancy matrix (and current submass fields) "
                "to count how many of each TF are bound; empty → no-op."
            ),
            'bulk': "provides the free active-TF pool that released TFs are added back to.",
            'global_time': "reads simulation time to decide (with next_update_time) whether this tick fires.",
            'timestep': "reads the tick length to schedule the next update time.",
            'next_update_time': "reads the scheduled fire time gating update_condition.",
        },
        outputs={
            'bulk': "adds n_released_j molecules of each TF j back to the free active pool.",
            'promoters': (
                "resets bound_TF to zero and decrements each promoter's submass "
                "fields by the released TF mass."
            ),
            'next_update_time': "writes the next scheduled fire time (global_time + timestep).",
        },
        config={
            'tf_ids': "list of transcription-factor species whose [c] active forms are released.",
            'active_tf_masses': "per-TF active-molecule mass m_j (fg) removed from promoter submass on release.",
            'submass_indices': "maps promoter submass fields to columns of the released-mass matrix.",
        },
        symbols={
            'n_released_j': "count of TF j molecules released from DNA to the free active pool",
            'bound_TF[i,j]': "1 if TF j is bound at promoter i, else 0 (indicator)",
            'Δmass_i': "mass removed from promoter i submass on release (fg)",
            'm_j': "mass of one active TF j molecule (fg)",
        },
        assumptions=[
            "Release is deterministic and total: every bound TF is freed each time the process fires.",
            "TF mass is conserved — it moves from promoter submass back to the bulk active pool.",
        ],
        references=[],
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'active_tf_masses': {'_type': 'array[float[fg]]', '_default': []},  # mass per TF molecule
        'emit_unique': {'_type': 'boolean', '_default': False},
        'submass_indices': 'map[integer]',
        'tf_ids': 'list[string]',
        'time_step': {'_type': 'integer[s]', '_default': 1},
    }


    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'promoters': {'_type': PROMOTER_ARRAY, '_default': []},
            'global_time': {'_type': 'float[s]', '_default': 0.0},
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'next_update_time': {'_type': 'overwrite[float[s]]', '_default': 1.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'promoters': PROMOTER_ARRAY,
            'next_update_time': 'overwrite[float[s]]',
        }


    def initialize(self, config):
        self.tf_ids = self.parameters["tf_ids"]
        self.submass_indices = self.parameters["submass_indices"]
        self.active_tf_masses = self.parameters["active_tf_masses"]

        # Numpy indices for bulk molecules
        self.active_tf_idx = None

    def update_condition(self, timestep, states):
        """
        See :py:meth:`~.Requester.update_condition`.
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
        # At t=0, convert all strings to indices
        if self.active_tf_idx is None:
            self.active_tf_idx = bulk_name_to_idx(
                [tf + "[c]" for tf in self.tf_ids], states["bulk"]["id"]
            )

        # Get attributes of all promoters
        (bound_TF,) = attrs(states["promoters"], ["bound_TF"])
        # If there are no promoters, return immediately
        if len(bound_TF) == 0:
            return {}

        # Calculate number of bound TFs for each TF prior to changes
        n_bound_TF = bound_TF.sum(axis=0)

        update = {
            # Free all DNA-bound TFs into free active TFs
            "bulk": [(self.active_tf_idx, n_bound_TF)],
            "promoters": {
                # Reset bound_TF attribute of promoters
                "set": {"bound_TF": np.zeros_like(bound_TF)}
            },
        }

        # Add mass_diffs array to promoter submass
        mass_diffs = bound_TF @ -self.active_tf_masses
        for submass, idx in self.submass_indices.items():
            update["promoters"]["set"][submass] = (
                attrs(states["promoters"], [submass])[0] + mass_diffs[:, idx]
            )

        update["next_update_time"] = states["global_time"] + states["timestep"]
        return update
