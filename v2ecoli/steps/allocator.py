"""
=========
Allocator
=========

Reads requests from PartionedProcesses, and allocates molecules according to
process priorities.
"""

import numpy as np
from v2ecoli.library.ecoli_step import EcoliStep as Step
from typing import Any

# topology_registry removed — topology defined as class attribute
from v2ecoli.library.schema import counts, bulk_name_to_idx

# Register default topology for this process, associating it with process name
NAME = "allocator"
TOPOLOGY = {
    "request": ("request",),
    "allocate": ("allocate",),
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "allocator_rng": ("allocator_rng",),
}
# Register "allocator-1", "allocator-2", "allocator-3" to support
# multi-tiered partitioning scheme

ASSERT_POSITIVE_COUNTS = True


class NegativeCountsError(Exception):
    pass


class Allocator(Step):
    """Allocator Step

    process-bigraph interface: reads request/bulk/allocator_rng,
    writes allocate/request/listeners.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'molecule_names': 'list[string]',
        'process_names': 'list[string]',
        'custom_priorities': 'map[integer]',
        'seed': 'integer{0}',
    }

    processes: dict[str, Any] = {}

    def inputs(self):
        return {
            'request': 'map[map[list[integer]]]',
            'bulk': 'bulk_array',
            'allocator_rng': 'random_state',
        }

    def outputs(self):
        # Note: do NOT wrap allocate/request in `overwrite[...]`. Bigraph-schema
        # promotes per-port schemas into self.schema during apply, and an
        # `overwrite[map[...]]` at the parent map level causes ANY sub-key write
        # (e.g. a per-process Requester writing only its own slot) to replace
        # the whole map, dropping siblings. With plain `map[...]`, Map.apply
        # walks update keys and preserves siblings — which is the behavior
        # the partitioned execution layers depend on.
        return {
            'allocate': 'map[map[list[integer]]]',
            'request': 'map[map[list[integer]]]',
            'listeners': {
                'atp': {
                    # length n_processes; written as numpy arrays
                    'atp_requested': f'array[{self.n_processes},integer]',
                    'atp_allocated_initial': f'array[{self.n_processes},integer]',
                },
            },
        }

    def initialize(self, config):
        self.moleculeNames = self.parameters["molecule_names"]
        self.n_molecules = len(self.moleculeNames)
        self.mol_name_to_idx = {
            name: idx for idx, name in enumerate(self.moleculeNames)
        }
        self.mol_idx_to_name = {
            idx: name for idx, name in enumerate(self.moleculeNames)
        }
        self.processNames = self.parameters["process_names"]
        self.n_processes = len(self.processNames)
        self.proc_name_to_idx = {
            name: idx for idx, name in enumerate(self.processNames)
        }
        self.proc_idx_to_name = {
            idx: name for idx, name in enumerate(self.processNames)
        }
        self.processPriorities = np.zeros(len(self.processNames))
        for process, custom_priority in self.parameters["custom_priorities"].items():
            if process not in self.proc_name_to_idx.keys():
                continue
            self.processPriorities[self.proc_name_to_idx[process]] = custom_priority
        self.seed = self.parameters["seed"]

        # Helper indices for Numpy indexing
        self.molecule_idx = None

        # Count of ticks on which an over-draft had to be clamped (a transient
        # infeasibility upstream). Surfaced via a rate-limited warning so a
        # persistent problem stays visible without flooding the log.
        self._overdraft_events = 0

    def update(self, states, interval=None):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.moleculeNames, states["bulk"]["id"]
            )
            self.atp_idx = bulk_name_to_idx("ATP[c]", states["bulk"]["id"])
        total_counts = counts(states["bulk"], self.molecule_idx)
        original_totals = total_counts.copy()
        counts_requested = np.zeros((self.n_molecules, self.n_processes), dtype=int)
        # Keep track of which process indices are in current partitioning layer
        proc_idx_in_layer = []
        for process in states["request"]:
            proc_idx = self.proc_name_to_idx[process]
            if len(states["request"][process]["bulk"]) > 0:
                proc_idx_in_layer.append(proc_idx)
            for req_idx, req in states["request"][process]["bulk"]:
                counts_requested[req_idx, proc_idx] += req

        if ASSERT_POSITIVE_COUNTS and np.any(counts_requested < 0):
            raise NegativeCountsError(
                "Negative value(s) in counts_requested:\n"
                + "\n".join(
                    "{} in {} ({})".format(
                        self.mol_idx_to_name[molIndex],
                        self.proc_idx_to_name[processIndex],
                        counts_requested[molIndex, processIndex],
                    )
                    for molIndex, processIndex in zip(*np.where(counts_requested < 0))
                )
            )

        # Calculate partition
        partitioned_counts = calculatePartition(
            self.processPriorities,
            counts_requested,
            total_counts,
            states["allocator_rng"],
        )

        partitioned_counts.astype(int, copy=False)

        if ASSERT_POSITIVE_COUNTS and np.any(partitioned_counts < 0):
            raise NegativeCountsError(
                "Negative value(s) in partitioned_counts:\n"
                + "\n".join(
                    "{} in {} ({})".format(
                        self.mol_idx_to_name[molIndex],
                        self.proc_idx_to_name[processIndex],
                        partitioned_counts[molIndex, processIndex],
                    )
                    for molIndex, processIndex in zip(*np.where(partitioned_counts < 0))
                )
            )

        # Resolve any over-draft gracefully instead of crashing the lineage. An
        # over-draft here is almost always a molecule pool already driven
        # negative upstream (e.g. PROTON[c] after an FBA GLP_NOFEAS tick), which
        # the allocator merely observes — a single infeasible tick must not kill
        # a multi-generation run. Genuine corruption (negative requests / negative
        # allocations) is still a hard error in the checks above.
        partitioned_counts, overdrafts = resolve_overdraft(
            partitioned_counts, original_totals
        )
        if overdrafts:
            self._overdraft_events += 1
            if self._overdraft_events <= 5 or self._overdraft_events % 100 == 0:
                detail = ", ".join(
                    "{} ({})".format(self.mol_idx_to_name[m], d)
                    for m, d in overdrafts
                )
                print(
                    "Warning: allocator '{}' clamped over-draft "
                    "(event #{}): {} — pool driven negative upstream "
                    "(e.g. FBA infeasibility); clamping to available and "
                    "continuing".format(
                        getattr(self, "name", "allocator"),
                        self._overdraft_events,
                        detail,
                    ),
                    flush=True,
                )

        # Only update listener ATP counts for processes in
        # current partitioning layer
        non_zero_mask = counts_requested[self.atp_idx, :] != 0
        curr_atp_req = np.array(states["listeners"]["atp"]["atp_requested"]).copy()
        curr_atp_alloc = np.array(
            states["listeners"]["atp"]["atp_allocated_initial"]
        ).copy()
        curr_atp_req[non_zero_mask] = counts_requested[self.atp_idx, non_zero_mask]
        curr_atp_alloc[non_zero_mask] = partitioned_counts[self.atp_idx, non_zero_mask]

        update = {
            "request": {process: {"bulk": []} for process in states["request"]},
            "allocate": {
                process: {"bulk": partitioned_counts[:, self.proc_name_to_idx[process]]}
                for process in states["request"]
            },
            "listeners": {
                "atp": {
                    "atp_requested": curr_atp_req,
                    "atp_allocated_initial": curr_atp_alloc,
                }
            },
        }

        return update


def resolve_overdraft(partitioned_counts, original_totals):
    """Clamp allocations so no molecule is handed out beyond its available pool,
    and report any over-draft.

    Returns ``(clamped_counts, overdrafts)`` where ``overdrafts`` is a list of
    ``(molecule_index, deficit)`` pairs and ``deficit < 0`` is the amount by
    which the pool was exceeded.

    Two cases produce an over-draft:

    - The pool was already negative on entry (a transient infeasibility upstream
      — e.g. PROTON[c] driven negative by an FBA ``GLP_NOFEAS`` tick). Nothing
      was allocated to claw back, so the deficit is reported and the (zero)
      allocation is left untouched; the negative pool is the upstream's to heal.
    - A positive pool was genuinely over-allocated. The offending molecule's
      allocations are scaled down proportionally so their sum equals the pool
      (defense-in-depth; the float partition above should already prevent this).
    """
    counts_unallocated = original_totals - partitioned_counts.sum(axis=1)
    over_idx = np.where(counts_unallocated < 0)[0]
    if len(over_idx) == 0:
        return partitioned_counts, []

    clamped = partitioned_counts.copy()
    overdrafts = []
    for mol in over_idx:
        overdrafts.append((int(mol), int(counts_unallocated[mol])))
        avail = max(int(original_totals[mol]), 0)
        row = clamped[mol, :]
        row_sum = int(row.sum())
        if row_sum <= avail:
            continue  # pool already negative; nothing allocated to claw back
        if avail == 0:
            clamped[mol, :] = 0
            continue
        # Proportional integer scale-down to the available pool, distributing the
        # rounding remainder to the largest fractional parts (full allocation).
        scaled = row.astype(np.float64) * avail / row_sum
        floored = np.floor(scaled).astype(clamped.dtype)
        deficit = avail - int(floored.sum())
        if deficit > 0:
            order = np.argsort(scaled - floored)[::-1][:deficit]
            floored[order] += 1
        clamped[mol, :] = floored
    return clamped, overdrafts


def calculatePartition(
    process_priorities, counts_requested, total_counts, random_state
):
    priorityLevels = np.sort(np.unique(process_priorities))[::-1]

    partitioned_counts = np.zeros_like(counts_requested)

    for priorityLevel in priorityLevels:
        processHasPriority = priorityLevel == process_priorities

        requests = counts_requested[:, processHasPriority].copy()

        total_requested = requests.sum(axis=1)
        excess_request_mask = (total_requested > total_counts) & (total_requested > 0)

        # Get fractional request for molecules that have excess request
        # compared to available counts
        # Cast to float before multiplying: `requests * total_counts` is an int64
        # intermediate that overflows for high-count molecules (e.g. PROTON[c],
        # pools > ~3e9), wrapping to garbage and corrupting the partition. The
        # division already yields float, so for all non-overflowing magnitudes
        # this is bit-identical to the prior int64 path.
        fractional_requests = (
            requests[excess_request_mask, :].astype(np.float64)
            * total_counts[excess_request_mask, np.newaxis]
            / total_requested[excess_request_mask, np.newaxis]
        )

        # Distribute fractional counts to ensure full allocation of excess
        # request molecules
        remainders = fractional_requests % 1
        options = np.arange(remainders.shape[1])
        for idx, remainder in enumerate(remainders):
            total_remainder = remainder.sum()
            count = int(np.round(total_remainder))
            if count > 0:
                allocated_indices = random_state.choice(
                    options, size=count, p=remainder / total_remainder, replace=False
                )
                fractional_requests[idx, allocated_indices] += 1
        requests[excess_request_mask, :] = fractional_requests

        allocations = requests.astype(np.int64)
        partitioned_counts[:, processHasPriority] = allocations
        total_counts -= allocations.sum(axis=1)
    return partitioned_counts
