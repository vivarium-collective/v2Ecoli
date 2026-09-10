"""
Cell division state splitting functions for v2ecoli.

Ports the division logic from vEcoli/ecoli/library/schema.py.
These are pure numpy functions that partition a mother cell's state
into two daughter cells.

Division strategy:
- Bulk molecules: binomial p=0.5
- Unique molecules: domain-based (chromosomes alternate, attached molecules follow)
- RNAs: full transcripts binomial, partial follow RNAP domain
- Ribosomes: follow their mRNA, degraded mRNA ribosomes binomial
"""

import copy
from typing import Dict, List, Any, Tuple

import numpy as np

from v2ecoli.library.schema import attrs


# Builtin computation/lookup errors that are ALWAYS real failures — never a
# division signal — even when their message happens to contain the substring
# "divide"/"division" (the canonical trap: ``ZeroDivisionError: float division
# by zero``). Callers that detect division from a raised exception must exclude
# these so a real bug is surfaced, not silently mislabeled as a division.
NON_DIVISION_ERRORS = (
    ArithmeticError, TypeError, KeyError, AttributeError, ValueError,
    IndexError, NameError, ImportError, AssertionError,
)


def is_division_exception(e: BaseException) -> bool:
    """True if an exception from ``composite.run()`` is a genuine division signal.

    A division surfaces as a structural agents-map update (mother removed,
    daughters added) that process-bigraph can raise through; its message
    mentions "divide"/"division". A genuine division is never one of
    :data:`NON_DIVISION_ERRORS` — those are real code errors that merely happen
    to contain the substring. Returns ``False`` for them (and for any exception
    without the token) so the caller re-raises instead of treating it as a
    phantom division.
    """
    if isinstance(e, NON_DIVISION_ERRORS):
        return False
    msg = str(e).lower()
    return "divide" in msg or "division" in msg

RAND_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# Domain tree helpers
# ---------------------------------------------------------------------------

def follow_domain_tree(domain, domain_index, child_domains, placeholder):
    """Recursively collect a domain and all its descendants."""
    idx = np.where(domain_index == domain)[0]
    if len(idx) == 0:
        return [domain]
    children = child_domains[idx[0]]
    if children[0] != placeholder:
        branches = []
        for child in children:
            branches.extend(
                follow_domain_tree(child, domain_index, child_domains, placeholder))
        branches.append(domain)
        return branches
    return [domain]


def get_descendent_domains(root_domains, domain_index, child_domains, placeholder):
    """Collect all descendant domain indexes for a set of root domains."""
    all_domains = []
    for root in root_domains:
        all_domains.extend(
            follow_domain_tree(root, domain_index, child_domains, placeholder))
    return np.array(all_domains)


# ---------------------------------------------------------------------------
# Bulk molecule division
# ---------------------------------------------------------------------------

def divide_bulk(state):
    """Divide bulk molecules using binomial distribution with p=0.5.

    Args:
        state: Structured numpy array with 'count' field.

    Returns:
        (daughter_1, daughter_2) structured arrays.
    """
    counts = state['count']
    seed = int(counts.sum()) % RAND_MAX
    rng = np.random.RandomState(seed=seed)
    d1 = state.copy()
    d2 = state.copy()
    d1['count'] = rng.binomial(counts, 0.5)
    d2['count'] = counts - d1['count']
    return d1, d2


# ---------------------------------------------------------------------------
# Domain-based unique molecule division
# ---------------------------------------------------------------------------

def divide_domains(unique_state):
    """Partition chromosome domains between daughters.

    Alternates full chromosomes: daughter 1 gets even-indexed (0, 2, ...),
    daughter 2 gets odd-indexed (1, 3, ...). Then collects all descendant
    domains for each daughter.

    Args:
        unique_state: Dict with 'full_chromosome' and 'chromosome_domain' arrays.

    Returns:
        Dict with 'd1_all_domain_indexes' and 'd2_all_domain_indexes'.
    """
    (domain_index_full,) = attrs(unique_state['full_chromosome'], ['domain_index'])
    domain_index_domains, child_domains = attrs(
        unique_state['chromosome_domain'], ['domain_index', 'child_domains'])

    d1_full = domain_index_full[0::2]
    d2_full = domain_index_full[1::2]

    d1_all = get_descendent_domains(d1_full, domain_index_domains, child_domains, -1)
    d2_all = get_descendent_domains(d2_full, domain_index_domains, child_domains, -1)

    assert np.intersect1d(d1_all, d2_all).size == 0

    return {
        'd1_all_domain_indexes': d1_all,
        'd2_all_domain_indexes': d2_all,
    }


def divide_by_domain(values, unique_state):
    """Divide chromosome-attached molecules by their domain assignment.

    Args:
        values: Structured array with 'domain_index' and '_entryState' fields.
        unique_state: Dict with 'full_chromosome' and 'chromosome_domain'.

    Returns:
        (daughter_1, daughter_2) arrays of active molecules.
    """
    domain_div = divide_domains(unique_state)
    active = values[values['_entryState'].view(np.bool_)]
    d1_bool = np.isin(active['domain_index'], domain_div['d1_all_domain_indexes'])
    d2_bool = np.isin(active['domain_index'], domain_div['d2_all_domain_indexes'])
    # Chromosome domains may lose some entries; skip assertion for domains
    if 'child_domains' not in values.dtype.names:
        assert d1_bool.sum() + d2_bool.sum() == len(active)
    return active[d1_bool], active[d2_bool]


def divide_RNAs_by_domain(values, unique_state):
    """Divide RNA molecules: full transcripts binomial, partial follow RNAP.

    Args:
        values: Structured array of RNA unique molecules.
        unique_state: Dict with 'active_RNAP', 'full_chromosome', 'chromosome_domain'.

    Returns:
        (daughter_1, daughter_2) arrays.
    """
    is_full_transcript, RNAP_index = attrs(values, ['is_full_transcript', 'RNAP_index'])
    n_molecules = len(is_full_transcript)

    if n_molecules == 0:
        return np.zeros(0, dtype=values.dtype), np.zeros(0, dtype=values.dtype)

    # Determine RNAP partitioning
    domain_div = divide_domains(unique_state)
    rnaps = unique_state['active_RNAP']
    rnaps = rnaps[rnaps['_entryState'].view(np.bool_)]
    d1_rnap_bool = np.isin(rnaps['domain_index'], domain_div['d1_all_domain_indexes'])
    d1_rnap_indexes = rnaps['unique_index'][d1_rnap_bool]
    d2_rnap_indexes = rnaps['unique_index'][~d1_rnap_bool]

    d1_bool = np.zeros(n_molecules, dtype=np.bool_)
    d2_bool = np.zeros(n_molecules, dtype=np.bool_)

    # Full transcripts: binomial split
    full_idxs = np.where(is_full_transcript)[0]
    if len(full_idxs) > 0:
        rng = np.random.RandomState(seed=n_molecules)
        n_full_d1 = rng.binomial(len(full_idxs), 0.5)
        full_d1 = rng.choice(full_idxs, size=n_full_d1, replace=False)
        full_d2 = np.setdiff1d(full_idxs, full_d1)
        d1_bool[full_d1] = True
        d2_bool[full_d2] = True

    # Partial transcripts: follow RNAP
    partial_idxs = np.where(~is_full_transcript)[0]
    rnap_idx_partial = RNAP_index[partial_idxs]
    d1_bool[partial_idxs[np.isin(rnap_idx_partial, d1_rnap_indexes)]] = True
    d2_bool[partial_idxs[np.isin(rnap_idx_partial, d2_rnap_indexes)]] = True

    assert n_molecules == d1_bool.sum() + d2_bool.sum()
    assert np.count_nonzero(np.logical_and(d1_bool, d2_bool)) == 0

    rnas = values[values['_entryState'].view(np.bool_)]
    return rnas[d1_bool], rnas[d2_bool]


def divide_ribosomes_by_RNA(values, unique_state):
    """Divide ribosomes to follow their mRNA destination.

    Ribosomes on degraded mRNAs (not in either daughter's RNA set)
    are split binomially.

    Args:
        values: Structured array of active ribosome unique molecules.
        unique_state: Dict with 'RNA', 'active_RNAP', 'full_chromosome', 'chromosome_domain'.

    Returns:
        (daughter_1, daughter_2) arrays.
    """
    (mRNA_index,) = attrs(values, ['mRNA_index'])
    n_molecules = len(mRNA_index)

    if n_molecules == 0:
        return np.zeros(0, dtype=values.dtype), np.zeros(0, dtype=values.dtype)

    # Divide RNAs first to know where each mRNA goes
    d1_rnas, d2_rnas = divide_RNAs_by_domain(unique_state['RNA'], unique_state)
    d1_bool = np.isin(mRNA_index, d1_rnas['unique_index'])
    d2_bool = np.isin(mRNA_index, d2_rnas['unique_index'])

    # Handle ribosomes on degraded mRNAs
    unassigned = ~(d1_bool | d2_bool)
    degraded_mRNA_indexes = np.unique(mRNA_index[unassigned])
    n_degraded = len(degraded_mRNA_indexes)

    if n_degraded > 0:
        rng = np.random.RandomState(seed=n_molecules)
        n_d1 = rng.binomial(n_degraded, 0.5)
        d1_degraded = rng.choice(degraded_mRNA_indexes, size=n_d1, replace=False)
        d2_degraded = np.setdiff1d(degraded_mRNA_indexes, d1_degraded)
        d1_bool[np.isin(mRNA_index, d1_degraded)] = True
        d2_bool[np.isin(mRNA_index, d2_degraded)] = True

    assert n_molecules == d1_bool.sum() + d2_bool.sum()
    assert np.count_nonzero(np.logical_and(d1_bool, d2_bool)) == 0

    ribosomes = values[values['_entryState'].view(np.bool_)]
    return ribosomes[d1_bool], ribosomes[d2_bool]


# ---------------------------------------------------------------------------
# Dispatch table: unique molecule name → divider function
# ---------------------------------------------------------------------------

UNIQUE_DIVIDERS = {
    'full_chromosome': divide_by_domain,
    'chromosome_domain': divide_by_domain,
    'active_replisome': divide_by_domain,
    'oriC': divide_by_domain,
    'promoter': divide_by_domain,
    'gene': divide_by_domain,
    'DnaA_box': divide_by_domain,
    'active_RNAP': divide_by_domain,
    'chromosomal_segment': divide_by_domain,
    'RNA': divide_RNAs_by_domain,
    'active_ribosome': divide_ribosomes_by_RNA,
}


# ---------------------------------------------------------------------------
# Injected agent-root ("extra") stores: one carry/division policy
# ---------------------------------------------------------------------------
#
# A cell's agent state has FOUR stores the biology proper owns and this module
# knows how to split -- ``bulk`` (binomial), ``unique`` (domain), ``environment``
# and ``boundary`` (copied to both daughters). Everything else at the agent root
# is either per-tick runtime bookkeeping (rebuilt identically every generation)
# or a store some INJECTED process wired there: sms-ecoli's ``fields`` (the drug
# field), ``imposed_flux_bounds``, ``<drug>_env`` / ``<drug>_exchange``,
# ``periplasm``, ``cytoplasm``, ``counts``, ``kinetic_parameters``, the
# peptidoglycan wall ``pg_cellwall``, ...
#
# Before this policy existed those injected roots were simply DROPPED at every
# division and at every generation boundary: the daughter document is rebuilt
# from ``baseline()`` (fresh injected stores) and only the four core keys were
# overlaid onto it. A dose delivered into ``fields`` was re-zeroed (and, because
# the injected process is rebuilt with the cumulative ``lineage_time_offset``,
# re-fired) every generation; mecillinam wall damage could not accumulate; a
# ``lysed`` latch un-latched. The fork (vEcoli-private) does not lose them: it
# copies field-like environment stores to both daughters and splits the wall
# with a registered divider.
#
# The policy, applied identically by :func:`divide_cell` (the in-composite
# Division step) and by ``workflow.lineage``'s generation boundary:
#
#   * DEFAULT -- copy (deepcopy) the store into both daughters / carry it
#     forward, exactly the way ``environment`` is handled.
#   * A store may declare a DIVIDER (see :func:`register_store_divider`) and is
#     then split the way the fork splits it.
#   * ``listeners`` is never carried wholesale (it is re-derived, and the lineage
#     runner resets ``listeners.mass`` right after carry); an individual listener
#     LEAF may opt in via :func:`register_carried_listener_path`.

CORE_DIVISIBLE_KEYS = ('bulk', 'unique', 'environment', 'boundary')

#: Agent-root keys that are NEVER treated as carryable "extra" stores.
#: Two groups, both deliberate:
#:  * per-tick runtime bookkeeping -- re-derived every tick, so carrying a raw
#:    snapshot of one can only ever drop its updater (see the
#:    ``_FRESH_ENVIRONMENT_SUBSTORES`` note in ``v2ecoli/workflow/lineage.py``);
#:  * stores ``baseline()`` rebuilds identically for each daughter -- carrying
#:    ``allocator_rng`` would pin a daughter to its MOTHER's RNG stream instead
#:    of its own seeded one, and the two feature-config stores are pure config.
NON_CARRIED_ROOT_KEYS = frozenset({
    'listeners', 'global_time', 'timestep', 'next_update_time', 'process_state',
    'process', 'exchange', 'divide', 'division_threshold', 'first_update',
    'agents', 'allocator_rng', 'ppgpp_state', 'attenuation_config',
    # a Division PORT name (wired to environment/media_id), never an agent root
    'media_id',
    # Per-tick PARTITION bookkeeping (steps/partition.py groups these with
    # ``process``/``listeners`` as re-derived node stores): every Requester
    # overwrites its own ``request[<process>]`` each tick and the Allocator
    # writes ``allocate`` from whatever ``request`` holds. Carrying the mother's
    # snapshot seeds the daughter's first tick with the MOTHER's full-size
    # demands for every process, so the Allocator partitions a half-size cell
    # against stale requests (sims 943/944/945, 2026-09-10; v2ecoli#769).
    'request', 'allocate',
})

#: Agent-root stores the carry policy copies into the daughter ON PURPOSE (no
#: divider registered, so mother and daughter both get an independent deepcopy).
#: This is the explicit ALLOW-list that makes the policy auditable: every root
#: store a step/process declares must be in CORE_DIVISIBLE_KEYS, here, in
#: NON_CARRIED_ROOT_KEYS, or have a registered divider --
#: ``tests/test_root_store_classification.py`` enforces it, and the lineage
#: runner's ``division`` event reports any root that reaches a division without
#: a classification (the class of bug #765 was: ``request``/``allocate`` were
#: neither listed nor excluded, and were silently copied).
#: One line of reason each:
CARRIED_BY_COPY = frozenset({
    # injected environment / dose fields (sms-ecoli field_timeline, well-mixed
    # fields): the daughter lives in the mother's medium; the runner's fire-once
    # timeline re-fires from the cumulative offset, so a copy is the intent
    'fields',
    # injected FBA bound overrides -- pure config-shaped state, valid for both
    'imposed_flux_bounds',
    # cell_geometry feature: per-compartment volumes, re-derived every tick from
    # the daughter's own mass; copying only seeds the first tick sensibly
    'periplasm', 'cytoplasm',
    # ecoli_millard / fba_flux_coupler: per-tick derived flux vectors, re-written
    # each tick by their owning step; a stale copy is harmless for one tick
    'central_fluxes', 'pinned_flux_targets', 'bridge_diagnostics',
})

_EDGE_TYPES = frozenset({'process', 'step', 'composite', 'edge'})


def is_edge_node(value) -> bool:
    """True if ``value`` is a process-bigraph EDGE (a process/step node) rather
    than a data store. A full agent node holds both; only stores are carried."""
    if not isinstance(value, dict):
        return False
    if 'address' in value or 'instance' in value:
        return True
    node_type = value.get('_type')
    return isinstance(node_type, str) and node_type in _EDGE_TYPES


def extra_store_keys(cell_state) -> Tuple[str, ...]:
    """Sorted agent-root store keys subject to the extra-store carry policy.

    Everything at the root that is not one of :data:`CORE_DIVISIBLE_KEYS`, not in
    :data:`NON_CARRIED_ROOT_KEYS`, not a schema key (``_``-prefixed) and not a
    process/step edge. Safe to call on a FULL agent node (edges filtered out) or
    on a bare snapshot dict."""
    if not isinstance(cell_state, dict):
        return ()
    return tuple(sorted(
        key for key, value in cell_state.items()
        if isinstance(key, str)
        and not key.startswith('_')
        and key not in CORE_DIVISIBLE_KEYS
        and key not in NON_CARRIED_ROOT_KEYS
        and not is_edge_node(value)
    ))


#: store name -> ``divider(value) -> (daughter_1_value, daughter_2_value)``.
#:
#: THE DIVIDER CONTRACT. A divider takes the MOTHER's value for that one root
#: store and returns the two daughters' values. It is called exactly once per
#: division, receives nothing else, and must not mutate its argument -- any
#: configuration it needs (the fork's ``_divider: {"divider": ..., "config":
#: {...}}`` block: seeds, index arrays, maxima) is bound by the registrant, via
#: a closure or ``functools.partial``. A divider that raises is NOT swallowed.
#:
#: Downstream adoption is one line. sms-ecoli's ``pg_maturation.py`` already
#: builds exactly that config for its ``pg_cellwall`` port; in ``initialize()``
#: (where ``self.seed`` and ``self.idx`` exist) it registers:
#:
#:     register_store_divider("pg_cellwall", lambda pg, _c=divider_config:
#:                            divide_pg_cellwall(pg, _c))
#:
#: and ``pg_cellwall`` is then split at division instead of copied, matching
#: vEcoli-private's ``divider_registry`` entry of the same name.
STORE_DIVIDERS: Dict[str, Any] = {}

#: Listener LEAF paths (tuples relative to the agent's ``listeners`` store) that
#: survive a division / generation boundary. ``listeners`` is otherwise entirely
#: re-derived; this is the narrow opt-out for a LATCH -- e.g. sms-ecoli's
#: ``("peptidoglycan_shape", "lysed")``, which must stay latched once set.
CARRIED_LISTENER_PATHS: List[Tuple[str, ...]] = []


def register_store_divider(store_name: str, divider) -> None:
    """Register ``divider`` for the agent-root store ``store_name``.

    See :data:`STORE_DIVIDERS` for the contract. Idempotent-friendly: a repeat
    registration of the SAME callable is a no-op, a different one replaces it
    (module-level registration re-runs whenever a daughter rebuild re-imports
    the declaring module)."""
    if not callable(divider):
        raise TypeError(f"divider for {store_name!r} is not callable: {divider!r}")
    STORE_DIVIDERS[store_name] = divider


def register_carried_listener_path(path) -> None:
    """Declare one listener LEAF (a path under ``listeners``) as carried.

    ``path`` is a tuple/list of keys, or a dotted string; a leading ``listeners``
    segment is accepted and stripped. See :data:`CARRIED_LISTENER_PATHS`."""
    if isinstance(path, str):
        path = tuple(p for p in path.split('.') if p)
    else:
        path = tuple(path)
    if path and path[0] == 'listeners':
        path = path[1:]
    if not path:
        raise ValueError("register_carried_listener_path: empty listener path")
    if path not in CARRIED_LISTENER_PATHS:
        CARRIED_LISTENER_PATHS.append(path)


def resolve_store_dividers(dividers=None) -> Dict[str, Any]:
    """The effective divider table: the module registry, overlaid by an explicit
    ``dividers`` mapping (a caller-supplied table always wins)."""
    resolved = dict(STORE_DIVIDERS)
    if dividers:
        resolved.update(dividers)
    return resolved


def divide_extra_stores(cell_state, dividers=None) -> Tuple[Dict, Dict]:
    """Apply the extra-store policy to every extra root of ``cell_state``.

    Returns ``(daughter_1_extras, daughter_2_extras)``: a registered divider's
    two values for a store that declares one, and an independent deepcopy each
    for every other store."""
    resolved = resolve_store_dividers(dividers)
    d1: Dict[str, Any] = {}
    d2: Dict[str, Any] = {}
    for key in extra_store_keys(cell_state):
        value = cell_state[key]
        divider = resolved.get(key)
        if divider is None:
            d1[key] = copy.deepcopy(value)
            d2[key] = copy.deepcopy(value)
            continue
        split = divider(value)
        d1[key], d2[key] = split[0], split[1]
    return d1, d2


def merge_carried_store(fresh, carried):
    """Overlay a CARRIED store value onto the freshly built node, typed-safely.

    The fresh daughter / next-generation document may represent an injected root
    as a dict that carries a ``_type`` key (sms-ecoli's
    ``_materialize_native_declared_state`` stamps e.g.
    ``fields: {"_type": "map[overwrite[array[float]]]", <mol>: zeros}``).
    REPLACING that node with the carried raw dict would drop the ``_type`` and
    with it the store's overwrite updater -- the same trap
    ``_FRESH_ENVIRONMENT_SUBSTORES`` guards for ``environment.exchange_data``,
    whose additive fallback silently balloons a per-tick bound. So merge LEAVES
    into the existing node instead: schema keys (``_``-prefixed) and any key the
    carried state lacks stay as the fresh build wrote them, and carried leaves
    replace the fresh zero seeds."""
    if not (isinstance(fresh, dict) and isinstance(carried, dict)):
        return carried
    merged = dict(fresh)
    for key, value in carried.items():
        if isinstance(key, str) and key.startswith('_'):
            continue                      # keep the FRESH node's schema keys
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_carried_store(merged[key], value)
        else:
            merged[key] = value
    return merged


def collect_carried_listeners(cell_state, paths=None) -> Dict:
    """Pluck the declared carried listener leaves out of ``cell_state`` as a
    nested dict shaped like ``listeners`` (``{}`` when none are present)."""
    listeners = (cell_state or {}).get('listeners')
    if not isinstance(listeners, dict):
        return {}
    out: Dict[str, Any] = {}
    for path in (CARRIED_LISTENER_PATHS if paths is None else paths):
        node = listeners
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is None:
            continue
        target = out
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = copy.deepcopy(node)
    return out


def apply_carried_listeners(agent, carried_listeners) -> None:
    """Merge carried listener leaves back into ``agent['listeners']`` in place."""
    if not carried_listeners:
        return
    listeners = agent.setdefault('listeners', {})
    if not isinstance(listeners, dict):
        return
    agent['listeners'] = merge_carried_store(listeners, carried_listeners)


# ---------------------------------------------------------------------------
# Top-level cell division
# ---------------------------------------------------------------------------

def daughter_phylogeny_id(mother_id):
    """Generate daughter IDs from mother ID."""
    return [str(mother_id) + '0', str(mother_id) + '1']


def divide_cell(cell_state, dividers=None):
    """Divide a cell's data stores into two daughter initial states.

    Args:
        cell_state: Dict with 'bulk', 'unique', 'environment', 'boundary' and any
            INJECTED agent-root stores ('fields', 'imposed_flux_bounds',
            'pg_cellwall', ...).
        dividers: Optional ``{store_name: divider}`` table overlaid on the module
            registry (:func:`register_store_divider`) for this call only. See
            :data:`STORE_DIVIDERS` for the divider contract.

    Returns:
        (daughter_1_state, daughter_2_state) — data stores only, no step instances.

    ``bulk`` splits binomially and ``unique`` by chromosome domain, as always.
    ``environment``/``boundary`` are deep-copied to both daughters, as always.
    Every OTHER root store (see :func:`extra_store_keys`) now follows the
    extra-store policy documented above: deep-copied to both daughters by
    default, or split by its registered divider. Declared carried listener
    leaves ride along under ``_carried_listeners``.
    """
    # 1. Divide bulk molecules
    d1_bulk, d2_bulk = divide_bulk(cell_state['bulk'])

    # 2. Divide unique molecules
    # Build the shared state dict needed by domain-based dividers
    unique = cell_state['unique']
    unique_state = {
        'full_chromosome': unique['full_chromosome'],
        'chromosome_domain': unique['chromosome_domain'],
        'active_RNAP': unique['active_RNAP'],
        'RNA': unique['RNA'],
    }

    d1_unique = {}
    d2_unique = {}
    for name, arr in unique.items():
        if not hasattr(arr, 'dtype'):
            # Not a numpy array, just copy
            d1_unique[name] = arr
            d2_unique[name] = arr
            continue

        divider = UNIQUE_DIVIDERS.get(name)
        if divider is None:
            # Unknown molecule type — just copy to both
            d1_unique[name] = arr.copy()
            d2_unique[name] = arr.copy()
            continue

        if divider == divide_by_domain:
            d1_unique[name], d2_unique[name] = divide_by_domain(arr, unique_state)
        elif divider == divide_RNAs_by_domain:
            d1_unique[name], d2_unique[name] = divide_RNAs_by_domain(arr, unique_state)
        elif divider == divide_ribosomes_by_RNA:
            d1_unique[name], d2_unique[name] = divide_ribosomes_by_RNA(arr, unique_state)

    # 3. Build daughter initial states
    d1_state = {
        'bulk': d1_bulk,
        'unique': d1_unique,
    }
    d2_state = {
        'bulk': d2_bulk,
        'unique': d2_unique,
    }

    # Copy environment (both daughters inherit the same environment)
    if 'environment' in cell_state:
        d1_state['environment'] = copy.deepcopy(cell_state['environment'])
        d2_state['environment'] = copy.deepcopy(cell_state['environment'])

    if 'boundary' in cell_state:
        d1_state['boundary'] = copy.deepcopy(cell_state['boundary'])
        d2_state['boundary'] = copy.deepcopy(cell_state['boundary'])

    # 4. Injected agent-root stores: copy by default, divider when declared.
    #    Without this every store an injected process wires at the agent root
    #    (fields / imposed_flux_bounds / <drug>_env / periplasm / pg_cellwall /
    #    ...) came back FRESH in both daughters, because the daughter document
    #    is rebuilt from baseline() and only the keys above were overlaid.
    d1_extra, d2_extra = divide_extra_stores(cell_state, dividers)
    d1_state.update(d1_extra)
    d2_state.update(d2_extra)

    # 5. Declared listener LEAVES (a latch such as
    #    listeners.peptidoglycan_shape.lysed). `listeners` as a whole is
    #    re-derived and must NOT be carried; only the declared leaves ride along.
    carried_listeners = collect_carried_listeners(cell_state)
    if carried_listeners:
        d1_state['_carried_listeners'] = carried_listeners
        d2_state['_carried_listeners'] = copy.deepcopy(carried_listeners)

    return d1_state, d2_state
