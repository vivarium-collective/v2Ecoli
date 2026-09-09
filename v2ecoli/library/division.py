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


def divide_nascent_flagellum(values, unique_state):
    """Divide in-progress flagella binomially between daughters.

    Added 2026-08-06, flagella-cascade investigation. Before this, any
    unique molecule type without an entry in UNIQUE_DIVIDERS fell through to
    divide_cell's catch-all "unknown type -> copy whole array to BOTH
    daughters" -- physically impossible (a single anchored structure can't
    become two) and not mass-conserving (every nascent flagellum would be
    duplicated at every division, inflating the count exponentially across
    generations and directly undermining any multi-generation test of
    whether flagella count self-limits).

    Real flagella are physically anchored to a fixed point on the cell
    envelope; division splits that envelope roughly in half, so each
    existing structure (complete or in-progress) ends up in whichever
    daughter inherits that patch of membrane -- for a peritrichous,
    ~randomly-distributed set of flagella, a binomial p=0.5 assignment per
    flagellum (each independently, entirely, to one daughter or the other --
    not split fractionally) is the natural approximation, mirroring how
    complete flagella (CPLX0-7452, an ordinary bulk molecule) already divide
    via divide_bulk's binomial split.
    """
    active = values[values['_entryState'].view(np.bool_)]
    n_molecules = len(active)
    if n_molecules == 0:
        return active, active
    seed = int(active['filament_length'].sum() + n_molecules) % RAND_MAX
    rng = np.random.RandomState(seed=seed)
    d1_bool = rng.random(n_molecules) < 0.5
    return active[d1_bool], active[~d1_bool]


def divide_internal_observables(state):
    """Binomial(n, 0.5) split of the NFsim complexation Step's
    internal-only observable tallies -- 'flagellar_hook' (completed-but-
    unconsumed hook count) and 'flagella' (cumulative hook-basal-body
    completions, the trigger for nascent_flagellum creation). See
    flagella_nfsim_complexation.py's docstring for why these have no real
    bulk molecule ID.

    Added 2026-08-19, flagella-cascade investigation. Before this, these
    two top-level agent-state ports had NO entry anywhere in the division
    machinery (divide_cell only ever produced 'bulk'/'unique'/'environment'/
    'boundary'), so daughters were silently rebuilt with baseline()'s
    schema default ({}) -- discarding all accumulated progress at every
    division, not a physically-motivated reset.

    Real flagellar assembly timing (revised 2026-08-24, see CORRECTION
    below): no direct E. coli measurement of total flagellum assembly time
    exists in the literature (checked directly, dedicated search). The
    closest real data is in Salmonella typhimurium -- a close relative with
    a near-identical peritrichous flagellar apparatus, routinely used
    interchangeably with E. coli in this exact research area (e.g. Renault
    et al. 2017, cited below, is already this codebase's own source for
    flagella_filament_elongation.py's elongation rate law): hook-basal-body
    completion takes ~30 min (Karlinsey et al. 2000, Mol Microbiol 37:1220),
    while full filament growth to ~10 um (~21,300 subunits) takes over 180
    minutes / 3+ hours (Renault et al. 2017, eLife 6:e23136). Since a
    fast-growing E. coli's own division cycle is ~40-44 min, if E. coli's
    kinetics are broadly similar to Salmonella's, full filament completion
    would typically NOT finish within a single division -- real, if
    cross-species, grounds for this port's design (carry state across
    divisions rather than discard it).

    CORRECTION (2026-08-24): a previous version of this docstring instead
    claimed "basal bodies assemble in <5 min but motility-capable filaments
    take >40 min," citing Sim et al. 2017 (PMC5259725, real E. coli) and
    Zhuang et al. 2020 (Mol Microbiol 114:279-291, Vibrio alginolyticus) for
    those specific numbers. Checked directly (Maya's request): Sim et al.
    2017 measures flagella COUNT vs. growth rate, not assembly timing, and
    Zhuang et al. 2020 is cited correctly elsewhere in this docstring for a
    DIFFERENT claim (old-pole inheritance) -- neither paper actually
    supports the timing numbers they were attached to. Replaced with the
    Salmonella data above, which is real and honestly cross-species rather
    than misattributed to the wrong E. coli papers.

    True biology has a documented old-pole inheritance bias for flagellar
    structures. Zhuang et al. 2020 (Mol Microbiol 114:279-291) is the
    clearest mechanistic account, but it's in Vibrio alginolyticus, not E.
    coli -- cross-species conceptual support, not a validated E. coli
    mechanism. There IS direct E. coli evidence too, just a different,
    weaker kind: Ping (2010) J Mol Biol, PMID 20156455, found ~75% of cells
    carry more flagella on their old-pole half than their new-pole half --
    a steady-state population snapshot, not a single flagellum tracked
    live across an actual division event. Reproducing either would require
    sub-cellular position tracking this (currently well-mixed) whole-cell
    model doesn't have, and a broader review
    (academic.oup.com/femsre/article/39/6/812/550689) calls the precise
    mechanism "largely unresolved" even for E. coli specifically, so
    there's no citable split ratio to implement against. Binomial p=0.5 is
    the same well-mixed-model approximation already used for every other
    discrete structural count in this file (divide_bulk,
    divide_nascent_flagellum) -- not a claim of spatial accuracy, just
    conservation of the real count instead of silent loss.
    """
    if not state:
        return {}, {}
    total = int(round(sum(state.values())))
    seed = total % RAND_MAX
    rng = np.random.RandomState(seed=seed)
    d1, d2 = {}, {}
    for key, val in state.items():
        n = int(round(val))
        d1_n = int(rng.binomial(n, 0.5)) if n > 0 else 0
        d1[key] = float(d1_n)
        d2[key] = float(n - d1_n)
    return d1, d2


def divide_scaffold_species(state):
    """Binomial(n, 0.5) split of each distinct NFsim scaffold occupancy-
    state count (e.g. 'Growing_CPLX0_7450_i(FLIF~17,...)': 2 -> one
    daughter might get 1, the other 1) -- same rationale and same
    well-mixed-model caveat as divide_internal_observables. Each entry's
    count is independently binomial-split; entries that round to 0 for a
    daughter are omitted rather than kept as an explicit zero, matching
    pbg_nfsim's own convention of only listing seed species with count>0.
    """
    if not state:
        return {}, {}
    total = int(round(sum(state.values())))
    seed = total % RAND_MAX
    rng = np.random.RandomState(seed=seed)
    d1, d2 = {}, {}
    for pattern, count in state.items():
        n = int(round(count))
        if n <= 0:
            continue
        d1_n = int(rng.binomial(n, 0.5))
        d2_n = n - d1_n
        if d1_n > 0:
            d1[pattern] = float(d1_n)
        if d2_n > 0:
            d2[pattern] = float(d2_n)
    return d1, d2


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
    'nascent_flagellum': divide_nascent_flagellum,
}


# ---------------------------------------------------------------------------
# Top-level cell division
# ---------------------------------------------------------------------------

def daughter_phylogeny_id(mother_id):
    """Generate daughter IDs from mother ID."""
    return [str(mother_id) + '0', str(mother_id) + '1']


def divide_cell(cell_state):
    """Divide a cell's data stores into two daughter initial states.

    Args:
        cell_state: Dict with 'bulk', 'unique', 'environment', 'listeners',
            and (when flagella_nfsim_complexation is active)
            'nfsim_scaffold_species'/'nfsim_internal_observables', etc.

    Returns:
        (daughter_1_state, daughter_2_state) — data stores only, no step instances.
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
        elif divider == divide_nascent_flagellum:
            d1_unique[name], d2_unique[name] = divide_nascent_flagellum(arr, unique_state)

    # 3. Build daughter initial states
    d1_state = {
        'bulk': d1_bulk,
        'unique': d1_unique,
    }
    d2_state = {
        'bulk': d2_bulk,
        'unique': d2_unique,
    }

    # NFsim complexation Step's own state ports (2026-08-19, see
    # divide_internal_observables/divide_scaffold_species docstrings) --
    # only present when flagella_nfsim_complexation is wired in, so guard
    # on key presence rather than assuming every composite has them.
    if 'nfsim_scaffold_species' in cell_state:
        d1_state['nfsim_scaffold_species'], d2_state['nfsim_scaffold_species'] = (
            divide_scaffold_species(cell_state['nfsim_scaffold_species']))
    if 'nfsim_internal_observables' in cell_state:
        d1_state['nfsim_internal_observables'], d2_state['nfsim_internal_observables'] = (
            divide_internal_observables(cell_state['nfsim_internal_observables']))

    # Copy environment (both daughters inherit the same environment)
    if 'environment' in cell_state:
        d1_state['environment'] = copy.deepcopy(cell_state['environment'])
        d2_state['environment'] = copy.deepcopy(cell_state['environment'])

    if 'boundary' in cell_state:
        d1_state['boundary'] = copy.deepcopy(cell_state['boundary'])
        d2_state['boundary'] = copy.deepcopy(cell_state['boundary'])

    return d1_state, d2_state
