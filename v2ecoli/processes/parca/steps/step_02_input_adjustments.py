"""Step 2 — input_adjustments.  Apply literature-curated overrides before fitting.

A handful of genes are known from experiment to need their RNA
expression, translation efficiency, or degradation rate multiplied by a
fixed factor before any fitting begins. This step reads those factors
from the ``adjustments`` dataclass and applies them in place.

Mathematical Model
------------------

Inputs:
- transcription.rna_expression (basal) and transcription.rna_deg_rates.
- translation.translation_efficiencies_by_monomer.
- adjustments: a lookup table of {gene_id: multiplier} for each kind of
  adjustment, shipped in ``flat/adjustments/``.
- tf_to_active_inactive_conditions (reduced to one entry in debug mode).

Parameters:
- debug (bool): when True, prunes tf_to_active_inactive_conditions to a
  single TF so step 4 only runs one condition-fit.

Calculation:
- adjust_translation_efficiencies: scalar multiplier per monomer id.
- balance_translation_efficiencies: renormalize so mean(eff) == 1.
- adjust_rna_expression: scalar multiplier per rna id, then renormalize
  so sum(expression) == 1.
- adjust_rna_deg_rates, adjust_protein_deg_rates: scalar multipliers.

Outputs:
- transcription (mutated): rna_expression + rna_deg_rates updated.
- translation (mutated): translation_efficiencies_by_monomer updated.
- tf_to_active_inactive_conditions (pruned when debug=True).
"""

import time
import warnings

import numpy as np

from process_bigraph import Step


# ============================================================================
# Pure sub-functions (unchanged — take explicit numpy arrays)
# ============================================================================

def adjust_translation_efficiencies(monomer_ids, efficiencies, adjustments):
    """Multiply translation efficiencies by specified per-monomer factors.

    Args:
        monomer_ids: array of monomer IDs aligned with ``efficiencies``.
        efficiencies: numpy array (may be mutated in place; caller copies).
        adjustments: dict {monomer_id: adjustment_factor}.
    Returns:
        the adjusted numpy array.
    """
    for monomer_id, adjustment in adjustments.items():
        idx = np.where(monomer_ids == monomer_id)[0]
        efficiencies[idx] = efficiencies[idx] * adjustment
    return efficiencies


def balance_translation_efficiencies(monomer_ids, efficiencies, groups):
    """Average translation efficiencies across balanced groups.

    Args:
        monomer_ids: monomer IDs aligned with ``efficiencies``.
        efficiencies: numpy array.
        groups: list of lists — each sub-list is a set of monomer IDs to average.
    Returns:
        the adjusted numpy array.

    The group IDs are *bare* monomer IDs (no compartment), while ``monomer_ids``
    carry a trailing compartment tag (e.g. ``'EG10866-MONOMER[c]'``). Strip the
    3-char ``[X]`` tag before matching so the groups actually match — mirroring
    vEcoli's ``set_balanced_translation_efficiencies`` (``monomer["id"][:-3]``).
    Without this, no group ever matched and the (ribosomal-protein) groups were
    left unbalanced, leaving raw per-gene efficiencies that diverged from vEcoli.
    """
    bare_ids = [m[:-3] for m in monomer_ids]
    for group in groups:
        group = set(group)
        idx = np.array([
            i for i, m in enumerate(bare_ids) if m in group
        ])
        if len(idx) > 0:
            efficiencies[idx] = np.mean(efficiencies[idx])
    return efficiencies


def _combine_geometric(factors):
    """Geometric mean — the default. Several cistrons of one operon are repeated
    observations of ONE transcript on a multiplicative scale, so the natural
    pooling is the mean of their logs. Direction-symmetric, and a zero factor
    (a knockout) survives: gm(0, x) == 0.
    """
    factors = np.asarray(factors, dtype=float)
    if np.any(factors == 0.0):
        return 0.0
    if np.any(factors < 0.0):
        raise ValueError(f"negative adjustment factor in {list(factors)}")
    return float(np.exp(np.mean(np.log(factors))))


def _combine_max_guarded(factors):
    """`max`, refusing the inputs on which `max` is not meaningful.

    Provided for faithful reproduction of upstream work that used a plain
    ``max``. ⛔ Bare ``max`` is directionally asymmetric — for up-regulation it
    takes the most extreme observation, for down-regulation the LEAST, and a
    knockout is erased outright by any co-located up-regulation
    (``max(0.0, 3.0) == 3.0``). This raises on exactly those inputs instead of
    silently returning one, so "faithful" cannot quietly become "wrong".
    """
    factors = np.asarray(factors, dtype=float)
    if np.any(factors < 0.0):
        raise ValueError(f"negative adjustment factor in {list(factors)}")
    up, down = np.any(factors > 1.0), np.any(factors < 1.0)
    if up and down:
        raise ValueError(
            f"max_guarded refuses a direction-discordant transcription unit: "
            f"{list(factors)}. `max` would return the up-regulated factor and "
            f"discard the down-regulated one (a 0.0 knockout included). Use the "
            f"default 'geometric' combiner, or resolve the disagreement upstream."
        )
    return float(np.max(factors))


#: How several adjusted cistrons on ONE transcription unit are combined into the
#: single factor that TU's expression is multiplied by. This is a modelling
#: choice, not an implementation detail — see the two functions above.
COMBINERS = {
    "geometric": _combine_geometric,
    "max_guarded": _combine_max_guarded,
}
DEFAULT_COMBINER = "geometric"


def adjust_rna_expression(
    rna_ids, cistron_ids, rna_expression, adjustments, cistron_to_rna_indexes,
    combine=DEFAULT_COMBINER,
):
    """Apply adjustments to RNA expression, renormalize.

    Each key is a cistron id, or a transcription-unit id. Every adjusted cistron
    resolves to the TU(s) carrying it, the factors landing on one TU are combined
    ONCE, and the whole vector is renormalized to sum to 1.

    ⛔ **Several adjusted cistrons on one TU are combined, not compounded.** The
    cistrons of an operon are carried by the *same molecule*, so several large
    measurements across one operon are several observations of one transcript,
    not several multiplicative ones. Multiplying per cistron — what this function
    used to do — takes the product, which on a differential-expression-derived
    table can exceed the intended factor by many orders of magnitude. Because the
    vector is renormalized immediately afterwards it stays a valid distribution
    and **raises nothing**: the result is a silently different organism, not an
    error.

    ``combine`` selects how (see :data:`COMBINERS`); the default geometric mean
    is direction-symmetric and preserves a knockout.

    ⚠ Scoped claim: the stock ``rna_expression_adjustments`` table is ~10
    hand-curated single-cistron entries with no shared TU, so for THAT table
    every combiner agrees with the product and existing builds are unchanged.
    ⛔ Do not generalise that to the sibling adjustment tables — the stock
    ``rna_deg_rates_adjustments`` table DOES contain two cistrons sharing two
    TUs, and :func:`adjust_rna_deg_rates` still compounds them.

    Args:
        rna_ids: RNA IDs aligned with ``rna_expression``.
        cistron_ids: cistron IDs.
        rna_expression: numpy array of basal RNA expression (mutated in place).
        adjustments: dict {cistron_id or rna_id: adjustment_factor}.
        cistron_to_rna_indexes: dict {cistron_id: array of RNA indexes}.
        combine: key into :data:`COMBINERS`.
    Returns:
        the adjusted (still-normalized) numpy array — the same object passed in.
    Raises:
        ValueError: an unknown id, an unknown combiner, or input the chosen
            combiner refuses.
    """
    try:
        combiner = COMBINERS[combine]
    except KeyError:
        raise ValueError(
            f"unknown combiner {combine!r}; expected one of {sorted(COMBINERS)}"
        ) from None

    # An id is looked up in the SAME mapping it will be fetched from, so the
    # function does not silently depend on the caller having built
    # `cistron_to_rna_indexes` from exactly `cistron_ids`.
    known_cistrons = set(map(str, cistron_to_rna_indexes))

    # RNA ids carry a compartment suffix (`EG10054_RNA[c]`). Accept the suffixed
    # form as a TU address; `setdefault` so a stripped alias can never displace
    # an exact id.
    # ⚠ A BARE id is resolved as a CISTRON first, and for a monocistronic TU the
    # two spellings coincide — so a bare id may reach every TU carrying that
    # cistron, not one. Address a TU by its suffixed id when you mean the TU.
    rna_index_by_id = {}
    for i, rna_id in enumerate(rna_ids):
        rna_id = str(rna_id)
        rna_index_by_id.setdefault(rna_id, i)
    for i, rna_id in enumerate(rna_ids):
        rna_id = str(rna_id)
        if rna_id.endswith("]") and "[" in rna_id:
            rna_index_by_id.setdefault(rna_id[: rna_id.rindex("[")], i)

    factors_by_index: dict[int, list[float]] = {}
    for mol_id, adjustment in adjustments.items():
        mol_id = str(mol_id)
        if mol_id in known_cistrons:
            rna_indexes = cistron_to_rna_indexes[mol_id]
        elif mol_id in rna_index_by_id:
            rna_indexes = [rna_index_by_id[mol_id]]
        else:
            raise ValueError(
                f"RNA expression adjustment {mol_id!r} is neither a known "
                "cistron id nor a known RNA id."
            )
        # `unique`: one cistron listing an index twice is ONE observation.
        for rna_index in np.unique(np.atleast_1d(rna_indexes)):
            factors_by_index.setdefault(int(rna_index), []).append(adjustment)

    shared = {i: f for i, f in factors_by_index.items() if len(f) > 1}
    if shared:
        # Loud on purpose, and a warning rather than a print so a caller can
        # assert on it: which cistrons share a TU is a property of the operon
        # structure, not of the adjustments table, so an author cannot see this
        # coming from their own file.
        warnings.warn(
            f"{len(shared)} of {len(factors_by_index)} adjusted transcription "
            f"unit(s) carry more than one adjusted cistron; combining each with "
            f"'{combine}' rather than compounding.",
            stacklevel=2,
        )

    for rna_index, factors in factors_by_index.items():
        # A single observation is passed through untouched rather than round-
        # tripped through the combiner: exp(log(f)) != f in floating point, and
        # the regression contract for tables with no shared TU is BIT-identity,
        # not approximate agreement.
        factor = factors[0] if len(factors) == 1 else combiner(factors)
        rna_expression[rna_index] = rna_expression[rna_index] * factor
    rna_expression /= rna_expression.sum()
    return rna_expression


def adjust_rna_deg_rates(
    rna_ids, cistron_ids, rna_deg_rates, cistron_deg_rates,
    adjustments, cistron_to_rna_indexes,
):
    """Apply per-cistron degradation-rate adjustments to both the RNA and
    cistron arrays (Unum / structured-array aware, handled by caller).

    Returns:
        (new_rna_deg_rates, new_cistron_deg_rates) pair.
    """
    for cistron_id, adjustment in adjustments.items():
        rna_indexes = cistron_to_rna_indexes[cistron_id]
        rna_deg_rates[rna_indexes] = rna_deg_rates[rna_indexes] * adjustment
        cistron_idx = np.where(cistron_ids == cistron_id)[0]
        cistron_deg_rates[cistron_idx] = cistron_deg_rates[cistron_idx] * adjustment
    return rna_deg_rates, cistron_deg_rates


def adjust_protein_deg_rates(monomer_ids, rates, adjustments):
    """Apply per-monomer degradation-rate adjustments.

    Returns:
        the adjusted numpy array.
    """
    for monomer_id, adjustment in adjustments.items():
        idx = np.where(monomer_ids == monomer_id)[0]
        rates[idx] = rates[idx] * adjustment
    return rates


# ============================================================================
# Step class
# ============================================================================

INPUT_PORTS = {
    'tick_1'                            : 'overwrite',
    'transcription':                    'sim_data.transcription',
    'translation':                      'sim_data.translation',
    'adjustments':                      'overwrite',
    'tf_to_active_inactive_conditions': 'overwrite',
}

OUTPUT_PORTS = {
    'tick_2'                            : 'overwrite',
    'transcription':                    'sim_data.transcription',
    'translation':                      'sim_data.translation',
    'tf_to_active_inactive_conditions': 'overwrite',
}


def select_debug_tf_conditions(tf_cond: dict) -> dict:
    """The single TF whose regulation a ``debug=True`` (fast) build applies.

    ⚠ The selection is POSITIONAL, and that is the whole hazard: it takes the
    first key in insertion order, and ``tf_to_active_inactive_conditions`` is
    built by iterating ``condition/tf_condition.tsv`` in row order
    (``simulation_data.py``, ``_add_condition_data``). So which transcription
    factor a fast build models is decided by *which row is first in a data
    file*, filtered to rows whose active TF also appears in the fold-change
    tables.

    At the time of writing that is ``trpR`` (``CPLX-125``) out of 23 declared
    TFs; every other TF's regulation is dropped. Treat that as a fact to look up
    rather than a guarantee — change the ordering, or the fold-change tables'
    membership, and fast builds silently model a different regulator. Nothing
    fails: the run completes and its numbers quietly stop meaning what they
    meant, and a knockout of a TF the fast regime dropped returns a plausible
    null rather than an error.

    ⇒ Do not read regulatory behaviour out of a fast-mode build without checking
    which TF survived.

    Extracted from ``update`` so the selection is named and testable;
    ``tests/test_fast_mode_tf_prune.py`` pins that it stays positional and
    single-valued, deliberately not which TF wins (that is upstream data).
    """
    first_key = next(iter(tf_cond))
    return {first_key: tf_cond[first_key]}


class InputAdjustmentsStep(Step):
    """Step 2 — input_adjustments.  See module docstring for port wiring."""

    description = (
        "Step 2 — input_adjustments.\n\n"
        "Applies literature-curated per-gene overrides before any fitting,\n"
        "from the adjustments table {gene_id: multiplier}:\n"
        "    rna_expression  *= f_expr,   then renormalize Σ = 1\n"
        "    translation_eff *= f_te,     then renormalize mean = 1\n"
        "    rna_deg_rates   *= f_deg;    protein_deg_rates *= f_pdeg\n"
        "Mutates the transcription and translation subsystems in place.\n"
        "In debug mode also prunes tf_to_active_inactive_conditions to one\n"
        "TF so Step 4 fits a single condition."
    )

    config_schema = {
        'debug': {'_type': 'boolean', '_default': False},
        # How co-located adjusted cistrons combine; see COMBINERS.
        'rna_expression_adjustment_combine': {
            '_type': 'string', '_default': DEFAULT_COMBINER},
    }

    def inputs(self):
        return dict(INPUT_PORTS)

    def outputs(self):
        return dict(OUTPUT_PORTS)

    def update(self, state):
        t0 = time.time()

        transcription = state['transcription']
        translation   = state['translation']
        adjustments   = state['adjustments']
        tf_cond       = state['tf_to_active_inactive_conditions']

        # --- debug: optionally trim TF conditions ---
        tf_cond_out = None
        if self.config.get('debug', False):
            print(
                "  Step 2: debug mode — reducing tf_to_active_inactive_conditions"
                " to a single key"
            )
            tf_cond_out = select_debug_tf_conditions(tf_cond)

        # --- translation efficiencies ---
        monomer_ids = translation.monomer_data['id']
        efficiencies = translation.translation_efficiencies_by_monomer.copy()
        efficiencies = adjust_translation_efficiencies(
            monomer_ids, efficiencies,
            dict(adjustments.translation_efficiencies_adjustments),
        )
        efficiencies = balance_translation_efficiencies(
            monomer_ids, efficiencies,
            list(adjustments.balanced_translation_efficiencies),
        )
        translation.translation_efficiencies_by_monomer[:] = efficiencies

        # --- RNA expression ---
        rna_ids     = transcription.rna_data['id']
        cistron_ids = transcription.cistron_data['id']
        cistron_to_rna_indexes = {
            cid: transcription.cistron_id_to_rna_indexes(cid)
            for cid in cistron_ids
        }
        new_rna_expr = adjust_rna_expression(
            rna_ids, cistron_ids,
            transcription.rna_expression['basal'].copy(),
            dict(adjustments.rna_expression_adjustments),
            cistron_to_rna_indexes,
            combine=self.config.get(
                'rna_expression_adjustment_combine', DEFAULT_COMBINER),
        )
        transcription.rna_expression['basal'][:] = new_rna_expr

        # --- RNA + cistron degradation rates ---
        new_rna_deg, new_cistron_deg = adjust_rna_deg_rates(
            rna_ids, cistron_ids,
            transcription.rna_data.struct_array['deg_rate'].copy(),
            transcription.cistron_data.struct_array['deg_rate'].copy(),
            dict(adjustments.rna_deg_rates_adjustments),
            cistron_to_rna_indexes,
        )
        transcription.rna_data.struct_array['deg_rate'][:]     = new_rna_deg
        transcription.cistron_data.struct_array['deg_rate'][:] = new_cistron_deg

        # --- protein degradation rates ---
        new_prot_deg = adjust_protein_deg_rates(
            translation.monomer_data['id'],
            translation.monomer_data.struct_array['deg_rate'].copy(),
            dict(adjustments.protein_deg_rates_adjustments),
        )
        translation.monomer_data.struct_array['deg_rate'][:] = new_prot_deg

        print(f"  Step 2 (input_adjustments) completed in {time.time() - t0:.1f}s")

        out = {
            'transcription': transcription,
            'translation':   translation,
        }
        if tf_cond_out is not None:
            out['tf_to_active_inactive_conditions'] = tf_cond_out
        return out
