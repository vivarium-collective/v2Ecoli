"""Shared data shims for native vEcoli analysis ports.

v2ecoli's parquet schema differs from vEcoli in four ways that matter for
the ptools analyses:

    Shim A – bulk count matrix
        vEcoli stores a single ``bulk`` list column (counts in sim_data order).
        v2ecoli stores ``bulk__id`` + ``bulk__count`` (both per-row lists,
        ordering may differ from sim_data).  ``bulk_count_matrix`` re-indexes
        the counts to sim_data column order so all downstream index arithmetic
        is identical to the vEcoli original.

    Shim B – active ribosome scalar
        vEcoli emits ``listeners__unique_molecule_counts__active_ribosome``
        (a scalar per row).  v2ecoli has no such column; the equivalent is the
        sum of ``listeners__ribosome_data__n_ribosomes_per_transcript``
        (a list per row).  Use the SQL snippet ``ACTIVE_RIBOSOME_SQL`` in
        SELECT lists and read the resulting ``active_ribosome`` column.

    Shim C – oriC count scalar
        vEcoli emits ``listeners__unique_molecule_counts__oriC`` (scalar).
        v2ecoli uses ``listeners__replication_data__number_of_oric`` (also
        scalar).  Use ``ORIC_SQL`` in SELECT lists and read ``oriC``.

    Shim D – active RNAP scalar
        vEcoli emits ``listeners__unique_molecule_counts__active_RNAP``
        (scalar).  v2ecoli has no such column; the equivalent is the LENGTH of
        ``listeners__rnap_data__active_rnap_unique_indexes`` (an INTEGER[]
        per row).  Use ``ACTIVE_RNAP_SQL`` in SELECT lists and read
        ``active_RNAP``.

    Shim E – external exchange flux ordering
        vEcoli explodes external exchange fluxes into one column per molecule
        (``listeners__fba_results__external_exchange_fluxes__<MOL>[p]``), so a
        port can find them by name.  v2ecoli emits a single 87-wide list column
        with no name metadata.  ``external_exchange_molecule_ids`` returns the
        ordering to index it by.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shim B — active-ribosome SQL snippet
# ---------------------------------------------------------------------------

ACTIVE_RIBOSOME_SQL = (
    "list_sum(listeners__ribosome_data__n_ribosomes_per_transcript)"
    " AS active_ribosome"
)
"""Drop this into a SELECT column list to synthesise the active-ribosome
scalar from v2ecoli's per-transcript ribosome count list."""

# ---------------------------------------------------------------------------
# Shim C — oriC count SQL snippet
# ---------------------------------------------------------------------------

ORIC_SQL = "listeners__replication_data__number_of_oric AS oriC"
"""Drop this into a SELECT column list to read the oriC count (scalar).

vEcoli used ``listeners__unique_molecule_counts__oriC``; v2ecoli exposes the
same value via ``listeners__replication_data__number_of_oric``."""

# ---------------------------------------------------------------------------
# Shim D — active-RNAP scalar SQL snippet
# ---------------------------------------------------------------------------

ACTIVE_RNAP_SQL = (
    "len(listeners__rnap_data__active_rnap_unique_indexes) AS active_RNAP"
)
"""Drop this into a SELECT column list to synthesise the active-RNAP scalar.

vEcoli used ``listeners__unique_molecule_counts__active_RNAP``; v2ecoli has
no such scalar column — the equivalent is the length of
``listeners__rnap_data__active_rnap_unique_indexes`` (an INTEGER[] per row)."""


# ---------------------------------------------------------------------------
# Shim E — external exchange flux ordering
# ---------------------------------------------------------------------------

def external_exchange_molecule_ids(sim_data) -> list[str]:
    """Molecule ids, in the order of the ``external_exchange_fluxes`` list column.

    The emitted vector is ``col_primals[external_exchange_idx]``
    (``library/fba_fast.py``), whose index array is built from the FBA model's
    ``_externalExchangeIDs``.  That list is appended index-parallel with
    ``_externalMoleculeIDs`` in a single loop over the ``externalExchangedMolecules``
    argument (``processes/parca/wholecell/utils/modular_fba.py:_initExternalExchange``),
    and Metabolism passes that argument **sorted**
    (``processes/metabolism.py``: ``exchange_molecules = sorted(...)``).  So the
    flux vector is indexed by the sorted external-exchange molecule ids.

    Verified against the three indices pinned independently in
    ``scripts/render_basal_vs_literature.py`` (``_GLC_IDX, _CO2_IDX, _ACET_IDX =
    37, 11, 3``, 1-indexed): sorted position 37 is ``GLC[p]``, 11 is
    ``CARBON-DIOXIDE[p]``, 3 is ``ACET[p]``.
    """
    return sorted(
        str(m) for m in sim_data.external_state.all_external_exchange_molecules
    )


# ---------------------------------------------------------------------------
# Shim A — bulk count matrix
# ---------------------------------------------------------------------------

def bulk_count_matrix(df: pd.DataFrame, sim_data) -> np.ndarray:
    """Return an (n_tp, n_bulk) count matrix in *sim_data* column order.

    vEcoli provides a single ``bulk`` list column already ordered to match
    ``sim_data.internal_state.bulk_molecules.bulk_data["id"]``.  v2ecoli's
    parquet stores ``bulk__id`` + ``bulk__count`` (per row, but with a stable
    ordering that may differ from sim_data).

    This function:
      1. Reads the parquet ordering from the first row of ``bulk__id``
         (verified constant across rows / files).
      2. Stacks ``bulk__count`` rows into an (n_tp, n_pq) matrix.
      3. Reorders columns to match sim_data order.

    A sim_data bulk molecule that is absent from the emitted ``bulk__id`` list
    (e.g. an injected new-gene monomer such as ``NG-GFP-MONOMER[c]`` that lives
    in sim_data but was not emitted by this run) is tolerated: its column is
    filled with zero counts rather than raising. The result keeps sim_data
    column order and width, which the callers depend on (downstream
    ``bulk_ids.index(...)`` lookups in ptools_rna / ptools_proteins index into
    this matrix by sim_data position). Same spirit as v2ecoli #685 tolerating
    injected reactions in the flux path.

    Parameters
    ----------
    df:
        DataFrame returned by ``read_outputs`` containing ``bulk__id`` and
        ``bulk__count`` columns.
    sim_data:
        Loaded sim_data object (provides the canonical bulk ordering).

    Returns
    -------
    np.ndarray of shape (n_tp, n_bulk) with counts in sim_data order.
    """
    sim_ids: list[str] = list(
        sim_data.internal_state.bulk_molecules.bulk_data["id"]
    )
    pq_ids: list[str] = list(df["bulk__id"].iloc[0])
    counts = np.stack(df["bulk__count"].values)          # (n_tp, n_pq)
    pos = {bid: i for i, bid in enumerate(pq_ids)}
    missing = [b for b in sim_ids if b not in pos]
    if not missing:
        col_idx = [pos[b] for b in sim_ids]
        return counts[:, col_idx]
    # Append a single zero-count sentinel column and route every missing
    # sim_id to it (all-zero, so sharing one column is exact).
    import warnings

    preview = ", ".join(str(b) for b in missing[:5])
    warnings.warn(
        f"bulk_count_matrix: {len(missing)} sim_data bulk molecule(s) absent "
        f"from the emitted bulk__id list; filling zero counts "
        f"(e.g. {preview}{'…' if len(missing) > 5 else ''}).",
        stacklevel=2,
    )
    counts = np.concatenate(
        [counts, np.zeros((counts.shape[0], 1), dtype=counts.dtype)], axis=1
    )
    sentinel = counts.shape[1] - 1
    col_idx = [pos.get(b, sentinel) for b in sim_ids]
    return counts[:, col_idx]
