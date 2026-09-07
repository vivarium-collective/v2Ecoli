"""Multi-scale registrations of the PathwayTools (ptools) TSV analyses.

In vEcoli the ptools analyses live at three scales (``single``,
``multigeneration``, ``multiseed``) as byte-identical source files — the
difference is only which cells the ``history_sql`` covers and how ``read_outputs``
aggregates them by ``time``.

**Multigeneration** (:class:`_MultigenMixin`): a single-daughter lineage has
exactly one cell per generation.  vEcoli's absolute, monotonic ``time`` makes
``read_outputs``' ``GROUP BY time`` an identity (no row merges); v2ecoli's
``global_time`` resets each generation, so we rebuild the absolute axis with
:func:`_helpers.cumulative_time_history` and then reuse the single-scale
``analyze`` verbatim.  Result: the TSV spans the whole lineage with correct
minute checkpoints (e.g. 0m … 88m across two generations).

**Multiseed** (:class:`_MultiseedMixin`): multiple seeds share the same
absolute ``time`` value.  v2ecoli's ``groupby("time").sum()`` over Python-list
columns (``bulk__id``, ``bulk__count``, flux arrays) *concatenates* instead of
adding element-wise — wrong.  This mixin overrides ``_do_read_outputs()`` to
fetch the raw per-seed rows and apply :func:`_helpers.collapse_cross_seed`:
string-list id columns are first-preserved (with length assertion), numeric
list/array columns are element-wise numpy-summed (with shape assertion), and
scalar columns are plain-summed.  The single-scale ``analyze`` body then runs
unchanged on the aggregated DataFrame.
"""

from __future__ import annotations

from typing import Any

from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analyses._helpers import (
    cumulative_time_history,
    collapse_cross_seed,
)
from v2ecoli.workflow.analyses.ptools_rna import PtoolsRna
from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns
from v2ecoli.workflow.analyses.ptools_proteins import PtoolsProteins


def drop_leading_generations(
    conn: DuckDBPyConnection, history_sql: str, skip: int
) -> str:
    """Return ``history_sql`` with the first ``skip`` generations removed.

    ``generation`` is 0-indexed in the parquet hive, and lineage/seed filters
    can raise the minimum generation above 0, so the cutoff is computed relative
    to the minimum generation actually present (``skip=1`` drops just the first
    generation, whatever its index). Returns ``history_sql`` unchanged — never an
    empty query — when there are not strictly more than ``skip`` generations, so
    a short run still yields a table instead of failing downstream.
    """
    if skip <= 0:
        return history_sql
    row = conn.sql(
        "SELECT MIN(generation) AS lo, COUNT(DISTINCT generation) AS n "
        f"FROM ({history_sql})"
    ).fetchone()
    lo, n_gens = row
    if lo is None or int(n_gens) <= skip:
        return history_sql
    cutoff = int(lo) + skip
    return f"SELECT * FROM ({history_sql}) WHERE generation >= {cutoff}"


# ---------------------------------------------------------------------------
# _MultigenMixin — absolute time axis for single-daughter lineages
# ---------------------------------------------------------------------------

class _MultigenMixin:
    """Rewrite history to an absolute time axis, then run the single analyze.

    Multigeneration is also where the ptools time-axis defaults live, because it
    is the only ptools scale that spans more than one generation:

    * ``per_generation=True`` — consolidate into ONE window per generation
      (aligned to generation boundaries) instead of ``n_tp`` evenly-spaced ticks.
    * ``skip_n_gens=1`` — DROP the first generation and start the table at the
      second, so the initial pre-steady-state generation does not bias the
      per-generation averages.

    Both are overridable via ``variant_metadata`` (analysis_options); the single
    and multiseed scales keep the old evenly-spaced, no-skip behaviour.

    ``generation`` is 0-indexed in the parquet hive, and the multiseed/lineage
    filters can raise the minimum above 0, so "drop the first ``skip``
    generations" is computed relative to the minimum generation actually present
    rather than by a fixed ``generation >`` cutoff. The drop is skipped when it
    would leave no generations.
    """

    def analyze(
        self,
        *,
        conn: DuckDBPyConnection,
        history_sql: str,
        sim_data,
        variant_metadata: dict[str, Any] | None = None,
        **ctx,
    ) -> dict:
        params = dict(variant_metadata or {})
        params.setdefault("per_generation", True)
        skip = int(params.get("skip_n_gens", 1))
        history_sql = drop_leading_generations(conn, history_sql, skip)
        abs_sql = cumulative_time_history(history_sql)
        return super().analyze(
            conn=conn,
            history_sql=abs_sql,
            sim_data=sim_data,
            variant_metadata=params,
            **ctx,
        )


class PtoolsRnaMultigeneration(_MultigenMixin, PtoolsRna):
    name = "ptools_rna_multigeneration"
    scale = "multigeneration"


class PtoolsRxnsMultigeneration(_MultigenMixin, PtoolsRxns):
    name = "ptools_rxns_multigeneration"
    scale = "multigeneration"


class PtoolsProteinsMultigeneration(_MultigenMixin, PtoolsProteins):
    name = "ptools_proteins_multigeneration"
    scale = "multigeneration"


# ---------------------------------------------------------------------------
# _MultiseedMixin — cross-seed element-wise aggregation
# ---------------------------------------------------------------------------

class _MultiseedMixin:
    """Override _do_read_outputs to collapse multiple seeds element-wise.

    Multiple seeds share the same absolute ``time`` value, so the per-seed
    rows must be aggregated before the single-scale ``analyze`` body runs.
    :func:`collapse_cross_seed` handles this correctly:

    * ``bulk__id`` (and any other ``id_cols``) — first-preserved (same across
      seeds; length-mismatch raises ``ValueError``).
    * list/array columns (``bulk__count``, FBA flux arrays, per-transcript
      lists) — element-wise numpy sum (shape-mismatch raises ``ValueError``).
    * scalar columns (``active_ribosome``, ``oriC``, ``active_RNAP``) — plain
      scalar sum.

    The cross-seed aggregated DataFrame is then passed verbatim to the
    inherited single-scale ``analyze`` body.
    """

    def _do_read_outputs(
        self,
        history_sql: str,
        conn: DuckDBPyConnection,
        columns=None,
    ):
        """Fetch raw per-seed rows and collapse to one row per time."""
        if columns is None:
            # Fallback: should not happen in practice (analyze always passes
            # explicit columns), but guard against bare calls.
            raise ValueError("_MultiseedMixin._do_read_outputs requires explicit columns")

        # Build the same SQL used by the single-scale read_outputs — but do NOT
        # apply groupby.sum() here; collapse_cross_seed handles aggregation.
        query_sql = (
            f"SELECT {','.join(columns)}, global_time AS time"
            f" FROM ({history_sql})"
            f" ORDER BY time"
        )
        raw_df = conn.sql(query_sql).df()
        return collapse_cross_seed(raw_df, id_cols=frozenset({"bulk__id"}))


class PtoolsRnaMultiseed(_MultiseedMixin, PtoolsRna):
    name = "ptools_rna_multiseed"
    scale = "multiseed"


class PtoolsRxnsMultiseed(_MultiseedMixin, PtoolsRxns):
    name = "ptools_rxns_multiseed"
    scale = "multiseed"


class PtoolsProteinsMultiseed(_MultiseedMixin, PtoolsProteins):
    name = "ptools_proteins_multiseed"
    scale = "multiseed"
