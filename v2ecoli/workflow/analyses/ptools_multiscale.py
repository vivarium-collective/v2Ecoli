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

**Multiseed** (:class:`_MultiseedMixin`): a variant's slice spans MULTIPLE
independent lineages (seeds).  vEcoli's multiseed ptools is the single-cell
``plot`` re-run over the pooled parquet — ``build_query`` bins the run's absolute
time span into ``n_tp`` bins and ``AVG``\\ s each feature over all rows in a bin.
This mixin reproduces that **cross-seed mean** on v2ecoli data, aligning every
seed on the cumulative clock first (:func:`_helpers.reconstruct_cumulative_time`,
so a multi-generation run's per-generation ``global_time`` resets do not pool
different generations together), and emits an extra **cross-seed spread** panel
(the SD across seeds of each seed's per-bin mean) so heterogeneity is visible
rather than averaged away.  It reuses each concrete class's ``_feature_matrix``
extraction seam.  The legacy element-wise-SUM collapse
(:class:`_MultiseedCollapseMixin`, via :func:`_helpers.collapse_cross_seed`) is
retained only for the combined Cellular-Overview upload.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from duckdb import DuckDBPyConnection

from v2ecoli.workflow.analyses._helpers import (
    available_columns,
    cumulative_time_history,
    collapse_cross_seed,
    ptools_heatmap_view,
    reconstruct_cumulative_time,
)
from v2ecoli.workflow.analyses.ptools_rna import PtoolsRna
from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns
from v2ecoli.workflow.analyses.ptools_proteins import PtoolsProteins
from v2ecoli.workflow.analyses.ptools_overview import PtoolsCellOverview


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
# _MultiseedCollapseMixin — legacy cross-seed element-wise SUM
# ---------------------------------------------------------------------------

class _MultiseedCollapseMixin:
    """Legacy cross-seed aggregation: element-wise SUM of every seed at each
    shared ``time`` via :func:`collapse_cross_seed`, then the single-scale
    ``analyze`` body runs unchanged.

    Retained for :class:`PtoolsOverviewMultiseed` (the combined genes+reactions
    +proteins Omics-Viewer upload), which forwards a single history to its
    sibling analyses and cannot use the per-feature ``_feature_matrix`` seam the
    mean/spread mixin needs.  Note this SUMS across seeds and does not
    reconstruct the cumulative clock, so on a multi-generation multiseed run it
    pools different generations — the per-category ptools use
    :class:`_MultiseedMixin` (cross-seed mean + spread on the cumulative axis)
    instead.
    """

    def _do_read_outputs(
        self,
        history_sql: str,
        conn: DuckDBPyConnection,
        columns=None,
    ):
        """Collapse to one row per time, streaming one seed at a time.

        The cross-seed collapse is a pure element-wise SUM per shared ``time``
        (:func:`collapse_cross_seed`), which is associative and commutative
        across seeds.  So instead of loading every seed's rows — including the
        wide ``DOUBLE[]`` list columns — into one ``.df()`` (the 78 GB /
        112.8 GB peak that killed the multiseed Omics-Viewer upload on a 10×10
        sweep, #786), we read each seed on its own and accumulate the running
        collapse.  Peak resident stays at one seed's rows plus the
        one-row-per-time accumulator.
        """
        if columns is None:
            # Fallback: should not happen in practice (analyze always passes
            # explicit columns), but guard against bare calls.
            raise ValueError(
                "_MultiseedCollapseMixin._do_read_outputs requires explicit columns"
            )

        id_cols = frozenset({"bulk__id"})
        # Partition on the seed axis. If there is no seed column (a synthetic or
        # narrowed history), there is nothing to stream over — one group.
        if "lineage_seed" in available_columns(conn, history_sql):
            seeds = [
                r[0] for r in conn.sql(
                    f"SELECT DISTINCT lineage_seed FROM ({history_sql})"
                    f" ORDER BY lineage_seed"
                ).fetchall()
            ]
        else:
            seeds = [None]

        def _read_collapsed(where_seed):
            seed_sql = (
                history_sql if where_seed is None
                else f"SELECT * FROM ({history_sql}) WHERE lineage_seed = {where_seed}"
            )
            # Same SQL the single-scale read_outputs builds — no groupby.sum()
            # here; collapse_cross_seed does the aggregation.
            query_sql = (
                f"SELECT {','.join(columns)}, global_time AS time"
                f" FROM ({seed_sql})"
                f" ORDER BY time"
            )
            return collapse_cross_seed(conn.sql(query_sql).df(), id_cols=id_cols)

        accumulated = None
        for seed in seeds:
            seed_collapsed = _read_collapsed(seed)
            if accumulated is None:
                accumulated = seed_collapsed
                continue
            # Sum-of-sums: concatenate the two already-per-time-collapsed frames
            # and re-run the identical collapse, so the incremental step reuses
            # the exact aggregation primitive rather than a parallel reduction
            # that could drift from it.
            accumulated = collapse_cross_seed(
                pd.concat([accumulated, seed_collapsed], ignore_index=True),
                id_cols=id_cols,
            )

        if accumulated is None:  # empty history (no seeds) — preserve old shape
            return _read_collapsed(None)
        return accumulated


# ---------------------------------------------------------------------------
# _MultiseedMixin — cross-seed mean + spread on the cumulative-time axis
# ---------------------------------------------------------------------------

# Default render spec for classes that do not declare their own.
_DEFAULT_MULTISEED_SPEC = {
    "filename": "ptools_multiseed.tsv",
    "title": "Feature",
    "color_label": "value",
    "log_color": False,
    "sort_rows": False,
    "take_abs": False,
}


def _distinct_seeds(conn, history_sql, avail):
    """Ordered list of lineage_seed values present, or ``[None]`` if the column
    is absent (a synthetic/narrowed history with no seed axis → one group)."""
    if "lineage_seed" not in avail:
        return [None]
    rows = conn.sql(
        f"SELECT DISTINCT lineage_seed FROM ({history_sql}) ORDER BY lineage_seed"
    ).fetchall()
    return [r[0] for r in rows]


def _cross_seed_mean_spread(per_seed_binner, seeds, n_tp, edges):
    """Aggregate per-seed binned matrices into ``(mean_panel, spread_panel,
    feature_ids)``, each panel ``(n_tp × F)``.

    ``per_seed_binner(seed) -> (matrix[T×F], time_vec[T], feature_ids)`` is
    called once per seed (sequentially, so only one seed's matrix is resident).
    ``mean_panel`` is the pooled cross-seed-and-time AVG per (bin, feature) —
    matching vEcoli's ``build_query``.  ``spread_panel`` is the std ACROSS seeds
    of each seed's per-bin mean (0 for a single seed).
    """
    feature_ids = None
    pooled_sum = None
    pooled_cnt = np.zeros(n_tp)
    seed_bin_means = []  # one (n_tp × F) per seed

    for seed in seeds:
        mtx, tvec, fids = per_seed_binner(seed)
        if feature_ids is None:
            feature_ids = fids
            pooled_sum = np.zeros((n_tp, mtx.shape[1]))
        elif mtx.shape[1] != pooled_sum.shape[1]:
            raise ValueError(
                f"feature width differs across seeds ({mtx.shape[1]} != "
                f"{pooled_sum.shape[1]}); seeds do not share a sim_data ordering"
            )
        # Bin each row by absolute (cumulative) time. searchsorted+clip puts the
        # final edge (t == tmax) into the last bin (vEcoli's `time < bin_end`
        # would drop that single row — a negligible boundary difference).
        bin_idx = np.clip(np.searchsorted(edges, tvec, side="right") - 1, 0, n_tp - 1)
        sbm = np.full((n_tp, pooled_sum.shape[1]), np.nan)
        for b in range(n_tp):
            m = bin_idx == b
            if m.any():
                block = mtx[m]
                pooled_sum[b] += block.sum(axis=0)
                pooled_cnt[b] += block.shape[0]
                sbm[b] = block.mean(axis=0)
        seed_bin_means.append(sbm)

    mean_panel = pooled_sum / np.where(pooled_cnt == 0, 1.0, pooled_cnt)[:, None]
    stacked = np.stack(seed_bin_means, axis=0)  # (n_seeds × n_tp × F)
    # std across seeds of each seed's per-bin mean; nan (bins a seed never
    # sampled) ignored, all-nan bins -> 0.
    with np.errstate(invalid="ignore"):
        spread_panel = np.nan_to_num(np.nanstd(stacked, axis=0), nan=0.0)
    return mean_panel, spread_panel, feature_ids


def _two_panel_view(mean_df, spread_df, spec):
    """Stack the mean heatmap over the cross-seed-spread heatmap in one HTML."""
    mean_view = ptools_heatmap_view(
        mean_df,
        f"{spec['title']} — cross-seed mean",
        log_color=spec["log_color"],
        sort_rows=spec["sort_rows"],
        color_label=spec["color_label"],
    )
    # Spread is a std (>= 0); linear color, same row order as the mean panel.
    spread_view = ptools_heatmap_view(
        spread_df,
        f"{spec['title']} — cross-seed spread (SD across seeds)",
        log_color=False,
        sort_rows=spec["sort_rows"],
        color_label=f"SD of {spec['color_label']}",
    )
    return (
        "<div class='ptools-multiseed'>"
        f"{mean_view}<hr/>{spread_view}"
        "</div>"
    )


class _MultiseedMixin:
    """Cross-seed mean + spread panels on the cumulative-time axis.

    Reproduces vEcoli's cross-seed mean (its multiseed ptools is the single-cell
    ``plot`` over the pooled multiseed parquet: ``build_query`` bins the run's
    absolute time span into ``n_tp`` bins and ``AVG``\\ s each feature over all
    rows in a bin) with two fixes over the legacy element-wise-sum collapse:

    * **Cumulative-time alignment** (:func:`reconstruct_cumulative_time`,
      multiseed-safe): v2ecoli's ``global_time`` resets each generation, so
      binning raw time would pool different generations of different seeds — the
      analysis-clock bug.  A no-op when the clock is already absolute.
    * **A cross-seed spread panel**: per feature and bin, the SD across seeds of
      the per-seed bin means, so heterogeneity across seeds is visible rather
      than averaged away.  A single-seed run degrades to zero spread.

    Reuses the concrete ptools class's single-scale ``_feature_matrix`` seam for
    feature extraction; only the cross-seed aggregation and two-panel rendering
    live here.  Both panels go into one combined TSV (mean columns + matching
    ``*_sd`` spread columns) and one stacked two-heatmap view.
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
        params.setdefault("n_tp", 8)
        params.setdefault("time_unit", "minutes")
        if params["time_unit"] not in ("minutes", "seconds"):
            params["time_unit"] = "minutes"
        n_tp = int(params["n_tp"])

        spec = {
            **_DEFAULT_MULTISEED_SPEC,
            **getattr(self, "_ptools_multiseed_spec", {}),
        }

        # Honor burn-in filters the way the cd1 multiseed modules do, BEFORE
        # cumulative-time reconstruction: drop the first ``skip_n_gens``
        # generations (relative to the min generation present) and/or keep
        # ``generation >= generation_lower_bound``.
        if "generation" in available_columns(conn, history_sql):
            skip = int(params.get("skip_n_gens", 0) or 0)
            if skip > 0:
                history_sql = drop_leading_generations(conn, history_sql, skip)
            lb = params.get("generation_lower_bound")
            if lb is not None:
                history_sql = (
                    f"SELECT * FROM ({history_sql}) WHERE generation >= {int(lb)}"
                )

        # Recover the absolute (cumulative) clock vEcoli assumes; multiseed-safe
        # (per-seed offsets), a no-op when global_time is already absolute.
        cum_sql = reconstruct_cumulative_time(conn, history_sql)
        avail = available_columns(conn, cum_sql)
        seeds = _distinct_seeds(conn, cum_sql, avail)

        # Global absolute-time bin edges over the pooled span (vEcoli convention:
        # [min_t, max_t] split into n_tp bins).
        tmin, tmax = conn.sql(
            f"SELECT min(global_time), max(global_time) FROM ({cum_sql})"
        ).fetchone()
        if tmin is None or tmax is None:
            raise ValueError("multiseed ptools: no rows in history")
        if tmax <= tmin:
            edges = np.linspace(float(tmin), float(tmin) + 1.0, n_tp + 1)
        else:
            edges = np.linspace(float(tmin), float(tmax), n_tp + 1)

        def _binner(seed):
            seed_sql = (
                cum_sql if seed is None
                else f"SELECT * FROM ({cum_sql}) WHERE lineage_seed = {seed}"
            )
            mtx, tvec, fids, _gens = self._feature_matrix(
                seed_sql, conn, sim_data, params
            )
            return mtx, np.asarray(tvec, dtype=float), fids

        mean_panel, spread_panel, feature_ids = _cross_seed_mean_spread(
            _binner, seeds, n_tp, edges
        )

        if spec["take_abs"]:
            mean_panel = np.abs(mean_panel)

        # Column labels = bin START times (vEcoli labels bins by bin_start).
        starts = edges[:-1]
        if params["time_unit"] == "minutes":
            starts = [round(x / 60) for x in starts]
        else:
            starts = [round(x) for x in starts]
        unit = params["time_unit"][0]
        mean_cols = [f"{t}{unit}" for t in starts]
        spread_cols = [f"{t}{unit}_sd" for t in starts]

        mean_df = pd.DataFrame(
            mean_panel.transpose(), index=feature_ids, columns=mean_cols
        )
        spread_df = pd.DataFrame(
            spread_panel.transpose(), index=feature_ids, columns=spread_cols
        )
        mean_df.index.name = "$"
        spread_df.index.name = "$"

        # One combined TSV: mean columns then the matching _sd spread columns.
        combined = pd.concat([mean_df, spread_df], axis=1)
        combined.index.name = "$"
        tsv = combined.to_csv(sep="\t", index=True, header=True, float_format="%.4f")

        # For the stacked view the spread panel shares the mean panel's labels.
        spread_view_df = pd.DataFrame(
            spread_panel.transpose(), index=feature_ids, columns=mean_cols
        )
        spread_view_df.index.name = "$"
        view = _two_panel_view(mean_df, spread_view_df, spec)

        n_seeds = len([s for s in seeds if s is not None]) or 1
        return {
            "data": {
                "filename": spec["filename"],
                "tsv": tsv,
                "n_seeds": n_seeds,
            },
            "view": view,
        }


class PtoolsRnaMultiseed(_MultiseedMixin, PtoolsRna):
    name = "ptools_rna_multiseed"
    scale = "multiseed"


class PtoolsRxnsMultiseed(_MultiseedMixin, PtoolsRxns):
    name = "ptools_rxns_multiseed"
    scale = "multiseed"


class PtoolsProteinsMultiseed(_MultiseedMixin, PtoolsProteins):
    name = "ptools_proteins_multiseed"
    scale = "multiseed"


# The combined Cellular-Overview upload (genes + reactions + proteins in one
# "a mixture" Omics-Viewer dataset) had only a single-scale registration, so a
# multi-generation or multi-seed sweep could only export one generation at a
# time. Register it at the aggregating scales too, the same way the per-category
# ptools analyses are: PtoolsCellOverview.analyze forwards the (already time-rewritten
# / cross-seed-collapsed) history_sql to its sibling analyses, so the mixins work
# unchanged — multigeneration lays the lineage on one absolute time axis, and
# multiseed element-wise aggregates across seeds.
class PtoolsOverviewMultigeneration(_MultigenMixin, PtoolsCellOverview):
    name = "ptools_overview_multigeneration"
    scale = "multigeneration"


class PtoolsOverviewMultiseed(_MultiseedCollapseMixin, PtoolsCellOverview):
    name = "ptools_overview_multiseed"
    scale = "multiseed"
