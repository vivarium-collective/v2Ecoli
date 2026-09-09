"""Shared helpers for native vEcoli analysis ports (the altair/polars family).

vEcoli's plot analyses lean on ``ecoli.library.parquet_emitter`` helpers
(``read_stacked_columns``, ``num_cells``, ``skip_n_gens``, ``named_idx``,
``ndidx_to_duckdb_expr``) and produce Altair charts saved to HTML.  Two things
differ in v2ecoli's parquet schema and must be bridged natively here:

  * **``time`` column** — vEcoli emits a ``time`` column; v2ecoli emits
    ``global_time``.  ``read_stacked_columns`` (and column expressions that
    reference ``time``) need it, so :func:`aliased_history` wraps any
    ``history_sql`` to expose ``global_time`` *also* as ``time``.

  * **``bulk`` column** — vEcoli emits a single ``bulk`` list column ordered to
    match ``sim_data`` bulk order.  v2ecoli emits ``bulk__id`` + ``bulk__count``
    (per-row lists in a parquet ordering that may differ from sim_data).
    :func:`bulk_field_ids` returns the parquet bulk ordering (the equivalent of
    ``field_metadata(conn, config_sql, "bulk")``), and
    :func:`bulk_count_idx_expr` builds a ``named_idx`` expression over
    ``bulk__count`` for a set of molecule ids — failing loudly if any id is
    absent (never silently dropping).

The pure SQL-expression builders ``named_idx`` and ``ndidx_to_duckdb_expr`` are
schema-agnostic, so they are re-exported from the vEcoli module unchanged.

Altair views are serialized with :func:`chart_to_html` (``chart.to_html()``),
the view-output analogue of ``chart.save(path)``.
"""

from __future__ import annotations

from typing import Optional

import duckdb

# Pure SQL-expression builders — no schema assumptions, reuse as-is.
from ecoli.library.parquet_emitter import named_idx, ndidx_to_duckdb_expr  # noqa: F401

# Shim B (see _shims): active-ribosome scalar, aliased to the vEcoli column name
# so ported code can keep referencing the original unique_molecule_counts name.
ACTIVE_RIBOSOME_AS_UMC = (
    "list_sum(listeners__ribosome_data__n_ribosomes_per_transcript)"
    " AS listeners__unique_molecule_counts__active_ribosome"
)


def aliased_history(history_sql: str) -> str:
    """Wrap ``history_sql`` so ``global_time`` is also exposed as ``time``."""
    return f"SELECT *, global_time AS time FROM ({history_sql})"


_ID_COLS = "experiment_id, variant, lineage_seed, generation, agent_id, time"


def read_stacked_columns(
    history_sql: str,
    columns: list[str],
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    order_results: bool = True,
    success_sql: Optional[str] = None,
    remove_first: bool = False,
):
    """Native port of ``ecoli.library.parquet_emitter.read_stacked_columns``.

    Selects ``columns`` (raw names or DuckDB expressions) plus the cell-id
    columns and ``time`` from ``history_sql``.  ``time`` is synthesised from
    ``global_time`` via :func:`aliased_history`, so column expressions may
    reference ``time`` freely.  Returns a polars DataFrame when ``conn`` is
    given, otherwise the SQL string (for use as a subquery).
    """
    base = aliased_history(history_sql)
    cols = ", ".join(columns)
    sql = f"SELECT {cols}, {_ID_COLS} FROM ({base})"
    if success_sql:
        sql = (
            f"SELECT * FROM ({sql}) SEMI JOIN ({success_sql}) "
            "USING (experiment_id, variant, lineage_seed, generation, agent_id)"
        )
    if remove_first:
        sql = f"""
            SELECT * FROM ({sql})
            ANTI JOIN (
                SELECT experiment_id, variant, lineage_seed, generation,
                    agent_id, MIN(global_time) AS time
                FROM ({history_sql})
                GROUP BY experiment_id, variant, lineage_seed, generation,
                    agent_id
            ) USING (experiment_id, variant, lineage_seed, generation,
                agent_id, time)
            """
    if order_results:
        sql = f"SELECT * FROM ({sql}) ORDER BY {_ID_COLS}"
    if conn is None:
        return sql
    return conn.sql(sql).pl()


def cumulative_time_history(history_sql: str) -> str:
    """Rewrite ``global_time`` to be cumulative (absolute) across generations.

    vEcoli emits a monotonic absolute ``time`` across a lineage, so a
    ``GROUP BY time`` over a multigeneration slice never merges rows from
    different generations.

    v2ecoli's emitter behaviour is path-dependent: the parquet emitter carries
    an **absolute** clock across generations (gen N+1's first row already comes
    after gen N's last row), while the xarray path emits ``global_time``
    relative to each generation's birth (resets to 0).  The earlier version of
    this helper *unconditionally* added the summed prior-generation durations,
    which on the already-absolute parquet data **double-offset** every
    generation after the first — inserting a spurious empty time gap between
    generations (visible as a break in the fork-position track and stretched
    sawteeth in dry mass / oriC) and inflating the lineage's total time.

    Fix: compute, per generation, the offset needed to make it start strictly
    after the previous generation's last row, then **clamp that offset at 0**.
    When ``global_time`` is already absolute (parquet path) the required offset
    is negative → clamped to 0 → the data is left untouched.  When it resets
    per generation (xarray path) the offset is positive → generations are
    stacked end-to-end with a 1-unit gap, exactly as before.

    Only valid for a single-lineage slice (one cell per generation, e.g. a
    multigeneration group); do NOT use where multiple cells share a generation
    (e.g. multiseed), as that genuinely requires cross-cell aggregation.
    """
    return f"""
        WITH RECURSIVE _gbounds AS (
            SELECT generation,
                   min(global_time) AS gmin,
                   max(global_time) AS gmax,
                   row_number() OVER (ORDER BY generation) AS rn
            FROM ({history_sql}) GROUP BY generation
        ),
        -- Walk generations in order, carrying the SHIFTED end of the previous
        -- generation forward so each step's offset is computed against the
        -- already-shifted timeline (correct for >2 generations in the
        -- reset-clock case).  The offset is clamped at 0 so already-absolute
        -- time (parquet path) is left untouched and never double-offset.
        _shifted AS (
            SELECT generation, gmin, gmax, rn,
                   CAST(0 AS DOUBLE) AS off,
                   gmax AS shifted_end
            FROM _gbounds WHERE rn = 1
            UNION ALL
            SELECT b.generation, b.gmin, b.gmax, b.rn,
                   GREATEST(s.shifted_end + 1 - b.gmin, 0) AS off,
                   b.gmax + GREATEST(s.shifted_end + 1 - b.gmin, 0) AS shifted_end
            FROM _gbounds b JOIN _shifted s ON b.rn = s.rn + 1
        )
        SELECT o.* EXCLUDE(global_time), o.global_time + s.off AS global_time
        FROM ({history_sql}) o JOIN _shifted s USING (generation)
    """


def num_cells(conn: duckdb.DuckDBPyConnection, subquery: str) -> int:
    """Distinct cell count in a subquery (vEcoli parity)."""
    return conn.sql(
        f"""SELECT count(DISTINCT (experiment_id, variant, lineage_seed,
        generation, agent_id)) FROM ({subquery})"""
    ).fetchone()[0]


def skip_n_gens(subquery: str, n: int) -> str:
    """Skip the first ``n`` generations of a subquery (vEcoli parity)."""
    return f"SELECT * FROM ({subquery}) WHERE generation > {n}"


def available_columns(conn: duckdb.DuckDBPyConnection, history_sql: str) -> set[str]:
    """Return the set of column names available in ``history_sql``."""
    rows = conn.sql(
        f"SELECT column_name FROM (DESCRIBE ({history_sql}))"
    ).fetchall()
    return {r[0] for r in rows}


def bulk_field_ids(conn: duckdb.DuckDBPyConnection, history_sql: str) -> list[str]:
    """Parquet bulk-molecule ordering (equivalent of ``field_metadata("bulk")``).

    v2ecoli stores per-row ``bulk__id`` lists; the ordering is stable across
    rows/files, so the first row defines the ``bulk__count`` column order.
    """
    row = conn.sql(
        f"SELECT bulk__id FROM ({history_sql}) LIMIT 1"
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("no bulk__id rows found in history_sql")
    return list(row[0])


def bulk_count_idx_expr(
    conn: duckdb.DuckDBPyConnection,
    history_sql: str,
    mol_ids: list[str],
    names: Optional[list[str]] = None,
    zero_to_null: bool = False,
) -> str:
    """``named_idx`` expression over ``bulk__count`` for ``mol_ids``.

    Maps each molecule id to its index in the parquet bulk ordering (NOT the
    sim_data ordering) and builds ``bulk__count[idx+1] AS name`` expressions.
    Raises ``ValueError`` if any requested id is absent — never silently drops.
    """
    order = bulk_field_ids(conn, history_sql)
    pos = {bid: i for i, bid in enumerate(order)}
    missing = [m for m in mol_ids if m not in pos]
    if missing:
        raise ValueError(
            f"{len(missing)} molecule id(s) not in parquet bulk ordering: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    idxs = [pos[m] for m in mol_ids]
    if names is None:
        names = list(mol_ids)
    return named_idx("bulk__count", names, [idxs], zero_to_null=zero_to_null)


def collapse_cross_seed(
    df,
    id_cols: frozenset = frozenset({"bulk__id"}),
):
    """Collapse a multi-seed DataFrame to one row per ``time``.

    At multiseed scale, multiple seeds share the same absolute ``time``
    value.  v2ecoli's pandas ``groupby.sum()`` *concatenates* Python-list
    columns instead of adding them element-wise — this function replaces
    that groupby with a correct aggregation:

    - **id columns** (``id_cols``, e.g. ``bulk__id``): take the first
      row's value; assert every row in the group has the same length
      (raises ``ValueError`` if not — never silently truncates/pads).
    - **list/array columns** (all other non-scalar columns): element-wise
      numpy sum across seeds.  Assert all arrays in a group share the same
      shape (raises ``ValueError`` if not).
    - **scalar columns**: plain sum.

    Returns a DataFrame with one row per ``time``, columns in the original
    order.
    """
    import numpy as np
    import pandas as pd

    if df.empty:
        return df.copy()

    first_row = df.iloc[0]

    list_cols: list[str] = []
    id_col_list: list[str] = []
    scalar_cols: list[str] = []

    for col in df.columns:
        if col == "time":
            continue
        val = first_row[col]
        if col in id_cols:
            id_col_list.append(col)
        elif isinstance(val, (list, np.ndarray)):
            list_cols.append(col)
        else:
            scalar_cols.append(col)

    result_rows: list[dict] = []
    for time_val, group in df.groupby("time", sort=True):
        row: dict = {"time": time_val}

        # id columns: take first, assert all same length
        for col in id_col_list:
            first_val = list(group[col].iloc[0])
            for other_val in group[col].iloc[1:]:
                other_list = list(other_val)
                if len(other_list) != len(first_val):
                    raise ValueError(
                        f"Mismatched length in id column '{col}' at "
                        f"time={time_val}: expected {len(first_val)}, "
                        f"got {len(other_list)}"
                    )
            row[col] = first_val

        # list/array columns: element-wise numpy sum across seeds
        for col in list_cols:
            arrays = [np.asarray(v) for v in group[col]]
            shapes = [a.shape for a in arrays]
            if len(set(shapes)) > 1:
                raise ValueError(
                    f"Mismatched array shapes in column '{col}' at "
                    f"time={time_val}: {shapes}"
                )
            row[col] = np.sum(np.stack(arrays), axis=0)

        # scalar columns: plain sum
        for col in scalar_cols:
            row[col] = group[col].sum()

        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    # Preserve original column order
    return result[list(df.columns)]


def cast_decimals(df):
    """Cast any polars ``Decimal`` columns to ``Float64``.

    DuckDB surfaces some numeric columns as ``DECIMAL``; Altair's JSON encoder
    cannot serialize Python ``Decimal``, so coerce them to float before
    rendering a chart.
    """
    import polars as pl

    dec = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Decimal]
    if dec:
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in dec])
    return df


_CD1_ID_COLS = ["experiment_id", "variant", "lineage_seed", "generation", "agent_id"]

# Default cells-per-round-trip for run_chunked/distinct_cell_filters (item 77).
# chunk_size=1 was item 38's original OOM-safety default; item 38's own later
# investigation found chunking barely moves peak memory (2-6% noise), while at
# the real 1000-seed x 10-generation production scale chunk_size=1 means
# ~40,000 sequential DuckDB round-trips across the 4 cd1_* modules that call
# run_chunked -- confirmed live as the direct cause of a 5+ hour analysis run.
# 100 cuts that to ~400. Still exposed as a per-module tunable `chunk_size`
# config_schema param, not hardcoded -- see cd1_fluxomics.py etc.
# Cells per DuckDB batch in the cd1 explode+aggregate modules. 100 looked
# generous until a real campaign had FEWER cells than that: CD2 Run 2 at 10
# seeds x 8 generations is 80 cells, so the "chunking" ran the whole variant
# as ONE batch and cd1_metabolomics / cd1_higher_order_properties still needed
# 44.8 GiB each on the gather (sim 742, 2026-09-09). A batch is bounded by
# chunk x rows-per-cell x list width (bulk__count is ~16k wide), so 8 keeps a
# 10 x 8 campaign near 2-3 GB per batch; a module's own `chunk_size` param
# still overrides. Results are identical for any chunk size: every cd1
# aggregate is GROUP BY cell, and a group never spans two batches.
DEFAULT_CD1_CHUNK_SIZE = 8


def _sql_literal(value) -> str:
    """Inline a Python scalar (str/int/float — cell-identity values only) as a
    DuckDB SQL literal."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def distinct_cell_filters(
    conn: duckdb.DuckDBPyConnection,
    filtered_sql: str,
    id_cols: Optional[list[str]] = None,
    chunk_size: int = DEFAULT_CD1_CHUNK_SIZE,
) -> list[str]:
    """Row-tuple ``WHERE`` fragments partitioning ``filtered_sql``'s distinct cells.

    Selects only ``id_cols`` (default the 5 cell-identity columns — never a
    measurement column), so DuckDB's projection pushdown means this scan never
    materializes the large per-timepoint list columns (``monomer_counts``,
    ``bulk__count``, etc.) that make the cd1 explode+aggregate queries OOM.
    Each returned fragment restricts a follow-up query to at most
    ``chunk_size`` cells via a tuple-``IN`` predicate.
    """
    cols = id_cols or _CD1_ID_COLS
    col_list = ", ".join(cols)
    rows = conn.sql(f"SELECT DISTINCT {col_list} FROM ({filtered_sql})").fetchall()
    fragments = []
    for i in range(0, len(rows), chunk_size):
        batch = rows[i : i + chunk_size]
        values = ", ".join(
            "(" + ", ".join(_sql_literal(v) for v in row) + ")" for row in batch
        )
        fragments.append(f"({col_list}) IN ({values})")
    return fragments


def run_chunked(
    conn: duckdb.DuckDBPyConnection,
    filtered_sql: str,
    build_batch_sql,
    id_cols: Optional[list[str]] = None,
    chunk_size: int = DEFAULT_CD1_CHUNK_SIZE,
):
    """Run ``build_batch_sql(cell_filter)`` once per cell chunk, concatenated.

    ``build_batch_sql`` takes one ``(id_cols...) IN (...)`` WHERE fragment (to
    be applied against ``filtered_sql``) and returns the full per-batch
    explode+aggregate SQL string. Every cd1 module's ``GROUP BY`` already
    partitions by the same cell-identity columns (a group never spans two
    cells), so per-batch results concatenate directly with zero cross-batch
    recombination — confirmed by the item-38 chunk-safety audit. This bounds
    peak memory to ``chunk_size`` cells' unnested arrays at a time instead of
    materializing the whole sweep at once, which is what OOM-killed the
    analysis DAG node on the full sweep.
    """
    import polars as pl

    fragments = distinct_cell_filters(conn, filtered_sql, id_cols, chunk_size)
    if not fragments:
        return pl.DataFrame()
    parts = [conn.sql(build_batch_sql(frag)).pl() for frag in fragments]
    return pl.concat(parts, how="vertical")


def cd1_filter_clause(params: Optional[dict] = None) -> str:
    """The ``WHERE`` clause the cd1 omics ports share, or ``""`` when unfiltered.

    All six vEcoli ``cd1_*`` analyses gate their aggregate on the same two
    optional params — ``generation_lower_bound`` (drop early generations as
    inoculation burn-in) and ``time_lower_bound`` (drop the start of each cell
    cycle).  ``time`` is available because :func:`read_stacked_columns` aliases
    v2ecoli's ``global_time`` to it.

    Deviation from the originals: ``cd1_fluxomics`` tested
    ``generation_lower_bound`` for *truthiness* while the other five tested
    ``is not None``, so a bound of ``0`` was silently ignored there.  This
    normalizes on ``is not None``.  The two agree in effect (``generation >= 0``
    matches every row), so no output changes — the ports just stop disagreeing.
    """
    params = params or {}
    filters = []
    gen_lb = params.get("generation_lower_bound")
    if gen_lb is not None:
        filters.append(f"generation >= {int(gen_lb)}")
    time_lb = params.get("time_lower_bound")
    if time_lb is not None:
        filters.append(f"time >= {float(time_lb)}")
    return ("WHERE " + " AND ".join(filters)) if filters else ""


def with_cross_cell_stats(wide, index_col: str):
    """Append across-cell ``mean``/``std`` to a per-cell wide table.

    The cd1 omics analyses all end the same way: a table with one row per
    entity (reaction / monomer / gene / compound / property) and one column per
    cell, to which they add the mean and standard deviation *across cells* and
    then front-load those two columns.  An entity table with no cell columns
    (an empty sweep slice) gets null ``mean``/``std`` and no per-cell columns,
    matching the originals' empty-input branch.
    """
    import polars as pl

    value_cols = [c for c in wide.columns if c != index_col]
    if not value_cols:
        return wide.with_columns(
            [pl.lit(None).alias("mean"), pl.lit(None).alias("std")]
        ).select([index_col, "mean", "std"])
    wide = wide.with_columns(
        [
            pl.concat_list(value_cols).list.mean().alias("mean"),
            pl.concat_list(value_cols).list.std().alias("std"),
        ]
    )
    return wide.select([index_col, "mean", "std", *value_cols])


def chart_to_html(chart, title: str = "") -> str:
    """Serialize an Altair chart to a self-contained HTML view fragment.

    Altair caps inline datasets at 5000 rows by default; simulation timeseries
    routinely exceed that, so the limit is disabled (data is embedded inline in
    the view HTML, matching vEcoli's ``chart.save(path)`` behaviour).
    """
    import altair as alt

    alt.data_transformers.disable_max_rows()
    html = chart.to_html()
    if title:
        return f'<div class="analysis-view"><h3>{title}</h3>{html}</div>'
    return f'<div class="analysis-view">{html}</div>'


def variant_label_map(variant_metadata: Optional[dict] = None) -> dict[int, str]:
    """Extract an ``int variant -> display name`` map from ``variant_metadata``.

    The analysis runner passes the per-analysis config dict through as
    ``variant_metadata`` (analysis_runner.py ~line 281).  For the
    variant-comparison analyses the caller is expected to supply the runner's
    ``variant_metadata`` int->name mapping (e.g.
    ``{0: "baseline", 1: "ppgpp-off", ...}``); the keys may arrive as ints or
    as JSON-stringified ints.  Any non-int-like keys (ordinary config options
    such as ``skip_n_gens``) are ignored.

    Returns ``{}`` when no variant entries are present — callers should then
    fall back to labelling variants by their integer id.
    """
    out: dict[int, str] = {}
    for k, v in (variant_metadata or {}).items():
        try:
            ik = int(k)
        except (TypeError, ValueError):
            continue
        out[ik] = str(v)
    return out


def variant_display_expr(
    variant_metadata: Optional[dict] = None,
    col: str = "variant",
    out_name: str = "variant_name",
) -> str:
    """A DuckDB CASE expression mapping ``col`` (int variant) to a display name.

    Falls back to ``'variant ' || CAST(col AS VARCHAR)`` for any variant not
    present in the map, so every variant always gets a legible label.
    """
    label_map = variant_label_map(variant_metadata)
    whens = " ".join(
        f"WHEN {int(k)} THEN '{str(v).replace(chr(39), chr(39) * 2)}'"
        for k, v in sorted(label_map.items())
    )
    fallback = f"'variant ' || CAST({col} AS VARCHAR)"
    if not whens:
        return f"{fallback} AS {out_name}"
    return f"CASE {col} {whens} ELSE {fallback} END AS {out_name}"


def variant_sort_order(df_variants, variant_metadata: Optional[dict] = None) -> list[str]:
    """Display-name sort order for a set of variant ints (ascending by id).

    ``df_variants`` is any iterable of variant integers present in the data.
    Returns the corresponding display names in ascending-variant order, so
    baseline (variant 0) sorts first and Altair legends/x-axes are stable.
    """
    label_map = variant_label_map(variant_metadata)
    order: list[str] = []
    for v in sorted({int(x) for x in df_variants}):
        order.append(label_map.get(v, f"variant {v}"))
    return order


def ptools_heatmap_view(
    df: "pd.DataFrame",  # noqa: F821
    title: str,
    log_color: bool = False,
    sort_rows: bool = False,
    color_label: str = "Value",
) -> str:
    """Render a frame-ID × timepoint matrix as a heatmap HTML fragment.

    Tries Plotly ``go.Heatmap`` first (efficient for ~4500 rows × ~8 cols);
    falls back to an Altair ``mark_rect`` heatmap when Plotly is absent.
    Y-axis tick labels are suppressed (too dense at 4500 rows); frame IDs
    appear on hover.

    Args:
        df: DataFrame with frame IDs as index and timepoint strings as columns.
        title: Human-readable chart title.
        log_color: When True, color on a log10 scale.  Heavy-tailed,
            non-negative matrices (e.g. metabolic fluxes — a handful of
            central-carbon reactions carry O(10) flux while hundreds carry
            <0.1) render as a uniform flat field on a linear scale, washing
            out all band structure.  A log scale spreads the lower decades so
            the structure across hundreds of reactions becomes visible.  A
            small positive floor is applied so exact-zero cells map to the
            bottom of the color range instead of ``-inf``.
        sort_rows: When True, reorder rows by descending per-row maximum, so
            high-magnitude reactions cluster at the top and the magnitude
            gradient is legible top-to-bottom.
        color_label: Colorbar / legend title.

    Returns:
        HTML fragment (``full_html=False`` for Plotly; ``chart.to_html()``
        div for Altair).
    """
    import numpy as np

    if sort_rows and df.shape[0] > 1:
        order = np.argsort(-df.max(axis=1).to_numpy(dtype=float))
        df = df.iloc[order]

    z = df.to_numpy(dtype=float)
    if log_color:
        # Floor at the smallest positive value (or a tiny epsilon) so zeros
        # don't blow up log10 and instead sit at the bottom of the scale.
        pos = z[z > 0]
        floor = float(pos.min()) if pos.size else 1e-9
        z_color = np.log10(np.clip(z, floor, None))
        cb_title = f"log10({color_label})"
    else:
        z_color = z
        cb_title = color_label

    try:
        import plotly.graph_objects as go

        # Color on z_color, but show the raw value on hover for interpretability.
        fig = go.Figure(
            go.Heatmap(
                z=z_color.tolist(),
                x=list(df.columns),
                y=list(df.index),
                customdata=z.tolist(),
                colorbar=dict(title=cb_title),
                hovertemplate=(
                    "Frame: %{y}<br>Time: %{x}<br>"
                    + f"{color_label}: %{{customdata:.4g}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title,
            yaxis=dict(showticklabels=False),
            margin=dict(l=60, r=60, t=60, b=60),
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except ImportError:
        pass

    # --- Altair fallback ---
    import altair as alt
    import pandas as pd  # noqa: F811

    # Use a safe column name — df.index.name is "$" which is fine in Altair
    # long-form encoding but we rename to avoid any shorthand ambiguity.
    id_col = "frame_id"
    color_df = pd.DataFrame(z_color, index=df.index, columns=df.columns)
    raw_df = df
    df_reset = color_df.copy().reset_index()
    df_reset = df_reset.rename(columns={df_reset.columns[0]: id_col})
    long = df_reset.melt(id_vars=[id_col], var_name="timepoint", value_name="value")
    raw_long = (
        raw_df.copy()
        .reset_index()
        .rename(columns={raw_df.reset_index().columns[0]: id_col})
        .melt(id_vars=[id_col], var_name="timepoint", value_name="raw_value")
    )
    long["raw_value"] = raw_long["raw_value"].to_numpy()

    alt.data_transformers.disable_max_rows()

    chart = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X(
                "timepoint:N",
                sort=list(df.columns),
                title="Timepoint",
            ),
            y=alt.Y(
                f"{id_col}:N",
                sort=list(df.index),
                axis=alt.Axis(labels=False, ticks=False, title=None),
            ),
            color=alt.Color("value:Q", title=cb_title),
            tooltip=[
                alt.Tooltip(f"{id_col}:N", title="Frame ID"),
                alt.Tooltip("timepoint:N", title="Timepoint"),
                alt.Tooltip("raw_value:Q", title=color_label, format=".4g"),
            ],
        )
        .properties(title=title, width=400, height=600)
    )

    return chart_to_html(chart, "")
