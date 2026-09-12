"""Streaming (per-seed) cross-seed collapse for ``_MultiseedCollapseMixin``.

``_MultiseedCollapseMixin._do_read_outputs`` (ptools_multiscale.py) is the reader
behind ``PtoolsOverviewMultiseed`` — the combined Cellular-Overview upload. The
original implementation materialises EVERY seed × generation × timestep row,
including the wide ``DOUBLE[]`` list columns, in one ``conn.sql(...).df()`` before
collapsing. On a 10-seed × 10-generation sweep that single frame is the 78 GB /
112.8 GB peak #786 dies at.

The cross-seed collapse is a pure element-wise SUM per shared ``time`` (see
:func:`collapse_cross_seed`), which is associative and commutative across seeds.
So the reader can stream one seed at a time and accumulate the running collapse,
never holding more than one seed's rows plus the one-row-per-time accumulator.

Two properties pin the fix:

* **Equivalence** — the streamed result equals the whole-frame collapse for every
  output column (list, scalar, and the ``bulk__id`` identity column), to floating
  point (per-seed summation order differs from the whole-frame order).
* **Memory bound** — no single ``.df()`` materialisation holds more than one
  seed's rows. This FAILS on the original one-shot reader (one ``.df()`` over all
  seeds) and PASSES once the read streams per seed.
"""

from __future__ import annotations

import duckdb
import numpy as np

from v2ecoli.workflow.analyses._helpers import collapse_cross_seed
from v2ecoli.workflow.analyses.ptools_multiscale import _MultiseedCollapseMixin


FLUX_COL = "listeners__fba_results__base_reaction_fluxes"
SCALAR_COL = "listeners__mass__cell_mass"
BULK_ID_COL = "bulk__id"
COLUMNS = [BULK_ID_COL, FLUX_COL, SCALAR_COL]

N_SEEDS = 4
TIMES = (0.0, 1.0, 2.0)
ROWS_PER_SEED = len(TIMES)


def _make_history():
    """In-memory history with a list id column, a list flux column, and a scalar,
    across ``N_SEEDS`` seeds sharing the same ``global_time`` values."""
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE hist ("
        "  experiment_id VARCHAR, variant BIGINT, lineage_seed BIGINT,"
        "  generation BIGINT, agent_id VARCHAR, global_time DOUBLE,"
        f'  "{BULK_ID_COL}" VARCHAR[],'
        f'  "{FLUX_COL}" DOUBLE[],'
        f'  "{SCALAR_COL}" DOUBLE'
        ")"
    )
    for seed in range(N_SEEDS):
        for t in TIMES:
            flux = [seed * 10 + t, seed * 100 + t]
            scalar = seed + t
            conn.execute(
                "INSERT INTO hist VALUES ('exp', 0, ?, 0, ?, ?, ?, ?, ?)",
                [seed, f"{seed}_0", t, ["A", "B"], flux, scalar],
            )
    return conn, "SELECT * FROM hist"


def _reference_collapse(conn, history_sql):
    """The whole-frame collapse the streamed read must reproduce."""
    raw = conn.sql(
        f"SELECT {','.join(COLUMNS)}, global_time AS time"
        f" FROM ({history_sql}) ORDER BY time"
    ).df()
    return collapse_cross_seed(raw, id_cols=frozenset({BULK_ID_COL}))


def _assert_collapsed_equal(got, ref):
    got = got.sort_values("time").reset_index(drop=True)
    ref = ref.sort_values("time").reset_index(drop=True)
    assert list(got.columns) == list(ref.columns), (
        f"columns differ: {list(got.columns)} != {list(ref.columns)}"
    )
    assert len(got) == len(ref), f"row count differs: {len(got)} != {len(ref)}"
    for col in ref.columns:
        for i in range(len(ref)):
            g, r = got[col].iloc[i], ref[col].iloc[i]
            if col == BULK_ID_COL:
                assert list(g) == list(r), f"id column '{col}' row {i} differs"
            elif isinstance(r, (list, np.ndarray)):
                np.testing.assert_allclose(
                    np.asarray(g, dtype=float), np.asarray(r, dtype=float),
                    rtol=1e-9, atol=0.0,
                    err_msg=f"list column '{col}' row {i} differs",
                )
            else:
                np.testing.assert_allclose(
                    float(g), float(r), rtol=1e-9, atol=0.0,
                    err_msg=f"scalar column '{col}' row {i} differs",
                )


class _CountingRelation:
    """Wraps a DuckDB relation, recording the row count of every ``.df()``."""

    def __init__(self, rel, sink):
        self._rel = rel
        self._sink = sink

    def df(self):
        frame = self._rel.df()
        self._sink.append(len(frame))
        return frame

    def __getattr__(self, name):
        return getattr(self._rel, name)


class _CountingConn:
    """Delegates to a real connection but records ``.df()`` materialisation sizes,
    so a test can assert no single frame holds more than one seed's rows."""

    def __init__(self, conn):
        self._conn = conn
        self.df_row_counts: list[int] = []

    def sql(self, query):
        return _CountingRelation(self._conn.sql(query), self.df_row_counts)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_streamed_collapse_matches_whole_frame_all_columns():
    conn, history_sql = _make_history()
    ref = _reference_collapse(conn, history_sql)
    got = _MultiseedCollapseMixin()._do_read_outputs(
        history_sql, conn, columns=COLUMNS
    )
    _assert_collapsed_equal(got, ref)


def test_collapse_never_materializes_more_than_one_seed():
    conn, history_sql = _make_history()
    spy = _CountingConn(conn)
    _MultiseedCollapseMixin()._do_read_outputs(history_sql, spy, columns=COLUMNS)
    assert spy.df_row_counts, "expected at least one .df() materialisation"
    assert max(spy.df_row_counts) <= ROWS_PER_SEED, (
        f"a single .df() materialised {max(spy.df_row_counts)} rows, more than "
        f"one seed's {ROWS_PER_SEED}: the read is not streaming per seed"
    )
