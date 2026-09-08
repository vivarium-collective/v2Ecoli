"""ptools_rxns handles a flux array wider than base_reaction_ids.

An engineered strain whose metabolism is BUILT with an extra reaction not in the
pickled ``base_reaction_ids`` — e.g. ``include_violacein_reactions`` appends a
violacein reaction to the FBA network — emits a ``base_reaction_fluxes`` array one
(or more) wider than ``base_reaction_ids``. The injected reactions are appended
after the base set, so the base reactions still pair 1:1 with the leading flux
columns; the analysis must keep the trailing flux and label it, not raise.

A flux array NARROWER than ``base_reaction_ids`` is still a hard error (a genuinely
wrong sim_data): truncation would silently mislabel reactions.
"""
from __future__ import annotations

import duckdb
import pytest
from bigraph_schema import allocate_core

from v2ecoli.workflow.analyses.ptools_rxns import PtoolsRxns


class _Met:
    def __init__(self, ids):
        self.base_reaction_ids = ids


class _Proc:
    def __init__(self, ids):
        self.metabolism = _Met(ids)


class _SD:
    def __init__(self, ids):
        self.process = _Proc(ids)


def _conn_with_flux(width: int) -> duckdb.DuckDBPyConnection:
    """4 timepoints, each carrying a flux list of ``width`` values."""
    rows = []
    for t in range(1, 5):
        flux = "[" + ", ".join(f"{float(t + i)}" for i in range(width)) + "]"
        rows.append(f"({float(t)}, {flux})")
    conn = duckdb.connect()
    conn.execute(
        'CREATE TABLE h AS SELECT * FROM (VALUES '
        + ", ".join(rows)
        + ') AS v(global_time, "listeners__fba_results__base_reaction_fluxes")'
    )
    return conn


def _tsv_labels(width: int, ids: list[str]) -> list[str]:
    res = PtoolsRxns({}, core=allocate_core()).update(
        {
            "conn": _conn_with_flux(width),
            "history_sql": "SELECT * FROM h",
            "sim_data": _SD(ids),
            "variant_metadata": {"n_tp": 2, "time_unit": "minutes"},
        }
    )
    body = [
        ln for ln in res["data"]["tsv"].splitlines() if ln and not ln.startswith("#")
    ][1:]  # drop the header row
    return [ln.split("\t")[0] for ln in body]


def test_flux_wider_labels_injected_reactions():
    labels = _tsv_labels(4, ["RXN-A", "RXN-B", "RXN-C"])  # 3 base + 1 injected
    assert labels == ["RXN-A", "RXN-B", "RXN-C", "injected-reaction-0"]


def test_flux_two_wider_labels_both():
    labels = _tsv_labels(5, ["RXN-A", "RXN-B", "RXN-C"])  # 3 base + 2 injected
    assert labels[-2:] == ["injected-reaction-0", "injected-reaction-1"]
    assert len(labels) == 5


def test_flux_exact_width_unchanged():
    labels = _tsv_labels(3, ["RXN-A", "RXN-B", "RXN-C"])
    assert labels == ["RXN-A", "RXN-B", "RXN-C"]
    assert not any("injected" in x for x in labels)


def test_flux_narrower_raises():
    with pytest.raises(ValueError, match="narrower"):
        _tsv_labels(3, ["RXN-A", "RXN-B", "RXN-C", "RXN-D"])  # flux < ids
