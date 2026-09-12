"""The per-cell seam contract.

★ What these tests are really defending: two engines produce this table, and a
later study compares their RANKINGS. Every violation below is one that leaves
each producer internally consistent while making the two incomparable — which is
why they have to be caught here rather than noticed downstream.
"""
from __future__ import annotations

import pytest

from v2ecoli.library.per_cell_seam import (
    KEY_COLUMNS, SeamViolation, cell_key, cells_per_variant, check,
    observable_columns, validate)


def _row(variant=0, seed=0, gen=0, agent="0", exp="exp-1", **obs):
    row = {"experiment_id": exp, "variant": variant, "lineage_seed": seed,
           "generation": gen, "agent_id": agent}
    row.update(obs or {"growth_rate_h": 0.7, "GLC": -9.7})
    return row


def _panel():
    return [_row(variant=v, seed=s, growth_rate_h=0.7 + 0.01 * v, GLC=-9.7)
            for v in (0, 1) for s in (0, 1)]


def test_a_conforming_table_has_no_violations():
    rows = _panel()
    assert check(rows) == []
    validate(rows, required_observables=["growth_rate_h", "GLC"])
    assert observable_columns(rows) == ["growth_rate_h", "GLC"]


def test_experiment_id_is_part_of_the_key():
    """⚠ The upstream docstring names four key columns and its code groups by
    five. The code is right: two runs can share a lineage seed, and dropping
    experiment_id merges their cells into one."""
    assert "experiment_id" in KEY_COLUMNS
    a = _row(exp="exp-1", variant=0, seed=3)
    b = _row(exp="exp-2", variant=0, seed=3)
    assert cell_key(a) != cell_key(b)
    assert check([a, b]) == []          # distinct cells, not a duplicate


def test_a_duplicated_cell_is_a_violation():
    """★ The one that silently corrupts a ranking rather than breaking anything:
    a cell counted twice is double-weighted in every statistic over the panel."""
    rows = _panel()
    rows.append(dict(rows[0]))
    v = check(rows)
    assert any("duplicate cell key" in x for x in v), v


def test_an_incomplete_key_is_a_violation_not_a_default():
    rows = _panel()
    rows[1]["lineage_seed"] = None
    assert any("incomplete cell key" in x for x in check(rows))


def test_missing_key_columns_are_reported():
    rows = [{k: v for k, v in r.items() if k != "agent_id"} for r in _panel()]
    assert any("missing key column" in x for x in check(rows))


def test_a_table_of_keys_alone_grades_nothing():
    rows = [{c: 0 for c in KEY_COLUMNS}]
    assert any("no observable columns" in x for x in check(rows))


def test_ragged_rows_are_reported():
    rows = _panel()
    rows[2] = dict(rows[2])
    rows[2]["extra_column"] = 1.0
    assert any("columns differ" in x for x in check(rows))


def test_a_required_observable_that_is_absent_is_named():
    v = check(_panel(), required_observables=["violacein_exchange"])
    assert any("violacein_exchange" in x for x in v)


def test_non_numeric_observables_are_reported_but_none_is_allowed():
    """An unobserved observable is honest and stays None — the landing check
    turns that into an explicit unjudgeable entry. A STRING in a numeric column
    is a producer bug and must not reach arithmetic."""
    ok = _panel()
    ok[0]["GLC"] = None
    assert check(ok) == []

    bad = _panel()
    bad[0]["GLC"] = "n/a"
    assert any("not numeric" in x for x in check(bad))

    booly = _panel()
    booly[0]["GLC"] = True           # bool is an int subclass — must not slip through
    assert any("not numeric" in x for x in check(booly))


def test_validate_raises_with_every_violation_not_just_the_first():
    rows = _panel()
    rows.append(dict(rows[0]))       # duplicate
    rows[1]["GLC"] = "n/a"           # non-numeric
    with pytest.raises(SeamViolation) as exc:
        validate(rows, required_observables=["absent_observable"])
    assert len(exc.value.violations) >= 3, exc.value.violations


def test_cells_per_variant_reports_the_replication_actually_achieved():
    """★ Not assumed uniform: a variant with fewer surviving cells has a wider
    interval, and a ranking has to respect that."""
    rows = _panel() + [_row(variant=1, seed=2)]
    assert cells_per_variant(rows) == {0: 2, 1: 3}


def test_an_empty_table_is_a_violation_not_an_empty_pass():
    assert check([]) == ["the table has no rows"]
