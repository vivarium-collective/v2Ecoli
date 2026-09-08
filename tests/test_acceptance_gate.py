"""Acceptance gate: content checks that turn silent-success into loud failure.

Covers the failure classes from sms-ecoli#210: a required column missing, present
but all-null, no hive parquet at all, and a declared process absent from what ran.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v2ecoli.workflow.acceptance_gate import (
    bulk_species_count_from_state,
    check_columns,
    check_composition,
    check_species_count,
    process_names_from_state,
    report_from_gate_verdict,
    run_gate,
    _self_test,
)

pytestmark = pytest.mark.fast


def _write_hive(root: Path, table: pa.Table) -> None:
    leaf = (
        root / "history" / "experiment_id=t" / "variant=0"
        / "lineage_seed=0" / "generation=1" / "agent_id=1"
    )
    leaf.mkdir(parents=True)
    pq.write_table(table, str(leaf / "0.pq"))


@pytest.fixture
def sweep(tmp_path):
    _write_hive(tmp_path, pa.table({
        "global_time": [0.0, 1.0, 2.0],
        "listeners__mass__dry_mass": [430.0, 431.0, 432.0],
        "environment__exchange__VIOLACEIN": [0.0, 0.01, 0.02],
        "all_null_col": pa.array([None, None, None], type=pa.float64()),
        "dead_const_col": [7.0, 7.0, 7.0],
    }))
    return str(tmp_path)


def test_present_columns_pass(sweep):
    r = check_columns(sweep, ["global_time", "listeners__mass__dry_mass"])
    assert r["passed"]
    assert r["columns"]["listeners__mass__dry_mass"]["nonnull_rows"] == 3


def test_missing_column_fails_and_is_named(sweep):
    r = check_columns(sweep, ["listeners__mass__dry_mass", "listeners__mass__cell_mass"])
    assert not r["passed"]
    assert r["columns"]["listeners__mass__cell_mass"] == {"present": False, "nonnull_rows": 0, "distinct": 0}


def test_all_null_column_fails(sweep):
    """Present but all-null is the silent case: the column exists, reads nothing."""
    r = check_columns(sweep, ["all_null_col"])
    assert not r["passed"]
    assert r["columns"]["all_null_col"]["present"] is True
    assert r["columns"]["all_null_col"]["nonnull_rows"] == 0


def test_constant_column_fails_when_must_vary(sweep):
    """A present, non-null, but constant column is a dead channel wearing a name."""
    r = check_columns(sweep, ["dead_const_col"], must_vary=["dead_const_col"])
    assert not r["passed"]
    assert r["columns"]["dead_const_col"] == {"present": True, "nonnull_rows": 3, "distinct": 1}


def test_constant_column_passes_when_not_required_to_vary(sweep):
    """distinct is always reported so a reviewer sees a dead channel, but it only
    fails when the column is declared must_vary."""
    r = check_columns(sweep, ["dead_const_col"])  # not in must_vary
    assert r["passed"]
    assert r["columns"]["dead_const_col"]["distinct"] == 1


def test_varying_column_passes_must_vary(sweep):
    r = check_columns(sweep, ["listeners__mass__dry_mass"],
                      must_vary=["listeners__mass__dry_mass"])
    assert r["passed"]
    assert r["columns"]["listeners__mass__dry_mass"]["distinct"] == 3


def test_no_hive_parquet_fails(tmp_path):
    r = check_columns(str(tmp_path), ["global_time"])
    assert not r["passed"]
    assert "no hive history parquet" in r["error"]


def test_a_stray_flat_parquet_is_not_counted(tmp_path):
    """history_files scopes to experiment_id=* hive trees; a flat default/history
    file must not make the gate think output landed."""
    flat = tmp_path / "default" / "history"
    flat.mkdir(parents=True)
    pq.write_table(pa.table({"global_time": [0.0]}), str(flat / "1.pq"))
    r = check_columns(str(tmp_path), ["global_time"])
    assert not r["passed"] and r["n_files"] == 0


def test_process_names_from_state_extracts_addresses():
    state = {
        "agents": {"0": {
            "ecoli-metabolism": {"_type": "process", "address": "local:MetabolismReduxClassic"},
            "division": {"_type": "step", "address": "local:Division"},
            "bulk": {"count": [1, 2, 3]},  # not a process
        }},
        "global_time": 0.0,
    }
    names = process_names_from_state(state)
    assert names == {"local:MetabolismReduxClassic", "local:Division"}


def test_composition_passes_when_declared_class_present():
    """Declared is the EXPECTED mounted class; matches the mounted address."""
    ran = ["local:MetabolismReduxClassic", "local:Division"]
    r = check_composition(ran, ["MetabolismReduxClassic"])
    assert r["passed"] and r["missing"] == []


def test_composition_fails_on_wild_type_run():
    """The declared swap class is entirely absent -> the run was wild-type."""
    ran = ["local:MetabolismFBA", "local:Division"]
    r = check_composition(ran, ["MetabolismReduxClassic"])
    assert not r["passed"] and r["missing"] == ["MetabolismReduxClassic"]


def test_run_gate_ands_both_checks(sweep):
    ran = ["local:MetabolismReduxClassic"]
    v = run_gate(sweep, ["listeners__mass__dry_mass"],
                 ran_processes=ran, declared_processes=["MetabolismReduxClassic"])
    assert v["passed"]
    # a wrong composition sinks an otherwise-good column check
    v2 = run_gate(sweep, ["listeners__mass__dry_mass"],
                  ran_processes=["local:MetabolismFBA"], declared_processes=["MetabolismReduxClassic"])
    assert not v2["passed"] and v2["columns_check"]["passed"]


def test_must_equal_passes_on_matching_constant(sweep):
    r = check_columns(sweep, [], must_equal={"dead_const_col": 7.0})
    assert r["passed"]
    assert r["columns"]["dead_const_col"]["mismatch_rows"] == 0
    assert r["columns"]["dead_const_col"]["expected"] == 7.0


def test_must_equal_fails_on_wrong_value(sweep):
    r = check_columns(sweep, [], must_equal={"dead_const_col": 999.0})
    assert not r["passed"]
    assert r["columns"]["dead_const_col"]["mismatch_rows"] == 3


def test_must_equal_fails_when_some_rows_differ(sweep):
    """dry_mass = 430,431,432; expecting 430 leaves 2 mismatching rows."""
    r = check_columns(sweep, [], must_equal={"listeners__mass__dry_mass": 430.0})
    assert not r["passed"]
    assert r["columns"]["listeners__mass__dry_mass"]["mismatch_rows"] == 2


def test_must_equal_fails_when_column_absent(sweep):
    """The wrong-condition column simply not being emitted must fail, not pass."""
    r = check_columns(sweep, [], must_equal={"environment__media_id": "basal"})
    assert not r["passed"]
    assert r["columns"]["environment__media_id"]["present"] is False


def test_composition_forbidden_class_present_fails():
    """A run carrying BOTH the redux and the stock class fails: the stock process
    should be gone after the swap."""
    ran = ["local:MetabolismReduxClassic", "local:MetabolismFBA"]
    r = check_composition(ran, ["MetabolismReduxClassic"], forbidden=["MetabolismFBA"])
    assert not r["passed"] and r["forbidden_present"] == ["MetabolismFBA"]


def test_composition_no_reverse_substring_false_pass():
    """A short mounted address that is a substring of the declared class must NOT
    satisfy it -- local:Metabolism does not prove MetabolismReduxClassic ran."""
    ran = ["local:Metabolism"]
    r = check_composition(ran, ["MetabolismReduxClassic"])
    assert not r["passed"] and r["missing"] == ["MetabolismReduxClassic"]


def test_bulk_species_count_from_state_dict_and_list():
    dict_state = {"agents": {"0": {"bulk": {"GLC[c]": 10, "ATP[c]": 5, "ADP[c]": 3}}}}
    assert bulk_species_count_from_state(dict_state) == 3
    list_state = {"agents": {"0": {"bulk": [["GLC[c]", 10], ["ATP[c]", 5]]}}}
    assert bulk_species_count_from_state(list_state) == 2
    assert bulk_species_count_from_state({"global_time": 0.0}) is None


def test_check_species_count_pass_fail_and_unreadable():
    assert check_species_count(16323, 16323)["passed"]
    stock = check_species_count(16321, 16323)
    assert not stock["passed"] and stock["count"] == 16321
    # an unreadable count is not a pass
    assert not check_species_count(None, 16323)["passed"]


def test_run_gate_species_check_sinks_wrong_strain(sweep):
    """Columns and (absent) composition look fine, but the bulk count is stock."""
    v = run_gate(sweep, ["listeners__mass__dry_mass"],
                 species_count=16321, expected_species_count=16323)
    assert v["columns_check"]["passed"]
    assert not v["species_check"]["passed"]
    assert not v["passed"]


def test_run_gate_must_equal_wrong_condition_sinks_good_columns(sweep):
    v = run_gate(sweep, ["listeners__mass__dry_mass"],
                 must_equal={"dead_const_col": 999.0})
    assert not v["passed"]


def test_report_translation_pass(sweep):
    """A clean gate verdict becomes a within_tol report in the card schema."""
    v = run_gate(sweep, ["listeners__mass__dry_mass"],
                 must_vary=["listeners__mass__dry_mass"])
    r = report_from_gate_verdict(v)
    assert r["overall"] == "within_tol"
    ax = r["axes"]["columns/listeners__mass__dry_mass"]
    assert ax["verdict"] == "within_tol" and ax["group"] == "Output columns"


def test_report_translation_missing_column_is_mismatch(sweep):
    r = report_from_gate_verdict(run_gate(sweep, ["listeners__mass__cell_mass"]))
    assert r["overall"] == "mismatch"
    assert r["axes"]["columns/listeners__mass__cell_mass"]["verdict"] == "mismatch"


def test_report_translation_dead_channel_composition_and_strain(sweep):
    v = run_gate(sweep, ["dead_const_col"], must_vary=["dead_const_col"],
                 ran_processes=["local:MetabolismFBA"],
                 declared_processes=["MetabolismReduxClassic"],
                 species_count=16321, expected_species_count=16323)
    r = report_from_gate_verdict(v)
    assert r["overall"] == "mismatch"
    assert r["axes"]["columns/dead_const_col"]["verdict"] == "mismatch"
    assert r["axes"]["composition/MetabolismReduxClassic"]["verdict"] == "mismatch"
    assert r["axes"]["strain/species_count"]["verdict"] == "mismatch"


def test_report_translation_wrong_condition(sweep):
    v = run_gate(sweep, ["listeners__mass__dry_mass"],
                 must_equal={"dead_const_col": 999.0})
    r = report_from_gate_verdict(v)
    assert r["overall"] == "mismatch"
    assert r["axes"]["columns/dead_const_col"]["verdict"] == "mismatch"


def test_report_translation_no_history_surfaces_error_axis(tmp_path):
    r = report_from_gate_verdict(run_gate(str(tmp_path), ["global_time"]))
    assert r["overall"] == "mismatch" and "output/history" in r["axes"]


def test_report_translation_empty_is_ungraded():
    r = report_from_gate_verdict({"columns_check": {"columns": {}, "must_vary": []}})
    assert r["overall"] == "ungraded"
    assert "acceptance/status" in r["axes"]


def test_self_test_validates_the_verifier():
    assert _self_test() == 0
