"""AcceptanceCard: the gate wired into the report-card path.

The verdict->report translation is covered run-free in test_acceptance_gate.py
(report_from_gate_verdict). This checks the card registers, selects on the
`acceptance:` block, and emits the shared report_card_verdict/v1 schema through
the same verdict_json / render_verdict_html the science cards use. Importing the
report_cards package pulls the report_card / viva_superpowers.card_grade layer,
so CI verifies these (a stale local venv can't import them).
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.fast


def _card():
    # applies()/build() don't touch instance state, so skip __init__ (which wants
    # a bigraph core) via __new__ -- the same trick keeps this test core-free.
    from v2ecoli.workflow.post_sim import REPORT_CARD_REGISTRY
    cls = REPORT_CARD_REGISTRY["acceptance"]
    return cls.__new__(cls)


def test_acceptance_card_registered_first_class():
    import v2ecoli.workflow.report_cards  # noqa: F401 -- import registers cards
    from v2ecoli.workflow.post_sim import REPORT_CARD_REGISTRY
    cls = REPORT_CARD_REGISTRY.get("acceptance")
    assert cls is not None and cls.__name__ == "AcceptanceCard"


def test_applies_only_when_acceptance_declared():
    from v2ecoli.workflow.report_cards import StudyContext
    card = _card()
    assert card.applies(StudyContext("s", Path("."), {}, Path("."))) is False
    assert card.applies(
        StudyContext("s", Path("."), {"acceptance": {"required_columns": ["x"]}},
                     Path("."))) is True


def test_build_emits_shared_v1_schema(tmp_path):
    from v2ecoli.workflow.report_cards import StudyContext
    leaf = (tmp_path / "out" / "history" / "experiment_id=t" / "variant=0"
            / "lineage_seed=0" / "generation=1" / "agent_id=1")
    leaf.mkdir(parents=True)
    pq.write_table(pa.table({"global_time": [0.0, 1.0],
                             "listeners__mass__dry_mass": [430.0, 431.0]}),
                   str(leaf / "0.pq"))
    spec = {"acceptance": {"sweep_dir": "out",
                           "required_columns": ["listeners__mass__dry_mass"]}}
    vjson, html = _card().build(StudyContext("s", tmp_path, spec, tmp_path))
    assert vjson["schema"] == "report_card_verdict/v1"
    assert vjson["overall"] == "within_tol"
    assert isinstance(html, str) and html


def test_build_failing_run_dominates_as_mismatch(tmp_path):
    from v2ecoli.workflow.report_cards import StudyContext
    leaf = (tmp_path / "out" / "history" / "experiment_id=t" / "variant=0"
            / "lineage_seed=0" / "generation=1" / "agent_id=1")
    leaf.mkdir(parents=True)
    pq.write_table(pa.table({"global_time": [0.0, 1.0]}), str(leaf / "0.pq"))
    spec = {"acceptance": {"sweep_dir": "out",
                           "required_columns": ["listeners__mass__cell_mass"]}}
    vjson, _ = _card().build(StudyContext("s", tmp_path, spec, tmp_path))
    assert vjson["overall"] == "mismatch"


def test_missing_sweep_dir_is_ungraded_not_crash():
    from v2ecoli.workflow.report_cards import StudyContext
    vjson, _ = _card().build(
        StudyContext("s", Path("."), {"acceptance": {"required_columns": ["x"]}},
                     Path(".")))
    assert vjson["overall"] == "ungraded"
