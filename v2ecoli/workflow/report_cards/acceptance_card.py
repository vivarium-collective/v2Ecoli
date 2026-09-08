# v2ecoli/workflow/report_cards/acceptance_card.py
"""The run-integrity gate as a first-class report card.

The acceptance gate (columns present/non-null/vary/equal, composition, strain
fingerprint) used to be a parallel, unwired system with its own verdict format.
This card wraps the SAME engine (``acceptance_gate.run_gate``) and expresses its
result through the SAME ``verdict_json`` / ``render_verdict_html`` every science
card uses -- one verdict schema (``report_card_verdict/v1``), one report surface,
one config block. It answers "is this run real and from the strain/condition we
declared?"; the science cards then grade whether the real measurement is good.

A gate FAIL surfaces as a failing card whose ``overall`` is ``mismatch``; because
the study rolls up the worst verdict across cards, an invalid run dominates the
overall rather than letting a science card show a confident PASS on a run that did
not happen (cplong90 #234/#544).

The study opts in by declaring an ``acceptance:`` block in ``study.yaml``::

    acceptance:
      sweep_dir: out/run                 # local path or s3://; relative -> study_dir
      required_columns: [listeners__mass__dry_mass, environment__exchange__VIOLACEIN]
      must_vary: [listeners__mass__dry_mass]
      must_equal: {environment__media_id: basal_with_trp}
      expected_species_count: 16323
      declared_processes: [MetabolismReduxClassic]
      forbidden_processes: [MetabolismFBA]
      process_table: out/run/final_state.json   # for composition + species count

Only the checks whose keys are present run; the rest are skipped. ``required_columns``
alone is a valid, useful contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from v2ecoli.library.report_card import render_verdict_html, verdict_json
from v2ecoli.workflow.acceptance_gate import (
    bulk_species_count_from_state,
    process_names_from_state,
    report_from_gate_verdict,
    run_gate,
)
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext

_NA = "ungraded"


def _resolve(path: str, study_dir: Path) -> str:
    """Leave s3:// and absolute paths as-is; resolve a relative path against the
    study dir so a study.yaml can name its output relative to itself."""
    if str(path).startswith("s3://") or os.path.isabs(path):
        return str(path)
    return str(study_dir / path)


def _ungraded(reason: str) -> dict:
    return {"overall": _NA, "axes": {"acceptance/status": {
        "group": "Acceptance", "label": "gate could not run", "verdict": _NA,
        "value": None, "meter": reason, "detail": {}, "path": "acceptance/status"}}}


class AcceptanceCard(ReportCardStep):
    name = "acceptance"

    def applies(self, study: StudyContext) -> bool:
        return bool(study.spec.get("acceptance"))

    def build(self, study: StudyContext):
        spec = study.spec.get("acceptance") or {}
        report = self._grade(spec, study)
        vjson = verdict_json(report, model_ref=study.study_name)
        html = render_verdict_html(vjson, title="Run acceptance")
        return vjson, html

    def _grade(self, spec: dict, study: StudyContext) -> dict:
        sweep = spec.get("sweep_dir")
        if not sweep:
            return _ungraded("no acceptance.sweep_dir declared")
        sweep = _resolve(sweep, study.study_dir)

        ran = None
        species_count = None
        pt = spec.get("process_table")
        if pt:
            ptp = _resolve(pt, study.study_dir)
            try:
                state = json.loads(Path(ptp).read_text(encoding="utf-8"))
            except OSError as e:
                return _ungraded(f"could not read process_table {ptp!r}: {e}")
            ran = sorted(process_names_from_state(state))
            species_count = bulk_species_count_from_state(state)

        try:
            verdict = run_gate(
                sweep,
                spec.get("required_columns", []),
                must_vary=spec.get("must_vary", ()),
                must_equal=spec.get("must_equal"),
                ran_processes=ran,
                declared_processes=spec.get("declared_processes"),
                forbidden_processes=spec.get("forbidden_processes", ()),
                species_count=species_count,
                expected_species_count=spec.get("expected_species_count"),
            )
        except Exception as e:  # noqa: BLE001 -- a gate failure must render, not crash the suite
            return _ungraded(f"acceptance gate raised: {type(e).__name__}: {e}")
        return report_from_gate_verdict(verdict)
