"""Gen-2 report card: panel screen — N designs ranked against a reference arm.

Sibling of ``acetate_overflow_card`` and built on the same three-way split: the
science core lives in ``v2ecoli/library/panel_screen.py``, grading + rendering in
``v2ecoli/library/report_card.py``, and this module is only the Step.

Gated on the study declaring ``report_card_refs.panel_screen``. Every input is a
study input — no observable, arm name, band or condition is a constant here::

    report_card_refs:
      panel_screen:
        panel_json: tests/fixtures/panel_screen/panel_baseline.json
        objective_observable: objective_titer
        growth_observable: growth_rate
        reference_arm: wt              # a DESIGN name, resolved per stratum
        higher_is_better: true         # required: not every objective is maximised
        strata: [media]                # required: this IS the testing family
        bands:                         # required, all three; never defaulted
          objective_vs_reference: {good: 1.20, warn: 1.05}
          growth_cost:            {good: 0.85, warn: 0.70}
          ranking_resolvable:     {good: 3.0,  warn: 2.0}

Fixture-graded like ``acetate_overflow_card``: the panel JSON holds the baked
per-arm per-cell values, so the card grades run-free from a ``StudyContext`` with no
sweep and no ParCa cache.

``applies`` is gated **only** on the ref block existing, deliberately. The runner
swallows exceptions out of ``build`` (``TestStep.invoke``), so a card that
disqualified itself on a malformed ref would make the stratification contract the
softest failure in the system: a mis-specified panel would look like "card not
applicable" instead of a red axis. Missing ``strata`` therefore grades as a failing
axis (in the library), and a malformed ref raises.
"""
from __future__ import annotations

from pathlib import Path

from v2ecoli.library import panel_screen as ps
from v2ecoli.library.report_card import grade_card, render_html, verdict_json
from v2ecoli.workflow.report_cards import ReportCardStep, StudyContext


class PanelScreenCard(ReportCardStep):
    name = "panel_screen"

    def _ref(self, study: StudyContext) -> "dict | None":
        cfg = (study.spec.get("report_card_refs") or {}).get(self.name)
        return cfg if isinstance(cfg, dict) else None

    def applies(self, study: StudyContext) -> bool:
        return self._ref(study) is not None

    def _panel_path(self, study: StudyContext, cfg: dict) -> Path:
        rel = cfg.get("panel_json")
        if not rel:
            raise ValueError("panel_screen: report_card_refs.panel_screen.panel_json "
                             "must point at the baked panel JSON")
        p = Path(rel) if str(rel).startswith("/") else (study.ws_root / rel)
        if not p.is_file():
            raise ValueError(f"panel_screen: panel_json not found: {p}")
        return p

    def build(self, study: StudyContext):
        cfg = self._ref(study)
        if cfg is None:
            return None
        panel = ps.load_panel(self._panel_path(study, cfg))
        card, reference = ps.build(
            panel,
            objective_observable=cfg.get("objective_observable"),
            growth_observable=cfg.get("growth_observable"),
            reference_arm=cfg.get("reference_arm"),
            strata=cfg.get("strata"),
            higher_is_better=cfg.get("higher_is_better"),
            bands=cfg.get("bands"),
            graded_axes=cfg.get("graded_axes"),
            title=cfg.get("title"),
        )
        report = grade_card(card, reference)
        model_ref = reference["stimulus"]["measured_model"]
        vjson = verdict_json(report, model_ref=model_ref,
                             reference_model=reference["stimulus"]["reference_model"])
        vjson["title"] = reference["title"]
        # No `generated=`: the runner commits this output, so it must stay
        # byte-deterministic across re-renders.
        html = render_html(card, reference, model_ref=model_ref)
        return vjson, html
