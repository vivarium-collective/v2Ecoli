"""Regenerate the panel card's committed verdict JSON and HTML from `panel.json`.

WHY THIS EXISTS. The verdict JSON is a COMMITTED artifact that the acceptance
criteria and the figure both read. Without a regeneration step in the study it
is a hand-carried file: a change to the card library leaves the committed
verdict encoding the OLD card, and nothing in the study can tell. That is
exactly the defect this study exists to catch in a screen, so it must not be
the study's own shape.

The card is a pure function of `panel.json` plus this study's
`report_card_refs.panel_screen` block, and deliberately carries no timestamp, so
re-running this is byte-deterministic. It performs no simulation.

Run:  .venv/bin/python workspace/studies/design-screen-panel-01-grading/sims/render_card.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from v2ecoli.library import panel_screen as ps
from v2ecoli.library.report_card import grade_card, render_html, verdict_json

STUDY = Path(__file__).resolve().parent.parent
WS_ROOT = STUDY.parent.parent.parent          # repo root (workspace/studies/<s>/sims)
OUT = STUDY / "viz" / "report_card"


def main() -> int:
    cfg = (yaml.safe_load((STUDY / "study.yaml").read_text(encoding="utf-8"))
           or {}).get("report_card_refs", {}).get("panel_screen")
    if not isinstance(cfg, dict):
        raise SystemExit("render_card: study.yaml has no report_card_refs.panel_screen")

    panel = ps.load_panel(WS_ROOT / cfg["panel_json"])
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

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "panel_screen.verdict.json").write_text(
        json.dumps(vjson, indent=1) + "\n", encoding="utf-8")
    (OUT / "panel_screen.html").write_text(
        render_html(card, reference, model_ref=model_ref), encoding="utf-8")

    graded = [a for a in vjson["groups"]["panel_screen"]["axes"]
              if a["detail"].get("graded")] if vjson.get("groups") else []
    print(f"overall       : {vjson['overall']}")
    print(f"graded axes   : {len(graded)} of "
          f"{len(vjson['groups']['panel_screen']['axes'])} "
          f"({', '.join(sorted({a['id'].rsplit('.', 1)[-1] for a in graded})) or 'none'})")
    print(f"wrote         : {OUT}/panel_screen.verdict.json, panel_screen.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
