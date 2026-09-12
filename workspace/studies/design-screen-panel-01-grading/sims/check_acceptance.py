"""Compute this study's four acceptance criteria from its committed artifacts.

Each criterion is a COUNT OF VIOLATIONS that must be zero, computed here rather
than asserted in prose, so the study's outcomes cite a number a reader can
recompute. Nothing here requires a design to succeed: a panel whose every design
underperforms, or whose ranking is unresolvable, scores zero on all four.

Run:  .venv/bin/python workspace/studies/design-screen-panel-01-grading/sims/check_acceptance.py

Emits data/acceptance.json and prints a table. Exit status is 0 whatever the
counts are — this reports, the study's authored outcomes decide.
"""
from __future__ import annotations

import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
DATA = STUDY / "data"
VERDICT = STUDY / "viz" / "report_card" / "panel_screen.verdict.json"
PRESENTATION = DATA / "panel_presentation.json"

#: A resolvability axis in any of these states does not support a ranking.
_NOT_PASSING = {"mismatch", "ungraded"}
#: Retained for the reference-arm check below.
_RANKING_AXES = ("objective_vs_reference", "growth_cost")


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def every_declared_arm_is_accounted_for(acct: dict) -> tuple[int, str]:
    """An arm is accounted for when it is in the panel, OR recorded as terminated.

    ⚠ Deliberately not a cell-count threshold. A lethal design and a lost task can
    produce the same low count, so counting cells would fail real results. What is
    graded is whether a reader could be misled about what ran.
    """
    unaccounted = [
        a["arm"] for a in acct["arms"]
        if not a.get("in_panel")
        and not (a.get("terminated_before_graded_window")
                 and a.get("generations_reached") is not None)
    ]
    detail = (f"{len(acct['arms'])} declared arms; "
              f"{sum(1 for a in acct['arms'] if a.get('in_panel'))} in panel, "
              f"{sum(1 for a in acct['arms'] if a.get('terminated_before_graded_window'))} "
              "recorded as terminated before the graded window")
    if unaccounted:
        detail += f"; UNACCOUNTED: {', '.join(unaccounted)}"
    return len(unaccounted), detail


def declared_perturbations_land(acct: dict) -> tuple[int, str]:
    """Observed protein ratio within tolerance of the configured ratio."""
    misses = [f"{a['arm']}:{name}"
              for a in acct["arms"]
              for name, t in sorted((a.get("targets") or {}).items())
              if not t.get("within_tolerance")]
    checked = sum(len(a.get("targets") or {}) for a in acct["arms"])
    detail = (f"{checked} (arm, target) pairs checked at tolerance "
              f"{acct.get('landing_tolerance')}; {len(misses)} outside")
    if misses:
        detail += f": {', '.join(misses)}"
    return len(misses), detail


def _axes(verdict: dict) -> list[dict]:
    return [a for g in (verdict.get("groups") or {}).values()
            for a in (g.get("axes") or [])]


def _strata(verdict: dict) -> list[str]:
    out = []
    for a in _axes(verdict):
        aid = a.get("id", "")
        if ".medium=" in aid:
            s = aid.split(".medium=", 1)[1].split(".", 1)[0]
            if s not in out:
                out.append(s)
    return out


def reference_resolves_in_every_stratum(verdict: dict, acct: dict) -> tuple[int, str]:
    """Every stratum must carry the named reference arm.

    Read from the accounting rather than the card, because the card can only
    report a reference it found — a stratum whose reference never ran produces no
    axis at all, and an absent axis is not a failing one.
    """
    strata = _strata(verdict)
    missing = [s for s in strata
               if not any(a["stratum"] == s and a["design"] == "reference"
                          and a.get("in_panel") for a in acct["arms"])]
    detail = (f"{len(strata)} strata graded ({', '.join(strata) or 'none'}); "
              f"reference arm present in {len(strata) - len(missing)}")
    if missing:
        detail += f"; MISSING in: {', '.join(missing)}"
    return len(missing), detail


def ranking_claims_do_not_exceed_resolvability(presentation: dict) -> tuple[int, str]:
    """Where resolvability does not pass, the FIGURE must not present a ranking.

    ⛔ THIS CHECK MOVED, and the move is the point. It used to read the card's
    verdicts: "where resolvability fails, no ranking axis may carry a verdict."
    Once the outcome axes became ungraded by policy they can NEVER carry one, so
    that check could not fail — vacuous, while still reporting PASS.

    The ranking claim now lives where it always really lived: in the finding. So
    the check reads what the figure actually presented. A screen may legitimately
    find its design space unresolvable; what it may not do is draw a ranked panel
    anyway.
    """
    strata = (presentation or {}).get("strata") or {}
    violations = [f"{k}:presented a ranking at resolvability="
                  f"{v.get('resolvability_verdict')}"
                  for k, v in sorted(strata.items())
                  if not v.get("resolvable") and v.get("ranking_presented")]
    unresolvable = sum(1 for v in strata.values() if not v.get("resolvable"))
    detail = (f"{len(strata)} strata; {unresolvable} not resolvable; "
              f"{len(violations)} presented a ranking anyway")
    if violations:
        detail += "; " + ", ".join(violations)
    if not strata:
        return 1, "no presentation record — run sims/render_panel.py first"
    return len(violations), detail


def main() -> None:
    acct = _load(DATA / "panel_accounting.json")
    verdict = _load(VERDICT)
    presentation = _load(PRESENTATION)

    results = {
        "every-declared-arm-is-accounted-for":
            every_declared_arm_is_accounted_for(acct),
        "declared-perturbations-land-within-tolerance":
            declared_perturbations_land(acct),
        "reference-arm-resolves-in-every-stratum":
            reference_resolves_in_every_stratum(verdict, acct),
        "ranking-claims-do-not-exceed-resolvability":
            ranking_claims_do_not_exceed_resolvability(presentation),
    }

    out = {
        "_comment": [
            "Computed by sims/check_acceptance.py from data/panel_accounting.json",
            "and viz/report_card/panel_screen.verdict.json.",
            "Each value is a COUNT OF VIOLATIONS; zero is the passing state.",
            "No criterion requires a design to succeed.",
        ],
        "card_overall": verdict.get("overall"),
        "criteria": {k: {"violations": v[0], "detail": v[1]}
                     for k, v in results.items()},
    }
    (DATA / "acceptance.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")

    width = max(len(k) for k in results)
    for name, (count, detail) in results.items():
        print(f"{'PASS' if count == 0 else 'FAIL'}  {name:<{width}}  "
              f"violations={count}")
        print(f"      {detail}")
    print(f"\ncard overall verdict: {verdict.get('overall')} "
          "(independent of the criteria above, by design)")


if __name__ == "__main__":
    main()
