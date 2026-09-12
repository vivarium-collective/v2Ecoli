"""Render the screen's panel — the ranked arms AND the arms that never arrived.

A screen's scientific output is the ranked panel, so the panel is the figure. But
a ranking alone is a misleading figure here: an arm that terminated before the
graded window contributes no row to the ranking, and a reader seeing N-1 rows has
no way to know N were declared. So the accounting sits in the same frame as the
ranking, which is the study's whole claim rendered as one picture.

⚠ SVG, not HTML, and that is not a style choice. Four independent consumers each
carry their own address-scheme set and ``image:`` is the only one all four accept;
``html:`` does not survive into the published report, and ``file:`` + ``.html``
renders nothing at all while passing both the linter and the audit. An
unrecognised scheme vanishes silently from the charts payload.

Reads only committed artifacts, so the figure re-renders without re-running
anything:
    data/panel_per_cell.tsv          the per-cell seam
    data/panel_accounting.json       declared-vs-observed, generations reached
    viz/report_card/panel_screen.verdict.json   the graded axes

Run:  .venv/bin/python workspace/studies/design-screen-panel-01-grading/sims/render_panel.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
DATA = STUDY / "data"
VERDICT = STUDY / "viz" / "report_card" / "panel_screen.verdict.json"

# Status vocabulary: icon + word + reserved colour. Never colour alone — a reader
# with a colour-vision difference, or a greyscale print, must still get the state.
STATUS = {
    "within_tol": ("✓", "within band", "#1a7f37"),
    "drift":      ("△", "drift",       "#9a6700"),
    "mismatch":   ("✕", "outside band", "#b3261e"),
    "ungraded":   ("○", "ungraded",    "#57606a"),
    "absent":     ("●", "no cells",    "#57606a"),
}

# ⛔ The design-target vocabulary is DELIBERATELY DISTINCT from STATUS above.
# STATUS renders CARD VERDICTS — rulings about whether the screen executed
# correctly. These render a FINDING: how the best arm sits against the design
# target the study declared. Sharing one glyph set would invite a reader to
# read a biology result as a card ruling, which is the confusion this whole
# study exists to prevent.
TARGET = {
    "meets":  ("▲", "meets target",   "#1a7f37"),
    "under":  ("▽", "under target",   "#9a6700"),
    "n/a":    ("·", "not compared",   "#8c959f"),
}

W, ROW, PAD = 1180, 26, 18
COLS = [(24, "arm"), (230, "cells"), (330, "objective"), (450, "growth"),
        (580, "vs design target"), (790, "note")]


def esc(t: str) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def load():
    rows = []
    for i, line in enumerate((DATA / "panel_per_cell.tsv").read_text().splitlines()):
        parts = line.split("\t")
        if i == 0:
            head = parts
            continue
        rows.append(dict(zip(head, parts)))
    acct = json.loads((DATA / "panel_accounting.json").read_text())
    verdict = json.loads(VERDICT.read_text())
    return rows, acct, verdict


def per_arm(rows):
    """Mean per arm, aggregating within cell then across cells."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["arm"], r["lineage_seed"], r["generation"], r["agent_id"])
        cells[r["arm"]][key].append((float(r["objective_titer"]),
                                     float(r["growth_rate"])))
    out = {}
    for arm, by_cell in cells.items():
        obj = [sum(v[0] for v in vals) / len(vals) for vals in by_cell.values()]
        gro = [sum(v[1] for v in vals) / len(vals) for vals in by_cell.values()]
        out[arm] = {"n": len(by_cell),
                    "objective": sum(obj) / len(obj),
                    "growth": sum(gro) / len(gro)}
    return out


def axis_verdicts(verdict):
    """{stratum: {axis: verdict}} from the card."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for g in (verdict.get("groups") or {}).values():
        for a in g.get("axes") or []:
            aid = a.get("id", "")
            if ".medium=" not in aid:
                continue
            stratum, _, axis = aid.split(".medium=", 1)[1].partition(".")
            out[stratum][axis] = a.get("verdict", "ungraded")
    return out


def design_targets(verdict) -> dict:
    """{stratum: {axis: design_target}} — the declared targets the card REPORTS
    against but does not grade."""
    out = {}
    for g in (verdict.get("groups") or {}).values():
        for a in g.get("axes", []):
            aid = a.get("id", "")
            if ".medium=" not in aid:
                continue
            stratum, _, axis = aid.split(".medium=", 1)[1].partition(".")
            tgt = (a.get("detail") or {}).get("design_target")
            if tgt:
                out.setdefault(stratum, {})[axis] = tgt
    return out


def render(rows, acct, verdict, presented: dict) -> str:
    stats = per_arm(rows)
    verdicts = axis_verdicts(verdict)
    targets = design_targets(verdict)
    strata = sorted({a["stratum"] for a in acct["arms"]})
    expected = acct["cells_expected_per_arm"]

    parts, y = [], 0
    parts.append('<text x="24" y="34" class="h1">Design panel — ranked arms, '
                 'and the arms that never arrived</text>')
    parts.append('<text x="24" y="56" class="sub">Synthetic panel. Each arm '
                 'compared to the named reference within its own stratum. '
                 'Design-target comparisons are FINDINGS, not card verdicts — '
                 'the card grades execution, not which design wins.</text>')
    y = 78

    for stratum in strata:
        y += 26
        parts.append(f'<text x="24" y="{y}" class="h2">stratum: {esc(stratum)}</text>')
        y += 8
        parts.append(f'<line x1="24" y1="{y}" x2="{W-24}" y2="{y}" class="rule"/>')
        y += 20
        for x, label in COLS:
            parts.append(f'<text x="{x}" y="{y}" class="th">{esc(label)}</text>')
        y += 6
        parts.append(f'<line x1="24" y1="{y}" x2="{W-24}" y2="{y}" class="rule"/>')

        entries = [a for a in acct["arms"] if a["stratum"] == stratum]
        ref = next((a for a in entries if a["design"] == "reference"), None)
        ref_stats = stats.get(ref["arm"]) if ref else None

        # Present arms first, ranked by objective; absent arms last.
        present = [a for a in entries if a["in_panel"]]
        absent = [a for a in entries if not a["in_panel"]]
        rv = verdicts.get(stratum, {}).get("ranking_resolvable", "ungraded")
        resolvable = rv not in ("mismatch", "ungraded")
        if resolvable:
            present.sort(key=lambda a: -stats[a["arm"]]["objective"])
        else:
            # ⛔ THE RANKING IS WITHHELD, not merely annotated. Rows ordered by
            # arm id, and no arm is named "best": an ordering drawn from noise
            # is a claim the evidence does not carry, and a reader takes row
            # order as the claim whatever the footnote says.
            present.sort(key=lambda a: a["arm"])
        best_arm = (next((a["arm"] for a in present
                          if a["design"] != "reference"), None)
                    if resolvable else None)
        presented[stratum] = {"resolvable": resolvable,
                              "ranking_presented": bool(best_arm),
                              "resolvability_verdict": rv,
                              "n_present": len(present), "n_absent": len(absent)}

        for a in present + absent:
            y += ROW
            st = stats.get(a["arm"])
            is_ref = a["design"] == "reference"
            if st and ref_stats:
                obj = f'{st["objective"] / ref_stats["objective"]:.2f}x'
                gro = f'{st["growth"] / ref_stats["growth"]:.2f}x'
                cells = f'{st["n"]} / {expected}'
            else:
                obj = gro = "—"
                cells = f'0 / {expected}'

            if not a["in_panel"]:
                icon, word, colour = STATUS["absent"]
                note = f'terminated at gen {a["generations_reached"]}'
            elif is_ref:
                icon, word, colour = "—", "reference", "#57606a"
                note = "the named comparator"
            elif a["arm"] == best_arm:
                # The FINDING for the top-ranked arm, against the declared design
                # target. Not a card verdict: the card grades execution, and this
                # column says nothing about whether the screen ran correctly.
                # An earlier version stamped the stratum's CARD verdict onto every
                # row, reporting a design as failing a band it was never graded
                # against -- moderate at 0.91x growth flagged for a 0.85 floor it
                # clears. Only the top-ranked arm carries a target comparison.
                gt = targets.get(stratum, {}).get("growth_cost")
                gv = stats[a["arm"]]["growth"] / ref_stats["growth"] if ref_stats else None
                if gt is None or gv is None:
                    icon, word, colour = TARGET["n/a"]
                    note = "highest objective"
                elif gv < gt.get("warn", float("-inf")):
                    icon, word, colour = TARGET["under"]
                    note = (f'highest objective; growth {gv:.2f}x under the '
                            f'{gt["warn"]:.2f}x design target')
                else:
                    icon, word, colour = TARGET["meets"]
                    note = "highest objective; growth meets target"
            else:
                icon, word, colour = TARGET["n/a"]
                note = ("ranking withheld — not resolvable" if not resolvable
                        else "ranked, not the top arm")

            label = a["design"] + ("  (reference)" if is_ref else "")
            parts.append(f'<text x="{COLS[0][0]}" y="{y}" class="td">{esc(label)}</text>')
            parts.append(f'<text x="{COLS[1][0]}" y="{y}" class="tdn">{esc(cells)}</text>')
            parts.append(f'<text x="{COLS[2][0]}" y="{y}" class="tdn">{esc(obj)}</text>')
            parts.append(f'<text x="{COLS[3][0]}" y="{y}" class="tdn">{esc(gro)}</text>')
            parts.append(f'<text x="{COLS[4][0]}" y="{y}" class="td" '
                         f'fill="{colour}">{icon} {esc(word)}</text>')
            parts.append(f'<text x="{COLS[5][0]}" y="{y}" class="tdm">{esc(note)}</text>')

        # ⚠ 26px, not 14: at 14 this line sat flush under the last row and read
        # as belonging to that ARM. It is a statement about the STRATUM, and a
        # scope-level claim rendered as a row-level one is the same category of
        # error this figure exists to avoid.
        y += 26
        icon, word, _ = STATUS.get(rv, STATUS["ungraded"])
        tail = ("rows are ordered by arm id, not ranked"
                if not resolvable else "rows are ranked by objective")
        parts.append(f'<text x="24" y="{y}" class="foot">stratum ranking '
                     f'resolvable: {icon} {esc(word)} — {tail}</text>')
        y += 12

    y += 30
    declared = len(acct["arms"])
    shown = sum(1 for a in acct["arms"] if a["in_panel"])
    parts.append(f'<line x1="24" y1="{y-18}" x2="{W-24}" y2="{y-18}" class="rule"/>')
    parts.append(f'<text x="24" y="{y}" class="foot">{declared} arms declared · '
                 f'{shown} reached the graded window · {declared - shown} recorded as '
                 f'terminated before it. Every declared arm is accounted for.</text>')
    height = y + 24

    style = ("<style>"
             ".h1{font:600 17px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#1f2328}"
             ".sub{font:400 12px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#57606a}"
             ".h2{font:600 13px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#1f2328}"
             ".th{font:600 11px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#57606a}"
             ".td{font:400 13px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#1f2328}"
             ".tdn{font:400 13px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#1f2328}"
             ".tdm{font:400 12px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#57606a}"
             ".foot{font:400 11px -apple-system,Segoe UI,Helvetica,sans-serif;fill:#57606a}"
             ".rule{stroke:#d0d7de;stroke-width:1}"
             "</style>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}">{style}'
            f'<rect width="{W}" height="{height}" fill="#ffffff"/>'
            + "".join(parts) + "</svg>\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(STUDY / "charts" / "panel.svg"))
    args = ap.parse_args()
    rows, acct, verdict = load()
    presented: dict = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, acct, verdict, presented), encoding="utf-8")
    # The machine-readable record of WHAT THE FIGURE CLAIMED. The acceptance
    # criterion on ranking discipline reads this: the claim now lives in the
    # finding, so the check has to look at the finding.
    (STUDY / "data" / "panel_presentation.json").write_text(
        json.dumps({"_comment": ["Written by sims/render_panel.py.",
                                 "Per stratum: was a ranking actually presented?"],
                    "strata": presented}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
