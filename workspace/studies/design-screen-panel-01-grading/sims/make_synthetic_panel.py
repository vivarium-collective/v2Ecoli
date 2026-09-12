"""Generate the synthetic panel this study grades, and the accounting beside it.

Why synthetic: the grading layer is engine-agnostic by design — it consumes a
per-cell table and knows nothing about what produced it. Proving that chain needs
a per-cell table, not a simulation, and building it from a fixture makes the
criteria testable without a sweep, a ParCa cache, or a run.

Three artifacts, in dependency order — and the order is the point:

  data/panel_per_cell.tsv      one row per (arm x lineage_seed x generation x
                               cell). THE SEAM. Whatever engine produces it,
                               everything downstream is written once.
  data/panel.json              the card's input, DERIVED FROM the TSV rather than
                               written alongside it, so the seam is load-bearing
                               here rather than decorative.
  data/panel_accounting.json   per-arm: declared vs observed cells, generations
                               reached, and per-target observed/expected protein
                               ratios. The evidence behind the accounting and
                               landing criteria.

⚠ Deterministic by construction — a fixed generator seed, and no wall-clock or
unordered iteration anywhere. The outputs are committed and re-rendered, so a
value that moved between runs would show up as a spurious diff and, worse, would
make the graded verdicts irreproducible.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"

#: Replicate structure. Four lineages x four graded generations = 16 cells per
#: arm, which is what the accounting compares against.
SEEDS = [0, 1, 2, 3]
GRADED_GENERATIONS = [5, 6, 7, 8]
EXPECTED_CELLS = len(SEEDS) * len(GRADED_GENERATIONS)

STRATA = ["medium_a", "medium_b"]

#: Per-design behaviour. ``objective`` and ``growth`` are means relative to the
#: reference; ``cv`` is the within-arm coefficient of variation.
#:
#: The set is chosen so each criterion has something to bite on:
#:   reference      the named comparator, resolved per stratum
#:   moderate       wins on the objective at a growth cost inside the floor
#:   strong         wins harder on the objective and FAILS the growth floor —
#:                  a design that "wins" by killing the cell is not a win
#:   nonviable      contributes NO cells to the graded window. It must appear in
#:                  the accounting with the generation it reached, never vanish
DESIGNS = {
    "reference":  {"objective": 1.00, "growth": 1.00, "cv": 0.04, "viable": True},
    "moderate":   {"objective": 1.42, "growth": 0.93, "cv": 0.04, "viable": True},
    "strong":     {"objective": 1.71, "growth": 0.64, "cv": 0.05, "viable": True},
    "nonviable":  {"objective": 0.00, "growth": 0.00, "cv": 0.00, "viable": False},
}

#: What each design declared, and what the run observed. The landing criterion
#: compares these; it is measured on observed protein because a declared
#: translation-efficiency multiplier is a weight, not an achieved ratio.
DECLARED_TARGETS = {
    "reference": {},
    "moderate":  {"target_gene_1": {"expected": 2.0, "observed": 1.86},
                  "target_gene_2": {"expected": 0.0, "observed": 0.0}},
    "strong":    {"target_gene_1": {"expected": 8.0, "observed": 7.10},
                  "target_gene_2": {"expected": 0.0, "observed": 0.0}},
    "nonviable": {"target_gene_3": {"expected": 0.0, "observed": 0.0}},
}

#: Inherited from the screen configuration rather than chosen here, so the
#: criterion adopts a threshold the screen's authors already committed to.
LANDING_TOLERANCE = 0.30

#: The generation a non-viable lineage reached before it stopped. Recorded rather
#: than inferred: "this design does not sustain growth" is a screen result.
NONVIABLE_LAST_GENERATION = 2


def build_per_cell_rows(rng: random.Random) -> list[dict]:
    """The seam: one row per cell, in a stable order."""
    rows: list[dict] = []
    for stratum in STRATA:
        for design, spec in DESIGNS.items():
            if not spec["viable"]:
                continue
            for seed in SEEDS:
                for generation in GRADED_GENERATIONS:
                    rows.append({
                        "arm": f"{design}|{stratum}",
                        "design": design,
                        "medium": stratum,
                        "lineage_seed": seed,
                        "generation": generation,
                        "agent_id": f"{seed}{generation}",
                        "objective_titer": round(
                            spec["objective"] * rng.gauss(1.0, spec["cv"]), 6),
                        "growth_rate": round(
                            spec["growth"] * rng.gauss(1.0, spec["cv"]), 6),
                    })
    return rows


def write_per_cell_tsv(rows: list[dict], path: Path) -> None:
    cols = ["arm", "design", "medium", "lineage_seed", "generation", "agent_id",
            "objective_titer", "growth_rate"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def panel_from_per_cell(rows: list[dict]) -> dict:
    """Derive the card's input FROM the per-cell table.

    ⚠ Deliberately reads the TSV's own rows rather than regenerating values. If
    this ever diverges from the table, the card is grading something the study's
    evidence does not show.
    """
    arms: dict[str, dict] = {}
    for r in rows:
        arm = arms.setdefault(r["arm"], {
            "arm": r["arm"],
            "design": r["design"],
            "strata": {"medium": r["medium"]},
            "observables": {"objective_titer": {"by_cell": []},
                            "growth_rate": {"by_cell": []}},
        })
        for obs in ("objective_titer", "growth_rate"):
            arm["observables"][obs]["by_cell"].append(
                [r["lineage_seed"], r["generation"], r[obs]])
    return {
        "_comment": [
            "Synthetic panel for design-screen-panel-01-grading.",
            "DERIVED from data/panel_per_cell.tsv by sims/make_synthetic_panel.py.",
            "Values are illustrative and carry no biological claim; the point is",
            "the grading chain, not the numbers.",
        ],
        "panel": "design-screen-panel-01-synthetic",
        "arms": [arms[k] for k in sorted(arms)],
    }


def accounting(rows: list[dict]) -> dict:
    """Per-arm accounting — the record that turns a dead arm into a datum."""
    seen: dict[str, list[dict]] = {}
    for r in rows:
        seen.setdefault(r["arm"], []).append(r)

    entries = []
    for stratum in STRATA:
        for design, spec in DESIGNS.items():
            arm = f"{design}|{stratum}"
            cells = seen.get(arm, [])
            gens = sorted({c["generation"] for c in cells})
            targets = {
                name: {
                    "expected": t["expected"],
                    "observed": t["observed"],
                    "within_tolerance": _within_tolerance(t["expected"], t["observed"]),
                }
                for name, t in sorted(DECLARED_TARGETS[design].items())
            }
            entries.append({
                "arm": arm,
                "design": design,
                "stratum": stratum,
                "declared": True,
                "in_panel": bool(cells),
                "cells_observed": len(cells),
                "cells_expected": EXPECTED_CELLS,
                "generations_reached": (max(gens) if gens
                                        else NONVIABLE_LAST_GENERATION),
                "terminated_before_graded_window": not cells,
                "termination_note": (
                    None if cells else
                    "Lineage did not sustain growth to the graded window. Recorded "
                    "rather than dropped: a design that does not grow is a screen "
                    "result, not a missing measurement."
                ),
                "targets": targets,
            })
    return {
        "_comment": [
            "Per-arm accounting for design-screen-panel-01-grading.",
            "An arm is ACCOUNTED FOR when in_panel is true, or when",
            "terminated_before_graded_window is true and generations_reached is",
            "recorded. Absent-without-record is the failure; dying is not.",
        ],
        "landing_tolerance": LANDING_TOLERANCE,
        "cells_expected_per_arm": EXPECTED_CELLS,
        "arms": entries,
    }


def _within_tolerance(expected: float, observed: float) -> bool:
    """Observed within tolerance of expected, as a ratio.

    A knockout (expected 0) is graded on absolute agreement — a relative
    tolerance around zero is undefined, and silently treating it as a pass is how
    a knockout that did not happen gets reported as one that did.
    """
    if expected == 0:
        return observed == 0
    return abs(observed / expected - 1.0) <= LANDING_TOLERANCE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260821,
                    help="generator seed; fixed so outputs are reproducible")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    rows = build_per_cell_rows(random.Random(args.seed))

    write_per_cell_tsv(rows, DATA / "panel_per_cell.tsv")
    (DATA / "panel.json").write_text(
        json.dumps(panel_from_per_cell(rows), indent=2) + "\n", encoding="utf-8")
    (DATA / "panel_accounting.json").write_text(
        json.dumps(accounting(rows), indent=2) + "\n", encoding="utf-8")

    print(f"per-cell rows : {len(rows)}")
    print(f"arms in panel : {len({r['arm'] for r in rows})}")
    print(f"arms declared : {len(STRATA) * len(DESIGNS)}")
    print(f"wrote         : {DATA}/panel_per_cell.tsv, panel.json, "
          "panel_accounting.json")


if __name__ == "__main__":
    main()
