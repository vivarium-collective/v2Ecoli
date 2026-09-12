"""Panel-screen grading — the science core.

A **panel screen** is N distinct designs, optionally crossed with M environmental
conditions, ranked against each other: which design wins, by how much, and at what
cost. That is a different readout from a *sweep* along one axis
(``library/phenotype_sweep``), which asks how an observable moves as one knob turns.

★ **The contract is the deliverable.** The failure this module exists to prevent is
not a missing statistic — it is **family construction**. In a prior panel analysis
Welch was applied correctly per arm, but the q-value family pooled two media, so
nearly every arm in the second medium read "significant" when most of that signal was
the medium rather than the design. A library function cannot prevent that, because the
*caller* decides the family. So here:

* ``strata`` is **required**. A panel cannot be graded without declaring what the
  testing family is (a missing declaration produces a visible failing axis, not a
  silent global family — see ``build``).
* **one BH family per stratum**, never one across the panel
  (``apply_stratified_qvalues``).
* the **reference arm is resolved within each stratum**, so a design is contrasted
  against the control it shares conditions with.
* an arm's identity (``arm``) must be unique and carry its stratum. If a label were
  derived from the design vector alone, the same design in two conditions would give
  two arms with one label, silently sharing a row. Duplicates are a hard error.

**Consumed, not reimplemented:** Welch's t-test from
``viva_superpowers.card_criteria._welch``; the verdict banding from ``_band``; the
BH-FDR step-up from ``scipy.stats.false_discovery_control``; cell-level aggregation
upstream of us (``by_cell`` is ``[[seed, gen, value], ...]``, the shape
``workflow/analysis.py`` emits and ``library/report_card._decompose`` reads).

Import-safe: no CLI, no duckdb, no sweep access. The Gen-2 Step is
``v2ecoli/workflow/report_cards/panel_screen_card.py``; observable ids, the reference
arm, the strata keys and all three bands are **study inputs** via
``report_card_refs.panel_screen`` — nothing here is organism-, product- or
pathway-specific.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from v2ecoli.library.card_criteria import _band, _welch

#: Every axis this card emits lands in one report-card group.
GROUP = "Panel screen"

#: The graded axes, per stratum. Bands for all three are REQUIRED in
#: ``report_card_refs`` — a default here would bake an unexamined number into every
#: future study.
AXES = ("objective_vs_reference", "growth_cost", "ranking_resolvable")

# ★ WHICH AXES CARRY A VERDICT — declarable, defaulted to the defensible policy.
#
# A design screen's report card grades whether the screen was EXECUTED and
# RECORDED faithfully; which design wins is a scientific FINDING. A verdict
# computed from the biology makes the card's overall a function of the result,
# so a screen that ran perfectly and found nothing good reads as FAILED — the
# instrument becomes uninterpretable exactly when it is working. Resolvability
# is the exception and therefore the default: it is a property of the screen's
# replicate count and separation, not of whether the designs were any good.
#
# Ungraded axes still carry value, meter, p/q and their declared `design_target`,
# so nothing is hidden — only the ruling is withheld. A caller that genuinely
# wants an outcome graded can say so via `graded_axes` rather than patching this
# module; the default is a policy, not a constraint.
DEFAULT_GRADED_AXES = ("ranking_resolvable",)


# ---------------------------------------------------------------------------
# BH-FDR: the family is ours, the arithmetic is scipy's
# ---------------------------------------------------------------------------

def bh_qvalues(pvalues: list) -> list:
    """Benjamini–Hochberg q-values for **one** family of p-values.

    The step-up arithmetic is delegated to ``scipy.stats.false_discovery_control``
    (BH 1995) — the statistic is not ours to reimplement, and it is available across
    our whole supported SciPy range. What *is* ours is the family: only finite
    entries are members. ``None`` / NaN (an arm too small to test) passes through as
    ``None`` and is **excluded from m**, so a degenerate arm cannot dilute the
    correction for the arms that were actually tested.

    Returns a list positionally aligned with ``pvalues``.
    """
    idx = [i for i, p in enumerate(pvalues)
           if isinstance(p, (int, float)) and not isinstance(p, bool)
           and math.isfinite(float(p))]
    out: list = [None] * len(pvalues)
    if not idx:
        return out
    from scipy.stats import false_discovery_control
    q = false_discovery_control([float(pvalues[i]) for i in idx], method="bh")
    for i, qi in zip(idx, list(q)):
        out[i] = float(qi)
    return out


def apply_stratified_qvalues(records: list) -> dict:
    """Assign ``q`` to each record with **one BH family per stratum**.

    ``records`` are the per-arm records from :func:`assemble` (each carrying
    ``stratum`` and ``p``). The reference arm of a stratum is never a member of its
    own family — it is the comparator, not a test. Mutates ``records`` in place
    (setting ``q`` and ``in_family``) and returns ``{stratum: m}``, the family size
    actually corrected over, so a reader can audit the family they got rather than
    trusting that one was declared.
    """
    families: dict = {}
    for rec in records:
        rec["q"] = None
        rec["in_family"] = False
        if rec.get("is_reference"):
            continue
        families.setdefault(rec["stratum"], []).append(rec)
    sizes: dict = {}
    for stratum, members in families.items():
        qs = bh_qvalues([m.get("p") for m in members])
        m_eff = 0
        for member, q in zip(members, qs):
            member["q"] = q
            member["in_family"] = q is not None
            m_eff += 1 if q is not None else 0
        sizes[stratum] = m_eff
    return sizes


# ---------------------------------------------------------------------------
# Cell-level stats
# ---------------------------------------------------------------------------

def _cell_values(node) -> list:
    """Per-cell values out of a ``{"by_cell": [[seed, gen, value], ...]}`` block.

    Cell-level n, never per-timepoint n: ``by_cell`` is already one row per cell
    (the shape ``workflow/analysis.py`` emits), so no re-aggregation happens here.
    """
    if not isinstance(node, dict):
        return []
    rows = node.get("by_cell") or []
    out = []
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 3 and row[2] is not None:
            out.append(float(row[2]))
    return out


def _stats(values: list) -> dict:
    """mean / sample sd / sem / n. ``sd`` and ``sem`` are ``None`` at n < 2 — with
    one cell there is no spread to report, and a fabricated 0 would read as
    infinite precision."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "sem": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "sem": None}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return {"n": n, "mean": mean, "sd": sd, "sem": sd / math.sqrt(n)}


def _median(values: list) -> "float | None":
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return None
    mid = n // 2
    return vs[mid] if n % 2 else 0.5 * (vs[mid - 1] + vs[mid])


def stratum_key(strata_values: dict, strata: list) -> str:
    """Stable id for one stratum, e.g. ``media=glucose`` or
    ``media=glucose|oxygen=aerobic``.

    Always contains ``=``, so a stratum id can never collide with the card's
    ``strata_declared`` node, and never contains ``.``, so it stays a single
    diggable segment of a dotted axis path.
    """
    parts = []
    for key in strata:
        val = str(strata_values.get(key, "")).strip().replace(".", "_")
        parts.append(f"{key}={val}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def load_panel(path) -> dict:
    """Read a committed panel fixture (the baked per-arm per-cell values)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assemble(panel: dict, *, objective_observable: str, growth_observable: str,
             reference_arm: str, strata: list, higher_is_better: bool) -> list:
    """Per-arm records: mean / sem / n, the contrast against the stratum's reference
    arm (Welch p, effect size), and the BH q-value from that stratum's family.

    ``panel`` is ``{"arms": [{arm, design, strata: {...},
    observables: {<id>: {"by_cell": [[seed, gen, value], ...]}}}, ...]}``.

    Raises ``ValueError`` on a malformed panel or a panel that does not satisfy the
    contract: a duplicate ``arm`` id, an arm missing a declared stratum key or the
    objective observable, or a stratum without exactly one arm carrying
    ``reference_arm`` as its design. The last one is deliberately fatal rather than
    ungraded — "compared against nothing" must not be reportable as a result.
    """
    arms = panel.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("panel_screen: panel has no `arms` list")

    records: list = []
    seen: dict = {}
    for i, arm in enumerate(arms):
        if not isinstance(arm, dict):
            raise ValueError(f"panel_screen: arms[{i}] is not a mapping")
        name = arm.get("arm")
        design = arm.get("design")
        if not name or not design:
            raise ValueError(f"panel_screen: arms[{i}] needs both `arm` and `design`")
        if name in seen:
            raise ValueError(
                f"panel_screen: duplicate arm id {name!r} — an arm id must be unique "
                "and carry its stratum, or the same design in two conditions "
                "silently shares one row")
        seen[name] = i
        svals = arm.get("strata") or {}
        missing = [k for k in strata if k not in svals]
        if missing:
            raise ValueError(
                f"panel_screen: arm {name!r} is missing declared stratum key(s) "
                f"{missing} — it cannot be placed in a testing family")
        obs = arm.get("observables") or {}
        if objective_observable not in obs:
            raise ValueError(
                f"panel_screen: arm {name!r} has no observable "
                f"{objective_observable!r} (the objective)")
        ovals = _cell_values(obs[objective_observable])
        gvals = _cell_values(obs.get(growth_observable))
        rec = {"arm": name, "design": design,
               "stratum": stratum_key(svals, strata), "strata": dict(svals),
               "is_reference": design == reference_arm,
               "values": ovals, **_stats(ovals)}
        rec["growth"] = _stats(gvals)
        records.append(rec)

    # Resolve the reference arm WITHIN each stratum (never one global control): a
    # design is only meaningfully contrasted against the control it shares
    # conditions with.
    by_stratum: dict = {}
    for rec in records:
        by_stratum.setdefault(rec["stratum"], []).append(rec)
    refs: dict = {}
    for stratum, members in by_stratum.items():
        hits = [m for m in members if m["is_reference"]]
        if len(hits) != 1:
            raise ValueError(
                f"panel_screen: stratum {stratum!r} has {len(hits)} arms with design "
                f"{reference_arm!r}; exactly one reference arm per stratum is required")
        refs[stratum] = hits[0]

    for rec in records:
        ref = refs[rec["stratum"]]
        rec["ref_arm"] = ref["arm"]
        rec["ref_mean"] = ref["mean"]
        rec["ref_n"] = ref["n"]
        rec["ratio"] = rec["improvement"] = rec["p"] = rec["cohens_d"] = None
        rec["growth_ratio"] = None
        if rec["mean"] is not None and ref["mean"]:
            rec["ratio"] = rec["mean"] / ref["mean"]
            # `improvement` is direction-corrected so higher is ALWAYS better; a
            # minimised objective (a by-product, say) would otherwise be graded
            # upside down by a band that reads high as good.
            if higher_is_better:
                rec["improvement"] = rec["ratio"]
            elif rec["ratio"]:
                rec["improvement"] = 1.0 / rec["ratio"]
        gm, rgm = rec["growth"]["mean"], ref["growth"]["mean"]
        if gm is not None and rgm:
            rec["growth_ratio"] = gm / rgm
        if not rec["is_reference"] and rec["n"] > 1 and ref["n"] > 1:
            w = _welch(rec["values"], ref["values"])
            p = w.get("p")
            rec["p"] = float(p) if p is not None and math.isfinite(float(p)) else None
            rec["cohens_d"] = w.get("cohens_d")

    apply_stratified_qvalues(records)
    return records


# ---------------------------------------------------------------------------
# Axis nodes (verdicts are computed here and carried as `type: status`)
# ---------------------------------------------------------------------------
# There is no "ratio vs a floor/band" criterion in `viva_superpowers.card_criteria`
# and the grader lives upstream, so the banding happens here with the shared `_band`
# and each node carries its own verdict. Precedent: `library/genotype_build.py` and
# `workflow/report_cards/tests_card.py`.

def _node(verdict: str, value, meter: str, detail: dict) -> dict:
    return {"verdict": verdict, "value": value, "meter": meter, "detail": detail}


def _band_str(name: str, good: float, warn: float) -> str:
    if name == "growth_cost":
        return f"growth ≥ {good:.3g}× reference (drift ≥ {warn:.3g}×)"
    if name == "ranking_resolvable":
        return f"between-arm SD ÷ median within-arm SEM ≥ {good:.3g} (drift ≥ {warn:.3g})"
    return f"best arm ≥ {good:.3g}× reference (drift ≥ {warn:.3g}×)"


def _objective_node(members: list, ref: str, bands: dict, family_size: int,
                    higher_is_better: bool) -> tuple:
    """(node, best_record). Grades the BEST arm's direction-corrected improvement
    factor over the stratum's reference arm."""
    good, warn = bands["objective_vs_reference"]
    gradeable = [m for m in members
                 if not m["is_reference"] and m["improvement"] is not None]
    tested = [m for m in members if not m["is_reference"]]
    excluded = [m["arm"] for m in tested if m["q"] is None]
    if not gradeable:
        return _node("ungraded", None,
                     f"no arm contrastable against {ref} (n<1 or reference mean 0)",
                     {"family_size": family_size, "excluded_arms": excluded}), None
    # Ties broken by arm id so the card is byte-deterministic across re-renders.
    best = sorted(gradeable, key=lambda m: (-m["improvement"], m["arm"]))[0]
    # The band verdict is computed here; whether it SURVIVES is the caller's
    # policy, applied centrally in `build` via `graded_axes`. By default this
    # axis is reported, not graded — see DEFAULT_GRADED_AXES.
    verdict = _band(best["improvement"], good, warn, higher_is_better=True)
    qtxt = f"q = {best['q']:.3g}" if best["q"] is not None else "q = n/a"
    raw = "" if higher_is_better else f" (raw {best['ratio']:.3g}×)"
    meter = (f"best {best['arm']}: {best['improvement']:.3g}× vs {ref}{raw} · "
             f"{qtxt} · m = {family_size}")
    detail = {
        "best_arm": best["arm"], "best_design": best["design"],
        "improvement": best["improvement"], "ratio": best["ratio"],
        "mean": best["mean"], "sem": best["sem"], "n": best["n"],
        "reference_arm": ref, "reference_mean": best["ref_mean"],
        "reference_n": best["ref_n"], "p": best["p"], "q": best["q"],
        "higher_is_better": higher_is_better,
        "design_target": {"good": good, "warn": warn},
        # `family_size` is the m BH actually corrected over, and `excluded_arms`
        # says who was too small to test: without both, a pass here is unauditable.
        "family_size": family_size, "excluded_arms": excluded,
        "arms": [{"arm": m["arm"], "design": m["design"], "mean": m["mean"],
                  "sem": m["sem"], "n": m["n"], "ratio": m["ratio"],
                  "improvement": m["improvement"], "p": m["p"], "q": m["q"],
                  "in_family": m["in_family"], "is_reference": m["is_reference"]}
                 for m in sorted(members, key=lambda m: (not m["is_reference"],
                                                         m["arm"]))],
    }
    return _node(verdict, best["improvement"], meter, detail), best


def _growth_node(best, ref: str, bands: dict) -> dict:
    """A design that "wins" by killing the cell is not a win: the best arm's growth
    as a fraction of the reference's, against a floor."""
    good, warn = bands["growth_cost"]
    if best is None or best["growth_ratio"] is None:
        return _node("ungraded", None,
                     "growth observable absent or reference growth 0",
                     {"reference_arm": ref})
    value = best["growth_ratio"]
    # Band verdict computed here; `graded_axes` decides whether it survives.
    # By default it does not: a growth floor is a biology/design decision, not
    # execution correctness.
    return _node(_band(value, good, warn, higher_is_better=True), value,
                 f"{best['arm']}: {value:.3g}× {ref} growth "
                 f"({best['growth']['mean']:.3g} vs {best['growth']['mean'] / value:.3g})",
                 {"best_arm": best["arm"], "growth_ratio": value,
                  "growth_mean": best["growth"]["mean"],
                  "growth_sem": best["growth"]["sem"],
                  "growth_n": best["growth"]["n"], "reference_arm": ref,
                  "design_target": {"good": good, "warn": warn}})


def _ranking_node(members: list, ref: str, bands: dict) -> dict:
    """★ The load-bearing axis. Separation between the arms vs within-arm noise. If
    this fails the ranking is noise and the other two axes say nothing, so it forces
    them to ``ungraded`` (see :func:`build`).

    The reference arm is **excluded** from the between-arm SD: a large
    control-vs-design gap would inflate "resolvability" without saying anything about
    resolving the designs from each other.
    """
    good, warn = bands["ranking_resolvable"]
    designs = [m for m in members if not m["is_reference"]]
    means = [m["mean"] for m in designs if m["mean"] is not None]
    sems = [m["sem"] for m in designs if m["sem"] is not None]
    # `reference_arm` is the RESOLVED arm id for this stratum. Every p, q and ratio
    # on this card is a contrast against that one declared control, so which arm it
    # resolved to must be readable off the card — a stratum whose reference resolved
    # to something unexpected has to be visible, not baked silently into the stats.
    detail = {"n_design_arms": len(designs), "reference_arm": ref,
              "reference_excluded": True, "arms": [m["arm"] for m in designs]}
    if len(means) < 2:
        return _node("ungraded", None,
                     f"{len(means)} design arm(s) with a mean — nothing to rank",
                     detail)
    mu = sum(means) / len(means)
    sd = math.sqrt(sum((m - mu) ** 2 for m in means) / (len(means) - 1))
    med_sem = _median(sems)
    detail.update({"sd_between": sd, "median_sem": med_sem, "n_ranked": len(means)})
    if not med_sem:
        return _node("ungraded", None,
                     "median within-arm SEM is 0 or unavailable (n<2 per arm)", detail)
    value = sd / med_sem
    detail["ratio"] = value
    return _node(_band(value, good, warn, higher_is_better=True), value,
                 f"SD {sd:.3g} ÷ median SEM {med_sem:.3g} = {value:.2f} "
                 f"({len(means)} design arms)", detail)


# ---------------------------------------------------------------------------
# Card + reference assembly
# ---------------------------------------------------------------------------

def _graded_axes(raw) -> frozenset:
    """Validate `graded_axes`. ``None`` -> the default policy; anything else must
    name only known axes, so a typo cannot silently ungrade the whole card."""
    if raw is None:
        return frozenset(DEFAULT_GRADED_AXES)
    if isinstance(raw, str) or not hasattr(raw, "__iter__"):
        raise ValueError("panel_screen: `graded_axes` must be a list of axis names")
    names = tuple(raw)
    unknown = [n for n in names if n not in AXES]
    if unknown:
        raise ValueError(
            f"panel_screen: graded_axes names unknown axis/axes {unknown!r} — "
            f"valid names are {list(AXES)}")
    return frozenset(names)


def _bands(raw) -> dict:
    """Validate the three REQUIRED bands. No defaults: a default band is an
    unexamined number baked into every future study."""
    if not isinstance(raw, dict):
        raise ValueError("panel_screen: `bands` must be a mapping with "
                         f"{list(AXES)}")
    out = {}
    for axis in AXES:
        spec = raw.get(axis)
        if not isinstance(spec, dict) or "good" not in spec or "warn" not in spec:
            raise ValueError(
                f"panel_screen: bands.{axis} must declare both `good` and `warn` "
                "— bands are required, never defaulted")
        out[axis] = (float(spec["good"]), float(spec["warn"]))
    return out


_AXIS_LABEL = {
    "objective_vs_reference": "Objective vs reference arm",
    "growth_cost": "Growth cost of the best arm",
    "ranking_resolvable": "Ranking resolvable above noise",
}

_AXIS_HOW = {
    "objective_vs_reference": (
        "Best design arm's objective observable as a direction-corrected factor of "
        "the reference arm's, both as cell-level means within this stratum. Welch "
        "p vs the reference arm, BH-FDR q over the arms of THIS stratum only "
        "(m reported). REPORTED against the design target the study declared, "
        "never graded: which design wins is a scientific finding, and a card "
        "verdict computed from it would fail a screen that ran perfectly and "
        "found nothing good."),
    "growth_cost": (
        "The best arm's growth observable as a fraction of the reference arm's, "
        "cell-level means within this stratum, REPORTED against the design target "
        "the study declared. Never graded: a growth floor is a biology/design "
        "decision, not execution correctness. Reported for the arm the objective "
        "axis selected."),
    "ranking_resolvable": (
        "Between-arm SD of the design arms' objective means divided by the median "
        "within-arm SEM, within this stratum; the reference arm is excluded. "
        "Graded against a floor supplied by the study — the ONLY graded axis on "
        "this card, because resolvability is a property of the screen's execution "
        "and replicate count, not of the biology. When it is a mismatch the other "
        "two axes are annotated unreadable: the ranking is noise, so a win over "
        "the reference cannot be read off it."),
}


def _strata_missing(reason: str) -> tuple:
    """The card for a panel graded with no ``strata`` declared.

    A visible failing axis, NOT a raise and NOT a silent global family: the runner
    swallows exceptions out of a card's ``build`` (``TestStep.invoke``), so a raise
    would make this project's headline contract the softest failure in the system.
    """
    card = {"panel": {"strata_declared": _node(
        "mismatch", None,
        f"{reason} — a panel cannot be graded without declaring its testing family",
        {"strata": None, "reason": reason})}}
    reference = {
        "title": "Panel screen — ungradeable: no stratification declared",
        "status": "populated",
        "stimulus": {"reference_model": "declared reference arm (per stratum)",
                     "measured_model": "panel screen", "summary": reason},
        "findings": [
            "This panel declared no `strata`, so no testing family is defined. "
            "Pooling q-values across conditions makes nearly every arm in the "
            "second condition read as significant when the signal is the condition, "
            "not the design — so the card refuses to grade rather than guess.",
        ],
        "footer": "Behavioral report card · panel screen (stratification contract).",
        "axes": {"panel.strata_declared": {
            "group": GROUP, "label": "Stratification declared", "units": "",
            "how": ("`report_card_refs.panel_screen.strata` must name the keys that "
                    "define the testing family (e.g. the medium). Required: the "
                    "caller decides the family, so only a declaration makes it "
                    "auditable."),
            "criterion": {"type": "status",
                          "criterion_str": "`strata` keys declared (required)"}}},
    }
    return card, reference


def build(panel: dict, *, objective_observable: str, growth_observable: str,
          reference_arm: str, strata, higher_is_better, bands,
          graded_axes=None, title: "str | None" = None) -> tuple:
    """``(card, reference)`` — the gradeable inputs for one panel screen.

    Emits the three axes **once per stratum** (paths
    ``panel.<stratum>.<axis>``): "the best arm overall" across strata would pool
    exactly what the strata contract forbids, one level up from the q-values.

    Raises ``ValueError`` on malformed inputs (missing observable ids, a missing or
    incomplete band, an unresolvable reference arm, a duplicate arm id). A *missing*
    ``strata`` declaration is different — it is the contract itself, and produces a
    failing ``strata_declared`` axis so a study visibly fails on it.
    """
    if not strata:
        return _strata_missing("no `strata` declared")
    if isinstance(strata, str) or not all(isinstance(k, str) for k in strata):
        raise ValueError("panel_screen: `strata` must be a list of key names")
    if not objective_observable or not growth_observable or not reference_arm:
        raise ValueError("panel_screen: `objective_observable`, `growth_observable` "
                         "and `reference_arm` are all required")
    if not isinstance(higher_is_better, bool):
        raise ValueError("panel_screen: `higher_is_better` is required (a screen "
                         "objective is not always maximised, and a silently "
                         "wrong-direction grade is the worst outcome)")
    band = _bands(bands)
    graded = _graded_axes(graded_axes)

    records = assemble(panel, objective_observable=objective_observable,
                       growth_observable=growth_observable,
                       reference_arm=reference_arm, strata=list(strata),
                       higher_is_better=higher_is_better)
    families = apply_stratified_qvalues(records)

    by_stratum: dict = {}
    for rec in records:
        by_stratum.setdefault(rec["stratum"], []).append(rec)

    # The reference arm each stratum RESOLVED to, reported alongside the family sizes
    # so both halves of "what was this compared against, over what family" are on the
    # card. `reference_arm` in the refs names a *design*; these are the concrete arms.
    resolved = {s: by_stratum[s][0]["ref_arm"] for s in sorted(by_stratum)}
    cnode: dict = {"strata_declared": _node(
        # Deliberately `ungraded`, never `within_tol`: declaring the family is a
        # precondition, and it must not be able to carry a passing verdict on its
        # own while every substantive axis is ungraded.
        "ungraded", len(list(strata)),
        f"strata {list(strata)} · {len(by_stratum)} stratum(a) · "
        f"{len(records)} arms · reference design {reference_arm!r} → "
        f"{', '.join(f'{s}: {a}' for s, a in resolved.items())}",
        {"strata": list(strata), "n_strata": len(by_stratum),
         "n_arms": len(records), "reference_design": reference_arm,
         "reference_arms": resolved, "family_sizes": families})}
    axes: dict = {"panel.strata_declared": {
        "group": GROUP, "label": "Stratification declared", "units": "",
        "how": ("The keys defining the testing family. Required, and reported so a "
                "reader can see which family each q-value was corrected over "
                "(one BH family per stratum, never one across the panel). "
                "Informational: this axis never grades."),
        "criterion": {"type": "status",
                      "criterion_str": "`strata` keys declared (required)"}}}

    for stratum in sorted(by_stratum):
        members = by_stratum[stratum]
        ref = members[0]["ref_arm"]
        obj, best = _objective_node(members, ref, band,
                                    families.get(stratum, 0), higher_is_better)
        nodes = {"objective_vs_reference": obj,
                 "growth_cost": _growth_node(best, ref, band),
                 "ranking_resolvable": _ranking_node(members, ref, band)}
        if nodes["ranking_resolvable"]["verdict"] == "mismatch":
            # The reported axes are ungraded by construction, so this no longer
            # changes a verdict — but the ANNOTATION is still load-bearing: an
            # unresolvable ranking makes those numbers unreadable, and a reader
            # (or a consumer withholding the ranking) has to be told so.
            for name in ("objective_vs_reference", "growth_cost"):
                node = nodes[name]
                node["meter"] = f"ranking unresolvable — {node['meter']}"
                node["detail"]["unreadable_because"] = "ranking_resolvable"
        # ★ THE POLICY, applied in ONE place. An axis the caller did not name as
        # graded keeps its number, its meter and its declared design target, and
        # gives up only the ruling. Recorded per node so a reader of the verdict
        # JSON can tell "not graded by policy" from "could not be computed".
        for name in AXES:
            node = nodes[name]
            node["detail"]["graded"] = name in graded
            if name not in graded and node["verdict"] != "ungraded":
                node["detail"]["reported_verdict"] = node["verdict"]
                node["verdict"] = "ungraded"
        cnode[stratum] = nodes
        for name in AXES:
            good, warn = band[name]
            axes[f"panel.{stratum}.{name}"] = {
                "group": GROUP, "label": f"{_AXIS_LABEL[name]} — {stratum}",
                "units": "", "how": _AXIS_HOW[name],
                "criterion": {"type": "status",
                              "criterion_str": _band_str(name, good, warn)}}

    reference = {
        "title": title or "Panel screen — designs ranked against a reference arm",
        "status": "populated",
        "stimulus": {
            "reference_model": f"reference arm {reference_arm!r} (resolved per stratum)",
            "measured_model": f"panel of {len(records)} arms · objective "
                              f"{objective_observable}",
            "summary": f"{len(by_stratum)} stratum(a) over {list(strata)} · "
                       f"BH-FDR family per stratum "
                       f"(m = {[families.get(s, 0) for s in sorted(by_stratum)]})",
        },
        "findings": [
            "Each stratum is graded on its own: the reference arm is resolved inside "
            "the stratum and the BH-FDR family is the arms of that stratum only. "
            "Pooling either one across conditions attributes the condition's effect "
            "to the design.",
            "`ranking_resolvable` gates the stratum: if between-arm separation does "
            "not exceed within-arm noise, the ranking is noise and the other two "
            "axes are reported but ungraded.",
        ],
        "footer": "Behavioral report card · panel screen (per-stratum ranking).",
        "axes": axes,
    }
    return {"panel": cnode}, reference
