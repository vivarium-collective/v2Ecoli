"""Tests for the panel-screen report card.

Covers both layers, mirroring ``tests/test_overflow_card.py``: the science core
(``v2ecoli.library.panel_screen``) and the Gen-2 Step (``PanelScreenCard``). Grading
is fixture-driven — no live run, no ParCa cache — from the committed synthetic
fixture ``tests/fixtures/panel_screen/panel_baseline.json``.

★ The load-bearing tests here are the family-construction ones. The defect this card
exists to prevent is a q-value family pooled across conditions, and a test that passed
under both a per-stratum and a pooled family would certify exactly that defect. So
``test_qvalue_family_is_per_stratum_not_pooled`` and
``test_pooling_would_change_a_recorded_qvalue`` are written to **straddle a decision
boundary**: they fail if the family is pooled.
"""
from pathlib import Path

import pytest

from v2ecoli.library import panel_screen as ps
from v2ecoli.library.report_card import grade_card, render_html, verdict_json
from v2ecoli.workflow.post_sim import REPORT_CARD_REGISTRY
from v2ecoli.workflow.report_cards import StudyContext
from v2ecoli.workflow.report_cards.panel_screen_card import PanelScreenCard

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/fixtures/panel_screen/panel_baseline.json"

# Illustrative bands, matching the fixture's docstring. NOT canonical — bands are a
# study input and this card never defaults them.
BANDS = {"objective_vs_reference": {"good": 1.20, "warn": 1.05},
         "growth_cost": {"good": 0.85, "warn": 0.70},
         "ranking_resolvable": {"good": 3.0, "warn": 2.0}}

REFS = {"objective_observable": "objective_titer", "growth_observable": "growth_rate",
        "reference_arm": "wt", "strata": ["media"], "higher_is_better": True,
        "bands": BANDS}


def _build(panel, **over):
    kw = {k: v for k, v in REFS.items()}
    kw.update(over)
    return ps.build(panel, **kw)


def _cells(mean, spread, n=8):
    """n cells around ``mean`` with a fixed zero-sum offset pattern (so the mean is
    exact and the sd is deterministic)."""
    offs = [(-1) ** i * (0.4 + 0.2 * i) for i in range(n)]
    shift = sum(offs) / n
    return [[i // 4, i % 4, mean + spread * (o - shift)] for i, o in enumerate(offs)]


def _arm(design, media, obj_mean, obj_spread=0.05, gro_mean=0.7, gro_spread=0.02,
         n=8):
    return {"arm": f"{design}|{media}", "design": design, "strata": {"media": media},
            "observables": {
                "objective_titer": {"by_cell": _cells(obj_mean, obj_spread, n)},
                "growth_rate": {"by_cell": _cells(gro_mean, gro_spread, n)}}}


def _panel(*arms):
    return {"arms": list(arms)}


@pytest.fixture(scope="module")
def fixture_panel():
    return ps.load_panel(FIXTURE)


# --- BH-FDR: a check on OUR wiring, not on scipy ------------------------------

def test_bh_matches_hand_worked_example():
    """The canonical Benjamini-Hochberg 1995 worked example (m=15).

    Expected values are the hand-computed step-up ``p·m/rank`` with the monotone
    (cumulative-minimum-from-the-top) enforcement applied — written out as literals
    rather than recomputed, so this checks our call and our scatter-back rather than
    re-running whatever algorithm the implementation happens to use. Ranks 6 and 8
    are the positions where monotone enforcement bites (raw 0.0695 -> 0.06386 and
    0.0645 -> 0.0645 capped by rank 7), which is where a wrong implementation shows.
    """
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
         0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    expected = [0.0015, 0.0030, 0.0095, 0.035625, 0.0603, 0.0638571, 0.0638571,
                0.0645, 0.0765, 0.4860, 0.5811818, 0.7148750, 0.7532308, 0.8132143,
                1.0]
    got = ps.bh_qvalues(p)
    assert got == pytest.approx(expected, rel=1e-5)
    # Monotone non-decreasing in p, and never below the raw p.
    assert got == sorted(got)
    assert all(q >= pi - 1e-12 for q, pi in zip(got, p))


def test_bh_is_order_independent():
    p = [0.04, 0.0001, 0.5, 0.02]
    q = ps.bh_qvalues(p)
    order = [2, 0, 3, 1]
    shuffled = ps.bh_qvalues([p[i] for i in order])
    assert shuffled == pytest.approx([q[i] for i in order])


def test_bh_excludes_none_and_nan_from_m():
    """A degenerate arm must not dilute the correction: with one testable p the
    q equals the p (m=1), not p·2."""
    assert ps.bh_qvalues([0.01, None]) == pytest.approx([0.01, None])
    assert ps.bh_qvalues([0.01, float("nan")])[0] == pytest.approx(0.01)
    assert ps.bh_qvalues([0.01, float("nan")])[1] is None
    assert ps.bh_qvalues([None, None]) == [None, None]


# --- ★ the family is per stratum, and that CHANGES the answer -----------------

def _records(*specs):
    """(arm, stratum, p, is_reference) -> the record shape apply_stratified_qvalues
    consumes."""
    return [{"arm": a, "stratum": s, "p": p, "is_reference": r}
            for a, s, p, r in specs]


def test_qvalue_family_is_per_stratum_not_pooled():
    """★ The discriminating test: pooling would flip a verdict.

    Stratum A carries six strongly significant arms; stratum B carries one moderate
    arm (p = 0.02) among five null ones. BH is a step-up, so pooling A's tiny
    p-values in with B's lifts B's single moderate arm to rank 7 of 12 and drags its
    q under 0.05 — the exact failure this card exists to prevent (signal from the
    *condition* read as significance for the *design*).

    The assertion straddles 0.05 in both directions, so it cannot pass under a pooled
    family: it requires the per-stratum q to be non-significant AND the pooled q for
    the same arm to be significant.
    """
    specs = [(f"a{i}|A", "A", 1e-8, False) for i in range(6)]
    specs += [("b0|B", "B", 0.02, False)]
    specs += [(f"b{i}|B", "B", 0.5 + 0.1 * i, False) for i in range(1, 6)]
    specs += [("wt|A", "A", None, True), ("wt|B", "B", None, True)]
    recs = _records(*specs)
    sizes = ps.apply_stratified_qvalues(recs)
    assert sizes == {"A": 6, "B": 6}

    b0 = next(r for r in recs if r["arm"] == "b0|B")
    pooled = ps.bh_qvalues([r["p"] for r in recs])
    pooled_b0 = pooled[[r["arm"] for r in recs].index("b0|B")]

    # Per stratum: m=6, rank 1 -> 0.02*6 = 0.12, not significant.
    assert b0["q"] == pytest.approx(0.12, rel=1e-6)
    # Pooled: m=12, rank 7 -> 0.02*12/7 = 0.0343, significant. Different verdict.
    assert pooled_b0 < 0.05 < b0["q"], (
        "the per-stratum family must not reproduce the pooled family: pooling "
        f"gives q={pooled_b0:.4f} (significant) where the correct per-stratum "
        f"family gives q={b0['q']:.4f}")


def test_reference_arm_is_not_a_member_of_its_own_family():
    recs = _records(("wt|A", "A", 0.001, True), ("d1|A", "A", 0.01, False),
                    ("d2|A", "A", 0.02, False))
    sizes = ps.apply_stratified_qvalues(recs)
    assert sizes == {"A": 2}                     # the comparator is not a test
    ref = next(r for r in recs if r["is_reference"])
    assert ref["q"] is None and ref["in_family"] is False
    # m=2, so q = p*2/rank: 0.02 and 0.02.
    assert [r["q"] for r in recs if not r["is_reference"]] == pytest.approx([0.02, 0.02])


def test_pooling_would_change_a_recorded_qvalue():
    """Through ``assemble`` (real Welch p-values, not injected ones): every recorded
    q must reproduce a per-stratum BH and NOT a pooled one."""
    arms = [_arm("wt", "glucose", 1.00, 0.02), _arm("design_a", "glucose", 1.60, 0.02),
            _arm("design_b", "glucose", 1.50, 0.02), _arm("design_c", "glucose", 1.40, 0.02),
            _arm("wt", "acetate", 1.00, 0.30), _arm("design_a", "acetate", 1.20, 0.30),
            _arm("design_b", "acetate", 1.15, 0.30), _arm("design_c", "acetate", 1.10, 0.30)]
    recs = ps.assemble(_panel(*arms), objective_observable="objective_titer",
                       growth_observable="growth_rate", reference_arm="wt",
                       strata=["media"], higher_is_better=True)
    designs = [r for r in recs if not r["is_reference"]]

    # Recomputing per stratum reproduces every recorded q exactly.
    for stratum in {r["stratum"] for r in designs}:
        members = [r for r in designs if r["stratum"] == stratum]
        assert ps.bh_qvalues([m["p"] for m in members]) == pytest.approx(
            [m["q"] for m in members])

    # Recomputing pooled does NOT — at least one arm differs materially. Without this
    # the test above would also pass on a pooled implementation.
    pooled = ps.bh_qvalues([r["p"] for r in designs])
    assert any(abs(pq - r["q"]) > 1e-9 for pq, r in zip(pooled, designs)), (
        "pooled and per-stratum q-values are indistinguishable in this panel — the "
        "test would not detect a pooled family")


def test_degenerate_arm_leaves_the_family_and_shrinks_m():
    """An untestable arm (n<2) gets p=q=None and is excluded from m, so the arms that
    WERE tested are corrected over the smaller family."""
    arms = [_arm("wt", "glucose", 1.00, 0.02),
            _arm("design_a", "glucose", 1.30, 0.02),
            _arm("design_b", "glucose", 1.20, 0.02),
            _arm("design_c", "glucose", 1.10, 0.02, n=1)]  # single cell: untestable
    recs = ps.assemble(_panel(*arms), objective_observable="objective_titer",
                       growth_observable="growth_rate", reference_arm="wt",
                       strata=["media"], higher_is_better=True)
    small = next(r for r in recs if r["arm"] == "design_c|glucose")
    assert small["n"] == 1 and small["p"] is None and small["q"] is None
    assert small["in_family"] is False
    tested = [r for r in recs if r["in_family"]]
    assert len(tested) == 2
    # m=2, not 3: a q corrected over the diluted family would be 1.5x larger.
    assert ps.bh_qvalues([t["p"] for t in tested]) == pytest.approx(
        [t["q"] for t in tested])

    card, ref = _build(_panel(*arms))
    detail = card["panel"]["media=glucose"]["objective_vs_reference"]["detail"]
    assert detail["family_size"] == 2
    assert detail["excluded_arms"] == ["design_c|glucose"]


# --- the reference arm is resolved per stratum --------------------------------

def test_reference_arm_is_resolved_within_each_stratum():
    """A global reference would give a different ratio. The acetate reference sits at
    2.0 while glucose's sits at 1.0, so a design at 2.4 in acetate must read 1.2x
    (its own control) and not 2.4x (the other stratum's)."""
    arms = [_arm("wt", "glucose", 1.00, 0.02), _arm("design_a", "glucose", 1.30, 0.02),
            _arm("design_b", "glucose", 1.10, 0.02),
            _arm("wt", "acetate", 2.00, 0.04), _arm("design_a", "acetate", 2.40, 0.04),
            _arm("design_b", "acetate", 2.10, 0.04)]
    recs = ps.assemble(_panel(*arms), objective_observable="objective_titer",
                       growth_observable="growth_rate", reference_arm="wt",
                       strata=["media"], higher_is_better=True)
    a_ac = next(r for r in recs if r["arm"] == "design_a|acetate")
    assert a_ac["ref_arm"] == "wt|acetate"
    assert a_ac["ratio"] == pytest.approx(1.2, rel=1e-9)   # not 2.4
    a_gl = next(r for r in recs if r["arm"] == "design_a|glucose")
    assert a_gl["ref_arm"] == "wt|glucose"
    assert a_gl["ratio"] == pytest.approx(1.3, rel=1e-9)


def test_every_axis_detail_names_the_resolved_reference_arm():
    """Every p, q and ratio is a contrast against one declared control, so which arm
    it resolved to must be readable off the card without opening the fixture."""
    card, _ = _build(_panel(
        _arm("wt", "glucose", 1.00, 0.02), _arm("design_a", "glucose", 1.30, 0.02),
        _arm("design_b", "glucose", 1.20, 0.02)))
    stratum = card["panel"]["media=glucose"]
    for axis in ps.AXES:
        assert stratum[axis]["detail"]["reference_arm"] == "wt|glucose", axis
    declared = card["panel"]["strata_declared"]["detail"]
    assert declared["reference_design"] == "wt"
    assert declared["reference_arms"] == {"media=glucose": "wt|glucose"}


def test_stratum_without_the_reference_arm_is_fatal():
    with pytest.raises(ValueError, match="reference arm per stratum"):
        _build(_panel(_arm("wt", "glucose", 1.0), _arm("design_a", "glucose", 1.2),
                      _arm("design_a", "acetate", 1.2)))


def test_two_reference_arms_in_one_stratum_is_fatal():
    dupe = _arm("wt", "glucose", 1.1)
    dupe["arm"] = "wt_replicate|glucose"
    with pytest.raises(ValueError, match="reference arm per stratum"):
        _build(_panel(_arm("wt", "glucose", 1.0), dupe,
                      _arm("design_a", "glucose", 1.2)))


# --- the strata contract ------------------------------------------------------

def test_missing_strata_grades_as_a_visible_failing_axis():
    """★ No `strata` -> a mismatch axis, NOT a silent global family and NOT a raise.
    The runner swallows exceptions out of build(), so a raise would make the card's
    headline contract the softest failure in the system."""
    panel = _panel(_arm("wt", "glucose", 1.0), _arm("design_a", "glucose", 1.4))
    for missing in (None, [], {}):
        card, reference = _build(panel, strata=missing)
        report = grade_card(card, reference)
        assert report["overall"] == "mismatch", missing
        assert list(report["axes"]) == ["panel.strata_declared"]
        assert report["axes"]["panel.strata_declared"]["verdict"] == "mismatch"
        # No stratum axes exist, so nothing was graded against an implicit family.
        assert not any(k.startswith("panel.media") for k in report["axes"])


def test_strata_declared_axis_can_never_carry_a_pass():
    """Declaring the family is a precondition, not an achievement: the axis is
    `ungraded` when satisfied, so it cannot manufacture a passing card while every
    substantive axis is ungraded."""
    card, reference = _build(_panel(
        _arm("wt", "glucose", 1.0), _arm("design_a", "glucose", 1.4)))
    node = card["panel"]["strata_declared"]
    assert node["verdict"] == "ungraded"
    assert node["detail"]["strata"] == ["media"]
    assert node["detail"]["family_sizes"] == {"media=glucose": 1}


def test_duplicate_arm_id_is_fatal():
    """The note's sibling defect: the same design in two conditions must not share
    one row."""
    a = _arm("design_a", "glucose", 1.2)
    b = _arm("design_a", "acetate", 1.4)
    b["arm"] = a["arm"]                      # a label derived from the design alone
    with pytest.raises(ValueError, match="duplicate arm id"):
        _build(_panel(_arm("wt", "glucose", 1.0), a, b))


def test_arm_missing_a_declared_stratum_key_is_fatal():
    bad = _arm("design_a", "glucose", 1.2)
    bad["strata"] = {}
    with pytest.raises(ValueError, match="missing declared stratum key"):
        _build(_panel(_arm("wt", "glucose", 1.0), bad))


def test_missing_objective_observable_is_fatal():
    bad = _arm("design_a", "glucose", 1.2)
    del bad["observables"]["objective_titer"]
    with pytest.raises(ValueError, match="objective"):
        _build(_panel(_arm("wt", "glucose", 1.0), bad))


@pytest.mark.parametrize("axis", list(ps.AXES))
def test_every_band_is_required(axis):
    """Bands are never defaulted — a default is an unexamined number baked into every
    future study."""
    bands = {k: dict(v) for k, v in BANDS.items()}
    del bands[axis]
    with pytest.raises(ValueError, match=f"bands.{axis}"):
        _build(_panel(_arm("wt", "glucose", 1.0), _arm("design_a", "glucose", 1.4)),
               bands=bands)


def test_partial_band_is_required():
    bands = {k: dict(v) for k, v in BANDS.items()}
    del bands["growth_cost"]["warn"]
    with pytest.raises(ValueError, match="bands.growth_cost"):
        _build(_panel(_arm("wt", "glucose", 1.0), _arm("design_a", "glucose", 1.4)),
               bands=bands)


def test_higher_is_better_is_required():
    with pytest.raises(ValueError, match="higher_is_better"):
        _build(_panel(_arm("wt", "glucose", 1.0), _arm("design_a", "glucose", 1.4)),
               higher_is_better=None)


def test_minimised_objective_is_REPORTED_in_the_right_direction():
    """With ``higher_is_better: false`` the winner is the LOWEST arm, and its
    improvement factor is the inverse ratio — a reader comparing against the
    declared design target would otherwise read a by-product minimisation upside
    down. Direction correction applies to the VALUE; the axis carries no verdict."""
    panel = _panel(_arm("wt", "glucose", 1.00, 0.02),
                   _arm("design_a", "glucose", 0.50, 0.02),
                   _arm("design_b", "glucose", 1.50, 0.02))
    card, reference = _build(panel, higher_is_better=False)
    node = card["panel"]["media=glucose"]["objective_vs_reference"]
    assert node["detail"]["best_arm"] == "design_a|glucose"
    assert node["detail"]["ratio"] == pytest.approx(0.5, rel=1e-9)
    assert node["value"] == pytest.approx(2.0, rel=1e-9)      # 1/0.5
    # ⛔ Reported, never graded — the direction correction must still be right,
    # because the number is what the finding is read off.
    assert node["verdict"] == "ungraded"
    assert node["detail"]["graded"] is False
    assert node["detail"]["design_target"] == {"good": 1.20, "warn": 1.05}
    # The same panel read as a maximisation picks the other arm.
    card_max, _ = _build(panel, higher_is_better=True)
    assert (card_max["panel"]["media=glucose"]["objective_vs_reference"]
            ["detail"]["best_arm"] == "design_b|glucose")


# --- ranking_resolvable: the load-bearing axis --------------------------------

def test_reference_arm_is_excluded_from_the_between_arm_sd():
    """★ Discriminating: the designs are mutually unrankable, but the reference sits
    far away. Including it in the between-arm SD would inflate the ratio into a pass;
    excluding it is a mismatch, because nothing here resolves the designs from each
    other."""
    panel = _panel(_arm("wt", "glucose", 5.00, 0.20),
                   _arm("design_a", "glucose", 1.02, 0.20),
                   _arm("design_b", "glucose", 1.00, 0.20),
                   _arm("design_c", "glucose", 0.98, 0.20))
    card, _ = _build(panel)
    node = card["panel"]["media=glucose"]["ranking_resolvable"]
    assert node["verdict"] == "mismatch"
    assert node["detail"]["reference_excluded"] is True
    assert node["detail"]["n_ranked"] == 3
    # Sanity: with the reference included the SD would be ~10x larger, i.e. a pass.
    means = [1.02, 1.00, 0.98, 5.00]
    mu = sum(means) / 4
    sd_with_ref = (sum((m - mu) ** 2 for m in means) / 3) ** 0.5
    assert sd_with_ref / node["detail"]["median_sem"] > 3.0
    assert node["detail"]["sd_between"] < sd_with_ref / 10


def test_unrankable_panel_annotates_the_reported_axes_unreadable():
    """★ If the ranking is noise, a win over the reference cannot be read off it.
    The reported axes are ungraded by construction, so the load-bearing part is the
    ANNOTATION — a consumer withholding the ranking keys off it — and the fact that
    the card still fails overall on resolvability alone."""
    panel = _panel(_arm("wt", "glucose", 1.00, 0.40),
                   _arm("design_a", "glucose", 1.30, 0.40),
                   _arm("design_b", "glucose", 1.28, 0.40),
                   _arm("design_c", "glucose", 1.26, 0.40))
    card, reference = _build(panel)
    stratum = card["panel"]["media=glucose"]
    assert stratum["ranking_resolvable"]["verdict"] == "mismatch"
    for axis in ("objective_vs_reference", "growth_cost"):
        node = stratum[axis]
        assert node["verdict"] == "ungraded", axis
        assert node["detail"]["unreadable_because"] == "ranking_resolvable"
        assert node["meter"].startswith("ranking unresolvable — ")
        assert node["value"] is not None          # the number is kept, not hidden
    # ★ Discriminating: resolvability is the ONLY graded axis, so the card's
    # overall must come from it alone. If a future change let a reported axis
    # carry a verdict again, a resolvable-but-poor panel would fail here too.
    assert grade_card(card, reference)["overall"] == "mismatch"

    good = _panel(_arm("wt", "glucose", 1.00, 0.02),
                  _arm("design_a", "glucose", 3.00, 0.02),
                  _arm("design_b", "glucose", 2.00, 0.02),
                  _arm("design_c", "glucose", 1.50, 0.02))
    card_g, ref_g = _build(good)
    assert card_g["panel"]["media=glucose"]["ranking_resolvable"]["verdict"] != "mismatch"
    # A panel whose arms are well separated passes even though its growth ratio
    # sits far under any plausible floor — the biology does not move the verdict.
    assert grade_card(card_g, ref_g)["overall"] != "mismatch"


def test_default_policy_grades_execution_not_outcome():
    """★ The §15 contract: by default the card grades ONLY resolvability. The
    outcome axes keep every number and their declared design target, and give up
    only the ruling."""
    panel = _panel(_arm("wt", "glucose", 1.00, 0.02, gro_mean=0.70),
                   _arm("design_a", "glucose", 3.00, 0.02, gro_mean=0.20),
                   _arm("design_b", "glucose", 2.00, 0.02, gro_mean=0.60),
                   _arm("design_c", "glucose", 1.50, 0.02, gro_mean=0.65))
    card, reference = _build(panel)
    stratum = card["panel"]["media=glucose"]
    for axis in ("objective_vs_reference", "growth_cost"):
        node = stratum[axis]
        assert node["verdict"] == "ungraded", axis
        assert node["detail"]["graded"] is False, axis
        assert node["value"] is not None, axis          # number kept
        assert "design_target" in node["detail"], axis  # target kept
    assert stratum["ranking_resolvable"]["detail"]["graded"] is True
    # ★ Discriminating: design_a's growth is 0.29x the reference, far under the
    # 0.70 floor, and the card STILL does not fail. Biology cannot move it.
    assert stratum["growth_cost"]["value"] < 0.30
    assert grade_card(card, reference)["overall"] != "mismatch"


def test_grading_policy_is_declarable_and_reversible():
    """★ The demotion is a POLICY, not a capability that was removed — a caller
    that wants an outcome graded says so, with no upstream change. This is what
    keeps a downstream consumer free to choose differently."""
    panel = _panel(_arm("wt", "glucose", 1.00, 0.02, gro_mean=0.70),
                   _arm("design_a", "glucose", 3.00, 0.02, gro_mean=0.20),
                   _arm("design_b", "glucose", 2.00, 0.02, gro_mean=0.60),
                   _arm("design_c", "glucose", 1.50, 0.02, gro_mean=0.65))
    card, reference = _build(panel, graded_axes=list(ps.AXES))
    stratum = card["panel"]["media=glucose"]
    assert stratum["growth_cost"]["verdict"] == "mismatch"   # 0.29 < warn 0.70
    assert stratum["growth_cost"]["detail"]["graded"] is True
    assert stratum["objective_vs_reference"]["verdict"] == "within_tol"
    assert grade_card(card, reference)["overall"] == "mismatch"
    # And the default really is the narrow one, not "everything".
    assert ps.DEFAULT_GRADED_AXES == ("ranking_resolvable",)


def test_unknown_graded_axis_is_rejected_not_silently_ignored():
    """A typo must not silently ungrade the whole card."""
    panel = _panel(_arm("wt", "glucose", 1.00, 0.02),
                   _arm("design_a", "glucose", 1.40, 0.02),
                   _arm("design_b", "glucose", 1.20, 0.02))
    with pytest.raises(ValueError, match="unknown axis"):
        _build(panel, graded_axes=["ranking_resolveable"])   # sic
    with pytest.raises(ValueError, match="must be a list"):
        _build(panel, graded_axes="ranking_resolvable")


def test_single_design_arm_is_ungraded_not_a_pass():
    card, _ = _build(_panel(_arm("wt", "glucose", 1.0, 0.02),
                            _arm("design_a", "glucose", 1.4, 0.02)))
    node = card["panel"]["media=glucose"]["ranking_resolvable"]
    assert node["verdict"] == "ungraded"
    assert "nothing to rank" in node["meter"]


def test_growth_cost_reports_the_arm_the_objective_axis_picked():
    """A design that "wins" by killing the cell is not a win."""
    panel = _panel(_arm("wt", "glucose", 1.00, 0.02, gro_mean=0.70),
                   _arm("design_a", "glucose", 1.80, 0.02, gro_mean=0.35),
                   _arm("design_b", "glucose", 1.10, 0.02, gro_mean=0.69),
                   _arm("design_c", "glucose", 1.05, 0.02, gro_mean=0.68))
    card, _ = _build(panel)
    stratum = card["panel"]["media=glucose"]
    assert stratum["objective_vs_reference"]["detail"]["best_arm"] == "design_a|glucose"
    growth = stratum["growth_cost"]
    assert growth["detail"]["best_arm"] == "design_a|glucose"
    assert growth["value"] == pytest.approx(0.5, rel=1e-6)
    # ⛔ 0.5 sits far below the declared floor of 0.70 and the axis STILL carries
    # no verdict — the target is reported for the finding to weigh, not graded.
    assert growth["verdict"] == "ungraded"
    assert growth["detail"]["design_target"] == {"good": 0.85, "warn": 0.70}
    assert growth["detail"]["graded"] is False


def test_missing_growth_observable_is_ungraded_not_a_pass():
    arms = [_arm("wt", "glucose", 1.00, 0.02), _arm("design_a", "glucose", 1.40, 0.02),
            _arm("design_b", "glucose", 1.20, 0.02)]
    for a in arms:
        del a["observables"]["growth_rate"]
    card, _ = _build(_panel(*arms))
    assert card["panel"]["media=glucose"]["growth_cost"]["verdict"] == "ungraded"


# --- the committed fixture (golden) ------------------------------------------

def test_fixture_grades_run_free(fixture_panel):
    """Fixture-graded: no sweep, no ParCa cache, no live run. The committed fixture
    deliberately exercises both paths — one stratum resolves, the other does not.

    ★ Every reported axis is `ungraded` in BOTH strata: the card grades execution
    (resolvability), never the biology. The overall comes from the unrankable
    stratum alone."""
    card, reference = _build(fixture_panel)
    report = grade_card(card, reference)
    verdicts = {p: a["verdict"] for p, a in report["axes"].items()}
    assert verdicts == {
        "panel.strata_declared": "ungraded",
        "panel.media=glucose.objective_vs_reference": "ungraded",
        "panel.media=glucose.growth_cost": "ungraded",
        "panel.media=glucose.ranking_resolvable": "within_tol",
        "panel.media=acetate.objective_vs_reference": "ungraded",
        "panel.media=acetate.growth_cost": "ungraded",
        "panel.media=acetate.ranking_resolvable": "mismatch",
    }
    assert report["overall"] == "mismatch"
    glucose = report["axes"]["panel.media=glucose.objective_vs_reference"]
    assert glucose["detail"]["best_arm"] == "design_a|glucose"
    assert glucose["detail"]["improvement"] == pytest.approx(1.45, rel=1e-3)
    assert glucose["detail"]["family_size"] == 4
    assert report["axes"]["panel.media=glucose.growth_cost"]["value"] == pytest.approx(
        0.9, rel=1e-3)


def test_fixture_carries_no_private_content(fixture_panel):
    """The fixture must stay public: generic arm and observable names only."""
    designs = {a["design"] for a in fixture_panel["arms"]}
    assert designs == {"wt", "design_a", "design_b", "design_c", "design_d"}
    media = {a["strata"]["media"] for a in fixture_panel["arms"]}
    assert media == {"glucose", "acetate"}
    obs = {k for a in fixture_panel["arms"] for k in a["observables"]}
    assert obs == {"objective_titer", "growth_rate"}


# --- the Gen-2 Step ----------------------------------------------------------

def _ctx(tmp_path, refs, name="panel-demo"):
    (tmp_path / "workspace" / "studies" / name).mkdir(parents=True, exist_ok=True)
    return StudyContext(study_name=name,
                        study_dir=tmp_path / "workspace" / "studies" / name,
                        spec={"report_card_refs": {"panel_screen": refs}} if refs
                        else {},
                        ws_root=tmp_path)


def _refs(**over):
    refs = {"panel_json": str(FIXTURE), **{k: v for k, v in REFS.items()}}
    refs.update(over)
    return refs


def test_card_is_registered():
    assert REPORT_CARD_REGISTRY.get("panel_screen") is PanelScreenCard


def test_applies_only_when_the_ref_block_is_present(tmp_path, core):
    card = PanelScreenCard({}, core=core)
    assert card.applies(_ctx(tmp_path, _refs())) is True
    assert card.applies(_ctx(tmp_path, None)) is False


def test_applies_even_when_the_refs_are_malformed(tmp_path, core):
    """Deliberate: `applies` is gated ONLY on the ref block existing. A card that
    disqualified itself on a bad ref would let a mis-specified panel look like "card
    not applicable" instead of surfacing."""
    card = PanelScreenCard({}, core=core)
    refs = _refs()
    del refs["panel_json"]
    ctx = _ctx(tmp_path, refs)
    assert card.applies(ctx) is True
    with pytest.raises(ValueError, match="panel_json"):
        card.build(ctx)


def test_missing_panel_json_file_raises(tmp_path, core):
    card = PanelScreenCard({}, core=core)
    with pytest.raises(ValueError, match="not found"):
        card.build(_ctx(tmp_path, _refs(panel_json="nope/missing.json")))


def test_step_builds_verdict_and_html(tmp_path, core):
    card = PanelScreenCard({}, core=core)
    vjson, html = card.build(_ctx(tmp_path, _refs()))
    assert vjson["schema"] == "report_card_verdict/v1"
    assert vjson["overall"] == "mismatch"
    assert vjson["title"]
    # One group, and the report_card_axis evaluator reads its axes list.
    assert list(vjson["groups"]) == ["panel_screen"]
    assert len(vjson["groups"]["panel_screen"]["axes"]) == 7
    assert vjson["groups"]["panel_screen"]["verdict"] == "mismatch"
    assert html.startswith("<!doctype html>")
    assert "Panel screen" in html and "Ranking resolvable above noise" in html


def test_step_output_is_byte_deterministic(tmp_path, core):
    """The runner commits the card, so two renders of the same fixture must be
    identical (no timestamp, ties broken by arm id)."""
    card = PanelScreenCard({}, core=core)
    a = card.build(_ctx(tmp_path, _refs()))
    b = card.build(_ctx(tmp_path, _refs()))
    assert a[1] == b[1]
    assert a[0] == b[0]


def test_step_resolves_a_relative_panel_json(tmp_path, core):
    card = PanelScreenCard({}, core=core)
    rel = "tests/fixtures/panel_screen/panel_baseline.json"
    ctx = _ctx(REPO, _refs(panel_json=rel))
    vjson, _ = card.build(ctx)
    assert vjson["overall"] == "mismatch"


def test_step_surfaces_a_missing_strata_declaration(tmp_path, core):
    card = PanelScreenCard({}, core=core)
    refs = _refs()
    del refs["strata"]
    vjson, html = card.build(_ctx(tmp_path, refs))
    assert vjson["overall"] == "mismatch"
    axes = vjson["groups"]["panel_screen"]["axes"]
    assert [a["id"] for a in axes] == ["panel.strata_declared"]
    assert "testing family" in axes[0]["meter"]


def test_render_html_needs_no_card_specific_plot(tmp_path):
    """The card declares no `plot`, so it is a pure addition: nothing in
    library/report_card.py had to change for it."""
    card, reference = _build(ps.load_panel(FIXTURE))
    assert all("plot" not in ax for ax in reference["axes"].values())
    html = render_html(card, reference, model_ref="test")
    assert "plot unavailable" not in html
    verdict_json(grade_card(card, reference))     # serialises without a plot node
