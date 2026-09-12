"""``adjust_rna_expression``: combining vs compounding on a shared transcription unit.

Every id here is invented. The point of the fixture is the OPERON STRUCTURE --
several adjusted cistrons carried by one transcription unit -- which is what the
stock adjustment table (single-cistron, no shared TU) cannot exercise.
"""

import warnings

import numpy as np
import pytest

from v2ecoli.processes.parca.steps.step_02_input_adjustments import (
    COMBINERS,
    DEFAULT_COMBINER,
    adjust_rna_expression,
)

# Three TUs. TU 0 is polycistronic and carries three of the adjusted cistrons;
# TUs 1 and 2 carry one each. Compartment suffixes match the real id shape.
RNA_IDS = ["some_operon_TU[c]", "solo_geneA_RNA[c]", "solo_geneB_RNA[c]"]
CISTRON_IDS = ["cisA", "cisB", "cisC", "solo_geneA", "solo_geneB"]
CISTRON_TO_RNA_INDEXES = {
    "cisA": np.array([0]),
    "cisB": np.array([0]),
    "cisC": np.array([0]),
    "solo_geneA": np.array([1]),
    "solo_geneB": np.array([2]),
}
BASE = np.array([0.5, 0.3, 0.2])


def _adjust(adjustments, base=None, combine=DEFAULT_COMBINER):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return adjust_rna_expression(
            RNA_IDS,
            CISTRON_IDS,
            (BASE if base is None else base).copy(),
            adjustments,
            CISTRON_TO_RNA_INDEXES,
            combine=combine,
        )


def test_one_cistron_per_tu_is_unchanged_by_the_fix():
    """The regression contract: the stock table's shape must not move."""
    out = _adjust({"solo_geneA": 10.0})
    expected = np.array([0.5, 3.0, 0.2])
    assert out == pytest.approx(expected / expected.sum())


def test_cistrons_sharing_a_tu_are_COMBINED_not_compounded():
    out = _adjust({"cisA": 2.0, "cisB": 5.0, "cisC": 3.0})
    # geometric mean of (2, 5, 3), applied once
    expected = np.array([0.5 * (2.0 * 5.0 * 3.0) ** (1 / 3), 0.3, 0.2])
    assert out == pytest.approx(expected / expected.sum())


# --- direction semantics: the axis a plain `max` gets wrong -------------------

def test_a_knockout_is_NOT_erased_by_a_co_located_upregulation():
    """`max(0.0, 3.0)` is 3.0 — the knockout vanishes. The default must not."""
    out = _adjust({"cisA": 0.0, "cisB": 3.0})
    expected = np.array([0.0, 0.3, 0.2])
    assert out == pytest.approx(expected / expected.sum())
    assert out[0] == 0.0


def test_two_down_regulated_cistrons_do_not_collapse_to_the_mildest():
    """`max(0.1, 0.01)` is 0.1 — the LEAST perturbed. Geometric mean is 0.0316."""
    out = _adjust({"cisA": 0.1, "cisB": 0.01})
    expected = np.array([0.5 * (0.1 * 0.01) ** 0.5, 0.3, 0.2])
    assert out == pytest.approx(expected / expected.sum())
    mildest = np.array([0.5 * 0.1, 0.3, 0.2])
    assert out != pytest.approx(mildest / mildest.sum())


def _combined_factor(out):
    """Recover the factor applied to TU 0 from a renormalized result.

    The trailing renormalization is a single global divisor, so it cancels out
    of a ratio against an UNadjusted gene — reading `out[0]` alone would measure
    the divisor as much as the combiner.
    """
    return (out[0] / out[2]) / (BASE[0] / BASE[2])


def test_the_combiner_is_direction_symmetric():
    """Inverting every factor must invert the combined factor.

    `max` fails this: max(2, 8) = 8 but max(0.5, 0.125) = 0.5, and 1/8 != 0.5.
    """
    up = _combined_factor(_adjust({"cisA": 2.0, "cisB": 8.0}))
    down = _combined_factor(_adjust({"cisA": 0.5, "cisB": 0.125}))
    assert up == pytest.approx(4.0)
    assert down == pytest.approx(0.25)
    assert down == pytest.approx(1.0 / up)


def test_max_guarded_reproduces_max_on_a_same_direction_set():
    out = _adjust({"cisA": 2.0, "cisB": 5.0, "cisC": 3.0}, combine="max_guarded")
    expected = np.array([0.5 * 5.0, 0.3, 0.2])
    assert out == pytest.approx(expected / expected.sum())


def test_max_guarded_REFUSES_a_direction_discordant_tu():
    with pytest.raises(ValueError, match="direction-discordant"):
        _adjust({"cisA": 0.0, "cisB": 3.0}, combine="max_guarded")
    with pytest.raises(ValueError, match="direction-discordant"):
        _adjust({"cisA": 0.5, "cisB": 3.0}, combine="max_guarded")


def test_an_unknown_combiner_raises():
    with pytest.raises(ValueError, match="unknown combiner"):
        _adjust({"solo_geneA": 2.0}, combine="mode")


def test_every_combiner_agrees_when_no_tu_is_shared():
    """The regression contract holds for ALL combiners, not just the default."""
    ref = _adjust({"solo_geneA": 10.0, "solo_geneB": 0.5})
    for name in COMBINERS:
        assert _adjust({"solo_geneA": 10.0, "solo_geneB": 0.5}, combine=name) == (
            pytest.approx(ref)
        )


def test_a_repeated_index_within_one_cistron_is_ONE_observation():
    """A cistron listing an index twice must not be treated as two cistrons."""
    c2r = dict(CISTRON_TO_RNA_INDEXES, solo_geneA=np.array([1, 1]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the shared-TU warning must NOT fire
        out = adjust_rna_expression(
            RNA_IDS, CISTRON_IDS, BASE.copy(), {"solo_geneA": 4.0}, c2r
        )
    expected = np.array([0.5, 0.3 * 4.0, 0.2])
    assert out == pytest.approx(expected / expected.sum())


def test_the_array_is_mutated_in_place_and_returned():
    arr = BASE.copy()
    out = adjust_rna_expression(
        RNA_IDS, CISTRON_IDS, arr, {"solo_geneA": 2.0}, CISTRON_TO_RNA_INDEXES
    )
    assert out is arr


def test_a_shared_tu_warns_so_the_caller_can_see_the_operon_structure():
    with pytest.warns(UserWarning, match="more than one adjusted cistron"):
        adjust_rna_expression(
            RNA_IDS, CISTRON_IDS, BASE.copy(),
            {"cisA": 2.0, "cisB": 5.0}, CISTRON_TO_RNA_INDEXES,
        )


def test_the_compounded_result_is_NOT_what_we_produce():
    """Paired control: pin the wrong answer so a regression cannot pass quietly.

    Compounding would multiply 2 * 5 * 3 = 30 onto one TU. Renormalizing keeps
    that a valid distribution, so only an explicit assertion catches it.
    """
    out = _adjust({"cisA": 2.0, "cisB": 5.0, "cisC": 3.0})
    compounded = np.array([0.5 * 30.0, 0.3, 0.2])  # 2 * 5 * 3
    compounded = compounded / compounded.sum()
    assert out != pytest.approx(compounded)
    assert out[0] < compounded[0]


def test_a_transcription_unit_id_addresses_its_tu_directly():
    with_suffix = _adjust({"some_operon_TU[c]": 4.0})
    without_suffix = _adjust({"some_operon_TU": 4.0})
    expected = np.array([2.0, 0.3, 0.2])
    assert with_suffix == pytest.approx(expected / expected.sum())
    assert without_suffix == pytest.approx(expected / expected.sum())


def test_an_unknown_id_raises_rather_than_being_skipped():
    with pytest.raises(ValueError, match="neither a known cistron"):
        _adjust({"not_a_real_id": 2.0})


def test_the_result_is_always_renormalized():
    for adjustments in ({"cisA": 1000.0}, {"cisA": 0.001, "solo_geneB": 7.0}):
        assert _adjust(adjustments).sum() == pytest.approx(1.0)
