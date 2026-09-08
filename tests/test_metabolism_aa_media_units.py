"""Regression: boundary.external amino-acid concentrations must compare against
import_constraint_threshold whether they arrive as a plain float or a pint
Quantity. The amino-acid-supplemented media arm (e.g. basal_with_trp) can leave a
unit-carrying Quantity in boundary.external; pint refuses ``Quantity > float``,
which crashed metabolism.py:841 on that arm while plain ``minimal`` (no AA keys)
never reached the code path. See _mM_magnitude.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast

from v2ecoli.processes.metabolism import _mM_magnitude
from v2ecoli.types.quantity import ureg as units


def test_plain_float_passes_through():
    assert _mM_magnitude(0.5) == 0.5
    assert _mM_magnitude(0.0) == 0.0


def test_pint_quantity_in_mM_returns_magnitude():
    assert abs(_mM_magnitude(units("0.7 mM")) - 0.7) < 1e-9


def test_pint_quantity_in_other_conc_unit_converts_to_mM():
    # 500 uM == 0.5 mM
    assert abs(_mM_magnitude(units("500 uM")) - 0.5) < 1e-9


def test_the_actual_failing_comparison_is_now_unit_safe():
    """metabolism.py:841 does ``external[aa] > import_constraint_threshold``.
    A pint Quantity on the left, against a nonzero plain-float threshold, raised
    ``ValueError: Cannot compare PlainQuantity and <class 'float'>`` (pint
    special-cases only comparison against zero, so a nonzero threshold is the
    real crashing case)."""
    threshold = 0.05  # a nonzero plain-float import_constraint_threshold
    present = units("0.7 mM")   # an AA that IS in the media
    absent = units("0.01 mM")   # below threshold
    assert (_mM_magnitude(present) > threshold) is True
    assert (_mM_magnitude(absent) > threshold) is False
    # the raw comparison (the pre-fix code path) genuinely raises, proving this
    # guards a real crash rather than a hypothetical one
    with pytest.raises(ValueError):
        _ = present > threshold
