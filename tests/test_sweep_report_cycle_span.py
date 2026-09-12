"""A generation's cycle duration is ``t[-1] - t[0]``, never ``t[-1]``.

``time`` reaches these reports under two conventions:

* **one continuous invocation** -> an ABSOLUTE clock that keeps rising across
  generations. ``run_comparison_ensemble.py`` refuses ``--initial-generation > 1``
  on the wrapped arm, so that arm is always this case.
* **chained / resumed invocations** (``--initial-generation`` /
  ``--daughter-state-out``) -> an INVOCATION-RELATIVE clock restarting near zero.

``t[-1]`` is a span only when a generation's clock happened to start at ~0. On an
absolute clock it is a TIMESTAMP, so the reported "cycle" grows without bound and
the cumulative trajectory offset double-counts.

These tests give both scripts the SAME BIOLOGY under BOTH conventions and assert
the reported spans are identical. That is the assertion ``t[-1]`` fails.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import sweep_report as sr  # noqa: E402
import sweep_report_xarray as sx  # noqa: E402

# Three generations, 40 / 44 / 43 minutes of identical biology.
SPANS_S = [2400.0, 2640.0, 2580.0]


def _t(span, start):
    return np.linspace(start, start + span, 5)


def _relative():
    """Every generation restarts near zero."""
    return [_t(s, 0.0) for s in SPANS_S]


def _absolute():
    """One monotone clock across all three generations."""
    out, clock = [], 0.0
    for s in SPANS_S:
        out.append(_t(s, clock))
        clock += s
    return out


def _cycles_xarray(times):
    cells = {}
    for i, t in enumerate(times):
        cells[(0, 0, i + 1)] = {
            "time": t,
            "dry_mass": np.linspace(350.0, 700.0, len(t)),
            "protein": np.linspace(150.0, 300.0, len(t)),
            "RNA": np.linspace(70.0, 140.0, len(t)),
            "DNA": np.linspace(7.0, 14.0, len(t)),
        }
    _p1, _p2, _frac, div_rows = sx._plots(cells)
    return [row[-1] for row in sorted(div_rows, key=lambda r: r[2])]


def _cycles_plain(times):
    cells = {}
    for i, t in enumerate(times):
        n = len(t)
        cells[(0, 0, i + 1)] = np.column_stack([
            t,
            np.linspace(350.0, 700.0, n),
            np.linspace(150.0, 300.0, n),
            np.linspace(70.0, 140.0, n),
            np.linspace(7.0, 14.0, n),
        ])
    _p1, _p2, _frac, div_rows = sr._plots(cells)
    return [row[-1] for row in sorted(div_rows, key=lambda r: r[2])]


def test_xarray_spans_are_convention_independent():
    rel, absolute = _cycles_xarray(_relative()), _cycles_xarray(_absolute())
    assert rel == absolute, (
        "the same biology reported different cycle durations under the two "
        "clock conventions -- the span is being read as a timestamp")
    assert np.allclose(rel, SPANS_S)


def test_plain_spans_are_convention_independent():
    rel, absolute = _cycles_plain(_relative()), _cycles_plain(_absolute())
    assert np.allclose(rel, absolute), (
        "the same biology reported different cycle durations under the two "
        "clock conventions -- the span is being read as a timestamp")
    assert np.allclose(rel, SPANS_S)


def test_the_absolute_convention_is_what_t_minus_1_gets_wrong():
    """Names the defect explicitly, so a future edit that reintroduces it fails
    with the reason rather than with a bare inequality."""
    times = _absolute()
    naive = [float(t[-1]) for t in times]          # the old computation
    correct = _cycles_xarray(times)
    assert naive != correct
    assert naive[-1] > sum(SPANS_S) * 0.9, (
        "sanity: on an absolute clock the naive value is a cumulative "
        "timestamp, which is why it inflates the lineage axis")
    assert np.allclose(correct, SPANS_S)


def test_a_single_point_generation_does_not_raise():
    """A generation with one sample has a zero span, not an IndexError."""
    one = [np.array([1234.0])]
    assert _cycles_xarray(one) == [0.0]
    assert np.allclose(_cycles_plain(one), [0.0])
