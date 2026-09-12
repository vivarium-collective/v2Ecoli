"""The equilibrium negative-steady-state guard names the offending species.

When the equilibrium ODE settles below -1 molecule for some species, the solver
raises. The bare "Negative values at equilibrium steady state." message forced a
whole investigation (v2ecoli#784) just to learn WHICH species and by how much —
small (numerical) vs gross (a real infeasibility). ``_negative_ss_detail`` turns
that opaque error into a self-diagnosing one: the worst offenders named, with
their molecule-count magnitudes.
"""

from __future__ import annotations

import numpy as np

from v2ecoli.library.function_registry import _negative_ss_detail


def test_names_worst_offenders_most_negative_first():
    mols = np.array([100.0, -5.0, 0.0, -137.0, -0.5])
    names = ["A[c]", "B[c]", "C[c]", "TRP[c]", "D[c]"]
    msg = _negative_ss_detail(mols, names)
    # Only species <= -1 molecule are offenders (0.0 and -0.5 are not).
    assert msg.startswith("2 species")
    # Most negative first.
    assert msg.index("TRP[c]") < msg.index("B[c]")
    assert "TRP[c]=-137" in msg
    assert "B[c]=-5" in msg
    assert "C[c]" not in msg and "D[c]" not in msg


def test_falls_back_to_index_when_names_absent():
    mols = np.array([1.0, -3.0])
    msg = _negative_ss_detail(mols, None)
    assert "index 1=-3" in msg


def test_truncates_and_counts_the_remainder():
    mols = -np.arange(2, 12, dtype=float)  # 10 species, all <= -1
    msg = _negative_ss_detail(mols, None, k=3)
    assert msg.startswith("10 species")
    assert "+7 more" in msg
    # Exactly k named offenders before the "+N more".
    assert msg.count("index ") == 3
