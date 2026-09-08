"""Behavior tests for FlagellaFilamentElongation's completion-consumption logic.

Directly targets the 2026-08-21 FliD double-consumption fix (see
MASTER_DOCUMENT.md History §3.1): NFsim's own flagellum reaction and this
Step's own completion event used to BOTH independently consume 5x FliD for
the same real completion (10 total instead of the real 5, Song et al.
2017/Postel et al. 2020: the FliD pentameric cap forms once, before
elongation, and cannot accept a second binding event). The fix kept the
consumption here and removed it from NFsim's own reaction
(generate_flagella_bngl.py) -- these tests pin THIS Step's own arithmetic
so a future edit can't silently reintroduce a doubled (or otherwise wrong)
FliD cost per completion. Scaling across multiple simultaneous completions
is checked specifically since that's the actual shape the original bug
had -- a fixed per-completion cost applied by two different Steps, not a
wrong constant in one place.
"""
import numpy as np

from v2ecoli.library.schema import MetadataArray
from v2ecoli.processes.flagella_filament_elongation import FlagellaFilamentElongation


_NASCENT_DTYPE = [
    ("_entryState", "i1"),
    ("filament_length", "<i8"),
    ("massDiff_protein", "f8"),
    ("unique_index", "<i8"),
]


def _bulk(triples):
    """triples: list of (id, count, protein_submass)."""
    return np.array(triples, dtype=[("id", "U40"), ("count", "i8"), ("protein_submass", "f8")])


def _nascent(lengths):
    rows = [(1, L, 0.0, i) for i, L in enumerate(lengths)]
    return MetadataArray(np.array(rows, dtype=_NASCENT_DTYPE), len(lengths))


def _proc():
    return FlagellaFilamentElongation(parameters={})


def _states(lengths, flic=100_000, flid=100, flis_flic_cplx=100_000, flis=0, flagella=0):
    return {
        "bulk": _bulk([
            ("EG10321-MONOMER[e]", flic, 1.0),           # free FliC
            ("EG10841-MONOMER[e]", flid, 1.0),           # FliD (cap)
            ("CPLX0-7452[j]", flagella, 1.0),            # complete flagellum
            ("EG11388-MONOMER[c]", flis, 1.0),           # FliS
            ("FLIS-FLIC-CPLX[e]", flis_flic_cplx, 1.0),  # protected FliC pool
        ]),
        "nascent_flagellum": _nascent(lengths),
        "timestep": 2.0,
        "global_time": 0.0,
        "next_update_time": 0.0,
    }


def test_single_completion_consumes_exactly_5_fliD():
    proc = _proc()
    # target_length defaults to 5000; one tick's growth from 4999 completes it.
    out = proc.update(_states([4999]))
    updates = dict(out["bulk"])
    assert updates[proc.fliD_idx] == -5, (
        f"expected -5 FliD for 1 completion, got {updates[proc.fliD_idx]}"
    )
    assert updates[proc.flagellum_idx] == 1


def test_two_simultaneous_completions_scale_to_10_not_20():
    """Direct regression guard for the 2026-08-21 double-consumption bug's
    actual shape: the bug wasn't a wrong constant, it was the same 5x-per-
    completion cost applied by two different Steps for the same event. This
    checks the scaling itself (2 completions -> 10, not 20), not just a
    single-completion constant, since a doubled-consumption regression would
    only show up in the multiplier, not in a 1-completion case."""
    proc = _proc()
    out = proc.update(_states([4999, 4998]))
    updates = dict(out["bulk"])
    assert updates[proc.fliD_idx] == -10, (
        f"expected -10 FliD for 2 completions, got {updates[proc.fliD_idx]}"
    )
    assert updates[proc.flagellum_idx] == 2


def test_no_completion_does_not_touch_fliD():
    proc = _proc()
    out = proc.update(_states([100]))  # far below target_length=5000
    updates = dict(out["bulk"])
    assert proc.fliD_idx not in updates
    assert proc.flagellum_idx not in updates
