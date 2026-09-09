"""Unit tests for the generic externally-imposed flux-bound hook in
metabolism.py (``Metabolism._apply_imposed_bounds``).

The hook lets an injected subsystem impose reaction upper/lower bounds on the
FBA before the solve, WITHOUT ecoli-metabolism knowing anything about what
imposes them (no drug-specific knowledge). The key contract:

  * an empty/absent store is a strict no-op (no setReactionFluxBounds calls, so
    the LP is byte-identical to a run with no injected subsystem);
  * a supplied ``{"upper_bound"/"lower_bound"}`` maps to the corresponding
    setReactionFluxBounds kwargs (raiseForReversible=False, matching the pin
    path);
  * unknown reaction ids are skipped, not raised.

Tested against a minimal fake fba so no real solver is needed, mirroring
test_metabolism_flux_pin.py.
"""

from types import SimpleNamespace

import numpy as np

from v2ecoli.processes.metabolism import Metabolism


class _FakeFBA:
    def __init__(self, reaction_ids):
        self._ids = list(reaction_ids)
        self.calls = []  # list of (rid, kwargs)

    def getReactionIDs(self):
        return np.array(self._ids)

    def setReactionFluxBounds(self, rid, **kwargs):
        self.calls.append((rid, kwargs))


def _apply(fba, imposed):
    # Call the method with a bare namespace as ``self`` — it only touches
    # ``self._pin_valid_reaction_ids`` (absent -> computed + cached here).
    Metabolism._apply_imposed_bounds(SimpleNamespace(), fba, imposed)
    return fba.calls


def test_empty_store_is_noop():
    fba = _FakeFBA(["R1", "R2"])
    assert _apply(fba, {}) == []


def test_absent_store_is_noop():
    fba = _FakeFBA(["R1", "R2"])
    assert _apply(fba, None) == []


def test_upper_bound_applied():
    fba = _FakeFBA(["H2PTEROATESYNTH-RXN", "R2"])
    calls = _apply(fba, {"H2PTEROATESYNTH-RXN": {"upper_bound": 1.5}})
    assert calls == [
        ("H2PTEROATESYNTH-RXN", {"raiseForReversible": False, "upperBounds": 1.5})
    ]


def test_upper_and_lower_bound_applied():
    fba = _FakeFBA(["R1"])
    calls = _apply(fba, {"R1": {"upper_bound": 2.0, "lower_bound": 0.5}})
    assert calls == [
        ("R1", {"raiseForReversible": False, "upperBounds": 2.0, "lowerBounds": 0.5})
    ]


def test_unknown_reaction_skipped():
    fba = _FakeFBA(["R1"])
    assert _apply(fba, {"NOT-A-REACTION": {"upper_bound": 1.0}}) == []


def test_no_drug_knowledge_in_metabolism_source():
    """Guard the separation-of-concerns principle: ecoli-metabolism must carry
    no antibiotic-specific identifiers."""
    import inspect

    src = inspect.getsource(Metabolism)
    for token in ("sulfadiazine", "mecillinam", "CPD-20940", "H2PTEROATESYNTH"):
        assert token not in src, f"antibiotic token {token!r} leaked into Metabolism"
