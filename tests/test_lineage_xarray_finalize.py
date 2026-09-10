"""LineageProcess._finalize_xarray must FAIL LOUD, not swallow.

One XArrayEmitter drives a whole lineage: at each division it is advanced in
place, finalizing the current generation (flush + mark_success + consolidate)
before the next opens. The next generation's ``_check_group`` requires that
consolidated previous generation.

Regression: the old code wrapped the advance/close in
``except Exception: warnings.warn(...); self._xarray_em = None`` -- swallowing a
failure and rebuilding a fresh emitter next generation. That does NOT heal the
un-consolidated generation; it defers the crash to the next generation's
``_open_store -> _check_group`` ("Missing path from previous generation"
FileNotFoundError), which on a multi-seed ray-mnp gang tears down the whole run
(observed on Run 2 J3 fix558 gen2 and Run 1 cell-only sim190 gen3). The fix
lets the failure propagate at the generation that could not persist.

Like tests/test_lineage_emitter_finalize.py, these call the unbound method with
a duck-typed ``self`` -- no cache, no core, no composite, no simulation.
"""

from types import SimpleNamespace

import pytest

from v2ecoli.workflow.lineage import LineageProcess


class _StubXEmitter:
    """Duck-typed XArrayEmitter: only advance_generation/close are exercised."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.advanced = 0
        self.closed = 0

    def advance_generation(self, *, agent_id, success=True):
        self.advanced += 1
        if self.fail:
            raise RuntimeError("consolidate failed: s3 unreachable")

    def close(self, success: bool = False):
        self.closed += 1
        if self.fail:
            raise RuntimeError("close failed: s3 unreachable")


def _lineage_x(agent_id: str, em, *, generations: int, is_xarray: bool = True):
    """A LineageProcess-shaped stand-in carrying only what _finalize_xarray reads."""
    return SimpleNamespace(
        _agent_id=agent_id,
        _generation=len(agent_id) - 1,
        config={"generations": generations},
        _xarray_em=em,
        _xarray_pending=True,
        _is_xarray=lambda: is_xarray,
    )


def test_advance_failure_propagates_not_swallowed():
    """A failing advance MUST raise here -- not warn-and-continue, which would
    defer to a cryptic _check_group crash next generation."""
    em = _StubXEmitter(fail=True)
    stub = _lineage_x("0", em, generations=5)  # gen 0, not last -> advance
    with pytest.raises(RuntimeError, match="consolidate failed"):
        LineageProcess._finalize_xarray(stub)
    assert em.advanced == 1


def test_advance_success_keeps_the_one_emitter():
    """On success the SAME emitter carries forward (not dropped) and the
    pending flag clears."""
    em = _StubXEmitter()
    stub = _lineage_x("0", em, generations=5)
    LineageProcess._finalize_xarray(stub)
    assert em.advanced == 1 and em.closed == 0
    assert stub._xarray_em is em            # not rebuilt
    assert stub._xarray_pending is False


def test_last_generation_closes_the_emitter():
    """generations == 1 -> gen 0 is the last -> close(), not advance."""
    em = _StubXEmitter()
    stub = _lineage_x("0", em, generations=1)
    LineageProcess._finalize_xarray(stub)
    assert em.closed == 1 and em.advanced == 0
    assert stub._xarray_em is None
    assert stub._xarray_pending is False


def test_last_generation_close_failure_propagates():
    em = _StubXEmitter(fail=True)
    stub = _lineage_x("0", em, generations=1)
    with pytest.raises(RuntimeError, match="close failed"):
        LineageProcess._finalize_xarray(stub)


def test_no_emitter_is_a_noop_but_clears_pending():
    stub = _lineage_x("0", None, generations=5)
    LineageProcess._finalize_xarray(stub)  # must not raise
    assert stub._xarray_pending is False


def test_non_xarray_run_is_untouched():
    em = _StubXEmitter()
    stub = _lineage_x("0", em, generations=5, is_xarray=False)
    LineageProcess._finalize_xarray(stub)
    assert em.advanced == 0 and em.closed == 0
    assert stub._xarray_pending is True     # left as-is for a non-xarray run
