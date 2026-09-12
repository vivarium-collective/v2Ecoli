"""LineageProcess._open_xarray_emitter: two open-path guards for the
multi-seed-gang residual to #777.

One XArrayEmitter drives a whole lineage. #777 fixed the FINALIZE path (the
``advance_generation`` swallow, tests/test_lineage_xarray_finalize.py). This
covers the OPEN path, where two remaining ways to break the next generation's
``_check_group`` linkage live:

1. **Empty first tick must not abandon the generation.** The docstring's
   contract is to open "on the first POPULATED tick". The old code set
   ``_xarray_pending = False`` on the FIRST empty tick, abandoning the whole
   generation even when a later tick in the same generation would populate.
   An abandoned generation writes NO group, which breaks the NEXT
   generation's ``_check_group`` ("Missing path from previous generation").
   The fix leaves ``_xarray_pending`` True and returns, so a later populated
   tick opens the emitter; a generation empty for ALL ticks is still caught
   loudly by ``_assert_generation_emitted`` at its own end.

2. **A FAILED fresh open at generation>0 must fail loud with context.** A
   fresh open at gen>0 means the one lineage emitter was NOT carried forward
   across the last division; the emitter's own ``_open_store -> _check_group``
   then raises a cryptic zarr FileNotFoundError on the missing prior-gen
   group. The guard re-raises with the exact gen/seed and the two suspected
   causes named, turning the next gang failure into a precise diagnostic
   instead of another opaque crash. This NAMES the residual; it is NOT the
   completion fix.

Like tests/test_lineage_xarray_finalize.py, these drive the unbound method
with a duck-typed ``self`` -- no cache, no core, no composite, no simulation.
The three ``v2ecoli.library.xarray_run`` helpers the method imports at call
time are monkeypatched on their source module so the ``from ... import`` inside
the method re-resolves to the stubs.
"""

from types import SimpleNamespace

import pytest

import v2ecoli.library.xarray_run as xr
from v2ecoli.workflow.lineage import LineageProcess


def _lineage_open(generation, *, store=None, out_dir):
    """A LineageProcess-shaped stand-in carrying only what _open_xarray_emitter
    reads before/at the _build_emitter call."""
    return SimpleNamespace(
        _generation=generation,
        _agent_id="0" * (generation + 1),
        _core=object(),
        _xarray_store=store,
        _xarray_view=None,
        _xarray_em=None,
        _xarray_pending=True,
        config={
            "emitter_arg": {},
            "out_dir": out_dir,
            "experiment_id": "exp",
            "variant_index": 0,
            "lineage_seed": 7,
            "time_step": 1.0,
            "max_duration_per_gen": 100.0,
        },
    )


def test_empty_first_tick_does_not_abandon_generation(tmp_path):
    """An empty view (no declared leaves present yet) must leave the emitter
    PENDING for a later populated tick -- not clear the pending flag, which
    would abandon the generation and break the next gen's _check_group."""
    stub = _lineage_open(0, out_dir=str(tmp_path))
    # emit_cell has none of the DEFAULT_XARRAY_VIEW leaves -> the real
    # filter_view_to_existing_leaves returns an empty view.
    LineageProcess._open_xarray_emitter(stub, {})
    assert stub._xarray_pending is True   # NOT abandoned
    assert stub._xarray_em is None        # nothing opened yet


def test_failed_fresh_open_at_gen_gt0_fails_loud_with_residual(tmp_path, monkeypatch):
    """A _build_emitter failure at generation>0 re-raises a RuntimeError that
    names the residual and the two suspected causes -- not the cryptic
    FileNotFoundError alone."""
    monkeypatch.setattr(
        xr, "filter_view_to_existing_leaves", lambda wrapped, view: view
    )
    monkeypatch.setattr(
        xr, "extract_output_metadata_from_state", lambda wrapped, view: {}
    )

    def _boom(**_kw):
        raise FileNotFoundError("Missing path from previous generation")

    monkeypatch.setattr(xr, "_build_emitter", _boom)

    stub = _lineage_open(2, out_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="residual to #777") as ei:
        LineageProcess._open_xarray_emitter(stub, {"listeners": {"mass": {}}})
    # the guard names the generation and preserves the original as __cause__
    assert "generation 2" in str(ei.value)
    assert isinstance(ei.value.__cause__, FileNotFoundError)


def test_failed_fresh_open_at_gen0_reraises_original(tmp_path, monkeypatch):
    """At generation 0 a fresh open is EXPECTED (start of the lineage), so a
    build failure is re-raised as-is -- the residual wrapper is gen>0 only."""
    monkeypatch.setattr(
        xr, "filter_view_to_existing_leaves", lambda wrapped, view: view
    )
    monkeypatch.setattr(
        xr, "extract_output_metadata_from_state", lambda wrapped, view: {}
    )

    def _boom(**_kw):
        raise FileNotFoundError("some other build failure")

    monkeypatch.setattr(xr, "_build_emitter", _boom)

    stub = _lineage_open(0, out_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="some other build failure"):
        LineageProcess._open_xarray_emitter(stub, {"listeners": {"mass": {}}})
