"""The post-run xarray flush must fire even when run() raises.

Without it, any exception out of ``composite.run`` (a process crash, an
OOM) skipped ``close_emitter()`` entirely: the up-to-600-row trailing
buffer was lost and the zarr store was never finalized (no consolidate, no
``division_reached`` attr) — the "empty group skeletons crash the next
generation" durability gap CD2 hit. The hook now flushes on both paths;
the run's own exception still propagates, and a flush failure on the
failure path is a warning rather than a mask over the cause.
"""
from __future__ import annotations

import types

import pytest

import v2ecoli as v2


class _Emitter:
    def __init__(self, fail=False):
        self.closed = 0
        self.fail = fail

    def close_emitter(self):
        self.closed += 1
        if self.fail:
            raise RuntimeError("flush boom")


def _hooked(monkeypatch, emitter, run):
    comp = types.SimpleNamespace(state={}, run=run)
    monkeypatch.setattr(v2, "_find_lazy_xarray_emitters",
                        lambda state: [emitter])
    v2._install_xarray_flush_hook(comp)
    return comp


def test_flush_fires_on_clean_run(monkeypatch):
    em = _Emitter()
    comp = _hooked(monkeypatch, em, lambda interval: "ok")
    assert comp.run(10) == "ok"
    assert em.closed == 1


def test_flush_fires_when_run_raises_and_the_cause_propagates(monkeypatch):
    em = _Emitter()

    def _boom(interval):
        raise ValueError("process crashed mid-generation")

    comp = _hooked(monkeypatch, em, _boom)
    with pytest.raises(ValueError, match="process crashed"):
        comp.run(10)
    assert em.closed == 1  # the trailing buffer was still flushed


def test_flush_failure_never_masks_the_run_exception(monkeypatch):
    em = _Emitter(fail=True)

    def _boom(interval):
        raise ValueError("the real cause")

    comp = _hooked(monkeypatch, em, _boom)
    with pytest.warns(UserWarning, match="also failed"):
        with pytest.raises(ValueError, match="the real cause"):
            comp.run(10)


def test_flush_failure_on_a_clean_run_stays_loud(monkeypatch):
    em = _Emitter(fail=True)
    comp = _hooked(monkeypatch, em, lambda interval: "ok")
    with pytest.raises(RuntimeError, match="flush boom"):
        comp.run(10)
