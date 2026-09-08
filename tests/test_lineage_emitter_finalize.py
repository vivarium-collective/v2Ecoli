"""LineageProcess must finalize its OWN generation's parquet emitter.

Regression test for v2ecoli#687. ``Division`` finalizes by looking the emitter
up in the process-global registry, but the key it derives is always "0" (inside
the composite the cell is the ``agents/0`` key, and the parquet override that
carries the runner's real identity is cleared before the composite is built).
Emitters are registered under the runner's per-generation id -- "0", "00",
"000" -- so from generation 1 on that lookup misses and the generation's
trailing batch and ``success/`` sentinel are silently lost.

These tests exercise ``LineageProcess._finalize_parquet`` directly against the
registry, so they need no ParCa cache and no simulation.
"""

from types import SimpleNamespace

import pytest

from v2ecoli.composites._helpers import (
    get_parquet_emitter,
    register_parquet_emitter,
    unregister_parquet_emitter,
)
from v2ecoli.workflow.lineage import LineageProcess


class _StubEmitter:
    """Duck-typed stand-in: the registry only requires ``close(success=...)``."""

    def __init__(self):
        self.closed_with = None
        self.close_calls = 0

    def close(self, success: bool = False):
        self.close_calls += 1
        self.closed_with = success


def _lineage(agent_id: str, composite_state: dict | None = None):
    """A LineageProcess-shaped object carrying only what _finalize_parquet reads.

    Built as a SimpleNamespace rather than a real process so the test needs no
    cache, no core and no composite -- the method under test touches exactly
    these three attributes.
    """
    return SimpleNamespace(
        _agent_id=agent_id,
        _generation=int(len(agent_id)) - 1,
        _composite=SimpleNamespace(state=composite_state or {}),
    )


@pytest.mark.parametrize("agent_id", ["0", "00", "000"])
def test_finalizes_its_own_generation_key(agent_id):
    """The emitter is closed for EVERY generation, not just the first.

    Before #687 this passed for "0" (where Division's fallback key happens to
    coincide) and failed for "00"/"000" -- the case that silently lost data.
    """
    emitter = _StubEmitter()
    register_parquet_emitter(agent_id, emitter)
    try:
        LineageProcess._finalize_parquet(_lineage(agent_id))
        assert emitter.close_calls == 1, "generation's emitter was not closed"
        assert emitter.closed_with is True, "success sentinel was not requested"
        assert get_parquet_emitter(agent_id) is None, "registry entry not popped"
    finally:
        unregister_parquet_emitter(agent_id)


def test_does_not_touch_another_generations_emitter():
    """Finalizing generation 1 must leave a still-running generation alone."""
    mine, other = _StubEmitter(), _StubEmitter()
    register_parquet_emitter("00", mine)
    register_parquet_emitter("000", other)
    try:
        LineageProcess._finalize_parquet(_lineage("00"))
        assert mine.close_calls == 1
        assert other.close_calls == 0, "closed an emitter belonging to another generation"
        assert get_parquet_emitter("000") is other
    finally:
        unregister_parquet_emitter("00")
        unregister_parquet_emitter("000")


def test_is_idempotent_when_division_already_finalized():
    """Generation 0 is finalized by Division first; the second call must no-op.

    Division's own lookup succeeds for "0", so by the time LineageProcess runs
    the registry entry is already gone. That must not raise or double-report.
    """
    lineage = _lineage("0")
    assert get_parquet_emitter("0") is None  # nothing registered: Division popped it
    LineageProcess._finalize_parquet(lineage)  # must not raise


def test_survives_an_emitter_that_raises_on_close():
    """A failing close is warned about, not propagated -- the summary still records
    the generation, and one bad emitter must not abort the lineage."""

    class _Boom(_StubEmitter):
        def close(self, success: bool = False):
            raise RuntimeError("s3 unreachable")

    register_parquet_emitter("00", _Boom())
    try:
        with pytest.warns(UserWarning, match="parquet finalize failed"):
            LineageProcess._finalize_parquet(_lineage("00"))
    finally:
        unregister_parquet_emitter("00")
