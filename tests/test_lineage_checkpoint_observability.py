"""Checkpoint/emitter fail-loud + observability (sms-ecoli#210 / dispatch 313).

Dispatch 313 stalled IDLE for 4+ hours, no error, right before a lineage
checkpoint write — the emitter flush and checkpoint write are both S3 I/O with
no timeout. These tests cover the two guards added for that failure mode:

1. ``v2ecoli.cache`` gives its checkpoint S3 client a bounded connect/read
   timeout + retries, so a stalled write fails loud in minutes instead of
   hanging forever.
2. ``LineageProcess`` has a cheap ``_estimate_state_mb`` used to log the carry
   state's size (and warn on runaway growth) at every checkpoint, so an
   over-growing or stalled write is visible in the run log rather than silent.
"""
from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.workflow.lineage import _estimate_state_mb

pytestmark = pytest.mark.fast


def test_estimate_state_mb_sums_bulk_and_unique_arrays():
    state = {
        "bulk": np.zeros(500_000, dtype="i8"),          # 4.0 MB
        "unique": {"ribosome": np.zeros(200_000, dtype="i8")},  # 1.6 MB
        "environment": {"media_id": "minimal"},          # no nbytes
        "boundary": {"volume": 1.2},
    }
    mb = _estimate_state_mb(state)
    assert mb == pytest.approx((500_000 * 8 + 200_000 * 8) / 1e6)


def test_estimate_state_mb_handles_non_dict_and_empty():
    assert _estimate_state_mb(None) == 0.0
    assert _estimate_state_mb({}) == 0.0
    # unique present but not arrays -> counted as zero, no crash
    assert _estimate_state_mb({"unique": {"x": {"nested": "dict"}}}) == 0.0


def test_estimate_state_mb_tracks_growth():
    """The per-generation growth signal the checkpoint log surfaces: a bigger
    unique population estimates proportionally bigger."""
    small = {"unique": {"r": np.zeros(100_000, dtype="i8")}}
    big = {"unique": {"r": np.zeros(300_000, dtype="i8")}}
    assert _estimate_state_mb(big) == pytest.approx(3 * _estimate_state_mb(small))


def test_checkpoint_s3_client_has_bounded_timeouts():
    """A bare boto3 client can hang indefinitely on a stalled connection; the
    checkpoint client must carry an explicit timeout + retry budget."""
    pytest.importorskip("boto3")
    from v2ecoli.cache import _s3_client

    cfg = _s3_client()._client_config
    assert cfg.connect_timeout is not None and cfg.connect_timeout <= 60
    assert cfg.read_timeout is not None and cfg.read_timeout <= 120
    # bounded retries (standard mode surfaces total_max_attempts)
    assert cfg.retries and cfg.retries.get("total_max_attempts", 0) >= 2


def test_checkpoint_write_is_an_event_with_size_and_seconds(monkeypatch, tmp_path, capsys):
    """The 313 stall point, as an event: ``checkpoint`` carries the path, the
    estimated MB and the wall seconds of the write."""
    import json

    pbg_events = pytest.importorskip("process_bigraph.events")
    monkeypatch.delenv("PBG_EVENT_SINKS", raising=False)
    monkeypatch.setenv("PBG_EVENT_HEARTBEAT_S", "0")
    pbg_events.configure("stdout")
    from v2ecoli.workflow.lineage import LineageProcess

    lp = LineageProcess.__new__(LineageProcess)
    out = tmp_path / "gen.pkl"
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "b", "config_overrides": {}, "generations": 1,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 10.0, "initial_carry_state_path": "",
        "initial_generation_index": 0, "daughter_state_out_path": str(out),
        "checkpoint_dir": "", "require_output": False,
    }
    lp.initialize(lp.config)
    monkeypatch.setattr(lp, "_build_generation", lambda: setattr(lp, "_gen_elapsed", 0.0))
    monkeypatch.setattr(lp, "_run_until_division",
                        lambda interval: (True, {"bulk": {}, "unique": {}}, 1.0))
    lp.update({}, 10.0)
    pbg_events.set_emitter(None)
    events = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    ck = [e for e in events if e["event"] == "checkpoint"]
    assert len(ck) == 1
    assert ck[0]["payload"]["path"] == str(out)
    assert ck[0]["payload"]["status"] == "written"
    assert ck[0]["payload"]["seconds"] >= 0
    assert out.exists()
