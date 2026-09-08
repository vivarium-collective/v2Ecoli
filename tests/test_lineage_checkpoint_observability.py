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
