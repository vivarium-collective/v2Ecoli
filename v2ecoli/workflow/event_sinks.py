"""Pluggable event sinks for the engine's event stream (``process_bigraph.events``).

The engine ships only ``stdout`` and ``file:`` sinks and never imports a cloud
SDK. This module is the *plugin* that adds an object-store sink, resolved by
spec through the ``process_bigraph.event_sinks`` entry-point group
(``[project.entry-points."process_bigraph.event_sinks"] s3 = ...``) or the
factory registered by :mod:`v2ecoli.workflow.events`::

    PBG_EVENT_SINKS="stdout,s3://bucket/prefix/events/"

``S3JsonlSink`` buffers events and rewrites ONE object per writer,
``<prefix>/<trace_id>/<source>.jsonl``, on a timer (``PBG_EVENT_FLUSH_S``,
default 60 s), so a task's progress is visible in the store *before* the task
exits -- the property S3 lacks today (nothing lands until the emitter closes).
Objects stay small (a 5-generation lineage is well under 1 MB), so a whole-
object rewrite is cheaper than any append emulation. Any fsspec URI works
(``file://`` in tests, ``memory://``); ``s3://`` needs ``s3fs``, which v2ecoli
already depends on. The sink never raises into the simulation: a failed
write is retried on the next flush and the engine disables a sink only when
``emit`` itself raises.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
from typing import Any

try:
    from process_bigraph.events import EventSink
except Exception:  # pragma: no cover - pre-#209 engine; the module is then unused
    class EventSink:  # type: ignore[no-redef]
        def emit(self, event):
            raise NotImplementedError

        def flush(self):
            return None

        def close(self):
            return None


DEFAULT_FLUSH_S = 60.0


def _default_source() -> str:
    # AWS_BATCH_JOB_ID is read only to make the object key unique per task
    # attempt; nothing else here knows or cares which cloud it runs on.
    return (
        os.environ.get("PBG_EVENT_SOURCE")
        or os.environ.get("AWS_BATCH_JOB_ID")
        or f"{socket.gethostname()}-{os.getpid()}"
    )


class S3JsonlSink(EventSink):
    """Timer-flushed JSON-lines object at ``<uri>/<trace_id>/<source>.jsonl``.

    Accepts the full sink spec (``s3://bucket/prefix``, ``s3:bucket/prefix``,
    ``file:///tmp/x``, ``memory://x``) as the engine's factory contract passes
    it. ``flush_s <= 0`` disables the timer (flush only on ``flush``/``close``).
    """

    def __init__(self, spec: str, *, flush_s: float | None = None, source: str | None = None):
        spec = (spec or "").strip()
        if "://" not in spec and ":" in spec:
            scheme, rest = spec.split(":", 1)
            spec = f"{scheme}://{rest.lstrip('/')}"
        self.uri = spec.rstrip("/")
        if flush_s is None:
            try:
                flush_s = float(os.environ.get("PBG_EVENT_FLUSH_S", DEFAULT_FLUSH_S))
            except ValueError:
                flush_s = DEFAULT_FLUSH_S
        self.flush_s = float(flush_s)
        self.source = source or _default_source()
        self.trace_id: str | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._dirty = False
        self._timer: threading.Timer | None = None
        self._closed = False
        self.last_error: str | None = None
        self.flush_count = 0
        atexit.register(self.close)

    # -- EventSink contract ---------------------------------------------- #

    def emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, default=str)
        with self._lock:
            if self.trace_id is None and event.get("trace_id"):
                self.trace_id = str(event["trace_id"])
            self._lines.append(line)
            self._dirty = True
            self._ensure_timer_locked()

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            payload = "\n".join(self._lines) + "\n"
            key = self.key
        try:
            import fsspec

            with fsspec.open(key, "wb") as fh:
                fh.write(payload.encode("utf-8"))
            with self._lock:
                self._dirty = False
                self.flush_count += 1
                self.last_error = None
        except Exception as exc:  # retried on the next flush; never raises
            with self._lock:
                self.last_error = repr(exc)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self.flush()

    # -- helpers ---------------------------------------------------------- #

    @property
    def key(self) -> str:
        return f"{self.uri}/{self.trace_id or 'untraced'}/{self.source}.jsonl"

    def _ensure_timer_locked(self) -> None:
        if self.flush_s <= 0 or self._closed or self._timer is not None:
            return
        timer = threading.Timer(self.flush_s, self._on_timer)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _on_timer(self) -> None:
        with self._lock:
            self._timer = None
        self.flush()
        # The next emit() re-arms the timer; an idle sink stays idle.
