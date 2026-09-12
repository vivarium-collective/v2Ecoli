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
object rewrite is cheaper than any append emulation.

*Per writer* means per OS process, not per task: see :func:`_default_source`.

Because the object is rewritten whole, the buffer is capped -- the first
``PBG_EVENT_MAX_HEAD_LINES`` events (setup, early decisions) plus a rolling
``PBG_EVENT_MAX_TAIL_LINES`` tail (where a failure lands), with an explicit
``sink.truncated`` marker between them carrying the drop count. An uncapped
buffer would hold the whole run in memory and make cumulative bytes-written
grow with the square of the event count -- tolerable for a lineage task
emitting thousands of events, not for a process running many composites at a
high tick rate. Any fsspec URI works
(``file://`` in tests, ``memory://``); ``s3://`` needs ``s3fs``, which v2ecoli
already depends on. The sink never raises into the simulation: a failed
write is retried on the next flush and the engine disables a sink only when
``emit`` itself raises.
"""

from __future__ import annotations

import atexit
import collections
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
# The object is rewritten WHOLE on every flush, so an unbounded buffer costs
# both memory and cumulative bytes-written that grow with the square of the
# run's event count. Keep the head (the run's setup and early decisions) and
# a rolling tail (where a failure lands), and say so in the object.
DEFAULT_MAX_HEAD_LINES = 2000
DEFAULT_MAX_TAIL_LINES = 20000


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _default_source() -> str:
    """A key fragment unique to THIS WRITER -- one OS process, not one task.

    The sink rewrites a whole object per writer, so two writers that resolve
    the same source silently clobber each other: each flush replaces the
    object with only that writer's buffer. A task id alone is therefore the
    wrong granularity the moment a task runs more than one Python process.

    That is not hypothetical. On the Ray/multi-node path the driver and every
    Ray worker process on a node share one ``AWS_BATCH_JOB_ID`` (Batch gives a
    multi-node child ``<mainJobId>#<nodeIndex>``, which is per *node*, not per
    process), and each Ray worker runs its own cell composites through the
    engine's tick path -- so they all emit. The pid disambiguates them; it
    costs nothing on the one-process-per-task paths (Nextflow, chain), where
    the job id still leads the key and keeps it readable.

    ``PBG_EVENT_SOURCE`` remains an exact operator override and is used
    verbatim: whoever sets it owns the uniqueness.
    """
    explicit = os.environ.get("PBG_EVENT_SOURCE")
    if explicit:
        return explicit
    # AWS_BATCH_JOB_ID is read only to make the object key unique and legible
    # per task attempt; nothing else here knows or cares which cloud it runs on.
    task = os.environ.get("AWS_BATCH_JOB_ID") or socket.gethostname()
    return f"{task}-{os.getpid()}"


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
        self._max_head = _int_env("PBG_EVENT_MAX_HEAD_LINES", DEFAULT_MAX_HEAD_LINES)
        self._max_tail = _int_env("PBG_EVENT_MAX_TAIL_LINES", DEFAULT_MAX_TAIL_LINES)
        self._head: list[str] = []
        self._tail: collections.deque[str] = collections.deque(maxlen=max(self._max_tail, 1))
        self.dropped = 0
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
            if len(self._head) < self._max_head:
                self._head.append(line)
            else:
                if len(self._tail) == self._tail.maxlen:
                    self.dropped += 1
                self._tail.append(line)
            self._dirty = True
            self._ensure_timer_locked()

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            payload = "\n".join(self._payload_lines_locked()) + "\n"
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

    def _payload_lines_locked(self) -> list[str]:
        """Head + (an explicit gap marker) + rolling tail. Never silent."""
        if not self.dropped:
            return self._head + list(self._tail)
        marker = json.dumps({
            "v": 1,
            "component": "v2ecoli.event_sink",
            "event": "sink.truncated",
            "level": "warning",
            "trace_id": self.trace_id,
            "source": self.source,
            "payload": {
                "dropped": self.dropped,
                "head_lines": len(self._head),
                "tail_lines": len(self._tail),
                "reason": "buffer cap; raise PBG_EVENT_MAX_TAIL_LINES or lower the event rate",
            },
        })
        return self._head + [marker] + list(self._tail)

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
