"""Runner-side observability for the lineage runner (the seams the engine cannot see).

The engine (``process_bigraph.events``, process-bigraph >= 1.9 / PR #209) emits
``run_start`` / ``tick`` / ``structural_change`` / ``exception`` / ``run_end`` for
every ``Composite.run``. This module adds the *runner* layer on top of it: the
per-generation decisions that only ``LineageProcess`` / ``LineageStep`` know --

* ``generation_start``  -- the generation seed, the cumulative lineage-time
  offset, the emitter target, and the previous division's carry report;
* ``division``          -- which signal fired (structural / divide flag /
  exception), when, at what mass, and exactly which agent-root stores were
  carried, dropped by policy, filtered as edges, or left UNCLASSIFIED (the
  #765 ``request``/``allocate`` class, now a warning-level event);
* ``generation_end`` / ``checkpoint`` / ``chunk_flushed`` / ``warning``;
* ``failure_record``    -- written next to the task's output *and* emitted,
  with the engine's ``pbg_context`` (process path, ``global_time``, state
  summary) when the exception carries one.

Design rules, inherited from the engine and load-bearing here too:

* **Never raises into the simulation.** Every helper is guarded; with the
  engine absent (an image whose process-bigraph pin predates #209) everything
  degrades to a no-op emitter, so the pin can lag one image.
* **Infrastructure-agnostic.** No cloud SDK here. The S3 sink lives in
  :mod:`v2ecoli.workflow.event_sinks` and is registered as a *plugin*
  (``process_bigraph.event_sinks`` entry point + ``register_sink_factory``).
* **Zero configuration is a valid configuration.** ``LineageStep`` turns on
  stdout + a per-task ``events.jsonl`` file sink; a laptop run of the local
  sweep runner gets nothing unless ``PBG_EVENT_SINKS`` says so.

Identity: the task entrypoint (``process_bigraph.run_step``) already parsed
``PBG_TRACEPARENT`` / ``PBG_TRACE_BAGGAGE``; the runner *binds* the fields only
it knows (``variant``, ``lineage_seed``, ``generation``, ``experiment_id``).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any

LAYER = "runner"

try:  # process-bigraph >= 1.9 (feat/events, #209)
    from process_bigraph import events as _pbg_events
except Exception:  # pragma: no cover - exercised only on a pre-#209 pin
    _pbg_events = None


# ---------------------------------------------------------------------------
# No-op fallback (engine predates the events module)
# ---------------------------------------------------------------------------


class _NullSpan:
    def end(self, status: str = "ok", error: str | None = None) -> None:  # noqa: ARG002
        return None


class _NullEmitter:
    """Behaves like ``process_bigraph.events.EventEmitter`` with no sinks."""

    enabled = False
    identity: dict[str, Any] = {}
    trace_id = None

    def bind(self, **identity):  # noqa: ARG002
        return self

    def event(self, *_a, **_k):
        return None

    def start_span(self, *_a, **_k):
        return _NullSpan()

    @contextlib.contextmanager
    def span(self, *_a, **_k):
        yield _NullSpan()

    def heartbeat(self, *_a, **_k):
        return False

    def exception(self, *_a, **_k):
        return None

    def flush(self):
        return None

    def current_traceparent(self):
        return ""


_NULL = _NullEmitter()


def engine_available() -> bool:
    return _pbg_events is not None


def get_emitter():
    """The process-wide engine emitter, or a no-op stand-in."""
    if _pbg_events is None:
        return _NULL
    try:
        return _pbg_events.get_emitter()
    except Exception:  # never let observability break the runner
        return _NULL


def _add_sink(emitter, sink) -> bool:
    """Attach an extra sink to an already-configured emitter."""
    sinks = getattr(emitter, "_sinks", None)
    if isinstance(sinks, list):
        sinks.append(sink)
        return True
    return False


def configure_for_task(out_dir: str | None = None, *, default: str = "stdout"):
    """Make sure a task has an emitter with a stdout sink (unless
    ``PBG_EVENT_SINKS`` says otherwise) plus ``<out_dir>/events.jsonl``.

    If the entrypoint (``run_step``) already configured one, its sinks and the
    open task span are kept and only the file sink is added; otherwise this is
    the configuration. Returns the emitter (or the no-op stand-in).
    """
    if _pbg_events is None:
        return _NULL
    try:
        emitter = _pbg_events.get_emitter()
        if not emitter.enabled:
            emitter = _pbg_events.configure(default=default)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "events.jsonl")
            already = any(
                getattr(s, "path", None) == path for s in getattr(emitter, "_sinks", [])
            )
            if not already:
                _add_sink(emitter, _pbg_events.FileSink(path))
        return emitter
    except Exception:
        return _NULL


# ---------------------------------------------------------------------------
# Identity + spans for a LineageProcess
# ---------------------------------------------------------------------------


def identity_from_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if config.get("experiment_id") is not None:
        out["experiment_id"] = str(config["experiment_id"])
    for src, dst in (("variant_index", "variant"), ("lineage_seed", "lineage_seed")):
        if config.get(src) is not None:
            try:
                out[dst] = int(config[src])
            except (TypeError, ValueError):
                out[dst] = config[src]
    return out


def bind_generation(lp):
    """Bind this generation's identity onto the emitter and return it."""
    emitter = get_emitter()
    try:
        identity = identity_from_config(lp.config)
        identity["generation"] = int(lp._generation)
        emitter.bind(**identity)
    except Exception:
        pass
    return emitter


def generation_span(lp):
    """Open the ``generation[g]`` span (child of the task span)."""
    emitter = get_emitter()
    try:
        return emitter.start_span(
            "generation",
            generation=int(lp._generation),
            agent_id=str(lp._agent_id),
            **identity_from_config(lp.config),
        )
    except Exception:
        return _NullSpan()


def emit(name: str, level: str = "info", **payload) -> None:
    """Emit a runner-layer event; never raises."""
    emitter = get_emitter()
    try:
        emitter.event(name, level=level, layer=LAYER, **payload)
    except Exception:
        pass


def events_enabled() -> bool:
    try:
        return bool(get_emitter().enabled)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The carry report (the #765 class, made visible)
# ---------------------------------------------------------------------------


def carry_report(mother_snapshot, carry) -> dict[str, Any]:
    """Classify every agent-root key of the mother snapshot against what the
    carry actually kept.

    ``carried`` = keys in the carry state; ``dropped`` = everything else, split
    into ``non_carried`` (deny-listed per-tick bookkeeping / rebuilt config),
    ``edges`` (process/step nodes) and ``unclassified`` -- a root store that was
    dropped for no declared reason (or carried without being classified). The
    latter is what would have flagged ``request``/``allocate`` before #765
    shipped; the caller emits it at warning level.
    """
    from v2ecoli.library.division import (
        CARRIED_BY_COPY,
        CORE_DIVISIBLE_KEYS,
        NON_CARRIED_ROOT_KEYS,
        STORE_DIVIDERS,
        is_edge_node,
    )

    if not isinstance(mother_snapshot, dict):
        return {"carried": [], "dropped": {"non_carried": [], "edges": [], "unclassified": []}}
    carry = carry if isinstance(carry, dict) else {}
    carried = sorted(k for k in carry if isinstance(k, str) and not k.startswith("_"))
    non_carried, edges, unclassified = [], [], []
    for key in sorted(mother_snapshot):
        if not isinstance(key, str) or key.startswith("_") or key in carry:
            continue
        if key in NON_CARRIED_ROOT_KEYS or key == "listeners":
            non_carried.append(key)
        elif is_edge_node(mother_snapshot[key]):
            edges.append(key)
        else:
            unclassified.append(key)
    known = set(CORE_DIVISIBLE_KEYS) | set(CARRIED_BY_COPY) | set(STORE_DIVIDERS)
    unknown_carried = [k for k in carried if k not in known]
    return {
        "carried": carried,
        "carried_unclassified": unknown_carried,
        "dropped": {"non_carried": non_carried, "edges": edges, "unclassified": unclassified},
    }


# ---------------------------------------------------------------------------
# Observing the (third-party) parquet emitter from outside
# ---------------------------------------------------------------------------


class _ObservedEmitter:
    """Wrap a viva_emitters ``ParquetEmitter`` instance without editing it.

    The composite holds the *inner* instance and calls its ``update`` directly,
    so the wrapper installs an instrumented ``update`` on the instance itself
    (an instance attribute shadows the class method) and otherwise delegates
    every attribute (``num_emits``, ``batch_size``, ``close``, ...). A
    ``chunk_flushed`` event is emitted each time ``num_emits // batch_size``
    advances -- i.e. right after the emitter submitted a chunk write. The
    write itself is asynchronous inside viva_emitters; the event brackets it.
    """

    def __init__(self, inner, emitter=None, *, batch_size: int | None = None):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_emitter", emitter or get_emitter())
        bs = batch_size or getattr(inner, "batch_size", None) or 400
        object.__setattr__(self, "_batch_size", int(bs))
        object.__setattr__(self, "_last_chunk", self._chunk_index())
        object.__setattr__(self, "_orig_update", inner.update)
        try:
            inner.update = self._update  # instance attribute shadows the method
        except Exception:
            pass

    def _chunk_index(self) -> int:
        try:
            return int(getattr(self._inner, "num_emits", 0) or 0) // self._batch_size
        except Exception:
            return 0

    def _update(self, *args, **kwargs):
        t0 = time.monotonic()
        result = self._orig_update(*args, **kwargs)
        try:
            idx = self._chunk_index()
            if idx > self._last_chunk:
                object.__setattr__(self, "_last_chunk", idx)
                self._emitter.event(
                    "chunk_flushed",
                    level="info",
                    layer=LAYER,
                    num_emits=int(getattr(self._inner, "num_emits", 0) or 0),
                    chunk=idx,
                    seconds=round(time.monotonic() - t0, 4),
                )
        except Exception:
            pass
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    @property
    def inner(self):
        return self._inner


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------


def failure_record(exc: BaseException, **extra) -> dict[str, Any]:
    if _pbg_events is not None:
        try:
            return _pbg_events.failure_record(exc, **extra)
        except Exception:
            pass
    import traceback

    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    record: dict[str, Any] = {
        "exc_type": type(exc).__name__,
        "exc_msg": str(exc)[:2000],
        "traceback_tail": "\n".join(text.splitlines()[-40:]),
        "pbg_context": getattr(exc, "pbg_context", None),
    }
    record.update(extra)
    return record


def write_failure_json(out_dir: str, record: dict[str, Any]) -> str | None:
    """Write ``<out_dir>/failure.json``; returns the path or None."""
    if not out_dir:
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "failure.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str)
        return path
    except Exception:
        return None


# Register the S3 sink factory eagerly (belt and braces next to the entry point)
try:  # pragma: no cover - trivial wiring
    if _pbg_events is not None:
        from v2ecoli.workflow.event_sinks import S3JsonlSink

        _pbg_events.register_sink_factory("s3", S3JsonlSink)
except Exception:
    pass
