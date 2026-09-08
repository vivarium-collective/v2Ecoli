"""
Base step class for v2ecoli.

Subclasses define inputs()/outputs() directly using bigraph-schema types.
"""

from __future__ import annotations

import warnings

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate

# Classes that have already had one swallowed-update warning emitted in this
# process. One line per class, not per tick: a per-tick listener that trips on
# missing data would otherwise flood the log, while zero lines is how the
# chain-dispatch batch runner lost whole generations without a trace.
_SWALLOW_WARNED: set[str] = set()


class V2Step(Step):
    """Step base class for v2ecoli.

    Subclasses override inputs()/outputs() with explicit bigraph-schema
    types.  Wraps update() in error handling so missing data doesn't
    crash the Composite's step cascade.

    ``raise_update_errors`` opts a subclass OUT of that swallow. The swallow
    is a per-tick listener convenience: a derived-value step that trips on a
    store that is not seeded yet should skip the tick, not abort the cell. It
    is the wrong default for an ORCHESTRATOR step whose single invocation IS
    the run -- ``BatchBaselineRunner`` dispatches every generation of every
    seed from one ``update()``, so an exception there (a stale ParCa cache, an
    injection-seam error, an S3 write failure) means the composite ran,
    produced nothing, and still reported success: the outer document's only
    artifact is the always-present global_time-only emitter row. That is the
    CD2 chain-dispatch "no emitted output" signature, reproduced locally
    (StaleCacheError inside the batch -> ``Composite.run()`` returned normally
    with an empty ``batch`` store). Such a step sets this True so the failure
    propagates out of ``Composite.run()`` with its real traceback.
    """

    config_schema = {}

    #: When True, an exception raised by ``update()`` propagates instead of
    #: being replaced by an empty update. Set on orchestrator steps.
    raise_update_errors: bool = False

    def inputs(self):
        return {}

    def outputs(self):
        return {}

    def invoke(self, state, interval=None):
        """Override invoke to catch errors from missing data (see class doc)."""
        try:
            update = self.update(state)
        except Exception as exc:
            if self.raise_update_errors:
                raise
            _warn_swallowed(type(self), exc)
            update = {}
        return SyncUpdate(update)


def _warn_swallowed(cls: type, exc: BaseException) -> None:
    """Warn ONCE per class that its update() raised and was replaced by {}.

    Before this, a swallowed exception left no trace at all. A single warning
    per class keeps a per-tick listener from flooding the log while making a
    step that silently does nothing visible in the run log.
    """
    key = f"{cls.__module__}.{cls.__qualname__}"
    if key in _SWALLOW_WARNED:
        return
    _SWALLOW_WARNED.add(key)
    warnings.warn(
        f"{key}.update() raised {type(exc).__name__}: {exc} -- the step's update "
        f"was replaced by {{}} (V2Step swallows update errors unless "
        f"raise_update_errors is set). Further occurrences for this class are "
        f"not reported.",
        RuntimeWarning,
        stacklevel=3,
    )
