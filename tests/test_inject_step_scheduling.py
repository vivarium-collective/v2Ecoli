"""Regression: a fork vivarium ``Step`` must be injected as a pbg STEP, not an
interval process.

If a vivarium ``Step`` (e.g. ``MetabolismRedux``) is injected as an interval
process, its ``next_update_time`` output (which ``GlobalClock`` minimises over)
is applied a tick late. On tick 2 the clock sees ``next_update_time ==
global_time`` for that process, ``calculate_timestep`` returns ``0``, and
simulation time never advances — the run hangs in ``Composite._run_inner``
(the metabolism_redux tick-2 hang). The fix classifies vivarium Steps as
``as_step=True`` in :func:`scripts._compare.inject._should_inject_as_step`.
"""
from scripts._compare import inject


def test_vivarium_step_subclass_injects_as_step():
    from vivarium.core.process import Step

    class MyDeriver(Step):
        name = "my-deriver"

        def ports_schema(self):
            return {"x": {"_default": 0}}

        def next_update(self, timestep, states):
            return {}

    assert inject._should_inject_as_step(MyDeriver) is True


def test_plain_vivarium_process_stays_a_process():
    from vivarium.core.process import Process

    class MyProcess(Process):
        name = "my-process"

        def ports_schema(self):
            return {"x": {"_default": 0}}

        def next_update(self, timestep, states):
            return {}

    # A genuine interval Process must NOT be forced into a step.
    assert inject._should_inject_as_step(MyProcess) is False


def test_force_step_attribute_wins_without_vivarium():
    # Duck-typed fork class (no vivarium base) with the explicit override.
    class Ducky:
        _force_step = True

    assert inject._should_inject_as_step(Ducky) is True


def test_duck_typed_non_step_is_not_a_step():
    class Ducky:
        name = "ducky"

        def ports_schema(self):
            return {}

        def next_update(self, timestep, states):
            return {}

    assert inject._should_inject_as_step(Ducky) is False


def test_native_deriver_with_update_condition_injects_as_step():
    """Regression (tick-2 non-advancing global_clock collapse): a v2ecoli-NATIVE
    deriver (e.g. native ecoli-metabolism-redux) is a pbg-native process, NOT a
    vivarium.Step subclass — so the old issubclass(cls, Step) test returned False
    and injected it as an interval process, deferring its next_update_time a tick
    and stalling GlobalClock on tick 2. It is update_condition-gated, so it must
    be classified as a step by that signature, independent of vivarium."""
    class NativeRedux:
        name = "ecoli-metabolism-redux"

        def inputs(self):
            return {}

        def outputs(self):
            return {}

        def update_condition(self, timestep, states):
            return True

    assert inject._should_inject_as_step(NativeRedux) is True


def test_update_condition_detection_needs_no_vivarium(monkeypatch):
    """The update_condition signal must win before the vivarium import, so a
    native/vecoli-free image (no vivarium installed) still classifies a deriver
    as a step instead of falling through to the import-failure `return False`."""
    import builtins
    real_import = builtins.__import__

    def _no_vivarium(name, *a, **k):
        if name.startswith("vivarium"):
            raise ImportError("simulated vivarium-free image")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_vivarium)

    class NativeRedux:
        name = "ecoli-metabolism-redux"
        def inputs(self):
            return {}
        def outputs(self):
            return {}
        def update_condition(self, timestep, states):
            return True

    assert inject._should_inject_as_step(NativeRedux) is True
