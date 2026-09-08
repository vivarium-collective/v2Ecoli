"""LineageStep: a whole lineage as ONE atomic task.

Each test here pins a property that, if it regressed, would fail SILENTLY --
producing a task that reports success while running the wrong thing. That is the
failure mode this wrapper exists to prevent, so the tests assert effects rather
than shapes.
"""

from __future__ import annotations

import pytest

from process_bigraph import Process

from v2ecoli.workflow.lineage_step import LineageStep


class _Recorded(LineageStep):
    """LineageStep with the biology stubbed: records what it WOULD have run."""

    def _run_lineage(self, config, interval):  # type: ignore[override]
        self.recorded = {"config": config, "interval": interval}


@pytest.fixture(scope="module")
def core():
    from v2ecoli.core import build_core
    from v2ecoli.workflow.meta_composite import register_workflow_processes

    c = build_core()
    register_workflow_processes(c)
    return c


def _step(core, **config):
    step = _Recorded(config=config, core=core)
    step.recorded = None
    return step


def test_runs_every_generation_in_one_invocation(core) -> None:
    """The reason this class exists. run_step invokes a Step ONCE, while
    LineageProcess advances one generation per update() -- so a task node must
    drive the whole lineage itself or it silently stops after generation 0."""
    step = _step(core, generations=8, max_duration_per_gen=3600.0, out_dir="", require_output=False)
    step.update({"cache_dir": "/staged/cache"})
    assert step.recorded["interval"] == 8 * 3600.0


def test_cache_dir_arrives_as_a_wire_and_the_input_wins(core) -> None:
    """cache_dir is an INPUT, not config: that is what makes ParCa -> lineage an
    edge a DAG engine can stage. A staged path must beat a stale configured one."""
    step = _step(core, cache_dir="/configured", out_dir="", require_output=False)
    step.update({"cache_dir": "/staged"})
    assert step.recorded["config"]["cache_dir"] == "/staged"


def test_missing_cache_dir_fails_loudly(core) -> None:
    """A lineage with no ParCa bundle has nothing to simulate. Failing here beats
    running with a default and emitting plausible-looking wrong biology."""
    with pytest.raises(ValueError, match="cache_dir"):
        _step(core, out_dir="", require_output=False).update({})


def test_declares_cache_dir_as_a_staged_file(core) -> None:
    assert LineageStep(core=core).inputs()["cache_dir"]["_is_file"] is True
    assert LineageStep(core=core).outputs()["sweep_dir"]["_is_file"] is True


def test_emits_one_result_port_not_loop_bookkeeping(core) -> None:
    """LineageProcess emits summary + complete; `complete` is loop state, not a
    result, and a task's interface should not expose it."""
    assert set(LineageStep(core=core).outputs()) == {"sweep_dir"}


# --- separability: a task reads its own config and nothing else -------------


def test_empty_swap_blocks_are_omitted_entirely(core) -> None:
    """'no swap requested' and 'swap requested, empty' are different downstream:
    an empty injected_processes is a config-less swap target (v2ecoli#682 now
    fails loud on it), and an empty override block is the shape that REPLACED a
    config's flat fields in viva-api#401. Absent is the honest encoding."""
    step = _step(core, out_dir="", require_output=False)
    step.update({"cache_dir": "/c"})
    assert "injected_processes" not in step.recorded["config"]
    assert "config_overrides" not in step.recorded["config"]


def test_a_real_swap_is_forwarded_verbatim(core) -> None:
    swap = {"ecoli-metabolism": {"process": "ecoli-metabolism-redux", "cache_dir": "/c"}}
    step = _step(core, injected_processes=swap, out_dir="", require_output=False)
    step.update({"cache_dir": "/c"})
    assert step.recorded["config"]["injected_processes"] == swap


def test_per_task_config_carries_the_variant_identity(core) -> None:
    """An N x M sweep is expressed by WHICH CONFIGS EXIST, not by a renderer
    feature -- so variant identity has to survive into the task's own config."""
    step = _step(core, seed=3, variant_index=1, variant_name="redux", out_dir="", require_output=False)
    step.update({"cache_dir": "/c"})
    cfg = step.recorded["config"]
    assert (cfg["seed"], cfg["variant_index"], cfg["variant_name"]) == (3, 1, "redux")


# --- go/no-go 6: a task that emits nothing must FAIL ------------------------


def test_task_that_emitted_nothing_fails(core, tmp_path) -> None:
    """run_composite ships no emitted-output guard, so without this a DAG-engine
    campaign can go green having produced no science."""
    step = _step(core, out_dir=str(tmp_path))
    with pytest.raises(SystemExit, match="no emitted output"):
        step.update({"cache_dir": "/c"})


def test_task_with_real_parquet_succeeds(core, tmp_path) -> None:
    part = tmp_path / "history" / "gen0"
    part.mkdir(parents=True)
    (part / "400.pq").write_bytes(b"not-really-parquet-but-nonempty")
    step = _step(core, out_dir=str(tmp_path))
    assert step.update({"cache_dir": "/c"}) == {"sweep_dir": str(tmp_path)}


def test_zero_byte_parquet_does_not_count(core, tmp_path) -> None:
    """An empty file is the artifact of a failed write, not evidence of output."""
    (tmp_path / "empty.pq").write_bytes(b"")
    with pytest.raises(SystemExit):
        _step(core, out_dir=str(tmp_path)).update({"cache_dir": "/c"})


# --- integration: the ONE property this class exists to guarantee -----------
#
# Every test above stubs _run_lineage, so none of them exercise the real
# Composite.run path. That gap hid an interval bug (@eagmon, review of #694):
# a process node with no `interval` defaults to 1.0, so Composite.run(total)
# ticks once per SIMULATED SECOND rather than once per generation. The result
# stayed correct because LineageProcess self-limits, which is exactly why a
# stub-only suite could not see it.


class _CountingLineage(Process):
    """Stands in for LineageProcess: counts update() calls, self-limits."""

    calls = 0
    intervals: list = []

    config_schema = {
        "generations": {"_type": "integer", "_default": 1},
        "cache_dir": {"_type": "string", "_default": ""},
        "seed": {"_type": "integer", "_default": 0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._generation = 0

    def inputs(self):
        return {}

    def outputs(self):
        return {"summary": "map", "complete": "boolean"}

    def update(self, state, interval):
        type(self).calls += 1
        type(self).intervals.append(interval)
        self._generation += 1
        if self._generation >= int(self.config.get("generations", 1)):
            return {"complete": True, "summary": {"generations": self._generation}}
        return {"complete": False, "summary": {"generations": self._generation}}


def test_one_invoke_drives_every_generation_through_composite_run(core, monkeypatch, tmp_path) -> None:
    """The closed loop: a REAL _run_lineage, a real Composite.run, and a fake
    lineage that counts. Asserts both halves of the contract -- every generation
    runs, and it costs one outer tick per generation rather than one per second."""
    import v2ecoli.workflow.lineage as lineage_mod

    _CountingLineage.calls = 0
    _CountingLineage.intervals = []
    monkeypatch.setattr(lineage_mod, "LineageProcess", _CountingLineage)

    (tmp_path / "history").mkdir()
    (tmp_path / "history" / "0.pq").write_bytes(b"nonempty")

    step = LineageStep(config={
        "generations": 3,
        "max_duration_per_gen": 600.0,
        "out_dir": str(tmp_path),
    }, core=core)
    result = step.update({"cache_dir": "/staged/cache"})

    assert result == {"sweep_dir": str(tmp_path)}
    # every generation ran -- not just the first (the silent truncation this
    # whole wrapper exists to prevent)
    assert _CountingLineage.calls == 3
    # and each tick advanced a GENERATION, not a second: with interval=1.0 this
    # would have been 3 * 600 = 1800 calls
    assert set(_CountingLineage.intervals) == {600.0}
