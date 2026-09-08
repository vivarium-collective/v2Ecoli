"""The emit path cannot run a composite that emits nothing and call it a success.

Three mechanisms, each reproduced locally against the CD2 dispatch shapes before
being closed here (see the branch's design report):

1. ``V2Step.invoke`` swallowed EVERY exception raised by ``update()``. For the
   chain-dispatch orchestrator (``BatchBaselineRunner``) that meant a
   StaleCacheError / injection-seam error / S3 failure inside the batch left
   ``Composite.run()`` returning normally with an empty ``batch`` store and only
   the outer document's global_time-only emitter row on disk.
2. A ``LineageProcess`` generation could end with its parquet sink having
   received zero rows (or its rows never landing) and still be recorded as a
   completed generation.
3. The parquet emit ROOT SET was a literal repeated at two call sites rather
   than the generator's own ``emitters=[...]`` declaration.

Every test here is fast: no ParCa cache, no simulation.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import fsspec
import pytest

from v2ecoli.composites import _helpers as H
from v2ecoli.steps.base import V2Step
from v2ecoli.workflow.lineage import LineageProcess


# ---------------------------------------------------------------------------
# 1. V2Step: the swallow is opt-out, and never silent
# ---------------------------------------------------------------------------


class _Boom(V2Step):
    def update(self, state, interval=None):
        raise RuntimeError("boom")


class _LoudBoom(_Boom):
    raise_update_errors = True


@pytest.fixture(scope="module")
def core():
    from v2ecoli.core import build_core

    return build_core()


def test_v2step_default_still_swallows_but_warns_once(core):
    """Per-tick listeners keep the historical skip-the-tick behaviour, but a
    swallowed exception is no longer invisible: one RuntimeWarning per class."""
    import v2ecoli.steps.base as base

    base._SWALLOW_WARNED.discard(f"{_Boom.__module__}.{_Boom.__qualname__}")
    step = _Boom({}, core=core)
    with pytest.warns(RuntimeWarning, match="update\\(\\) raised RuntimeError: boom"):
        assert step.invoke({}).get() == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert step.invoke({}).get() == {}  # second time: silent, still swallowed


def test_v2step_raise_update_errors_propagates(core):
    step = _LoudBoom({}, core=core)
    with pytest.raises(RuntimeError, match="boom"):
        step.invoke({})


# ---------------------------------------------------------------------------
# 1b. BatchBaselineRunner: a failed batch is a failed Step, not an empty update
# ---------------------------------------------------------------------------


def _runner(**config):
    from v2ecoli.core import build_core
    from v2ecoli.composites._helpers import _make_instance
    from v2ecoli.steps.batch_baseline_runner import BatchBaselineRunner

    base = {"n_seeds": 1, "n_generations": 1, "parallel": "", "analyses": "none"}
    return _make_instance(BatchBaselineRunner, {**base, **config}, build_core())


def test_batch_runner_reraises_a_failed_dispatch(monkeypatch):
    """FAILS without the fix: invoke() returned SyncUpdate({}) and the composite
    went on to report success with nothing emitted."""
    import v2ecoli.workflow.run as wr

    def _boom(config, **kw):
        raise RuntimeError("StaleCacheError stands in here")

    monkeypatch.setattr(wr, "run_workflow", _boom)
    with pytest.raises(RuntimeError, match="StaleCacheError stands in here"):
        _runner().invoke({"batch": {}})


def test_chain_dispatch_shaped_composite_fails_loud_when_the_batch_fails(monkeypatch, tmp_path):
    """The exact document viva-api's chain dispatch builds (``baseline(n_seeds=1,
    n_generations=1, stop_at_division=True, ...)``), run the way run_pbg runs it
    (``Composite.run(1)``): a failure inside the batch must surface from run().

    Before the fix this returned normally and left only the outer emitter's
    global_time-only ``history/1.pq`` -- the CD2 chain-dispatch artifact."""
    import v2ecoli.workflow.run as wr
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline

    def _boom(config, **kw):
        raise RuntimeError("the batch failed")

    monkeypatch.setattr(wr, "run_workflow", _boom)
    core = build_core()
    doc = baseline(core=core, n_seeds=1, n_generations=1, stop_at_division=True,
                   cache_dir=str(tmp_path / "no-cache"), out_dir=str(tmp_path / "out"),
                   experiment_id="chain", analyses="none", parallel="")
    composite = Composite(doc, core=core)
    with pytest.raises(RuntimeError, match="the batch failed"):
        composite.run(1)


def test_dispatch_batch_refuses_a_batch_in_which_no_seed_reported():
    from v2ecoli.steps.batch_baseline_runner import dispatch_batch

    with pytest.raises(RuntimeError, match="no result for ANY"):
        dispatch_batch(n_seeds=2, n_generations=1, analyses="none",
                       run_workflow_fn=lambda config: {"complete": False, "branches": {}})


def test_dispatch_batch_keeps_a_partial_batch_visible_not_fatal():
    """One seed reporting, one missing: still the existing visible-but-not-fatal
    contract (tests/test_batch_baseline.py pins the per-seed ``error`` entry)."""
    from v2ecoli.steps.batch_baseline_runner import dispatch_batch

    def _one_seed(config):
        return {"complete": True, "branches": {
            "variant=0/seed=0": {"complete": True, "summary": {"generations": [{}]}}}}

    batch = dispatch_batch(n_seeds=2, n_generations=1, analyses="none",
                           run_workflow_fn=_one_seed)
    assert batch["seeds"]["00"]["complete"] is True
    assert batch["seeds"]["01"] == {"error": "run produced no result"}


# ---------------------------------------------------------------------------
# 2. LineageProcess: a generation that emitted nothing is not a completed one
# ---------------------------------------------------------------------------


class _FakeParquetEmitter:
    """What _assert_generation_emitted reads off a real ParquetEmitter."""

    def __init__(self, out_dir, num_emits, *, experiment_id="t",
                 partitioning_path="experiment_id=t/variant=0/lineage_seed=0/generation=0/agent_id=0"):
        self.out_uri = str(out_dir)
        self.filesystem = fsspec.filesystem("file")
        self.num_emits = num_emits
        self.experiment_id = experiment_id
        self.partitioning_path = partitioning_path

    def history_dir(self, tmp_path):
        return tmp_path / self.experiment_id / "history" / self.partitioning_path

    def close(self, success=False):
        pass


def _lineage(monkeypatch, *, emitter="parquet", parquet_em="unset", xarray_emits=0,
             require_output=None):
    """A LineageProcess with the biology stubbed (as tests/test_workflow_lineage.py
    does) whose generation ends on the first update, so the end-of-generation
    check runs against the given emitter bookkeeping."""
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "baseline", "config_overrides": {}, "generations": 1,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 10.0, "emitter": emitter,
        "initial_carry_state_path": "", "initial_generation_index": 0,
        "daughter_state_out_path": "", "checkpoint_dir": "",
    }
    if require_output is not None:
        lp.config["require_output"] = require_output
    lp.initialize(lp.config)

    def fake_build():
        lp._gen_elapsed = 0.0
        if parquet_em != "unset":
            lp._parquet_em = parquet_em
        lp._xarray_emits = xarray_emits

    def fake_run_until_division(interval):
        lp._gen_elapsed += interval
        return True, {"bulk": {}, "unique": {}}, 100.0

    monkeypatch.setattr(lp, "_build_generation", fake_build)
    monkeypatch.setattr(lp, "_run_until_division", fake_run_until_division)
    monkeypatch.setattr(lp, "_finalize_parquet", lambda: None)
    return lp


def test_generation_whose_sink_received_no_rows_fails(monkeypatch, tmp_path):
    """FAILS without the fix: the generation was recorded as complete."""
    lp = _lineage(monkeypatch, parquet_em=_FakeParquetEmitter(tmp_path, num_emits=0))
    with pytest.raises(RuntimeError, match="received 0 rows"):
        lp.update({}, 10.0)


def test_generation_built_without_a_parquet_sink_fails(monkeypatch):
    lp = _lineage(monkeypatch, parquet_em=None)
    with pytest.raises(RuntimeError, match="built WITHOUT the lineage's parquet emitter"):
        lp.update({}, 10.0)


def test_generation_whose_rows_never_landed_fails(monkeypatch, tmp_path):
    """num_emits > 0 but nothing under the history partition: a failed background
    write (or a sink pointed elsewhere) is not a completed generation."""
    em = _FakeParquetEmitter(tmp_path, num_emits=37)
    em.history_dir(tmp_path).mkdir(parents=True)  # partition exists, but is empty
    lp = _lineage(monkeypatch, parquet_em=em)
    with pytest.raises(RuntimeError, match="never reached storage"):
        lp.update({}, 10.0)


def test_generation_with_real_rows_on_disk_completes(monkeypatch, tmp_path):
    em = _FakeParquetEmitter(tmp_path, num_emits=37)
    hist = em.history_dir(tmp_path)
    hist.mkdir(parents=True)
    (hist / "37.pq").write_bytes(b"not-really-parquet-but-nonempty")
    lp = _lineage(monkeypatch, parquet_em=em)
    out = lp.update({}, 10.0)
    assert out["complete"] is True
    assert out["summary"]["generations"][0]["divided"] is True


def test_zero_byte_history_does_not_count(monkeypatch, tmp_path):
    em = _FakeParquetEmitter(tmp_path, num_emits=37)
    hist = em.history_dir(tmp_path)
    hist.mkdir(parents=True)
    (hist / "37.pq").write_bytes(b"")
    lp = _lineage(monkeypatch, parquet_em=em)
    with pytest.raises(RuntimeError, match="never reached storage"):
        lp.update({}, 10.0)


def test_unlistable_destination_warns_instead_of_failing(monkeypatch, tmp_path):
    """'could not look' is not 'no output': an S3 listing that errors must not
    fail a generation whose emitter did receive rows."""
    em = _FakeParquetEmitter(tmp_path, num_emits=5)

    class _BrokenFS:
        def ls(self, *a, **k):
            raise PermissionError("s3 listing denied")

    em.filesystem = _BrokenFS()
    lp = _lineage(monkeypatch, parquet_em=em)
    with pytest.warns(UserWarning, match="UNVERIFIED"):
        out = lp.update({}, 10.0)
    assert out["complete"] is True


def test_xarray_only_generation_with_no_populated_emit_fails(monkeypatch):
    lp = _lineage(monkeypatch, emitter="xarray", xarray_emits=0)
    with pytest.raises(RuntimeError, match="0 populated emits"):
        lp.update({}, 10.0)


def test_xarray_only_generation_with_emits_completes(monkeypatch):
    lp = _lineage(monkeypatch, emitter="xarray", xarray_emits=3)
    assert lp.update({}, 10.0)["complete"] is True


def test_null_emitter_lineage_is_not_checked(monkeypatch):
    """emitter='null' emits nothing BY DESIGN (model browsing, division tests)."""
    lp = _lineage(monkeypatch, emitter="null", parquet_em=None)
    assert lp.update({}, 10.0)["complete"] is True


def test_require_output_false_opts_out(monkeypatch):
    lp = _lineage(monkeypatch, parquet_em=None, require_output=False)
    assert lp.update({}, 10.0)["complete"] is True


def test_require_output_is_on_by_default_in_the_schema():
    assert LineageProcess.config_schema["require_output"]["_default"] is True


# ---------------------------------------------------------------------------
# 3. The parquet emit root set comes from the generator's declaration
# ---------------------------------------------------------------------------


@pytest.fixture
def _decl():
    """Set/restore the generator-declared default emitter around a test."""
    saved = H._DEFAULT_EMITTER_DECL

    def _set(decl):
        H.set_default_emitter_decl(decl)

    yield _set
    H.set_default_emitter_decl(saved)


LISTENERS = {"mass": {"dry_mass": "float"}}


def test_root_set_is_derived_from_the_declared_paths(_decl):
    _decl({"address": "local:ParquetEmitter", "config": {},
           "paths": ["global_time", "bulk", "listeners.mass", "listeners.rna_counts",
                     "compartment/global/volume"]})
    emit_schema, topo = H._parquet_emit_set(LISTENERS)
    assert list(topo) == ["global_time", "bulk", "listeners", "compartment"]
    assert emit_schema["global_time"] == "float"
    assert emit_schema["bulk"] == "array[integer]"     # typed, not a bare node
    assert emit_schema["listeners"] is LISTENERS
    assert emit_schema["compartment"] == "node"        # a declared domain root
    assert topo["compartment"] == ("compartment",)


def test_ecoli_baseline_declaration_yields_todays_exact_set(_decl):
    from viva_superpowers.composite_generator import emitter_defaults
    from v2ecoli.composites.ecoli_baseline import baseline

    _decl(emitter_defaults(baseline)[0])
    emit_schema, topo = H._parquet_emit_set(LISTENERS)
    assert topo == {"global_time": ("global_time",), "bulk": ("bulk",),
                    "listeners": ("listeners",)}
    assert emit_schema == {"global_time": "float", "bulk": "array[integer]",
                           "listeners": LISTENERS}


def test_no_declaration_in_scope_keeps_the_baseline_set(_decl):
    _decl(None)
    _, topo = H._parquet_emit_set(LISTENERS)
    assert list(topo) == ["global_time", "bulk", "listeners"]


def test_a_declared_emitter_with_no_paths_is_refused(_decl):
    """A sink with no declared paths captures only global_time -- the 1-column
    parquet viva-api #475 had to special-case as 'nothing emitted'."""
    _decl({"address": "local:ParquetEmitter", "config": {}, "paths": []})
    with pytest.raises(ValueError, match="declares no emit paths"):
        H._parquet_emit_set(LISTENERS)


def test_explicit_extra_emit_paths_still_win_on_top(_decl):
    _decl(None)
    emit_schema, topo = H._parquet_emit_set(LISTENERS, [["some_store", "sub_key"]])
    assert emit_schema["some_store"] == {"sub_key": "node"}
    assert topo["some_store"] == ("some_store",)
    assert emit_schema["bulk"] == "array[integer]"     # declared set untouched


def test_an_extra_path_naming_a_declared_root_does_not_downgrade_its_schema():
    emit_schema = {"bulk": "array[integer]"}
    topo = {"bulk": ("bulk",)}
    H._merge_emit_paths(emit_schema, topo, [["bulk"]])
    assert emit_schema["bulk"] == "array[integer]"


# ---------------------------------------------------------------------------
# 4. lineage_ray_batch: the document says how long it must be run
# ---------------------------------------------------------------------------


def test_lineage_ray_batch_document_declares_its_required_run_interval():
    from v2ecoli.workflow.batch_lineage_ray import (
        build_lineage_ray_batch_document, required_run_interval)

    doc = build_lineage_ray_batch_document(
        n_seeds=2, n_generations=4, max_duration_per_gen=3600.0, out_dir="o")
    assert doc["required_run_interval"] == 4 * 3600.0
    assert required_run_interval(n_generations=4, max_duration_per_gen=3600.0) == 14400.0
    # every node's interval is one generation: anything shorter invokes nothing
    for name, node in doc["state"].items():
        if name != "lineages":
            assert node["interval"] == 3600.0


def test_composite_run_shorter_than_one_generation_invokes_nothing():
    """The mechanism behind the K4 canary / dispatch 438: with interval =
    max_duration_per_gen, Composite.run(n < interval) never invokes the lineage,
    and the extra top-level key is tolerated by Composite (so a runner can read it)."""
    from process_bigraph import Composite, Process
    from v2ecoli.core import build_core

    class _Counting(Process):
        calls = 0
        config_schema = {}

        def inputs(self):
            return {}

        def outputs(self):
            return {"complete": "boolean"}

        def update(self, state, interval):
            type(self).calls += 1
            return {"complete": True}

    core = build_core()
    core.register_link("_Counting", _Counting)
    doc = {"state": {"lineages": {}, "lineage_0000": {
        "_type": "process", "address": "local:_Counting", "config": {},
        "interval": 3600.0, "inputs": {},
        "outputs": {"complete": ["lineages", "lineage_0000", "complete"]}}},
        "required_run_interval": 3600.0}
    c = Composite(doc, core=core)
    c.run(1)
    assert _Counting.calls == 0 and c.state["global_time"] == 1.0
    c = Composite(doc, core=core)
    c.run(doc["required_run_interval"])
    assert _Counting.calls == 1
