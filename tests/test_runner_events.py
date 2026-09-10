"""Runner-layer observability (docs/plan-observability.md, D3).

Every test here reads the JSON-lines event stream the runner emits and asserts
on the *events* -- the contract a consumer (viva-api's ingester, `atlantis
simulation events`) will read -- not on log text. Events go to a stdout sink and
are parsed from ``capsys``.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.fast

pbg_events = pytest.importorskip(
    "process_bigraph.events", reason="process-bigraph >= 1.9 (feat/events) required"
)

from v2ecoli.workflow import events as revents  # noqa: E402
from v2ecoli.workflow.lineage import LineageProcess, _derive_generation_seed  # noqa: E402


def _ident(e, key):
    """Domain identity lives in the engine's opaque ``baggage`` map (the engine
    schema has no v2ecoli fields); older engine builds carried the same keys at
    the top level. Read either."""
    bag = e.get("baggage")
    if isinstance(bag, dict) and key in bag:
        return str(bag[key])            # strings on the wire, by design
    v = e.get(key)
    return None if v is None else str(v)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def stdout_events(monkeypatch, capsys):
    """Configure a stdout-only emitter for the test and return a reader that
    parses every JSON line printed so far."""
    for key in ("PBG_EVENT_SINKS", "PBG_TRACEPARENT", "PBG_TRACE_BAGGAGE", "PBG_EVENT_TAGS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PBG_EVENT_HEARTBEAT_S", "0")
    pbg_events.configure("stdout")

    def read():
        out = capsys.readouterr().out
        events = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events

    yield read
    pbg_events.set_emitter(None)


def _make(monkeypatch, generations, divide_after=2, **cfg):
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 3, "lineage_seed": 7, "variant_index": 2,
        "variant_name": "baseline", "config_overrides": {}, "generations": generations,
        "single_daughters": True, "experiment_id": "exp-t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0, "initial_carry_state_path": "",
        "initial_generation_index": 0, "daughter_state_out_path": "",
        "checkpoint_dir": "", "require_output": False,
    }
    lp.config.update(cfg)
    lp.initialize(lp.config)

    def fake_build():
        # what the real _build_generation does at its seam boundaries
        revents.bind_generation(lp)
        lp._gen_span = revents.generation_span(lp)
        lp._gen_elapsed = 0.0
        gen_seed = _derive_generation_seed(lp.config["seed"], lp.config["lineage_seed"], lp._generation)
        revents.emit("lineage.generation.start", generation=lp._generation, gen_seed=gen_seed,
                     lineage_offset=float(lp._lineage_offset))

    def fake_run(interval):
        lp._gen_elapsed += interval
        divided = lp._gen_elapsed >= divide_after
        daughter = {"bulk": {}, "unique": {}} if divided else None
        return divided, daughter, 100.0 + lp._generation

    monkeypatch.setattr(lp, "_build_generation", fake_build)
    monkeypatch.setattr(lp, "_run_until_division", fake_run)
    return lp


# ---------------------------------------------------------------------------
# generation_start / generation_end
# ---------------------------------------------------------------------------


def test_one_generation_start_and_end_per_generation(monkeypatch, stdout_events):
    lp = _make(monkeypatch, generations=3, divide_after=2)
    for _ in range(20):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break
    events = stdout_events()
    starts = [e for e in events if e["event"] == "lineage.generation.start"]
    ends = [e for e in events if e["event"] == "lineage.generation.end"]
    assert [e["payload"]["generation"] for e in starts] == [0, 1, 2]
    assert [e["payload"]["generation"] for e in ends] == [0, 1, 2]
    # identity bound by the runner, not by any dispatcher env
    assert all((e.get("component") or e.get("layer")) == "v2ecoli.lineage" for e in starts + ends)
    assert all(_ident(e, "variant") == "2" and _ident(e, "lineage_seed") == "7" for e in starts + ends)
    assert all(_ident(e, "experiment_id") == "exp-t" for e in starts + ends)
    # the seed is the real one (#766's combiner), not the base seed
    for e in starts:
        g = e["payload"]["generation"]
        assert e["payload"]["gen_seed"] == _derive_generation_seed(3, 7, g)
    # duration is the booked elapsed time (the #767/#771 number), never 0
    assert [e["payload"]["duration"] for e in ends] == [2.0, 2.0, 2.0]
    assert [round(e["payload"]["lineage_offset_after"]) for e in ends] == [2, 4, 6]
    # the generation spans opened and closed, nested under a common trace
    span_ends = [e for e in events if e["event"] in ("span_end", "span.end") and e["payload"]["name"] == "generation"]
    assert len(span_ends) == 3
    assert len({e["trace_id"] for e in events}) == 1


def test_events_off_keeps_the_legacy_log_lines(monkeypatch, capsys):
    pbg_events.set_emitter(None)
    monkeypatch.delenv("PBG_EVENT_SINKS", raising=False)
    lp = _make(monkeypatch, generations=1, divide_after=1)
    lp.update({}, 1.0)
    out = capsys.readouterr().out
    assert "[LineageProcess] gen 0: end" in out
    assert "emitters flushed" in out
    assert not [ln for ln in out.splitlines() if ln.startswith("{")]


# ---------------------------------------------------------------------------
# the division event and its carry report
# ---------------------------------------------------------------------------


def test_carry_report_classifies_every_root():
    from v2ecoli.library.division import NON_CARRIED_ROOT_KEYS

    mother = {
        "bulk": "M", "unique": {}, "environment": {}, "boundary": {},
        "request": {"p": {"bulk": [[1, 2]]}},          # non-carried (the #765 class)
        "listeners": {"mass": {}},                     # never wholesale
        "lineage.division": {"address": "local:Division"},     # an edge
        "fields": {"drug": 1.0},                       # carried by copy
        "mystery_root": {"x": 1},                      # nobody classified this
    }
    carry = {"bulk": "D0", "unique": {}, "environment": {}, "boundary": {}, "fields": {"drug": 1.0}}
    report = revents.carry_report(mother, carry)
    assert "request" in NON_CARRIED_ROOT_KEYS
    assert report["carried"] == ["boundary", "bulk", "environment", "fields", "unique"]
    assert report["dropped"]["non_carried"] == ["listeners", "request"]
    assert report["dropped"]["edges"] == ["lineage.division"]
    assert report["dropped"]["unclassified"] == ["mystery_root"]
    assert report["carried_unclassified"] == []


def test_division_event_reports_signal_and_report(monkeypatch, stdout_events):
    """Drive the REAL _run_until_division with a fake inner composite whose
    agents map changes (structural division)."""
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 1, "variant_index": 0,
        "variant_name": "b", "config_overrides": {}, "generations": 2,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0, "initial_carry_state_path": "",
        "initial_generation_index": 0, "daughter_state_out_path": "",
        "checkpoint_dir": "", "require_output": False, "emitter": "parquet",
    }
    lp.initialize(lp.config)

    mother = {
        "bulk": "M", "unique": {}, "environment": {}, "boundary": {},
        "request": {"p": {"bulk": []}}, "fields": {"drug": 2.0},
        "listeners": {"mass": {"dry_mass": 500.0}},
        "lineage.division": {"address": "local:Division"},
    }

    class _FakeComposite:
        def __init__(self):
            self.state = {"global_time": 0.0, "agents": {"0": mother}}

        def run(self, interval):
            # NB: no ``global_time`` stamp on the daughters -- the real Division
            # step rebuilds them from baseline() with 0.0, which #767 mistook for
            # a division time (#771 ignores a non-advancing stamp). Leaving it
            # out makes this test valid on both sides of #771.
            d = {"bulk": "D0", "unique": {}, "environment": {}, "boundary": {},
                 "request": {}, "fields": {"drug": 0.0},
                 "listeners": {"mass": {"dry_mass": 250.0}}}
            self.state = {"global_time": 42.0, "agents": {"00": d, "01": dict(d)}}

    lp._composite = _FakeComposite()
    lp._gen_elapsed = 0.0
    divided, daughter, _ = lp._run_until_division(100.0)
    assert divided and daughter is not None
    assert "request" not in daughter and daughter["fields"] == {"drug": 2.0}

    events = stdout_events()
    div = [e for e in events if e["event"] == "lineage.division"]
    assert len(div) == 1
    p = div[0]["payload"]
    assert p["signal"] == "structural"
    assert p["t_division"] == 42.0            # the inner clock (#771), not the window
    assert p["carried"] == ["boundary", "bulk", "environment", "fields", "unique"]
    assert p["dropped"]["non_carried"] == ["listeners", "request"]
    assert p["dropped"]["edges"] == ["lineage.division"]
    assert p["dropped"]["unclassified"] == []
    assert div[0]["level"] == "info"
    assert lp._last_carry_report["carried"] == p["carried"]


def test_division_with_an_unclassified_root_is_a_warning(monkeypatch, stdout_events):
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "b", "config_overrides": {}, "generations": 2,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0, "initial_carry_state_path": "",
        "initial_generation_index": 0, "daughter_state_out_path": "",
        "checkpoint_dir": "", "require_output": False,
    }
    lp.initialize(lp.config)
    mother = {"bulk": "M", "unique": {}, "environment": {}, "boundary": {},
              "not_a_known_root": {"x": 1}, "listeners": {"mass": {"dry_mass": 1.0}}}

    class _C:
        state = {"global_time": 0.0, "agents": {"0": mother}}

        def run(self, interval):
            self.state = {"global_time": 5.0, "agents": {"00": {"bulk": "D", "unique": {},
                          "environment": {}, "boundary": {}, "listeners": {"mass": {"dry_mass": 1.0}}}}}

    lp._composite = _C()
    lp._gen_elapsed = 0.0
    lp._run_until_division(100.0)
    div = [e for e in stdout_events() if e["event"] == "lineage.division"][0]
    # the policy COPIED it (extras are copied by default) but nothing classified it
    assert div["level"] == "warning"
    assert div["payload"]["carried_unclassified"] == ["not_a_known_root"]


# ---------------------------------------------------------------------------
# chunk_flushed from outside the third-party emitter
# ---------------------------------------------------------------------------


def test_observed_emitter_emits_chunk_flushed_on_batch_boundaries(stdout_events):
    class _Stub:
        batch_size = 4
        num_emits = 0
        out_uri = "x"

        def update(self, state):
            self.num_emits += 1
            return {}

    inner = _Stub()
    wrapped = revents._ObservedEmitter(inner, batch_size=4)
    for _ in range(9):
        inner.update({})          # the composite calls the INNER instance's update
    assert wrapped.num_emits == 9  # delegation
    chunks = [e for e in stdout_events() if e["event"] == "lineage.chunk.flushed"]
    assert [c["payload"]["chunk"] for c in chunks] == [1, 2]
    assert [c["payload"]["num_emits"] for c in chunks] == [4, 8]


# ---------------------------------------------------------------------------
# the S3 sink plugin, on a local fsspec target
# ---------------------------------------------------------------------------


def test_s3_jsonl_sink_rewrites_one_object_per_writer(tmp_path):
    from v2ecoli.workflow.event_sinks import S3JsonlSink

    sink = S3JsonlSink(f"file://{tmp_path}/events", flush_s=0, source="host-1")
    sink.emit({"event": "a", "trace_id": "abc123"})
    sink.emit({"event": "b", "trace_id": "abc123"})
    assert sink.key == f"file://{tmp_path}/events/abc123/host-1.jsonl"
    sink.flush()
    path = tmp_path / "events" / "abc123" / "host-1.jsonl"
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == ["a", "b"]
    sink.emit({"event": "c", "trace_id": "abc123"})
    sink.close()
    assert [json.loads(line)["event"] for line in path.read_text().splitlines()] == ["a", "b", "c"]
    assert sink.flush_count == 2 and sink.last_error is None


def test_s3_sink_resolves_from_the_engine_registry(tmp_path):
    import v2ecoli.workflow.events  # noqa: F401 -- registers the factory

    sink = pbg_events.resolve_sink(f"file://{tmp_path}/x")  # engine's own file: handles file:
    assert sink is not None
    s3 = pbg_events.resolve_sink("s3://bucket/prefix/")
    from v2ecoli.workflow.event_sinks import S3JsonlSink

    assert isinstance(s3, S3JsonlSink)
    assert s3.uri == "s3://bucket/prefix"


def test_s3_sink_write_failure_never_raises(tmp_path):
    from v2ecoli.workflow.event_sinks import S3JsonlSink

    sink = S3JsonlSink("bogus-scheme://nowhere/x", flush_s=0)
    sink.emit({"event": "a", "trace_id": "t"})
    sink.flush()          # unknown protocol -> recorded, not raised
    assert sink.last_error is not None
    assert sink.flush_count == 0


# ---------------------------------------------------------------------------
# time_step reaches the inner baseline
# ---------------------------------------------------------------------------


def test_time_step_is_forwarded_to_the_inner_baseline(monkeypatch):
    import v2ecoli.composites.ecoli_baseline as eb

    captured = {}

    def fake_baseline(core=None, seed=None, **kwargs):
        captured.update(kwargs)
        return {"state": {"agents": {"0": {"listeners": {"mass": {}}}}}}

    class _FakeComposite:
        def __init__(self, doc, core=None):
            self.state = doc["state"]

    monkeypatch.setattr(eb, "baseline", fake_baseline)
    monkeypatch.setattr(eb, "seed_mass_listener", lambda agent, core: None)
    import process_bigraph

    monkeypatch.setattr(process_bigraph, "Composite", _FakeComposite)
    from v2ecoli.composites import _helpers

    monkeypatch.setattr(_helpers, "set_null_emitter_override", lambda v: None)

    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "b", "config_overrides": {}, "generations": 1,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0, "initial_carry_state_path": "",
        "initial_generation_index": 0, "daughter_state_out_path": "",
        "checkpoint_dir": "", "require_output": False, "emitter": "xarray",
        "time_step": 2.5,
    }
    lp.initialize(lp.config)
    monkeypatch.setattr("v2ecoli.core.build_core", lambda: object())
    lp._build_generation()
    assert captured["time_step"] == 2.5


# ---------------------------------------------------------------------------
# no engine: everything degrades to a no-op
# ---------------------------------------------------------------------------


def test_runner_helpers_are_noops_without_the_engine(monkeypatch):
    monkeypatch.setattr(revents, "_pbg_events", None)
    em = revents.get_emitter()
    assert em.enabled is False
    em.bind(generation=1)
    span = em.start_span("x")
    span.end()
    revents.emit("anything", a=1)
    assert revents.events_enabled() is False
    assert revents.configure_for_task("/tmp/does-not-matter").enabled is False


def test_report_never_names_agent_ids_on_a_real_shaped_agents_map(monkeypatch, stdout_events):
    """The report is computed on the CELL (mother node / carry dict), never on
    the agents map: with agents {"0"} -> {"00", "01"} no agent id may appear
    anywhere in it (sim 956 review question)."""
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "b", "config_overrides": {}, "generations": 2,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0, "initial_carry_state_path": "",
        "initial_generation_index": 0, "daughter_state_out_path": "",
        "checkpoint_dir": "", "require_output": False,
    }
    lp.initialize(lp.config)
    cell = {"bulk": "M", "unique": {}, "environment": {}, "boundary": {}, "fields": {"d": 1.0},
            "listeners": {"mass": {"dry_mass": 1.0}}, "division": {"address": "local:Division"}}

    class _C:
        state = {"global_time": 0.0, "agents": {"0": cell}}

        def run(self, interval):
            d = {"bulk": "D", "unique": {}, "environment": {}, "boundary": {},
                 "listeners": {"mass": {"dry_mass": 0.5}}}
            self.state = {"global_time": 7.0, "agents": {"00": d, "01": dict(d)}}

    lp._composite = _C()
    lp._gen_elapsed = 0.0
    lp._run_until_division(100.0)
    div = [e for e in stdout_events() if e["event"] == "lineage.division"][0]["payload"]
    named = set(div["carried"]) | set(div["carried_unclassified"]) | set(div["daughter_keys"])
    for bucket in div["dropped"].values():
        named |= set(bucket)
    assert not named & {"0", "00", "01"}, named
    assert div["carried"] == ["boundary", "bulk", "environment", "fields", "unique"]


def test_a_root_store_literally_named_0_is_reported_with_a_summary():
    """sim 956 (2026-09-10): an injected composite carried an agent-root store
    named "0". That is a finding about the composite, not a bug in the report --
    the report says so and describes what the store is."""
    mother = {"bulk": "M", "unique": {}, "environment": {}, "boundary": {},
              "0": {"volume": 1.0, "counts": {}}, "listeners": {}}
    carry = {"bulk": "D", "unique": {}, "environment": {}, "boundary": {}, "0": {"volume": 1.0, "counts": {}}}
    report = revents.carry_report(mother, carry)
    assert report["carried_unclassified"] == ["0"]
    assert report["unclassified_summary"]["0"] == {"type": "dict", "n_keys": 2, "keys": ["volume", "counts"], "is_edge": False}


def test_downstream_can_register_its_copied_roots(monkeypatch):
    from v2ecoli.library import division as div

    monkeypatch.setattr(div, "CARRIED_BY_COPY_REGISTERED", set())
    mother = {"bulk": "M", "unique": {}, "environment": {}, "boundary": {}, "kinetic_parameters": {"k": 1}}
    carry = dict(mother)
    assert revents.carry_report(mother, carry)["carried_unclassified"] == ["kinetic_parameters"]
    div.register_carried_by_copy("kinetic_parameters")
    assert revents.carry_report(mother, carry)["carried_unclassified"] == []
    with pytest.raises(TypeError):
        div.register_carried_by_copy("")
