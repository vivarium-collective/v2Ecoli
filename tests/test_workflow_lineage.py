import pytest
from v2ecoli.workflow.lineage import LineageProcess


def _make(monkeypatch, generations, divide_after=2, **wave_kwargs):
    """Build a LineageProcess whose _build_generation/_run_until_division are
    stubbed so we can test generation counting without a real cell composite.

    ``wave_kwargs`` accepts the per-generation checkpoint/resume keys
    (initial_carry_state_path / initial_generation_index / daughter_state_out_path,
    backlog item 34; checkpoint_dir, item 115) -- omitted, they default to
    "" / 0 / "" / "", i.e. today's unchanged single-invocation-runs-every-
    generation behavior.
    """
    lp = LineageProcess.__new__(LineageProcess)
    # Minimal config + state normally set by Process.__init__/initialize.
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "baseline", "config_overrides": {}, "generations": generations,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0,
        "initial_carry_state_path": wave_kwargs.get("initial_carry_state_path", ""),
        "initial_generation_index": wave_kwargs.get("initial_generation_index", 0),
        "daughter_state_out_path": wave_kwargs.get("daughter_state_out_path", ""),
        "checkpoint_dir": wave_kwargs.get("checkpoint_dir", ""),
        # The biology is stubbed below, so no emitter is ever built: opt out of
        # the end-of-generation emitted-output check (tests/test_emit_path_robustness.py
        # covers it against a real emitter).
        "require_output": False,
    }
    lp.initialize(lp.config)
    calls = {"built": 0}

    def fake_build():
        calls["built"] += 1
        lp._gen_elapsed = 0.0

    def fake_run_until_division(interval):
        lp._gen_elapsed += interval
        divided = lp._gen_elapsed >= divide_after
        daughter = {"bulk": {}, "unique": {}} if divided else None
        return divided, daughter, 100.0 + lp._generation

    monkeypatch.setattr(lp, "_build_generation", fake_build)
    monkeypatch.setattr(lp, "_run_until_division", fake_run_until_division)
    return lp, calls


def test_completes_after_generations(monkeypatch):
    lp, calls = _make(monkeypatch, generations=3, divide_after=2)
    out = {}
    for _ in range(20):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break
    assert out["complete"] is True
    assert len(lp._summaries) == 3            # 3 generations recorded
    assert [s["generation"] for s in lp._summaries] == [0, 1, 2]
    # agent_id advanced one phylogeny step per completed generation: 0 -> 00 -> 000
    assert [s["agent_id"] for s in lp._summaries] == ["0", "00", "000"]
    assert all(s["divided"] for s in lp._summaries)


def test_lineage_drives_one_advance_generation_emitter(monkeypatch):
    """The whole lineage is driven by ONE XArrayEmitter: at each generation
    boundary it is ADVANCED in place (advance_generation) to the next
    generation's partition, not closed and rebuilt; only the LAST generation
    closes it. This is what keeps a single emitter (and its per-generation
    flush+consolidate durability) across the lineage -- Eran's "same emitter,
    launch a new internal ecoli model per generation"."""
    from unittest.mock import MagicMock

    lp, _calls = _make(monkeypatch, generations=3, divide_after=2)
    # The stubbed biology never builds an emitter; inject one and force the
    # xarray branch so we can observe the generation-boundary handling.
    monkeypatch.setattr(lp, "_is_xarray", lambda: True)
    monkeypatch.setattr(lp, "_is_parquet", lambda: False)
    em = MagicMock()
    lp._xarray_em = em

    out = {}
    for _ in range(30):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break
    assert out["complete"] is True

    # advance_generation at each NON-final boundary, walking the phylogeny
    # (0 -> 00 -> 000), success=True; never a per-generation close before the end.
    advanced = [c.kwargs["agent_id"] for c in em.advance_generation.call_args_list]
    assert advanced == ["00", "000"]
    assert all(c.kwargs.get("success") is True for c in em.advance_generation.call_args_list)
    # The LAST generation closes the one emitter exactly once, and nulls it.
    em.close.assert_called_once_with(success=True)
    assert lp._xarray_em is None


def test_single_daughters_false_not_implemented(monkeypatch):
    lp, _ = _make(monkeypatch, generations=2)
    lp.config["single_daughters"] = False
    with pytest.raises(NotImplementedError):
        lp.update({}, 1.0)


def test_daughter_carry_forward_orchestration(monkeypatch):
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "baseline", "config_overrides": {}, "generations": 2,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0,
        "require_output": False,  # biology stubbed: no emitter is built
    }
    lp.initialize(lp.config)

    builds = []  # (generation, agent_id, carry_state) seen at each build

    def fake_build():
        builds.append((lp._generation, lp._agent_id, lp._carry_state))
        lp._gen_elapsed = 0.0

    daughter = {"bulk": {"marker": 1}, "unique": {}}

    def fake_run_until_division(interval):
        lp._gen_elapsed += interval
        # Always "divide" after one tick, handing back a synthetic daughter.
        return True, daughter, 100.0

    monkeypatch.setattr(lp, "_build_generation", fake_build)
    monkeypatch.setattr(lp, "_run_until_division", fake_run_until_division)

    out = {}
    for _ in range(10):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break

    assert out["complete"] is True
    # Generation 0 built with no carry; generation 1 built carrying the daughter.
    assert len(builds) == 2
    assert builds[0] == (0, "0", None)
    assert builds[1][0] == 1
    assert builds[1][1] == "00"            # agent_id advanced via daughter_phylogeny_id
    assert builds[1][2] is daughter        # carry_state handed to the next build


def test_select_carry_daughter_uses_inner_daughter_unchanged():
    """Regression: when the inner Division step has already produced daughters
    (…0 / …1), carry the …0 daughter's state DIRECTLY — do NOT re-divide it
    (re-dividing an already-divided daughter yielded quarter-mass cells)."""
    from v2ecoli.workflow.lineage import select_carry_daughter

    bulk00 = ["sentinel-bulk-00"]          # identity-checked: must pass through
    agents_now = {
        "00": {"bulk": bulk00, "unique": {"u": 1}, "environment": {"e": 2}, "boundary": {}},
        "01": {"bulk": ["other"], "unique": {}, "environment": {}, "boundary": {}},
    }
    carry = select_carry_daughter({"0"}, agents_now, mother_snapshot=None)
    assert carry["bulk"] is bulk00         # the …0 daughter's bulk, unmodified
    assert carry["unique"] == {"u": 1}
    assert carry["environment"] == {"e": 2}


def test_select_carry_daughter_fallback_divides_mother_once(monkeypatch):
    """When no structural daughter surfaced (divide-flag / exception signal,
    agents map unchanged), fall back to dividing the pre-run mother snapshot
    exactly ONCE."""
    import v2ecoli.library.division as division_mod
    calls = []

    def fake_divide(cell_data):
        calls.append(cell_data)
        return {"bulk": "D1", "unique": {}}, {"bulk": "D2", "unique": {}}

    monkeypatch.setattr(division_mod, "divide_cell", fake_divide)
    from v2ecoli.workflow.lineage import select_carry_daughter

    mother = {"bulk": "MOTHER_BULK", "unique": {}, "environment": {}, "boundary": {}}
    carry = select_carry_daughter({"0"}, {"0": {}}, mother_snapshot=mother)
    assert len(calls) == 1                  # divided exactly once
    assert calls[0] == mother               # divided the MOTHER snapshot
    assert carry["bulk"] == "D1"


def test_select_carry_daughter_none_when_nothing_to_carry():
    from v2ecoli.workflow.lineage import select_carry_daughter
    assert select_carry_daughter({"0"}, {"0": {}}, mother_snapshot=None) is None


def test_apply_carry_state_preserves_fresh_exchange_data():
    """Regression: the rebuilt daughter must take ``environment.exchange_data``
    from its FRESH build, not inherit the mother's raw substore.

    Carrying the mother's raw ``exchange_data`` dict drops the store's overwrite
    (ListenerStore) updater, so the daughter falls back to ``map[float]`` ACCUMULATE
    semantics: the per-tick FBA bound write (ExchangeData's glucose uptake = cap)
    then ADDS up instead of overwriting, ballooning the bound across the generation
    and silently voiding every exchange constraint in gens >= 1. The biological
    state still comes from the daughter; only the derived substore is kept fresh."""
    from v2ecoli.workflow.lineage import apply_carry_state

    fresh_exchange_data = {"constrained": {"GLC[p]": 20.0}, "unconstrained": []}
    agent = {
        "bulk": "FRESH_BULK",
        "unique": {"u": "fresh"},
        "environment": {"media_id": "FRESH", "exchange_data": fresh_exchange_data},
        "boundary": {"external": {"GLC": 11.1}},
        "listeners": {"mass": {}},
    }
    carry_state = {
        "bulk": "CARRIED_BULK",
        "unique": {"u": "carried"},
        "environment": {
            "media_id": "CARRIED",
            # a mother store whose per-tick bound has already ballooned
            "exchange_data": {"constrained": {"GLC[p]": 9999.0}, "unconstrained": ["GLC[p]"]},
        },
        "boundary": {"external": {"GLC": 5.0}},
    }
    apply_carry_state(agent, carry_state)

    # biological + environmental state comes from the daughter ...
    assert agent["bulk"] == "CARRIED_BULK"
    assert agent["unique"] == {"u": "carried"}
    assert agent["boundary"]["external"]["GLC"] == 5.0
    assert agent["environment"]["media_id"] == "CARRIED"
    # ... but exchange_data is the FRESH typed substore (identity-checked), NOT the
    # ballooned carried one — this is the fix.
    assert agent["environment"]["exchange_data"] is fresh_exchange_data
    assert agent["environment"]["exchange_data"]["constrained"]["GLC[p]"] == 20.0


def test_apply_carry_state_carries_when_no_fresh_exchange_data():
    """If the fresh build has no exchange_data substore (e.g. a composite without
    ExchangeData), fall back to carrying the whole environment unchanged."""
    from v2ecoli.workflow.lineage import apply_carry_state

    agent = {"environment": {"media_id": "FRESH"}, "bulk": "F"}
    carry_state = {"environment": {"media_id": "CARRIED", "other": 1}, "bulk": "C"}
    apply_carry_state(agent, carry_state)
    assert agent["environment"] == {"media_id": "CARRIED", "other": 1}
    assert agent["bulk"] == "C"


def test_divide_flag_detected_when_agent_id_diverges_from_inner_cell():
    """Regression: gen 0 divides but gens >= 1 run to the duration cap.

    The inner baseline composite always names its single cell "0", while
    ``self._agent_id`` accumulates phylogeny suffixes across generations
    ("0" -> "00" -> ...).  MarkDPeriod sets a ``divide`` flag on the inner
    "0" cell without changing the agents map, so ``_run_until_division`` must
    look the survivor up by the inner key (falling back to the sole agent),
    not by ``self._agent_id``.  Before the fix it did ``agents.get("00")`` for
    generation 1, missed the flag, and the generation never divided.
    """
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {"emitter": "parquet", "single_daughters": True,
                 "generations": 3, "max_duration_per_gen": 100.0}
    lp.initialize(lp.config)
    lp._agent_id = "00"          # generation >= 1: diverges from the inner "0"

    class _FakeComposite:
        # Inner composite always names its single cell "0"; MarkDPeriod has set
        # the divide flag there without adding/removing agents.
        state = {"agents": {"0": {"divide": True,
                                  "listeners": {"mass": {"dry_mass": 500.0}}}}}

        def run(self, interval):  # no-op; flag is already set
            pass

    lp._composite = _FakeComposite()
    lp._gen_elapsed = 0.0

    divided, _daughter, _dry_mass = lp._run_until_division(1.0)
    assert divided is True       # False before the fix (looked up agents["00"])


# --- per-generation checkpoint/resume (backlog item 34) ----------------------
#
# Contract: 3 new optional LineageProcess config keys let a wave orchestrator
# run ONE generation per invocation, chained via daughter-state S3 handoff --
# initial_carry_state_path / initial_generation_index seed a resumed wave,
# daughter_state_out_path persists this invocation's own daughter for the
# NEXT wave to pick up. All three default to ""/0/"", which must reproduce
# today's unchanged single-invocation-runs-every-generation behavior exactly.

def test_backward_compatible_defaults_start_fresh_with_no_carry_state(monkeypatch):
    lp, _ = _make(monkeypatch, generations=1)
    assert lp._generation == 0
    assert lp._carry_state is None


def test_initial_generation_index_requires_carry_state_path():
    """A nonzero start with no state to seed it would silently mislabel a
    fresh cell as a later generation (wrong parquet/zarr partition, wrong
    summary["generation"]) -- must fail loudly instead."""
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "baseline", "config_overrides": {}, "generations": 1,
        "single_daughters": True, "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100.0, "initial_carry_state_path": "",
        "initial_generation_index": 3, "daughter_state_out_path": "",
    }
    with pytest.raises(ValueError, match="initial_generation_index"):
        lp.initialize(lp.config)


def test_resume_loads_carry_state_and_starts_at_given_generation(monkeypatch):
    import v2ecoli.cache as cache_mod
    loaded = {"bulk": "RESUMED_BULK", "unique": {}}
    calls = {"path": None}

    def fake_load(path):
        calls["path"] = path
        return loaded

    monkeypatch.setattr(cache_mod, "load_initial_state", fake_load)
    lp, _ = _make(monkeypatch, generations=1,
                  initial_carry_state_path="s3://bucket/seed0/gen4/daughter.json",
                  initial_generation_index=5)
    assert lp._generation == 5
    assert lp._carry_state is loaded
    assert calls["path"] == "s3://bucket/seed0/gen4/daughter.json"
    # Regression (task #14): a resumed process's agent_id must match the
    # phylogeny depth a continuous single-process run would have reached by
    # generation 5 ("0"*6), not restart at "0" (depth 1). The xarray/zarr
    # emitter derives its own generation number from len(agent_id), so a
    # wrong-depth agent_id makes every resumed generation misresolve as
    # "generation 1" and collide with the real prior generation's S3 content.
    assert lp._agent_id == "0" * 6


def test_daughter_state_persisted_when_configured_and_divided(monkeypatch):
    import v2ecoli.cache as cache_mod
    saved = {}

    def fake_save(initial_state, path):
        saved["state"] = initial_state
        saved["path"] = path

    monkeypatch.setattr(cache_mod, "save_initial_state", fake_save)
    lp, _ = _make(monkeypatch, generations=2, divide_after=1,
                  daughter_state_out_path="s3://bucket/seed0/gen0/daughter.json")
    out = {}
    for _ in range(10):
        out = lp.update({}, 1.0)
        if out.get("summary") or out.get("complete"):
            break
    assert saved["path"] == "s3://bucket/seed0/gen0/daughter.json"
    # The fake divide()'s daughter, PLUS the generation-0 summary accumulated so
    # far (backlog item 35: a per-generation job's saved daughter state must
    # also carry the running summary history, or the NEXT generation's job has
    # no way to reconstruct a complete per-seed summary.json across separate
    # process invocations).
    assert saved["state"]["bulk"] == {}
    assert saved["state"]["unique"] == {}
    assert [s["generation"] for s in saved["state"]["_prior_summaries"]] == [0]


def test_daughter_state_carries_prior_summaries_forward_across_resume(monkeypatch):
    """A resumed generation's own saved daughter state must include BOTH the
    summaries it restored from the carry-state AND its own new entry -- the
    real regression test for the per-seed summary.json accumulation fix
    (without this, every chained job's summary.json only ever reflects the
    single generation IT computed, and each subsequent job's write silently
    discards every prior generation's history)."""
    import v2ecoli.cache as cache_mod

    prior_summary = {"generation": 0, "agent_id": "0", "duration": 1.0,
                      "dry_mass": 100.0, "divided": True}
    monkeypatch.setattr(cache_mod, "load_initial_state", lambda path: {
        "bulk": {}, "unique": {}, "_prior_summaries": [dict(prior_summary)]})
    saved = {}
    monkeypatch.setattr(cache_mod, "save_initial_state",
                         lambda state, path: saved.update(state=state, path=path))

    lp, _ = _make(monkeypatch, generations=1, divide_after=1,
                  initial_carry_state_path="s3://bucket/seed0/gen0/daughter.json",
                  initial_generation_index=1,
                  daughter_state_out_path="s3://bucket/seed0/gen1/daughter.json")
    # Restored on initialize(), before any tick runs.
    assert lp._summaries == [prior_summary]
    assert lp._carry_state is not None
    assert "_prior_summaries" not in lp._carry_state  # popped, not left for apply_carry_state

    out = {}
    for _ in range(10):
        out = lp.update({}, 1.0)
        if out.get("summary") or out.get("complete"):
            break
    assert [s["generation"] for s in saved["state"]["_prior_summaries"]] == [0, 1]


def test_checkpoint_dir_derives_a_distinct_per_generation_path(monkeypatch):
    """Item 115: a pbg-native lineage has no external scheduler to pre-compute
    each generation's own literal daughter_state_out_path (unlike chain-dispatch,
    where JobScheduler computes it once per generation's own separate job) --
    LineageProcess must derive it itself, and a DIFFERENT path per generation,
    so a write failure at generation N can never corrupt generation N-1's
    already-durable checkpoint."""
    import v2ecoli.cache as cache_mod
    saved_paths = []
    monkeypatch.setattr(
        cache_mod, "save_initial_state",
        lambda state, path: saved_paths.append(path))

    lp, _ = _make(monkeypatch, generations=3, divide_after=1,
                  checkpoint_dir="s3://bucket/seed0/checkpoints")
    out = {}
    for _ in range(30):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break
    assert out["complete"] is True
    assert saved_paths == [
        "s3://bucket/seed0/checkpoints/gen_0000.pkl",
        "s3://bucket/seed0/checkpoints/gen_0001.pkl",
        "s3://bucket/seed0/checkpoints/gen_0002.pkl",
    ], saved_paths
    assert len(set(saved_paths)) == 3, "each generation must write a DISTINCT key"


def test_checkpoint_dir_strips_a_trailing_slash(monkeypatch):
    """A caller-supplied prefix with a trailing slash must not produce a
    double-slash in the derived path."""
    import v2ecoli.cache as cache_mod
    saved = {}
    monkeypatch.setattr(cache_mod, "save_initial_state",
                         lambda state, path: saved.update(path=path))
    lp, _ = _make(monkeypatch, generations=1, divide_after=1,
                  checkpoint_dir="s3://bucket/seed0/checkpoints/")
    lp.update({}, 1.0)
    assert saved["path"] == "s3://bucket/seed0/checkpoints/gen_0000.pkl"


def test_checkpoint_dir_takes_priority_over_daughter_state_out_path(monkeypatch):
    """Both set is a real, meaningful precedence, not an ambiguity -- a literal
    single path can only ever describe ONE generation's own destination, so
    checkpoint_dir (which can describe all of them) must win."""
    import v2ecoli.cache as cache_mod
    saved = {}
    monkeypatch.setattr(cache_mod, "save_initial_state",
                         lambda state, path: saved.update(path=path))
    lp, _ = _make(monkeypatch, generations=1, divide_after=1,
                  checkpoint_dir="s3://bucket/checkpoints",
                  daughter_state_out_path="s3://bucket/legacy/daughter.json")
    lp.update({}, 1.0)
    assert saved["path"] == "s3://bucket/checkpoints/gen_0000.pkl"


def test_checkpoint_dir_empty_falls_back_to_daughter_state_out_path_unchanged(monkeypatch):
    """The byte-identical regression: checkpoint_dir omitted (today's default,
    "") must reproduce EXACTLY chain-dispatch's own existing behavior -- a
    single literal path, unchanged by this feature's existence."""
    import v2ecoli.cache as cache_mod
    saved = {}
    monkeypatch.setattr(cache_mod, "save_initial_state",
                         lambda state, path: saved.update(path=path))
    lp, _ = _make(monkeypatch, generations=1, divide_after=1,
                  daughter_state_out_path="s3://bucket/seed0/gen0/daughter.json")
    lp.update({}, 1.0)
    assert saved["path"] == "s3://bucket/seed0/gen0/daughter.json"


def test_daughter_state_not_persisted_without_a_daughter(monkeypatch):
    """Timed out without dividing -> nothing to hand off, mirrors
    self._carry_state staying None in that case."""
    import v2ecoli.cache as cache_mod
    calls = {"n": 0}
    monkeypatch.setattr(cache_mod, "save_initial_state",
                         lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))

    lp, _ = _make(monkeypatch, generations=1, divide_after=10_000,  # never divides
                  daughter_state_out_path="s3://bucket/seed0/gen0/daughter.json")
    lp.config["max_duration_per_gen"] = 1.0  # times out on the first tick
    out = lp.update({}, 1.0)
    assert out.get("complete") is True
    assert calls["n"] == 0


def test_single_wave_invocation_completes_after_one_generation_labeled_correctly(monkeypatch):
    """The wave-orchestrator contract: generations=1 always completes after
    exactly the ONE generation at initial_generation_index, and the summary
    reports the real (resumed) generation number, not a within-invocation 0."""
    import v2ecoli.cache as cache_mod
    monkeypatch.setattr(cache_mod, "load_initial_state",
                         lambda path: {"bulk": {}, "unique": {}})
    lp, _ = _make(monkeypatch, generations=1, divide_after=1,
                  initial_carry_state_path="s3://bucket/seed0/gen6/daughter.json",
                  initial_generation_index=7)
    out = {}
    for _ in range(10):
        out = lp.update({}, 1.0)
        if out.get("complete"):
            break
    assert out["complete"] is True
    assert len(lp._summaries) == 1
    assert lp._summaries[0]["generation"] == 7   # real generation number, not 0
    assert lp._summaries[0]["agent_id"] == "0" * 8


@pytest.mark.parametrize("gen_index", [0, 1, 2, 7])
def test_agent_id_depth_matches_resumed_generation(monkeypatch, gen_index):
    """Regression test for task #14 (backlog item 34's per-generation
    chain-dispatch bug). ``LineageProcess.initialize`` used to hardcode
    ``self._agent_id = "0"`` regardless of ``initial_generation_index``, so
    every chain job resolved to the SAME agent_id no matter which generation
    it actually resumed. The xarray/zarr emitter reads ``len(agent_id)`` as
    the generation number, so every generation past 0 misresolved as
    "generation 1" (a fresh-lineage store) and collided with the real prior
    generation's content sitting at the shared per-seed S3 prefix -- the
    actual bug behind every real gen1+ chain job silently no-op'ing while
    reporting SUCCEEDED. Under single_daughters=True (the only supported
    mode), the phylogeny walk always keeps the "...0" daughter
    (select_carry_daughter), so the correct depth is exactly gen_index + 1.
    """
    import v2ecoli.cache as cache_mod
    monkeypatch.setattr(cache_mod, "load_initial_state",
                         lambda path: {"bulk": {}, "unique": {}})
    kwargs = {}
    if gen_index:
        kwargs = {"initial_carry_state_path": "s3://bucket/seed0/gen/daughter.json",
                  "initial_generation_index": gen_index}
    lp, _ = _make(monkeypatch, generations=1, divide_after=1, **kwargs)
    assert lp._agent_id == "0" * (gen_index + 1)


def _stub_xarray_run(monkeypatch, captured):
    """Stub the three v2ecoli.library.xarray_run symbols _open_xarray_emitter
    imports locally, isolating its own writer-defaulting logic (the thing
    under test) from the rest of the real emitter-building pipeline."""
    import v2ecoli.library.xarray_run as xarray_run_mod

    def fake_build_emitter(**kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(xarray_run_mod, "_build_emitter", fake_build_emitter)
    monkeypatch.setattr(xarray_run_mod, "filter_view_to_existing_leaves",
                         lambda wrapped, raw_view: raw_view)
    monkeypatch.setattr(xarray_run_mod, "extract_output_metadata_from_state",
                         lambda wrapped, view: {})


def test_xarray_emitter_defaults_buffers_per_chunk_to_one(monkeypatch):
    """Backlog item 105 / Boyan Beronov's report: build_emitter_config's own
    shared default (buffers_per_chunk=10) is wrong for immutable object
    storage (S3 Standard, our backend for this dispatch path) -- it means
    every chunk flush re-copies previously-written objects instead of
    appending cleanly. ecoli_baseline.py's single-cell path already overrides
    this to 1; this path silently inherited the shared default of 10 instead.
    """
    lp, _ = _make(monkeypatch, generations=1, divide_after=1)
    lp._core = object()
    captured: dict = {}
    _stub_xarray_run(monkeypatch, captured)

    lp._open_xarray_emitter(emit_cell={"bulk": {}})

    assert captured["kwargs"]["writer"] == {"buffers_per_chunk": 1}


def test_xarray_emitter_caller_writer_override_still_wins(monkeypatch):
    """setdefault, not assignment: an explicit caller-supplied buffers_per_chunk
    (or any other writer key) must not be silently clobbered by the new default."""
    lp, _ = _make(monkeypatch, generations=1, divide_after=1)
    lp._core = object()
    lp.config["emitter_arg"] = {"writer": {"buffers_per_chunk": 4, "backend": "zarr"}}
    captured: dict = {}
    _stub_xarray_run(monkeypatch, captured)

    lp._open_xarray_emitter(emit_cell={"bulk": {}})

    assert captured["kwargs"]["writer"] == {"buffers_per_chunk": 4, "backend": "zarr"}


# --- injected agent-root stores survive the generation boundary --------------
# sms-ecoli#166 P0 items 2 and 3: a native lineage lost EVERY injected
# agent-root store at each generation boundary, because only
# bulk/unique/environment/boundary were selected from the surviving daughter and
# overlaid onto the next generation's freshly built document. A `fields` dose
# was re-zeroed (and re-fired) every generation, wall damage could not
# accumulate, and a `lysed` latch un-latched.


def _restore_division_registries(monkeypatch):
    """Keep the module-level divider / carried-listener registries per-test."""
    import v2ecoli.library.division as _div

    monkeypatch.setattr(_div, "STORE_DIVIDERS", dict(_div.STORE_DIVIDERS))
    monkeypatch.setattr(
        _div, "CARRIED_LISTENER_PATHS", list(_div.CARRIED_LISTENER_PATHS))


def test_select_carry_daughter_returns_extra_root_stores():
    """The …0 daughter's carry state includes the injected agent-root stores,
    taken from the mother snapshot (the rebuilt daughter's are fresh/zeroed)."""
    import numpy as np
    from v2ecoli.workflow.lineage import select_carry_daughter

    dosed = np.array([[7.5]])
    mother_snapshot = {
        "bulk": "M", "unique": {}, "environment": {}, "boundary": {},
        "fields": {"_type": "map[overwrite[array[float]]]", "tetracycline": dosed},
        "imposed_flux_bounds": {"RXN": 3.0},
        "periplasm": {"global": {"volume": 0.2}},
    }
    agents_now = {
        "00": {
            "bulk": "D0", "unique": {"u": 1}, "environment": {}, "boundary": {},
            # what the Division step's baseline() rebuild produces: FRESH zeros
            "fields": {"_type": "map[overwrite[array[float]]]",
                       "tetracycline": np.zeros((1, 1))},
            "imposed_flux_bounds": {},
            # a process EDGE on the daughter node must never be carried
            "division": {"address": "local:Division", "config": {}},
        },
        "01": {"bulk": "D1", "unique": {}, "environment": {}, "boundary": {}},
    }
    carry = select_carry_daughter({"0"}, agents_now, mother_snapshot)

    assert carry["bulk"] == "D0"                 # core state still the daughter's
    assert carry["fields"]["tetracycline"] == dosed
    assert carry["imposed_flux_bounds"] == {"RXN": 3.0}
    assert carry["periplasm"] == {"global": {"volume": 0.2}}
    assert "division" not in carry               # edges filtered out
    assert "listeners" not in carry              # never carried wholesale


def test_select_carry_daughter_applies_a_registered_divider(monkeypatch):
    """A store that declares a divider is SPLIT (the fork's pg_cellwall
    behaviour), not copied; everything else is copied."""
    _restore_division_registries(monkeypatch)
    from v2ecoli.library.division import register_store_divider
    from v2ecoli.workflow.lineage import select_carry_daughter

    register_store_divider("pg_cellwall", lambda v: (["half-a"], ["half-b"]))
    mother_snapshot = {
        "bulk": "M", "unique": {}, "environment": {}, "boundary": {},
        "pg_cellwall": ["whole"], "fields": {"drug": 1.0},
    }
    carry = select_carry_daughter(
        {"0"}, {"00": {"bulk": "D0", "unique": {}}}, mother_snapshot)
    assert carry["pg_cellwall"] == ["half-a"]    # divided, daughter 1's share
    assert carry["fields"] == {"drug": 1.0}      # copied


def test_select_carry_daughter_carries_a_declared_listener_leaf(monkeypatch):
    """A declared listener LEAF (the lysed latch) rides along; the rest of
    ``listeners`` does not."""
    _restore_division_registries(monkeypatch)
    from v2ecoli.library.division import register_carried_listener_path
    from v2ecoli.workflow.lineage import select_carry_daughter

    register_carried_listener_path(("peptidoglycan_shape", "lysed"))
    mother_snapshot = {
        "bulk": "M", "unique": {}, "environment": {}, "boundary": {},
        "listeners": {
            "peptidoglycan_shape": {"lysed": True, "murein": 42},
            "mass": {"dry_mass": 500.0},
        },
    }
    carry = select_carry_daughter(
        {"0"}, {"00": {"bulk": "D0", "unique": {}}}, mother_snapshot)
    assert carry["_carried_listeners"] == {"peptidoglycan_shape": {"lysed": True}}
    assert "listeners" not in carry


def test_apply_carry_state_merges_extra_store_and_keeps_its_type():
    """The typed-node trap: the fresh document represents an injected root as a
    dict carrying ``_type`` (sms-ecoli's ``_materialize_native_declared_state``
    stamps ``fields`` as ``map[overwrite[array[float]]]`` and pre-seeds the
    molecule keys as zero arrays). Overlaying the carried store must MERGE
    leaves into that node — replacing it with a raw dict would drop the
    ``_type`` and with it the overwrite updater, the exact failure
    ``_FRESH_ENVIRONMENT_SUBSTORES`` guards for ``exchange_data``."""
    import numpy as np
    from v2ecoli.workflow.lineage import apply_carry_state

    fresh_exchange_data = {"constrained": {"GLC[p]": 20.0}}
    agent = {
        "bulk": "FRESH", "unique": {},
        "environment": {"exchange_data": fresh_exchange_data},
        "boundary": {},
        "listeners": {"mass": {}},
        "fields": {
            "_type": "map[overwrite[array[float]]]",
            "tetracycline": np.zeros((1, 1)),
            "glucose": np.zeros((1, 1)),        # key the carry state lacks
        },
        "imposed_flux_bounds": {},
    }
    dosed = np.array([[7.5]])
    carry_state = {
        "bulk": "CARRIED", "unique": {},
        "environment": {"exchange_data": {"constrained": {"GLC[p]": 9999.0}}},
        "boundary": {},
        "fields": {"tetracycline": dosed},
        "imposed_flux_bounds": {"RXN": 3.0},
        "counts": {"x": 1},                      # root absent from the fresh doc
    }
    apply_carry_state(agent, carry_state)

    assert agent["fields"]["_type"] == "map[overwrite[array[float]]]"  # type kept
    assert agent["fields"]["tetracycline"] == dosed                    # seed replaced
    assert agent["fields"]["glucose"] == np.zeros((1, 1))              # fresh key kept
    assert agent["imposed_flux_bounds"] == {"RXN": 3.0}
    assert agent["counts"] == {"x": 1}
    assert agent["bulk"] == "CARRIED"
    # the exchange_data guard is untouched by the new overlay
    assert agent["environment"]["exchange_data"] is fresh_exchange_data


def test_apply_carry_state_merges_a_carried_listener_leaf():
    """``_carried_listeners`` merges into ``listeners`` without disturbing the
    fresh listener tree (the caller resets ``listeners.mass`` right after)."""
    from v2ecoli.workflow.lineage import apply_carry_state

    agent = {
        "bulk": "F", "unique": {}, "environment": {}, "boundary": {},
        "listeners": {"mass": {"dry_mass": 0.0},
                      "peptidoglycan_shape": {"lysed": False, "murein": 0}},
    }
    apply_carry_state(agent, {
        "bulk": "C",
        "_carried_listeners": {"peptidoglycan_shape": {"lysed": True}},
    })
    assert agent["listeners"]["peptidoglycan_shape"]["lysed"] is True
    assert agent["listeners"]["peptidoglycan_shape"]["murein"] == 0
    assert agent["listeners"]["mass"] == {"dry_mass": 0.0}
    assert "_carried_listeners" not in agent


class _FakeComposite:
    def __init__(self, state):
        self.state = state


def test_elapsed_after_run_prefers_the_daughters_division_stamp(monkeypatch):
    """sim 898 / sms-ecoli#166: a generation that divides at 2,528 s inside a
    3,600 s window must book 2,528 s, not the window -- otherwise
    lineage_time_offset drifts by (window - division time) per generation and a
    cumulative-time dose fires early."""
    lp, _ = _make(monkeypatch, generations=2)
    lp._gen_elapsed = 0.0
    lp._composite = _FakeComposite({
        "global_time": 3600.0,
        "agents": {"00": {"global_time": 2528.0}, "01": {"global_time": 2528.0}},
    })
    assert lp._elapsed_after_run(3600.0, {"0"}, lp._composite.state["agents"]) == 2528.0


def test_elapsed_after_run_ignores_a_daughter_stamp_that_does_not_advance(monkeypatch):
    """sims 946/947 (2026-09-10): the Division step rebuilds each daughter from
    baseline(), whose global_time is 0.0, so the "stamp" on a real daughter is 0.0.
    Trusting it booked 0 s per generation: lineage_time_offset never advanced and
    Run 3's cumulative 10,000 s dose never fired. A non-advancing stamp must be
    ignored in favour of the inner clock, which stops at the division (2,528 s)."""
    lp, _ = _make(monkeypatch, generations=2)
    lp._gen_elapsed = 0.0
    lp._composite = _FakeComposite({
        "global_time": 2528.0,
        "agents": {"00": {"global_time": 0.0}, "01": {"global_time": 0.0}},
    })
    assert lp._elapsed_after_run(3600.0, {"0"}, lp._composite.state["agents"]) == 2528.0
    # and with neither a usable stamp nor an advancing clock, the window (stubs)
    lp._composite = _FakeComposite({"global_time": 0.0,
                                    "agents": {"00": {"global_time": 0.0}}})
    assert lp._elapsed_after_run(3600.0, {"0"}, lp._composite.state["agents"]) == 3600.0


def test_elapsed_after_run_uses_the_inner_clock_without_daughters(monkeypatch):
    lp, _ = _make(monkeypatch, generations=2)
    lp._gen_elapsed = 0.0
    lp._composite = _FakeComposite({"global_time": 1734.0, "agents": {"0": {}}})
    assert lp._elapsed_after_run(3600.0, {"0"}, {"0": {}}) == 1734.0
    # A window that ran to its end books the whole window.
    lp._composite = _FakeComposite({"global_time": 3600.0, "agents": {"0": {}}})
    assert lp._elapsed_after_run(3600.0, {"0"}, {"0": {}}) == 3600.0


def test_elapsed_after_run_falls_back_to_the_window_for_stubs(monkeypatch):
    lp, _ = _make(monkeypatch, generations=2)
    lp._gen_elapsed = 40.0
    lp._composite = _FakeComposite({"agents": {"0": {}}})
    assert lp._elapsed_after_run(10.0, {"0"}, {"0": {}}) == 50.0
    # A stale clock (not past what is already booked) never rewinds.
    lp._composite = _FakeComposite({"global_time": 5.0, "agents": {"0": {}}})
    assert lp._elapsed_after_run(10.0, {"0"}, {"0": {}}) == 50.0
