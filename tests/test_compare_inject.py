"""Regression tests for config-adapter process-set key carry-through."""
import pytest
from scripts._compare.config_adapter import translate_vecoli_config


def test_translate_preserves_process_set_keys():
    vecoli = {
        "experiment_id": "x", "generations": 2,
        "add_processes": ["example-secretion"],
        "swap_processes": {}, "exclude_processes": [],
        "process_configs": {"example-secretion": {"rate": 2.0}},
        "topology": {"example-secretion": {"counts": ["bulk"]}},
        "emitter": "parquet",            # vEcoli-only -> dropped
    }
    v2 = translate_vecoli_config(vecoli)
    assert v2["add_processes"] == ["example-secretion"]
    assert v2["process_configs"]["example-secretion"] == {"rate": 2.0}
    assert v2["topology"]["example-secretion"] == {"counts": ["bulk"]}
    assert "emitter" not in v2                      # still dropped
    assert v2["_dropped_vecoli_keys"]["emitter"] == "parquet"


# ---------------------------------------------------------------------------
# Task 2: classify_process + resolve_injections
# ---------------------------------------------------------------------------
import os

from scripts._compare import inject

FORK = os.path.join(os.path.dirname(__file__), "fixtures", "fork_example")

def test_classify_vivarium_and_partitioned():
    # Route through _fork_registry (not a raw ``from ecoli.processes import``):
    # it saves/restores the real ecoli.* around the fork import, so it returns
    # the fork's registry order-independently WITHOUT desyncing the vivarium
    # singleton registry (which a raw sys.modules pop would not clear).
    reg = inject._fork_registry(FORK)
    assert inject.classify_process(reg.access("example-secretion")) == "vivarium_1"
    assert inject.classify_process(reg.access("bad-partitioned")) == "partitioned"

def test_resolve_injections_builds_spec():
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {"example-secretion": {"rate": 3.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    assert len(specs) == 1
    s = specs[0]
    assert s["name"] == "example-secretion"
    assert s["kind"] == "vivarium_1"
    assert s["config"] == {"rate": 3.0}
    assert s["topology"] == {"counts": ["bulk"]}
    assert s["qualname"] == "ExampleSecretion"

def test_resolve_rejects_partitioned():
    cfg = {"add_processes": ["bad-partitioned"], "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="partitioned"):
        inject.resolve_injections(FORK, cfg)

def test_resolve_rejects_sim_data_config():
    cfg = {"add_processes": ["example-secretion"],
           "process_configs": {"example-secretion": "sim_data"},
           "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="sim_data"):
        inject.resolve_injections(FORK, cfg)

def test_resolve_rejects_unknown_name():
    cfg = {"add_processes": ["no-such-process"], "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="not in fork registry"):
        inject.resolve_injections(FORK, cfg)


def test_resolve_rejects_config_less_native_swap_target():
    """FAIL LOUD (sms-ecoli#210 Gate 0): a swap TARGET with no config on the native
    path (no explicit process_config, no fork_sim_data) would run on config_schema
    defaults -- for metabolism-redux an empty stoichiometry, which collapses the
    generation to one tick while reporting success. Refuse instead."""
    cfg = {"swap_processes": {"some-baseline": "example-secretion"}, "time_step": 1.0}
    with pytest.raises(inject.InjectionError, match="swap target but has NO config"):
        inject.resolve_injections(FORK, cfg)


def test_native_swap_target_with_explicit_config_is_allowed():
    """An explicit process_config on the swap target satisfies the guard -- the
    process gets a real config, so it is NOT config-less."""
    cfg = {"swap_processes": {"some-baseline": "example-secretion"},
           "process_configs": {"example-secretion": {"rate": 1.5}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    assert specs[0]["name"] == "example-secretion"
    assert specs[0]["config"] == {"rate": 1.5}


def test_resolve_injections_memoized(monkeypatch):
    """resolve_injections must import the fork only once per unique config."""
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {"example-secretion": {"rate": 3.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    inject._RESOLVE_CACHE.clear()

    specs1 = inject.resolve_injections(FORK, cfg)
    specs2 = inject.resolve_injections(FORK, cfg)
    assert specs1 == specs2

    # Prove the cache is in use: _fork_registry must NOT be called on a
    # repeated call with an already-cached key.
    def _should_not_be_called(repo):
        raise RuntimeError("cache miss — _fork_registry called unexpectedly")

    monkeypatch.setattr(inject, "_fork_registry", _should_not_be_called)
    specs3 = inject.resolve_injections(FORK, cfg)
    assert specs3 == specs1


# ---------------------------------------------------------------------------
# Task 3: apply_injected_processes
# ---------------------------------------------------------------------------
def test_apply_injects_edge_and_flow_order():
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "process_configs": {"example-secretion": {"rate": 2.0}},
           "topology": {"example-secretion": {"counts": ["bulk"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    cell_state = {"bulk": {}}      # a 'bulk' store exists
    flow_order = ["ecoli-metabolism"]
    added = inject.apply_injected_processes(cell_state, flow_order, core, specs)
    assert added == ["example-secretion"]
    assert "example-secretion" in cell_state
    edge = cell_state["example-secretion"]
    assert edge["_type"] in ("process", "step")
    assert edge["inputs"]["counts"] == ["bulk"]
    assert flow_order[-1] == "example-secretion"

def test_apply_introduces_process_owned_store():
    """A vivarium_1 process whose topology references a store the composite does
    not yet own now INTRODUCES that store — created and materialized from the
    process's own port schema — instead of being rejected. This is what lets a
    fork subsystem carrying its own private stores (e.g. the cell-wall model's
    murein_state/wall_state) inject and run unattended. (Trade-off: a genuinely
    mis-typed store path creates an isolated store rather than erroring; the
    process simply reads/writes its own orphan store.)"""
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "topology": {"example-secretion": {"counts": ["nonexistent_store"]}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    cell_state = {"bulk": {}}
    added = inject.apply_injected_processes(cell_state, [], core, specs)
    assert "example-secretion" in added
    assert "nonexistent_store" in cell_state  # auto-introduced, not rejected


# ---------------------------------------------------------------------------
# Config initial_state honoring + schema-default backfill
# ---------------------------------------------------------------------------
def test_resolve_config_initial_state_merges_inline_and_overrides(tmp_path):
    """initial_state (inline) + initial_state_overrides (files under
    <fork>/data) merge, overrides on top."""
    fork = tmp_path / "fork"
    (fork / "data" / "overrides").mkdir(parents=True)
    (fork / "data" / "overrides" / "red.json").write_text(
        '{"murein_state": {"unincorporated_murein": 2234940}, "wall_state": {}}')
    cfg = {"initial_state": {"murein_state": {"shadow_murein": 0}},
           "initial_state_overrides": ["overrides/red"]}
    merged = inject.resolve_config_initial_state(str(fork), cfg)
    assert merged["murein_state"]["unincorporated_murein"] == 2234940
    assert merged["murein_state"]["shadow_murein"] == 0     # inline preserved
    assert merged["wall_state"] == {}


def test_apply_seeds_new_store_from_config_initial_state_not_baseline():
    """Config initial_state seeds an injected NEW store, but never clobbers a
    pre-existing v2 baseline store (e.g. the structured bulk array)."""
    from v2ecoli.core import build_core
    core = build_core()
    cfg = {"add_processes": ["example-secretion"],
           "topology": {"example-secretion": {"counts": ["new_store"]}},
           "initial_state": {"new_store": {"seeded": 7}, "bulk": {"X": 1}},
           "time_step": 1.0}
    specs = inject.resolve_injections(FORK, cfg)
    cell_state = {"bulk": {"_real_array": True}}      # baseline store v2 owns
    inject.apply_injected_processes(cell_state, [], core, specs)
    assert cell_state["new_store"]["seeded"] == 7     # new store seeded
    assert cell_state["bulk"] == {"_real_array": True}  # baseline NOT clobbered


def test_bridge_backfills_sentinel_none_leaf():
    """The bridge presents a v1 process its declared leaves — including a
    sentinel-None default the pbg store can't persist — so reads don't KeyError."""
    from v2ecoli.library.vivarium_bridge import _backfill_schema_defaults
    schema = {"wall_state": {"lattice": {"_default": None, "_updater": "set"},
                             "rows": {"_default": 0}}}
    state = {"wall_state": {"rows": 5}}   # 'lattice' dropped by the store
    _backfill_schema_defaults(state, schema)
    assert "lattice" in state["wall_state"]
    assert state["wall_state"]["lattice"] is None     # sentinel present
    assert state["wall_state"]["rows"] == 5           # existing value untouched


# ---------------------------------------------------------------------------
# build_fork_config resolves the FORK, not the installed vEcoli
#
# A bare ``import ecoli.library.sim_data`` inside resolve_injections resolves to
# site-packages, because _fork_registry restores the installed ecoli.* as soon as
# it has the registry handle. Building a swapped process's config from the wrong
# vEcoli drops every key the fork added, and the process then falls back to its
# own class default with no error raised anywhere.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_resolve_cache():
    # resolve_injections memoizes into a plain module-level dict (_RESOLVE_CACHE),
    # NOT lru_cache, and its key omits fork_sim_data. Two tests whose configs are
    # equal therefore share a memo entry and the second never calls
    # build_fork_config at all — so a guard test can pass merely because an
    # earlier test cached a successful spec. Clear it around every test here.
    inject._RESOLVE_CACHE.clear()
    yield
    inject._RESOLVE_CACHE.clear()


def test_build_fork_config_reads_the_fork_not_installed_vecoli():
    cfg = inject.build_fork_config(FORK, "unused.cPickle", "example-secretion")
    # Only the fixture fork's getter emits this key.
    assert cfg["fork_only_key"] == "present"
    assert cfg["rate"] == 1.0


def test_guard_raises_when_the_import_resolves_outside_the_fork(monkeypatch):
    # The fork HAS a sim_data module but the import lands elsewhere — exactly what
    # the installed-vEcoli shadow does. Resolving the name to this test module
    # stands in for that, since it is outside the fork.
    import sys as _sys
    monkeypatch.setattr("importlib.import_module", lambda n: _sys.modules[__name__])
    with pytest.raises(inject.InjectionError, match="outside fork"):
        inject.build_fork_config(FORK, "unused.cPickle", "example-secretion")


def test_fork_with_no_sim_data_module_behaves_the_same_however_the_env_is_installed(
        tmp_path):
    # NOT InjectionError. With a vEcoli installed the import would succeed and
    # resolve outside the fork; with none it would raise ModuleNotFoundError —
    # the same fork and call giving opposite outcomes based on an unrelated
    # package. Decided from the fork's own files, so both environments agree, and
    # the caller's normal not-fork-configurable fallback handles it.
    with pytest.raises(ModuleNotFoundError):
        inject.build_fork_config(str(tmp_path), "unused.cPickle", "example-secretion")


def test_fork_resolution_guard_is_not_downgraded_to_the_default_config():
    # resolve_injections falls back to a default config when the fork cannot
    # configure a process. That fallback must NOT swallow the resolution guard —
    # otherwise the guard is decorative and the silent-wrong-config path returns.
    # Raise InjectionError directly, so this pins the re-raise in
    # resolve_injections rather than any particular way of provoking the guard.
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {}, "topology": {}, "time_step": 1.0,
           "fork_sim_data": "unused.cPickle",
           # Distinct from the other resolve_injections test's config: the memo
           # key is content-derived, so identical dicts collide across tests.
           "output_ports": {"_k": "guard-test"}}
    orig = inject.build_fork_config

    def _guard_trips(fork_repo, sim_data_path, name):
        raise inject.InjectionError(
            f"{name!r}: ecoli.library.sim_data resolved to '/elsewhere', "
            "outside fork")

    inject.build_fork_config = _guard_trips
    try:
        with pytest.raises(inject.InjectionError, match="outside fork"):
            inject.resolve_injections(FORK, cfg)
    finally:
        inject.build_fork_config = orig


def test_fork_config_still_falls_back_when_the_fork_lacks_a_getter():
    # The legitimate fallback must survive: a process the fork cannot configure
    # gets the default config, not an exception.
    cfg = {"add_processes": ["example-secretion"], "swap_processes": {},
           "process_configs": {}, "topology": {}, "time_step": 1.0,
           "fork_sim_data": "unused.cPickle",
           "output_ports": {"_k": "fallback-test"}}
    orig = inject.build_fork_config

    def _no_getter(fork_repo, sim_data_path, name):
        raise KeyError(f"Process of name {name} is not known")

    inject.build_fork_config = _no_getter
    try:
        specs = inject.resolve_injections(FORK, cfg)
        assert specs[0]["config"] is None      # default, and no exception
    finally:
        inject.build_fork_config = orig


# ---------------------------------------------------------------------------
# Explicitly-named companion fork processes (--inject-process / inject_processes)
#
# A swapped fork process can read a store written by a COMPANION fork process
# that v2ecoli has no equivalent of. On the fork the companion arrives via the
# config's standard `processes` list, which injection does not carry, so the
# store is created from schema defaults and the swapped process later fails
# INSIDE the fork on a lookup against empty contents. Naming the companion is
# explicit by design: inferring "this port has no writer" from a vivarium-1.0
# ports_schema means guessing at port direction.
# ---------------------------------------------------------------------------
from scripts.run_comparison_ensemble import _injected_from_resolved


def test_extra_processes_are_added_to_the_declared_ones():
    inj = _injected_from_resolved(
        {"add_processes": ["example-secretion"], "swap_processes": {}},
        FORK, None, extra_processes=["companion-listener"])
    assert inj["add_processes"] == ["example-secretion", "companion-listener"]


def test_extra_processes_do_not_duplicate_an_already_declared_one():
    inj = _injected_from_resolved(
        {"add_processes": ["example-secretion"], "swap_processes": {}},
        FORK, None, extra_processes=["example-secretion"])
    assert inj["add_processes"] == ["example-secretion"]


def test_extra_processes_alone_still_produce_an_injection_block():
    # A config that only SWAPS declares no add_processes; the companion must
    # still get through, so the "nothing to inject" early return has to account
    # for it.
    inj = _injected_from_resolved(
        {"swap_processes": {"a": "b"}}, FORK, None,
        extra_processes=["companion-listener"])
    assert inj["add_processes"] == ["companion-listener"]
    inj2 = _injected_from_resolved({}, FORK, None, extra_processes=["companion"])
    assert inj2 is not None and inj2["add_processes"] == ["companion"]


def test_no_extra_processes_leaves_the_declaration_untouched():
    inj = _injected_from_resolved(
        {"add_processes": ["example-secretion"], "swap_processes": {}}, FORK, None)
    assert inj["add_processes"] == ["example-secretion"]
    assert _injected_from_resolved({}, FORK, None) is None       # still nothing to do


def test_an_unknown_companion_fails_loud_rather_than_being_ignored():
    # The whole point is that a typo must not silently produce a run missing the
    # companion — which is the failure this feature exists to prevent.
    cfg = {"add_processes": ["example-secretion", "no-such-process"],
           "swap_processes": {}, "process_configs": {}, "topology": {},
           "time_step": 1.0, "output_ports": {"_k": "unknown-companion"}}
    with pytest.raises(inject.InjectionError):
        inject.resolve_injections(FORK, cfg)
# Execution order of injected processes
#
# make_edge gives every Step the DEFAULT priority 1.0, and inject_flow_
# dependencies — which replaces that with distinct descending values — runs
# BEFORE apply_injected_processes in the baseline generator. So injected steps
# used to keep 1.0: tied with each other AND with the last baseline step (whose
# priority is float(total_steps - step_idx) = 1.0 at the final index).
#
# The consequence is intermittent, not a clean failure: whichever tied step the
# scheduler happens to run first decides whether a consumer reads a populated
# store or an empty one.
# ---------------------------------------------------------------------------

def _cell_state_with_baseline():
    # Mirrors what inject_flow_dependencies leaves behind: distinct descending
    # priorities, the last baseline step sitting at exactly 1.0.
    return {"base-a": {"priority": 3.0}, "base-b": {"priority": 2.0},
            "base-last": {"priority": 1.0}}


def _apply(cell_state, names):
    specs = inject.resolve_injections(FORK, {
        "add_processes": ["example-secretion"], "swap_processes": {},
        "process_configs": {"example-secretion": {"rate": 1.0}},
        "topology": {"example-secretion": {"counts": ["bulk"]}}, "time_step": 1.0})
    spec = specs[0]
    out = []
    for n in names:                       # one spec per requested name
        s = dict(spec); s["name"] = n
        # Inject as a STEP. make_edge writes `priority` for steps and `interval`
        # for processes, and the scheduler orders steps by priority -- so a
        # process-edge fixture would assert on a field nothing reads, and would
        # fail against the pre-fix code with a KeyError (absence) rather than
        # with the 1.0-vs-1.0 tie these tests exist to pin. The real injected
        # pair here (a listener and MetabolismRedux) are both steps.
        s["as_step"] = True
        out.append(s)
    flow_order = list(cell_state)
    from v2ecoli.core import build_core
    inject.apply_injected_processes(cell_state, flow_order, build_core(), out)
    return cell_state, flow_order


def test_injected_processes_get_distinct_priorities_in_declaration_order():
    cell_state, _ = _apply(_cell_state_with_baseline(),
                           ["companion-listener", "the-consumer"])
    p_first = cell_state["companion-listener"]["priority"]
    p_second = cell_state["the-consumer"]["priority"]
    # Strictly ordered: a companion declared first must run first. Higher
    # priority runs earlier (see inject_flow_dependencies' float(n - i)).
    assert p_first > p_second, (p_first, p_second)


def test_injected_processes_still_run_after_every_baseline_step():
    # Appending to flow_order already put them last; the tie-break must not
    # promote them above baseline work.
    cell_state, _ = _apply(_cell_state_with_baseline(),
                           ["companion-listener", "the-consumer"])
    lowest_baseline = min(cell_state[n]["priority"] for n in
                          ("base-a", "base-b", "base-last"))
    for n in ("companion-listener", "the-consumer"):
        assert cell_state[n]["priority"] < lowest_baseline


def test_a_single_injected_process_is_no_longer_tied_with_the_last_baseline_step():
    # The shape every existing single-swap study uses. Before the fix this sat
    # at 1.0, exactly equal to the final baseline step.
    cell_state, _ = _apply(_cell_state_with_baseline(), ["the-consumer"])
    assert cell_state["the-consumer"]["priority"] < cell_state["base-last"]["priority"]


def test_seed_exchange_species_survives_the_config_to_baseline_assembly():
    """⛔ `_injected_from_resolved` is an ALLOWLIST, so a key it does not name is
    dropped one layer above baseline()'s own guard — and the symptom is a clean
    run with a zero product, never an error.

    This pins the forwarding so the next refactor of that dict cannot silently
    remove it with a green suite.
    """
    inj = _injected_from_resolved(
        {"swap_processes": {"a": "b"},
         "seed_exchange_species": ["MY-PRODUCT"]}, FORK, None)
    assert inj["seed_exchange_species"] == ["MY-PRODUCT"]


def test_seed_exchange_species_defaults_to_empty_rather_than_missing():
    """Absent in the config => present-and-empty in the block, so a downstream
    reader never has to distinguish 'not declared' from 'declared empty'."""
    inj = _injected_from_resolved(
        {"swap_processes": {"a": "b"}}, FORK, None)
    assert inj["seed_exchange_species"] == []
