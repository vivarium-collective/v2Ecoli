"""The wrapped fork's GENERATION INDEX is ``len(agent_id)`` — guard every hop.

⛔ The defect these tests kill: a lineage whose staged induction never fires.
``LoadSimData`` (vEcoli fork) computes ``generation = len(kwargs["agent_id"])``
and applies each ``sim_data.internal_shift_dict`` entry whose ``shift_gen <=
generation``. That is the ONLY mechanism by which a config's ``induction_gen``
takes effect. The chain was broken in TWO independent places, and fixing either
alone changes nothing:

  1. ``agent_id`` was not in ``VivariumEcoliProcess.config_schema``, so it was
     dropped before it could reach the fork at all; and
  2. the pbg lineage driver pinned ``composite_agent_id = "0"`` for the whole
     lineage — only the emitter's ``partition_agent_id`` advanced.

⚠ Measured symptom: a multi-generation run whose config's schedule WAS
registered (the build logs the applied variant and its ``induction_gen``) and
whose product exchange stayed FLAT across the induction generation and beyond. A
fired shift raises new-gene expression by ~6 orders of magnitude, so nothing in
the output distinguishes "not induced yet" from "can never induce" — the run
completes, every observable populates, and the lineage is the wrong model.

⭐ These are hermetic: the hops are exercised with the fork stubbed out, because
the failure is a dropped keyword, not a modelling error — and a keyword that
stops being forwarded satisfies every signature check while every run silently
reverts to the un-induced baseline.
"""
import inspect

import pytest

from v2ecoli.library import vivarium_ecoli_engine as ve


@pytest.fixture(autouse=True)
def _no_pending_handle_leak():
    """``_PENDING_HANDLE`` is CLASS state, consumed once by the next
    ``VivariumEcoliProcess.__init__``. A test that monkeypatches ``__init__``
    never consumes it, and ``monkeypatch`` restores the method, not the class
    attribute — so the handle survives into whatever runs next, which then
    silently takes the injected-handle branch instead of building. Invisible in a
    full-file run (the following test resets it); it escapes under ``-k``,
    ``--lf``, ``-x`` or any reordering."""
    yield
    ve.VivariumEcoliProcess._PENDING_HANDLE = None


def test_build_vivarium_ecoli_accepts_agent_id():
    assert "agent_id" in inspect.signature(ve.build_vivarium_ecoli).parameters


def test_agent_id_is_in_the_process_config_schema():
    """Hop 1 of 2. pbg validates a process config against its schema, so a key
    the schema does not declare is DROPPED — silently, with the default "0"
    served in its place, which is exactly the founder value."""
    assert "agent_id" in ve.VivariumEcoliProcess.config_schema


def test_composite_builder_forwards_agent_id_to_the_build_AND_the_process(monkeypatch):
    """⚠ Tested at a NON-founder value on purpose. "0" is the no-op: every
    mutant here — drop the keyword, hardcode "0", forward to one consumer and
    not the other — passes when the id under test is the default."""
    seen_build = {}

    def _fake_build(**kw):
        seen_build.update(kw)
        return type("_H", (), {})()

    monkeypatch.setattr(ve, "build_vivarium_ecoli", _fake_build)

    seen_cfg = {}

    def _fake_init(self, config=None, core=None):
        seen_cfg.update(config or {})

    monkeypatch.setattr(ve.VivariumEcoliProcess, "__init__", _fake_init)
    monkeypatch.setattr(ve.VivariumEcoliProcess, "interface",
                        lambda self: {"inputs": {}, "outputs": {}})

    from v2ecoli.core import build_core
    comp, info = ve.build_vivarium_ecoli_composite(
        sim_data_path="x", agent_id="000", core=build_core())

    assert seen_build["agent_id"] == "000", (
        "the engine build ran as the founder while the caller asked for "
        "generation 3 — a staged shift can never fire")
    assert seen_cfg["agent_id"] == "000", (
        "the process config lost the agent_id, so a process that builds its own "
        "handle (no _PENDING_HANDLE) reverts to the founder")
    assert info["agent_id"] == "000"
    assert "000" in comp.state["agents"]


def test_the_process_self_build_path_carries_agent_id_through_pbg_validation(monkeypatch):
    """Both halves of hop 1 at once, through the REAL pbg config machinery: the
    schema must declare the key (or validation drops it) AND ``__init__`` must
    forward it (or declaring it achieves nothing)."""
    seen = {}

    def _fake_build(**kw):
        seen.update(kw)
        return type("_H", (), {})()

    monkeypatch.setattr(ve, "build_vivarium_ecoli", _fake_build)
    from v2ecoli.core import build_core
    ve.VivariumEcoliProcess._PENDING_HANDLE = None
    ve.VivariumEcoliProcess(
        config={"sim_data_path": "x", "agent_id": "00"}, core=build_core())

    assert seen["agent_id"] == "00"


# ---------------------------------------------------------------------------
# Hop 2: the lineage drivers must WALK the phylogeny
# ---------------------------------------------------------------------------

class _FakeProc:
    _handle = None

    def divide(self):
        return {"bulk": {}, "unique": {}, "environment": {}, "boundary": {}}


def _drive_pbg_lineage(monkeypatch, tmp_path, generations=3):
    """Run the real pbg lineage loop with the fork stubbed out; return the
    ``agent_id`` seen at each generation's BUILD and at each generation's
    EMITTER, plus the emitter's generation labels."""
    builds, emits = [], []

    def _fake_composite(**kw):
        agent_id = kw["agent_id"]
        builds.append(agent_id)
        agent_state = {"listeners": {"mass": {"cell_mass": 400.0}}}

        class _Comp:
            # ⚠ Keyed by the id it was BUILT with: the loop reads state back by
            # the same variable it builds with, so a driver that advances one
            # and not the other fails here rather than passing quietly.
            state = {"agents": {agent_id: agent_state}}

            def run(self, n):
                return None

        return _Comp(), {"process": _FakeProc(), "agent_id": agent_id}

    class _FakeEmitter:
        def __init__(self, agent_id, generation):
            emits.append((agent_id, generation))

        def update(self, payload):
            return None

        def advance_generation(self, *, agent_id, success=True, metadata=None):
            # Lineage-scoped emitter: the driver builds ONE emitter (generation
            # 1) and advances its partition label in place at each division,
            # instead of building a fresh emitter per generation. Record the
            # advance exactly as a build would be, so the generation-progression
            # invariant below still sees one (agent_id, generation) per
            # generation — the emitter's label must still walk the phylogeny.
            emits.append((agent_id, len(agent_id)))

        def close(self, success=True):
            return None

    import v2ecoli.library.xarray_run as xr

    # ⚠ These two are imported INSIDE the driver, so they must be patched on
    # their defining module — a patch on `ve` never binds.
    monkeypatch.setattr(xr, "_build_emitter",
                        lambda **kw: _FakeEmitter(kw["agent_id"], kw["generation"]))
    monkeypatch.setattr(xr, "_filter_agent_state", lambda state, view: state)
    monkeypatch.setattr(ve, "build_vivarium_ecoli_composite", _fake_composite)
    monkeypatch.setattr(ve, "_dperiod_should_divide", lambda handle: (True, 2))
    monkeypatch.setattr(ve, "_vecoli_config_summary", lambda *a, **k: {})

    out = ve.run_vivarium_ecoli_pbg_multigen(
        store_path=str(tmp_path / "store.zarr"), sim_data_path="x",
        max_generations=generations, max_steps_per_gen=40, chunk=20)
    return builds, emits, out


def test_pbg_lineage_advances_the_agent_id_it_BUILDS_with(monkeypatch, tmp_path):
    builds, _emits, out = _drive_pbg_lineage(monkeypatch, tmp_path)
    assert out["generations"] == 3
    assert builds == ["0", "00", "000"], (
        "the wrapped fork was rebuilt as the founder every generation, so "
        "`generation = len(agent_id)` reads 1 forever")


def test_the_zarr_GENERATION_LABEL_and_the_fork_GENERATION_INDEX_agree(
        monkeypatch, tmp_path):
    """⭐ The invariant that makes a graded result mean what it says.

    The emitter labels partitions 1-based (generation 1 = founder) and the fork
    compares ``induction_gen`` against ``len(agent_id)``. If those two ever
    disagree, a card reports induction in a generation the model never induced —
    a shift-by-one nobody can see in the numbers, because both sides look
    internally consistent."""
    builds, emits, _out = _drive_pbg_lineage(monkeypatch, tmp_path)
    # ⚠ ASSERT THE LENGTHS FIRST. `zip` truncates to the shorter list. The
    # lineage-scoped emitter is built ONCE (generation 1) and advanced per
    # division; `emits` records the build plus each advance_generation, so it
    # must still hold one (agent_id, generation) per generation. If a division
    # ever stops calling advance_generation, every later generation collapses
    # into generation 1's partition and `emits` shrinks — without this length
    # check the loop would compare exactly one pair and call it agreement.
    assert len(builds) == len(emits) == 3
    for agent_id, (emit_id, generation) in zip(builds, emits):
        assert len(agent_id) == generation
        assert emit_id == agent_id, (
            "the emitter's phylogeny key and the cell's own key diverged — the "
            "partition would be labelled with a different cell's generation")


def test_standalone_multigen_driver_advances_it_too(monkeypatch):
    """The local (non-pbg) driver has the same fork underneath and had the same
    pin. Left alone it is a second way to produce an un-induced lineage that
    looks like a real one."""
    builds = []

    class _Handle:
        engine = None
        dry_mass_inc_dict = {}
        media_id = "minimal"

    def _fake_build(**kw):
        builds.append(kw["agent_id"])
        return _Handle()

    monkeypatch.setattr(ve, "build_vivarium_ecoli", _fake_build)
    # Divide immediately: dry_mass over threshold with >= 2 chromosomes.
    monkeypatch.setattr(ve, "cell_observables", lambda engine: {
        "dry_mass": 400.0, "cell_mass": 800.0, "bulk": {}, "unique": {},
        "environment": {}, "boundary": {}, "listeners": {}})
    monkeypatch.setattr(ve, "_n_chromosomes", lambda unique: 2)
    monkeypatch.setattr(ve, "_inc_to_fg", lambda inc: 0.0)
    monkeypatch.setattr(ve, "divide_cell", lambda state: ({}, {}))

    class _Engine:
        def run_for(self, t):
            return None

    _Handle.engine = _Engine()

    out = ve.run_vivarium_ecoli_multigen(
        sim_data_path="x", max_generations=3, max_steps_per_gen=40, chunk=20)
    assert out["generations"] == 3
    assert builds == ["0", "00", "000"]


# ---------------------------------------------------------------------------
# The hop that actually reaches the fork
# ---------------------------------------------------------------------------

def test_agent_id_lands_on_the_forks_own_config_key_BEFORE_the_build(monkeypatch):
    """⭐⭐ THE HOP THE OTHER TESTS CANNOT SEE.

    Everything above proves the id travels through v2ecoli. This proves it is
    handed to the fork on the key the fork actually reads: ``LoadSimData`` is
    constructed as ``LoadSimData(**self.config)`` from ``Ecoli(config)``, so
    ``sim.config["agent_id"]`` is the whole interface — and it must be set
    BEFORE ``build_ecoli()``, because that is when the composer loads sim_data
    and applies the staged shift. Set it afterwards and every hop still passes
    while nothing induces.

    ⚠ Captured INSIDE the fake ``build_ecoli`` for exactly that reason: asserting
    on ``sim.config`` after the call cannot distinguish the two orders.
    """
    at_build = {}

    class _FakeSim:
        def __init__(self):
            self.config = {"processes": {}, "emit_paths": []}
            self.generated_initial_state = {"environment": {"media_id": "minimal"}}
            self.ecoli = type("_C", (), {"processes": {}, "steps": {}, "flow": {},
                                         "topology": {}})()

        @classmethod
        def from_cli(cls):
            return cls()

        def build_ecoli(self):
            at_build["agent_id"] = self.config.get("agent_id", "<absent>")

    class _StubEngine:
        def __init__(self, **kw):
            pass

    import vivarium.core.engine as vengine
    monkeypatch.setattr(vengine, "Engine", _StubEngine)
    monkeypatch.setattr(ve, "_ensure_upstream", lambda: {"EcoliSim": _FakeSim})

    ve.build_vivarium_ecoli(sim_data_path="/nonexistent.cPickle", agent_id="00")

    assert at_build["agent_id"] == "00", (
        "the fork was built without the caller's agent_id, so LoadSimData read "
        "generation 1 and no internal_shift_dict entry could apply")


# ---------------------------------------------------------------------------
# The DECLARED composite — a second, independent surface
# ---------------------------------------------------------------------------

def test_declared_vecoli_composite_forwards_agent_id(monkeypatch):
    """⭐ The `vecoli` composite generator is a SEPARATE hop with its own callers
    (study YAML and the workbench), and nothing above this test touches it —
    deleting its two forwarding lines left the whole suite green."""
    from v2ecoli.composites import vecoli as vc

    seen_build, seen_cfg = {}, {}

    def _fake_build(**kw):
        seen_build.update(kw)
        return type("_H", (), {})()

    def _fake_init(self, config=None, core=None):
        seen_cfg.update(config or {})

    monkeypatch.setattr(ve, "build_vivarium_ecoli", _fake_build)
    monkeypatch.setattr(ve.VivariumEcoliProcess, "__init__", _fake_init)
    monkeypatch.setattr(ve.VivariumEcoliProcess, "interface",
                        lambda self: {"inputs": {}, "outputs": {}})
    monkeypatch.setattr(vc, "_resolve_fork_config", lambda repo, cfg: (None, None))

    doc = vc.vecoli(agent_id="000")

    assert seen_build["agent_id"] == "000", (
        "the declared composite built the fork as the founder while declaring "
        "generation 3")
    assert seen_cfg["agent_id"] == "000"
    assert "000" in doc["state"]["agents"]


def test_declared_agent_id_param_states_it_is_the_generation_index():
    """⚠ The `parameters` description — NOT the function docstring — is what a
    study author and the workbench read. While it called the key 'the agent key
    under agents', a study could reasonably name two nodes `ref` and `test` and
    silently get generations 3 and 4 of a staged schedule."""
    from viva_superpowers.composite_generator import _REGISTRY, discover_generators
    if not _REGISTRY:
        discover_generators()
    import v2ecoli.composites  # noqa: F401 — force registration
    desc = _REGISTRY["v2ecoli.composites.vecoli.vecoli"].parameters["agent_id"]["description"]
    low = desc.lower()
    assert "generation" in low and "length" in low, (
        "the declared description does not say the id's LENGTH is the fork's "
        "generation index — the one thing a caller has to know before setting it")


# ---------------------------------------------------------------------------
# A saved initial state is keyed by the FOUNDER's id
# ---------------------------------------------------------------------------

def test_a_state_file_keyed_for_the_founder_fails_by_NAME_not_by_KeyError(monkeypatch):
    """The fork indexes a saved state as `full_initial_state["agents"][agent_id]`.
    Those files are written by the founder and carry "0", so a non-founder
    generation raises a bare `KeyError: '00'` from inside the composer with
    nothing naming the cause. Name it — and only when the missing key IS ours, so
    an unrelated KeyError still propagates untouched."""
    class _FakeSim:
        def __init__(self):
            self.config = {"processes": {}, "emit_paths": []}
            self.generated_initial_state = {}
            self.ecoli = type("_C", (), {"processes": {}, "steps": {}, "flow": {},
                                         "topology": {}})()

        @classmethod
        def from_cli(cls):
            return cls()

        def build_ecoli(self):
            raise KeyError(self.config["agent_id"])

    monkeypatch.setattr(ve, "_ensure_upstream", lambda: {"EcoliSim": _FakeSim})
    with pytest.raises(KeyError, match="no key|founder"):
        ve.build_vivarium_ecoli(sim_data_path="/nonexistent.cPickle", agent_id="00")


def test_an_unrelated_KeyError_is_NOT_relabelled(monkeypatch):
    """⚠ The other half. A rescue that swallows every KeyError would hide real
    composer failures behind a confident, wrong explanation."""
    class _FakeSim:
        def __init__(self):
            self.config = {"processes": {}, "emit_paths": []}
            self.generated_initial_state = {}
            self.ecoli = type("_C", (), {"processes": {}, "steps": {}, "flow": {},
                                         "topology": {}})()

        @classmethod
        def from_cli(cls):
            return cls()

        def build_ecoli(self):
            raise KeyError("some_other_missing_key")

    monkeypatch.setattr(ve, "_ensure_upstream", lambda: {"EcoliSim": _FakeSim})
    with pytest.raises(KeyError) as ei:
        ve.build_vivarium_ecoli(sim_data_path="/nonexistent.cPickle", agent_id="00")
    assert "some_other_missing_key" in str(ei.value)
    assert "founder" not in str(ei.value)
