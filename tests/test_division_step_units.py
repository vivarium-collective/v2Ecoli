"""Regression tests for the Division / MarkDPeriod steps.

Two concerns are covered:

1. Quantity[fg] handling — under units-on-ports, listeners.mass.dry_mass is a
   pint Quantity[fg]. Division.next_update must do its arithmetic/comparison in
   plain fg floats; process-bigraph swallows exceptions inside next_update, so a
   raise would silently leave division_threshold stuck and the cell never split.

2. d_period gating — vEcoli divides D_period after replication completes (the
   flag MarkDPeriod raises at the chromosome's division_time) and IGNORES the
   dry-mass threshold (ecoli/processes/cell_division.py). v2's Division step
   defaults to that behavior (``d_period=True``); the legacy mass-distribution
   threshold is used only when ``d_period=False``. Before the fix the mass
   threshold always won, dividing slow-growth cells ~D_period too early.
"""
import numpy as np

from v2ecoli.steps.division import Division, MarkDPeriod
from v2ecoli.types.quantity import ureg as units


def _make_division(d_period=False):
    d = Division.__new__(Division)
    d.initialize({"dry_mass_inc_dict": {"minimal": units.Quantity(150.0, "fg")},
                  "d_period": d_period})
    return d


def _chroms(n):
    # _entryState is int8 in real MetadataArrays (attrs() views it as bool).
    arr = np.zeros(n, dtype=[("division_time", "f8"),
                             ("has_triggered_division", "?"),
                             ("_entryState", "i1")])
    arr["_entryState"] = 1
    return arr


# --- legacy mass-distribution path (d_period=False) --------------------------

def test_threshold_init_handles_quantity_dry_mass():
    """The threshold-init branch returns a numeric threshold (not raise) when
    dry_mass is a pint Quantity[fg]."""
    d = _make_division(d_period=False)
    states = {
        "division_threshold": "mass_distribution",
        "media_id": "minimal",
        "listeners": {"mass": {"dry_mass": units.Quantity(400.0, "fg")}},
    }
    update = d.next_update(1.0, states)
    thr = update["division_threshold"]
    assert isinstance(thr, float)
    assert thr == 550.0          # 400 + 150 * multiplier(=1)


def test_division_check_handles_quantity_dry_mass():
    """dry_mass < threshold must compare without raising (Quantity vs float)."""
    d = _make_division(d_period=False)
    states = {
        "division_threshold": 760.0,
        "listeners": {"mass": {"dry_mass": units.Quantity(400.0, "fg")}},
        "unique": {"full_chromosome": None},
    }
    assert d.next_update(1.0, states) == {}


# --- D-period path (d_period=True, the vEcoli default) ------------------------

def test_d_period_ignores_mass_threshold():
    """Under d_period, a cell over its mass threshold with 2 chromosomes must
    NOT divide unless the D-period `divide` flag is set — the dry-mass threshold
    is ignored. (This is the fix: previously the mass threshold force-divided
    slow-growth cells ~D_period early.)"""
    d = _make_division(d_period=True)
    states = {
        "division_threshold": 100.0,                 # far below dry_mass
        "listeners": {"mass": {"dry_mass": units.Quantity(99999.0, "fg")}},
        "unique": {"full_chromosome": _chroms(2)},   # replication done
        "divide": False,                             # D-period not elapsed
        "global_time": 5000.0,
    }
    assert d.next_update(1.0, states) == {}          # mass ignored -> no division


def test_mark_d_period_raises_divide_at_division_time():
    """MarkDPeriod sets the `divide` flag once global_time reaches the
    chromosome's division_time (with >= 2 full chromosomes)."""
    m = MarkDPeriod.__new__(MarkDPeriod)
    m.initialize({})
    chrom = _chroms(2)
    chrom["division_time"] = [3000.0, 5064.0]
    # before division_time -> no divide
    before = m.next_update(1.0, {"full_chromosome": chrom, "global_time": 2000.0})
    assert not before.get("divide")
    # at/after the earliest untriggered division_time -> divide
    after = m.next_update(1.0, {"full_chromosome": chrom, "global_time": 3000.0})
    assert after.get("divide") is True


# --- injected_processes survive division -------------------------------------

def test_division_threads_injected_processes_to_daughter_baseline(monkeypatch):
    """The DivisionStep must re-supply its ``injected_processes`` spec to each
    daughter's ``baseline()`` rebuild, so a swapped/injected process (e.g.
    metabolism-redux) survives cell division instead of reverting to the plain
    FBA baseline. Before the fix ``_build_daughter_doc`` called ``baseline(...)``
    without ``injected_processes`` and the daughter re-realization crashed with
    ``KeyError: 'current_timeline'`` at the first division.

    Fast proxy for the 40-min multi-gen run: monkeypatch ``baseline`` (and the
    heavy biology helpers) so a division trigger drives ``_build_daughter_doc``
    without any real ParCa/sim, then assert the captured kwargs carry the spec.
    """
    injected = {
        "fork_repo": "/Users/x/vEcoli",
        "fork_sim_data": "/Users/x/simData.cPickle",
        "swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"},
    }

    # 1. initialize() captures the spec off the config.
    d = Division.__new__(Division)
    d.core = None  # normally set by the Edge constructor; unused here (baseline patched)
    d.initialize({"agent_id": "0", "d_period": True,
                  "injected_processes": injected})
    assert d._injected_processes == injected

    # Capture every baseline() call the daughter rebuild makes.
    calls = []

    def _fake_baseline(**kwargs):
        calls.append(kwargs)
        # Minimal doc shape _build_daughter_doc consumes.
        agent = {"listeners": {}, "division": None}
        return {"state": {"agents": {"0": agent}}}

    # next_update imports these lazily from their source modules at call time,
    # so patch the SOURCE symbols (patching the division module would miss them).
    import v2ecoli.composites.ecoli_baseline as _eb
    import v2ecoli.library.division as _libdiv
    import v2ecoli.composites._helpers as _h
    monkeypatch.setattr(_eb, "baseline", _fake_baseline)
    monkeypatch.setattr(_eb, "seed_mass_listener", lambda *a, **k: None)
    _proxy_bulk = {"count": np.array([1, 2, 3])}
    monkeypatch.setattr(_libdiv, "divide_cell",
                        lambda cell_data: ({"bulk": dict(_proxy_bulk)},
                                           {"bulk": dict(_proxy_bulk)}))
    monkeypatch.setattr(_h, "finalize_emitter_for_agent", lambda *a, **k: None)

    # Fabricate a state that fires the d_period division trigger.
    states = {
        "bulk": {}, "unique": {"full_chromosome": _chroms(2)},
        "listeners": {"mass": {"dry_mass": units.Quantity(500.0, "fg")}},
        "environment": {}, "boundary": {},
        "global_time": 3600.0, "divide": True,
    }
    update = d.next_update(1.0, states)

    # Division actually fired (two daughters added), and BOTH daughter baseline()
    # calls received the injected_processes spec unchanged (incl. fork_sim_data,
    # the path build_fork_config needs to re-supply current_timeline).
    assert "_add" in update["agents"]
    assert len(calls) == 2
    for kwargs in calls:
        assert kwargs.get("injected_processes") == injected
        assert kwargs["injected_processes"]["fork_sim_data"] == injected["fork_sim_data"]


# --- injected agent-root stores survive division ------------------------------
# sms-ecoli#166 P0 items 2 and 3. divide_cell() used to return ONLY bulk /
# unique / environment / boundary, so every store an injected process wires at
# the agent root came back fresh (zeroed) in both daughters.


def _restore_division_registries(monkeypatch):
    import v2ecoli.library.division as _div

    monkeypatch.setattr(_div, "STORE_DIVIDERS", dict(_div.STORE_DIVIDERS))
    monkeypatch.setattr(
        _div, "CARRIED_LISTENER_PATHS", list(_div.CARRIED_LISTENER_PATHS))


def _minimal_cell_state():
    """A cell_state divide_cell can split without a real ParCa build."""
    bulk = np.zeros(3, dtype=[("id", "U8"), ("count", "i8")])
    bulk["count"] = [10, 20, 30]
    chroms = np.zeros(2, dtype=[("domain_index", "i8"), ("_entryState", "i1")])
    chroms["domain_index"] = [0, 1]
    chroms["_entryState"] = 1
    domains = np.zeros(2, dtype=[("domain_index", "i8"),
                                 ("child_domains", "i8", (2,)),
                                 ("_entryState", "i1")])
    domains["domain_index"] = [0, 1]
    domains["child_domains"] = -1
    domains["_entryState"] = 1
    return {
        "bulk": bulk,
        "unique": {
            "full_chromosome": chroms,
            "chromosome_domain": domains,
            "active_RNAP": np.zeros(
                0, dtype=[("domain_index", "i8"), ("unique_index", "i8"),
                          ("_entryState", "i1")]),
            "RNA": np.zeros(
                0, dtype=[("is_full_transcript", "?"), ("RNAP_index", "i8"),
                          ("unique_index", "i8"), ("_entryState", "i1")]),
        },
        "environment": {"media_id": "minimal"},
        "boundary": {"external": {"GLC": 1.0}},
    }


def test_divide_cell_copies_extra_root_stores_to_both_daughters():
    from v2ecoli.library.division import divide_cell

    cell = _minimal_cell_state()
    dosed = np.array([[7.5]])
    cell["fields"] = {"_type": "map[overwrite[array[float]]]",
                      "tetracycline": dosed}
    cell["imposed_flux_bounds"] = {"RXN": 3.0}
    cell["periplasm"] = {"global": {"volume": 0.2}}
    # never carried: rebuilt per daughter by baseline()
    cell["allocator_rng"] = np.random.RandomState(seed=7)
    cell["process_state"] = {"x": 1}

    d1, d2 = divide_cell(cell)
    for d in (d1, d2):
        assert d["fields"]["tetracycline"] == dosed
        assert d["fields"]["_type"] == "map[overwrite[array[float]]]"
        assert d["imposed_flux_bounds"] == {"RXN": 3.0}
        assert d["periplasm"] == {"global": {"volume": 0.2}}
        assert "allocator_rng" not in d
        assert "process_state" not in d
    # independent deep copies, not a shared reference
    d1["imposed_flux_bounds"]["RXN"] = 99.0
    assert d2["imposed_flux_bounds"]["RXN"] == 3.0


def test_divide_cell_applies_a_supplied_divider_for_a_named_store():
    """A store may declare a divider and is then SPLIT the way the fork splits
    ``pg_cellwall`` — copy stays the default for everything else."""
    from v2ecoli.library.division import divide_cell

    cell = _minimal_cell_state()
    cell["pg_cellwall"] = np.array([8, 8, 8])
    cell["fields"] = {"drug": 1.0}
    seen = []

    def _fake_divider(value):
        seen.append(value)
        return value // 2, value - value // 2

    d1, d2 = divide_cell(cell, dividers={"pg_cellwall": _fake_divider})
    assert len(seen) == 1                          # called once, on the mother
    assert list(d1["pg_cellwall"]) == [4, 4, 4]
    assert list(d2["pg_cellwall"]) == [4, 4, 4]
    assert d1["fields"] == {"drug": 1.0}            # default policy unchanged


def test_divide_cell_carries_declared_listener_leaves_only(monkeypatch):
    _restore_division_registries(monkeypatch)
    from v2ecoli.library.division import (
        divide_cell, register_carried_listener_path)

    register_carried_listener_path("listeners.peptidoglycan_shape.lysed")
    cell = _minimal_cell_state()
    cell["listeners"] = {"peptidoglycan_shape": {"lysed": True, "murein": 3},
                         "mass": {"dry_mass": 500.0}}
    d1, d2 = divide_cell(cell)
    for d in (d1, d2):
        assert d["_carried_listeners"] == {"peptidoglycan_shape": {"lysed": True}}
        assert "listeners" not in d


def test_division_step_daughter_overlay_includes_extra_stores(monkeypatch):
    """The Division step's daughter document overlay is no longer limited to the
    four core keys: an injected agent-root store visible on the step's ports is
    divided (copy by default) and MERGED onto the fresh daughter node, keeping
    that node's declared ``_type``."""
    d = Division.__new__(Division)
    d.core = None
    d.initialize({"agent_id": "0", "d_period": True})

    built = []

    def _fake_baseline(**kwargs):
        # What a fresh build looks like: `fields` typed and zero-seeded.
        agent = {
            "listeners": {}, "division": None,
            "fields": {"_type": "map[overwrite[array[float]]]",
                       "tetracycline": np.zeros((1, 1))},
        }
        built.append(agent)
        return {"state": {"agents": {"0": agent}}}

    import v2ecoli.composites.ecoli_baseline as _eb
    import v2ecoli.composites._helpers as _h
    monkeypatch.setattr(_eb, "baseline", _fake_baseline)
    monkeypatch.setattr(_eb, "seed_mass_listener", lambda *a, **k: None)
    monkeypatch.setattr(_h, "finalize_emitter_for_agent", lambda *a, **k: None)

    dosed = np.array([[7.5]])
    states = {
        "bulk": {}, "unique": {"full_chromosome": _chroms(2)},
        "listeners": {"mass": {"dry_mass": units.Quantity(500.0, "fg")}},
        "environment": {}, "boundary": {},
        "media_id": "minimal",
        "global_time": 3600.0, "divide": True,
        "fields": {"_type": "map[overwrite[array[float]]]",
                   "tetracycline": dosed},
        "imposed_flux_bounds": {"RXN": 3.0},
    }
    _proxy_bulk = {"count": np.array([1, 2, 3])}

    import v2ecoli.library.division as _libdiv
    _real_divide_cell = _libdiv.divide_cell

    def _light_divide_cell(cell_data, dividers=None):
        d1, d2 = ({"bulk": dict(_proxy_bulk)}, {"bulk": dict(_proxy_bulk)})
        e1, e2 = _libdiv.divide_extra_stores(cell_data, dividers)
        d1.update(e1)
        d2.update(e2)
        return d1, d2

    monkeypatch.setattr(_libdiv, "divide_cell", _light_divide_cell)
    update = d.next_update(1.0, states)

    assert "_add" in update["agents"]
    assert len(built) == 2
    for agent in built:
        assert agent["fields"]["_type"] == "map[overwrite[array[float]]]"
        assert agent["fields"]["tetracycline"] == dosed
        assert agent["imposed_flux_bounds"] == {"RXN": 3.0}
        # a Division PORT name is not an agent-root store
        assert "media_id" not in agent
    assert _real_divide_cell is not None
