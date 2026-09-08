"""Cumulative lineage-time offset exposed to injected processes.

The lineage rebuilds a FRESH inner composite each generation, whose
``global_time`` restarts at 0 (capped by ``max_duration_per_gen``). An injected
process that needs to reason about ABSOLUTE / cumulative lineage time (beyond
one generation's local clock) therefore cannot get it from ``global_time``
alone. ``LineageProcess`` accumulates the summed duration of the prior
generations (``_lineage_offset``) and exposes it to EVERY injected process as
``lineage_time_offset`` (see ``_apply_lineage_offset``), so such a process can
read ``global_time + lineage_time_offset``. Emitted ``global_time`` stays
per-generation (unchanged). Domain- and process-agnostic: no process is named.
"""
from v2ecoli.workflow.lineage import LineageProcess, _apply_lineage_offset


def _injected(*process_names):
    return {
        "add_processes": list(process_names),
        "process_configs": {name: {"knob": name} for name in process_names},
    }


# ---- the pure threading helper -------------------------------------------

def test_apply_lineage_offset_sets_offset_on_every_injected_process():
    inj = _injected("proc_a", "proc_b", "proc_c")
    out = _apply_lineage_offset(inj, 9500.0)
    for name in ("proc_a", "proc_b", "proc_c"):
        assert out["process_configs"][name]["lineage_time_offset"] == 9500.0
        # the process's own config is otherwise untouched
        assert out["process_configs"][name]["knob"] == name


def test_apply_lineage_offset_is_non_destructive():
    inj = _injected("proc_a")
    _apply_lineage_offset(inj, 9500.0)
    # the caller's original config is not mutated
    assert "lineage_time_offset" not in inj["process_configs"]["proc_a"]


def test_apply_lineage_offset_skips_non_dict_process_config():
    # a "default"-sentinel (non-dict) process config is left alone, not crashed
    inj = {"add_processes": ["p"], "process_configs": {"p": "default"}}
    out = _apply_lineage_offset(inj, 1234.0)
    assert out["process_configs"]["p"] == "default"


def test_apply_lineage_offset_handles_missing_block():
    assert _apply_lineage_offset(None, 5.0) in (None, {})
    assert _apply_lineage_offset({}, 5.0) == {}


# ---- LineageProcess accumulates + exposes the offset ----------------------

def _make(monkeypatch, generations, divide_after, injected):
    lp = LineageProcess.__new__(LineageProcess)
    lp.config = {
        "cache_dir": "x", "seed": 0, "lineage_seed": 0, "variant_index": 0,
        "variant_name": "baseline", "config_overrides": {},
        "generations": generations, "single_daughters": True,
        "experiment_id": "t", "out_dir": "out/t",
        "max_duration_per_gen": 100000.0,
        "initial_carry_state_path": "", "initial_generation_index": 0,
        "daughter_state_out_path": "", "injected_processes": injected,
    }
    lp.initialize(lp.config)
    seen = []

    def fake_build():
        # record the offset injected processes WOULD be built with this gen
        seen.append(lp._lineage_offset)
        lp._gen_elapsed = 0.0

    def fake_run_until_division(interval):
        lp._gen_elapsed += interval
        divided = lp._gen_elapsed >= divide_after
        daughter = {"bulk": {}, "unique": {}} if divided else None
        return divided, daughter, 100.0

    monkeypatch.setattr(lp, "_build_generation", fake_build)
    monkeypatch.setattr(lp, "_run_until_division", fake_run_until_division)
    return lp, seen


def test_lineage_offset_accumulates_prior_generation_durations(monkeypatch):
    # each generation runs exactly `divide_after` seconds before dividing
    lp, seen = _make(monkeypatch, generations=4, divide_after=2500,
                     injected=_injected("proc_a"))
    for _ in range(50):
        # interval == the generation's duration: one update completes one gen
        if lp.update({}, 2500.0).get("complete"):
            break
    # gen0 built with offset 0; gen1 with 2500; gen2 with 5000; gen3 with 7500
    assert seen == [0.0, 2500.0, 5000.0, 7500.0]


def test_lineage_offset_starts_at_zero_first_generation(monkeypatch):
    lp, seen = _make(monkeypatch, generations=1, divide_after=10,
                     injected=_injected("proc_a"))
    for _ in range(50):
        if lp.update({}, 1.0).get("complete"):
            break
    assert seen == [0.0]
