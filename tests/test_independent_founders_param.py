"""Independent per-seed founders as an opt-in ``ecoli_baseline`` param.

By default every seed of a multiseed ensemble starts from the SAME cached
founder (``initial_state``), so the spread reflects only downstream per-process
stochasticity, not cell-to-cell founder variability. ``independent_founders`` +
``founder_sim_data`` opt in to re-drawing the founder per lineage_seed from the
given v2 ``simData.cPickle`` via ``_independent_founder_state``
(``LoadSimData(sim_data_path, seed).generate_initial_state()`` + a
save/load round-trip for byte-format parity). Unlike ``match_simdata`` (a
build-time single-reference overlay that is rejected in batch mode), this
threads through the batch runner so each per-seed baseline() build re-draws.

Hermetic: the cache loader + the 55-process execution layer are monkeypatched,
and ``_independent_founder_state`` (the only heavy piece — it loads sim_data) is
stubbed, so the real ``baseline()`` document builder runs end-to-end quickly.
"""
from __future__ import annotations

import numpy as np
import pytest

import v2ecoli.composites.ecoli_baseline as eb

BULK_DTYPE = np.dtype([('id', 'U20'), ('count', 'i8')])


def _fake_bundle():
    return {
        "initial_state": {
            "bulk": np.array([('A', 10), ('B', 20)], dtype=BULK_DTYPE),
            "environment": {"media_id": "minimal"},
        },
        "configs": {}, "unique_names": [], "dry_mass_inc_dict": {},
    }


@pytest.fixture
def hermetic_baseline_build(monkeypatch):
    monkeypatch.setattr(eb, "load_cache_bundle", lambda cache_dir: _fake_bundle())
    monkeypatch.setattr(eb, "build_execution_layers", lambda features=None: [])


def test_independent_founders_params_declared():
    entry = eb.baseline._composite_generator_entry
    assert entry.parameters["independent_founders"]["type"] == "boolean"
    assert entry.parameters["independent_founders"]["default"] is False
    assert entry.parameters["founder_sim_data"]["type"] == "string"
    assert entry.parameters["founder_sim_data"]["default"] == ""


def test_default_off_uses_cached_founder(hermetic_baseline_build, monkeypatch):
    """Default path: the per-seed re-draw is never invoked; t=0 bulk comes
    straight from the cache bundle, unchanged."""
    def _must_not_run(*a, **k):
        raise AssertionError("_independent_founder_state must not be called "
                             "when independent_founders is off")
    monkeypatch.setattr(eb, "_independent_founder_state", _must_not_run)

    doc = eb.baseline(cache_dir="fake_cache", seed=0)  # flag omitted
    bulk = doc["state"]["agents"]["0"]["bulk"]
    assert list(bulk["count"]) == [10, 20]  # from the cache, untouched


def test_flag_without_sim_data_is_noop(hermetic_baseline_build, monkeypatch):
    """independent_founders=True but founder_sim_data empty → still the cached
    founder (no path to re-draw from)."""
    monkeypatch.setattr(eb, "_independent_founder_state",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-draw without sim_data")))
    doc = eb.baseline(cache_dir="fake_cache", seed=0, independent_founders=True)
    assert list(doc["state"]["agents"]["0"]["bulk"]["count"]) == [10, 20]


def test_on_redraws_founder_per_seed(hermetic_baseline_build, monkeypatch):
    """With the flag + a sim_data path, the founder is re-drawn via
    _independent_founder_state (seeded by this lineage's seed) and REPLACES the
    cached one."""
    calls = []

    def fake_redraw(sim_data_path, seed, condition="basal"):
        calls.append((sim_data_path, seed, condition))
        return {"bulk": np.array([('A', 777), ('B', 888)], dtype=BULK_DTYPE),
                "environment": {"media_id": "minimal"}}

    monkeypatch.setattr(eb, "_independent_founder_state", fake_redraw)
    doc = eb.baseline(cache_dir="fake_cache", seed=5,
                      independent_founders=True,
                      founder_sim_data="/fake/simData.cPickle")
    bulk = doc["state"]["agents"]["0"]["bulk"]
    assert dict(zip(bulk["id"], bulk["count"])) == {"A": 777, "B": 888}
    assert len(calls) == 1
    path, seed, _cond = calls[0]
    assert path == "/fake/simData.cPickle" and seed == 5  # this lineage's seed


def test_flag_threads_through_batch_workflow_config():
    """Unlike match_simdata (rejected in batch), independent_founders threads
    into the batch runner config so every per-seed baseline() build re-draws."""
    from v2ecoli.steps.batch_baseline_runner import build_workflow_config
    cfg = build_workflow_config(
        cache_dir="fake_cache", n_seeds=3, n_generations=1,
        independent_founders=True, founder_sim_data="/fake/simData.cPickle")
    assert cfg.get("independent_founders") is True
    assert cfg.get("founder_sim_data") == "/fake/simData.cPickle"
