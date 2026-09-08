"""Regression tests for item 101's n_workers durable fix.

An earlier concrete default of ``n_workers=2`` silently shadowed the cluster-derived
``RAY_SHARDS_DEFAULT`` env var viva-api's own dispatch code computes (``RayProtocolRuntime``
only reads it when given ``None`` explicitly) — see ``batch_lineage_ray``'s own module
docstring. These tests pin the fix: the default is ``None``, negative/zero overrides still
reject, and ``prewarm_lineage_pool`` forwards whatever it's given to the protocol runtime
unchanged.
"""
import inspect
from unittest.mock import patch

import pytest

from v2ecoli.composites.lineage_ray_batch import lineage_ray_batch
from v2ecoli.workflow.batch_lineage_ray import (
    build_lineage_ray_batch_document,
    prewarm_lineage_pool,
)


def test_lineage_ray_batch_n_workers_defaults_to_none():
    sig = inspect.signature(lineage_ray_batch)
    assert sig.parameters["n_workers"].default is None


def test_prewarm_lineage_pool_rejects_zero():
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        prewarm_lineage_pool(core=object(), n_workers=0)


def test_prewarm_lineage_pool_rejects_negative():
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        prewarm_lineage_pool(core=object(), n_workers=-1)


def test_prewarm_lineage_pool_forwards_none_to_runtime():
    # None must NOT be rejected by the >= 1 check, and must reach the runtime
    # unchanged so it can fall through to RAY_SHARDS_DEFAULT — the actual fix.
    core = object()
    with patch("process_bigraph.protocols.ray.get_or_create_runtime") as mock_runtime:
        prewarm_lineage_pool(core=core, n_workers=None)
    mock_runtime.assert_called_once_with(core, n_shards_default=None)


def test_prewarm_lineage_pool_forwards_explicit_override():
    # An explicit override must still reach the runtime unchanged (regression
    # guard: the fix must not break the deliberate-cap use case).
    core = object()
    with patch("process_bigraph.protocols.ray.get_or_create_runtime") as mock_runtime:
        prewarm_lineage_pool(core=core, n_workers=5)
    mock_runtime.assert_called_once_with(core, n_shards_default=5)


# ---------------------------------------------------------------------------
# N2: variant_grid crosses (variant, seed) into real per-node lineages
#     (on top of #663's injected_processes/emitter_arg exposure).
# N4: required_leaves makes a missing KPI leaf raise instead of silently skipping.
# ---------------------------------------------------------------------------

def _nodes(doc):
    # Process nodes are top-level state keys (address ray:LineageProcess); "lineages"
    # is the output target sub-store their outputs write into, not a node itself.
    return {k: v for k, v in doc["state"].items()
            if isinstance(v, dict) and v.get("address") == "ray:LineageProcess"}


def test_no_variant_grid_preserves_seeds_only_shape():
    doc = build_lineage_ray_batch_document(n_seeds=2, n_generations=1, base_seed=5)
    nodes = _nodes(doc)
    assert set(nodes) == {"lineage_0005", "lineage_0006"}
    assert all(n["config"]["variant_index"] == 0 for n in nodes.values())


def test_variant_grid_crosses_variants_and_seeds():
    grid = [
        {"variant_name": "control", "config_overrides": {"a": 1}},
        {"variant_name": "highVioA", "config_overrides": {"a": 2}},
    ]
    doc = build_lineage_ray_batch_document(
        n_seeds=2, n_generations=1, variant_grid=grid, config_overrides={"shared": 9})
    nodes = _nodes(doc)
    # 2 variants x 2 seeds = 4 real nodes, variant+seed encoded in the name.
    assert set(nodes) == {
        "lineage_v000_s0000", "lineage_v000_s0001",
        "lineage_v001_s0000", "lineage_v001_s0001",
    }
    v0 = nodes["lineage_v000_s0001"]["config"]
    v1 = nodes["lineage_v001_s0000"]["config"]
    assert v0["variant_index"] == 0 and v0["variant_name"] == "control"
    assert v1["variant_index"] == 1 and v1["variant_name"] == "highVioA"
    # shared + per-variant overrides merge; per-variant wins on conflict.
    assert v0["config_overrides"] == {"shared": 9, "a": 1}
    assert v1["config_overrides"] == {"shared": 9, "a": 2}
    assert all(n["address"] == "ray:LineageProcess" for n in nodes.values())


def test_variant_grid_honours_explicit_variant_index():
    doc = build_lineage_ray_batch_document(
        n_seeds=1, n_generations=1, variant_grid=[{"variant_index": 7}])
    (node,) = _nodes(doc).values()
    assert node["config"]["variant_index"] == 7


def test_variant_grid_threads_injection_and_emitter_arg_per_node():
    # The cross-product loop must carry #663's injected_processes + emitter_arg
    # into EVERY (variant, seed) node, not just the first.
    inj = {"swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"}}
    ea = {"view": [{"root": ("listeners", "fba_results"),
                    "variables": {"violacein_production_flux": [{"path": "violacein_production_flux"}]}}]}
    doc = build_lineage_ray_batch_document(
        n_seeds=2, n_generations=1,
        variant_grid=[{"variant_name": "a"}, {"variant_name": "b"}],
        injected_processes=inj, emitter_arg=ea)
    nodes = _nodes(doc)
    assert len(nodes) == 4
    for cfg in (n["config"] for n in nodes.values()):
        assert cfg["injected_processes"] == inj
        assert cfg["emitter_arg"] == ea


def test_required_leaves_raises_on_missing_kpi():
    from v2ecoli.core import build_core
    from v2ecoli.workflow.lineage import LineageProcess
    core = build_core()
    cfg = {
        "cache_dir": "out/cache", "seed": 0, "generations": 1, "emitter": "xarray",
        "emitter_arg": {
            "view": [{"root": ("listeners", "fba_results"),
                      "variables": {"violacein_production_flux": [{"path": "violacein_production_flux"}]}}],
            "required_leaves": ["listeners.fba_results.violacein_production_flux"],
        },
    }
    proc = LineageProcess(cfg, core=core)
    # emit_cell has only mass -> the KPI leaf is absent -> must raise, not skip.
    with pytest.raises(ValueError, match="required emitter leaf"):
        proc._open_xarray_emitter({"listeners": {"mass": {"dry_mass": 300.0}}})
