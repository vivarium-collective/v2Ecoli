"""reactor_bird_coupled declares -- and its own sink emits -- the cell's
molecular state alongside the reactor/population stores.

Before this, the coupled generator had no ``emitters=`` declaration at all:
its only in-document sink was the per-agent emitter ``baseline()`` builds from
ecoli_baseline's OWN declaration (``bulk`` + ``listeners``, agent-relative),
so the composite itself never emitted ``reactor`` / ``population``, and every
runner carried its own allow-list (``scripts/run_mbp_tracked.py``'s
``COMMON_AGENT_PATHS`` + ``extra_root_paths`` -- the reduced
reactor/population/mass-subset/boundary set the K4 coupling ensemble shipped
with, no bulk, no FBA fluxes, so the ptools omics views had nothing to read).

The fix routes the OUTER generator's declaration to the per-agent sink
(``_helpers.set_enclosing_emitter_decl``): agent-relative roots stay in the
cell's frame, document-level roots (``reactor`` / ``population`` / ``lineage``)
are wired upward with ``('..', '..', root)``. The tests below pin both the
declaration and the emitted parquet columns; they fail on a tree without the
``emitters=`` declaration (no reactor/population columns) and on one without
the upward wiring (a build-time ``cannot go above the top`` or a missing
column).
"""
from __future__ import annotations

import glob
import os

import pytest

from v2ecoli.composites import _helpers as H

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")
needs_cache = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")

# The columns the ptools omics views need (rna / rxns / proteins / metabolites
# BinderException'd on exactly these for the K4 coupling ensemble) plus the
# reactor/population columns the coupled analyses already read.
MOLECULAR_COLUMNS = (
    "bulk__id",
    "bulk__count",
    "listeners__fba_results__base_reaction_fluxes",
    "listeners__mass__protein_mass",
    "listeners__mass__cell_mass",
)
COUPLING_COLUMNS = (
    "reactor__dissolved_o2",
    "reactor__dissolved_co2",
    "reactor__biomass",
    "population__cell_count",
    "population__biomass_concentration_gL",
)


def _coupled_decl() -> dict:
    from viva_superpowers.composite_generator import emitter_defaults
    from v2ecoli.composites.reactor_bird_coupled import reactor_bird_coupled

    decls = emitter_defaults(reactor_bird_coupled)
    assert decls, "reactor_bird_coupled declares no emitters= (regression)"
    return decls[0]


def test_reactor_bird_coupled_declares_molecular_and_document_roots():
    """The generator advertises a ParquetEmitter whose paths cover the cell's
    bulk + listeners AND the document-level reactor/population stores."""
    from v2ecoli.composites.reactor_bird_coupled import COUPLED_DOCUMENT_EMIT_ROOTS

    decl = _coupled_decl()
    assert decl["address"] == "local:ParquetEmitter"
    paths = set(decl["paths"])
    assert {"global_time", "bulk", "listeners"} <= paths
    assert {"reactor", "population"} <= paths
    # Every root the builder wires upward is one the generator declares.
    assert set(COUPLED_DOCUMENT_EMIT_ROOTS) <= paths


def test_enclosing_decl_wires_document_roots_upward_and_clears():
    """With the coupled declaration published as the enclosing decl, the parquet
    emit set keeps the cell roots agent-relative and wires the document roots
    out of the agent frame; clearing restores the baseline set."""
    from v2ecoli.composites.reactor_bird_coupled import COUPLED_DOCUMENT_EMIT_ROOTS

    decl = _coupled_decl()
    saved = (H._DEFAULT_EMITTER_DECL, H._ENCLOSING_EMITTER_DECL)
    try:
        H.set_default_emitter_decl(decl)
        H.set_enclosing_emitter_decl(decl, document_roots=COUPLED_DOCUMENT_EMIT_ROOTS)
        schema, topo = H._parquet_emit_set({"mass": {"cell_mass": "float"}})
        assert topo["bulk"] == ("bulk",)
        assert topo["listeners"] == ("listeners",)
        assert schema["listeners"] == {"mass": {"cell_mass": "float"}}
        for root in COUPLED_DOCUMENT_EMIT_ROOTS:
            assert topo[root] == ("..", "..", root)
            assert schema[root] == "node"

        H.set_enclosing_emitter_decl(None)
        assert H._ENCLOSING_EMITTER_DECL is None
        _, topo = H._parquet_emit_set({})
        # No enclosing decl -> a declared "reactor" root would be agent-relative.
        assert topo["reactor"] == ("reactor",)
    finally:
        H.set_default_emitter_decl(saved[0])
        H._ENCLOSING_EMITTER_DECL = saved[1]


def _columns_under(out_dir: str) -> set[str]:
    import polars as pl

    files = glob.glob(os.path.join(out_dir, "**", "history", "**", "*.pq"),
                      recursive=True)
    assert files, f"no history parquet written under {out_dir}"
    cols: set[str] = set()
    for f in files:
        cols |= set(pl.scan_parquet(f).collect_schema().names())
    return cols


def _close_agent_emitters(composite) -> None:
    """Flush the trailing partial batch of every per-agent ParquetEmitter."""
    for agent in (composite.state.get("agents") or {}).values():
        inst = (agent.get("emitter") or {}).get("instance")
        if inst is not None and hasattr(inst, "close"):
            inst.close()


@needs_cache
@pytest.mark.sim
def test_reactor_bird_coupled_declared_sink_emits_bulk_fba_and_reactor(tmp_path, monkeypatch):
    """The composite's OWN declared sink (no external override) writes one hive
    row per tick with bulk__id/bulk__count + FBA fluxes + mass AND the
    reactor/population columns."""
    from v2ecoli import build_composite

    # Sink under tmp: the declared ParquetEmitter resolves out_dir to
    # <workspace>/.pbg/parquet-runs.
    monkeypatch.setattr(H, "_find_workspace_root", lambda *a, **k: tmp_path)
    monkeypatch.setenv("V2ECOLI_EMITTER_EXPERIMENT_ID", "coupled-emit-test")
    H.set_parquet_emitter_override(None)
    H.set_emitter_override(None)
    H.set_null_emitter_override(False)

    c = build_composite("reactor_bird_coupled", seed=0, cache_dir=CACHE,
                        single_daughters=True)
    # The declarations are published only for the duration of the build.
    assert H._ENCLOSING_EMITTER_DECL is None
    assert H._DEFAULT_EMITTER_DECL is None

    edge = c.state["agents"]["0"]["emitter"]
    wires = edge["inputs"]
    assert list(wires["reactor"]) == ["..", "..", "reactor"]
    assert list(wires["population"]) == ["..", "..", "population"]
    assert list(wires["bulk"]) == ["bulk"]

    c.run(2)
    _close_agent_emitters(c)

    cols = _columns_under(str(tmp_path / ".pbg" / "parquet-runs"))
    missing = [k for k in MOLECULAR_COLUMNS + COUPLING_COLUMNS if k not in cols]
    assert not missing, f"missing emitted columns: {missing}"


@needs_cache
@pytest.mark.sim
def test_reactor_bird_coupled_parquet_override_keeps_document_roots(tmp_path):
    """The external parquet-override path (what the lineage runner uses) derives
    the same root set, so a coupled lineage also lands reactor + population
    next to bulk/listeners in one hive."""
    from v2ecoli import build_composite
    from v2ecoli.library.emitter_presets import parquet_vecoli

    H.set_parquet_emitter_override(parquet_vecoli(
        out_dir=str(tmp_path), experiment_id="coupled-override", agent_id="0"))
    try:
        c = build_composite("reactor_bird_coupled", seed=0, cache_dir=CACHE,
                            single_daughters=True)
    finally:
        H.set_parquet_emitter_override(None)
    c.run(2)
    _close_agent_emitters(c)

    cols = _columns_under(str(tmp_path))
    missing = [k for k in MOLECULAR_COLUMNS + COUPLING_COLUMNS if k not in cols]
    assert not missing, f"missing emitted columns: {missing}"
