"""``run_multigen_parquet``: an EMPTY ``emit_paths`` is the composite's DECLARED
emit set; a non-empty one stays an explicit allow-list -- and the mbp runner
opts ONLY its single-lineage coupled variant into the declared set.

The K4 reactor-coupling ensemble (Run 1) went through
``scripts/run_mbp_tracked.py`` -> ``run_multigen_parquet(emit_paths=
COMMON_AGENT_PATHS + ...)`` and so shipped the reduced
reactor/population/mass-subset/boundary hive: no ``bulk__*``, no
``listeners__fba_results__*``, nothing for the ptools omics views to read.
v2ecoli#743 gave the coupled generator an ``emitters=`` declaration; this
makes the runner honor it -- for that ONE variant.

The OOM guard is the point of the second half: the population variants keep
their explicit reduced list, and an explicit list must never grow bulk. Full
bulk x agents x generations is the emit-path blow-up the reduced list exists
to avoid (#754/#776), so widening ``COMMON_AGENT_PATHS`` globally is exactly
the regression these tests refuse.
"""
from __future__ import annotations

import glob
import importlib.util
import inspect
import os
from pathlib import Path

import pytest

CACHE = os.environ.get("V2ECOLI_CACHE", "out/cache")
needs_cache = pytest.mark.skipif(
    not os.path.isdir(CACHE), reason=f"ParCa cache {CACHE} not present")

COUPLED_VARIANT = "reactor-bird-coupled-batch-multigen"
POPULATION_VARIANT = "aggregator-cpa1e6-multigen"

MOLECULAR_COLUMNS = (
    "bulk__id",
    "bulk__count",
    "listeners__fba_results__base_reaction_fluxes",
    "listeners__mass__protein_mass",
)
COUPLING_COLUMNS = (
    "reactor__dissolved_o2",
    "reactor__biomass",
    "population__cell_count",
    "population__biomass_concentration_gL",
)
# What the reduced allow-list carries (COMMON_AGENT_PATHS + population roots).
REDUCED_COLUMNS = (
    "listeners__mass__cell_mass",
    "listeners__mass__dry_mass",
    "population__cell_count",
)


def _runner_module():
    p = Path(__file__).resolve().parents[1] / "scripts" / "run_mbp_tracked.py"
    spec = importlib.util.spec_from_file_location("_mbp_tracked_declared_emit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _variant(mod, name):
    return next(v for v in mod.VARIANTS if v[0] == name)


def _columns_under(root: str) -> set[str]:
    import polars as pl

    files = glob.glob(os.path.join(root, "**", "history", "**", "*.pq"),
                      recursive=True)
    assert files, f"no history parquet under {root}"
    cols: set[str] = set()
    for f in files:
        cols |= set(pl.scan_parquet(f).collect_schema().names())
    return cols


# --------------------------------------------------------------------------
# Fast: the split + the runner's dispatch decision (no cache, no sim)
# --------------------------------------------------------------------------

class _FakeComposite:
    def __init__(self, state):
        self.state = state


def test_declared_emit_set_splits_agent_and_document_roots():
    """The coupled generator's declared roots resolve against the built state:
    cell stores agent-relative, document stores as root paths, global_time
    skipped (the runner writes its own)."""
    from v2ecoli.library.parquet_run import declared_emit_set
    from v2ecoli.composites.reactor_bird_coupled import reactor_bird_coupled

    fake = _FakeComposite({
        "global_time": 0.0,
        "reactor": {"dissolved_o2": 1.0},
        "population": {"cell_count": 1.0},
        "lineage": {"doublings": 0.0},
        "agents": {"0": {
            "global_time": 0.0, "bulk": [1], "listeners": {"mass": {}},
            "boundary": {"external": {}},
        }},
    })
    agent_leaves, root_leaves = declared_emit_set(fake, reactor_bird_coupled)
    assert agent_leaves == [("bulk",), ("listeners",), ("boundary",)]
    assert root_leaves == [("reactor",), ("population",), ("lineage",)]


def test_declared_emit_set_refuses_a_generator_that_declares_nothing():
    from v2ecoli.library.parquet_run import _declared_emit_roots
    from v2ecoli.composites.ecoli_population import baseline_population

    # baseline_population declares no emitters= -> "empty means declared" has
    # nothing to mean; refuse rather than silently emit nothing.
    with pytest.raises(ValueError, match="declares no emitter paths"):
        _declared_emit_roots(baseline_population)


def test_declared_emit_roots_default_is_the_baseline_set():
    from v2ecoli.library.parquet_run import _declared_emit_roots

    assert _declared_emit_roots(None) == ["global_time", "bulk", "listeners"]


class TestRunnerOptsInOnlyTheCoupledVariant:
    """Drive the real ``_run_one_variant`` with the builder and runner spied so
    the emit set it selects per variant is pinned without a simulation."""

    @staticmethod
    def _dispatch(mod, monkeypatch, tmp_path, *, sim_name, builder_fn, emitter="parquet"):
        def spy_builder(core, cache_dir, **kw):
            return {"state": {}}

        spy_builder.__signature__ = inspect.signature(builder_fn)
        # The declared path resolves the generator from the builder identity;
        # keep that identity mapping intact for the coupled builder.
        monkeypatch.setattr(mod, "_composite_of",
                            lambda fn: mod.__dict__["_composite_of_orig"](builder_fn))
        monkeypatch.setattr("process_bigraph.Composite", lambda doc, core=None: object())
        runner_got = {}

        def spy_parquet(*a, **kw):
            runner_got.update(kw)
            return {"generations": 0, "steps": 0}

        def spy_sqlite(*a, **kw):
            runner_got.update(kw)
            return {"generations": 0, "steps": 0}

        monkeypatch.setattr(mod, "run_multigen_parquet", spy_parquet)
        monkeypatch.setattr(mod, "run_multigen_sqlite", spy_sqlite)
        monkeypatch.setattr(mod, "_count_parquet_rows", lambda *a, **kw: 0)
        monkeypatch.setattr(mod, "write_run_identity", lambda *a, **kw: None)
        for name in ("_ensure_study_columns", "_register_simulation_row",
                     "_stamp_study_metadata"):
            monkeypatch.setattr(mod, name, lambda *a, **kw: None)
        monkeypatch.setattr(mod, "STUDIES_ROOT", tmp_path)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(mod, "DB_PATH", tmp_path / "runs.db")
        if emitter == "sqlite":
            import sqlite3
            conn = sqlite3.connect(str(tmp_path / "runs.db"))
            conn.execute("CREATE TABLE history (simulation_id TEXT, step INTEGER)")
            conn.commit()
            conn.close()
        mod._run_one_variant(
            sim_name=sim_name, study_slug="s", builder_fn=spy_builder,
            builder_kwargs={}, extra_root_paths=["population/cell_count"],
            duration_sec=1, max_generations=1, chunk=1, cache_dir="out/cache",
            core=None, emitter=emitter,
        )
        return runner_got

    @pytest.fixture
    def mod(self):
        m = _runner_module()
        m._composite_of_orig = m._composite_of
        return m

    def test_coupled_variant_requests_the_declared_set(self, mod, monkeypatch, tmp_path):
        from v2ecoli.composites.reactor_bird_coupled import reactor_bird_coupled

        got = self._dispatch(mod, monkeypatch, tmp_path, sim_name=COUPLED_VARIANT,
                             builder_fn=mod._build_reactor_bird_coupled)
        assert got["emit_paths"] == []
        assert got["declared_generator"] is reactor_bird_coupled
        assert got["extra_root_paths"] == ["population/cell_count"]

    def test_population_variant_keeps_the_explicit_reduced_list(self, mod, monkeypatch, tmp_path):
        got = self._dispatch(mod, monkeypatch, tmp_path, sim_name=POPULATION_VARIANT,
                             builder_fn=mod._build_baseline_population)
        assert got["emit_paths"] == list(mod.COMMON_AGENT_PATHS)
        assert got["declared_generator"] is None
        assert "bulk" not in got["emit_paths"]

    def test_every_other_variant_keeps_an_explicit_list(self, mod):
        """The opt-in is one name; the sweep's other variants are population
        runs and must not drift onto the declared (whole-cell) set."""
        assert mod.DECLARED_EMIT_VARIANTS == frozenset({COUPLED_VARIANT})
        for name, _slug, _fn, _kw, _extra in mod.VARIANTS:
            if name != COUPLED_VARIANT:
                assert name not in mod.DECLARED_EMIT_VARIANTS

    def test_sqlite_path_is_untouched(self, mod, monkeypatch, tmp_path):
        got = self._dispatch(mod, monkeypatch, tmp_path, sim_name=COUPLED_VARIANT,
                             builder_fn=mod._build_reactor_bird_coupled,
                             emitter="sqlite")
        assert got["emit_paths"] == list(mod.COMMON_AGENT_PATHS) + list(
            mod.EXTRA_AGENT_PATHS[COUPLED_VARIANT])
        assert "declared_generator" not in got


# --------------------------------------------------------------------------
# Real runner path (cache + sim): the columns that land in the hive
# --------------------------------------------------------------------------

def _run_variant_for_real(mod, monkeypatch, tmp_path, name, *, ticks=2):
    from v2ecoli.core import build_core

    _name, slug, builder_fn, kwargs, extra_root_paths = _variant(mod, name)
    monkeypatch.setattr(mod, "STUDIES_ROOT", tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    mod._run_one_variant(
        sim_name=name, study_slug=slug, builder_fn=builder_fn,
        builder_kwargs=dict(kwargs), extra_root_paths=list(extra_root_paths),
        duration_sec=ticks, max_generations=1, chunk=1, cache_dir=CACHE,
        core=build_core(), emitter="parquet",
    )
    return _columns_under(str(tmp_path / slug / "parquet-runs"))


@needs_cache
@pytest.mark.sim
def test_runner_coupled_single_lineage_lands_bulk_fba_and_reactor(monkeypatch, tmp_path):
    """Through the runner's real call path (the one Run 1 dispatches), the
    coupled single-lineage hive now carries bulk + FBA + mass + reactor."""
    mod = _runner_module()
    cols = _run_variant_for_real(mod, monkeypatch, tmp_path, COUPLED_VARIANT)
    missing = [c for c in MOLECULAR_COLUMNS + COUPLING_COLUMNS if c not in cols]
    assert not missing, f"missing: {missing}"
    # The variant's own extra observables still land (EXTRA_AGENT_PATHS are
    # subsumed by the whole listeners/boundary capture).
    assert "listeners__fba_results__external_exchange_fluxes" in cols
    assert "boundary__external__OXYGEN-MOLECULE" in cols


@needs_cache
@pytest.mark.sim
def test_runner_population_variant_stays_reduced(monkeypatch, tmp_path):
    """The population variant's explicit allow-list still yields ONLY the
    reduced set: bulk and the FBA fluxes are absent (the OOM guard)."""
    mod = _runner_module()
    cols = _run_variant_for_real(mod, monkeypatch, tmp_path, POPULATION_VARIANT)
    for c in REDUCED_COLUMNS:
        assert c in cols, f"reduced column {c} missing"
    leaked = [c for c in cols if c.startswith("bulk__")
              or c.startswith("listeners__fba_results__")]
    assert not leaked, f"explicit allow-list leaked whole-cell columns: {leaked[:5]}"
