"""Tests for the cross-seed (multiseed) ptools variants.

Exercises the _MultiseedMixin end to end on a synthetic in-memory DuckDB
history: cross-seed MEAN matching vEcoli's pooled per-bin AVG, a cross-seed
SPREAD (std across seeds) that is nonzero when seeds differ and zero for a
single seed, and cumulative-time reconstruction when global_time resets per
generation.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from bigraph_schema import allocate_core


FLUX_COL = "listeners__fba_results__base_reaction_fluxes"


def _fake_sim_data(rxn_ids):
    return SimpleNamespace(
        process=SimpleNamespace(
            metabolism=SimpleNamespace(base_reaction_ids=list(rxn_ids))
        )
    )


def _make_history(rows):
    """rows: list of (lineage_seed, generation, global_time, [flux...]).

    Returns (conn, history_sql) for an in-memory table with the partition +
    flux columns the ptools analyses read.
    """
    conn = duckdb.connect()
    conn.execute(
        f'CREATE TABLE hist ('
        f'  experiment_id VARCHAR, variant BIGINT, lineage_seed BIGINT,'
        f'  generation BIGINT, agent_id VARCHAR, global_time DOUBLE,'
        f'  "{FLUX_COL}" DOUBLE[]'
        f')'
    )
    for seed, gen, t, flux in rows:
        arr = "[" + ", ".join(f"{v}" for v in flux) + "]"
        conn.execute(
            f"INSERT INTO hist VALUES ('exp', 0, {seed}, {gen}, "
            f"'{seed}_{gen}', {t}, {arr})"
        )
    return conn, "SELECT * FROM hist"


def _parse_tsv(tsv):
    return pd.read_csv(io.StringIO(tsv), sep="\t", index_col=0)


def _run(rows, rxn_ids, n_tp=1, time_unit="seconds"):
    from v2ecoli.workflow.analyses.ptools_multiscale import PtoolsRxnsMultiseed

    conn, history_sql = _make_history(rows)
    step = PtoolsRxnsMultiseed({}, core=allocate_core())
    out = step.analyze(
        conn=conn,
        history_sql=history_sql,
        sim_data=_fake_sim_data(rxn_ids),
        variant_metadata={"n_tp": n_tp, "time_unit": time_unit},
    )
    return out


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_multiseed_classes_registered():
    from v2ecoli.workflow.analyses import ptools_multiscale  # noqa: F401 (register)
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, ANALYSIS_SCALES
    for name in ("ptools_rna_multiseed", "ptools_rxns_multiseed",
                 "ptools_proteins_multiseed"):
        cls = ANALYSIS_REGISTRY[name]
        assert cls.scale == "multiseed"
    assert "multiseed" in ANALYSIS_SCALES


# ---------------------------------------------------------------------------
# Cross-seed mean matches vEcoli's pooled AVG; spread is std across seeds
# ---------------------------------------------------------------------------

def test_mean_is_pooled_avg_and_spread_is_cross_seed_std():
    # 3 seeds, 1 generation each, 2 timepoints, 2 reactions. One time bin.
    rows = [
        (0, 0, 0.0, [10.0, 100.0]), (0, 0, 1.0, [20.0, 200.0]),  # seed 0 mean [15,150]
        (1, 0, 0.0, [30.0, 300.0]), (1, 0, 1.0, [40.0, 400.0]),  # seed 1 mean [35,350]
        (2, 0, 0.0, [50.0, 500.0]), (2, 0, 1.0, [60.0, 600.0]),  # seed 2 mean [55,550]
    ]
    out = _run(rows, ["R1", "R2"], n_tp=1)
    df = _parse_tsv(out["data"]["tsv"])
    assert out["data"]["n_seeds"] == 3

    # vEcoli pooled AVG over all 6 rows per reaction.
    mean_col = [c for c in df.columns if not c.endswith("_sd")]
    sd_col = [c for c in df.columns if c.endswith("_sd")]
    assert len(mean_col) == 1 and len(sd_col) == 1
    np.testing.assert_allclose(df.loc["R1", mean_col[0]], 35.0, atol=1e-6)
    np.testing.assert_allclose(df.loc["R2", mean_col[0]], 350.0, atol=1e-6)

    # spread = population std (ddof=0) across the three seed bin-means.
    np.testing.assert_allclose(
        df.loc["R1", sd_col[0]], np.std([15.0, 35.0, 55.0]), atol=1e-4
    )
    np.testing.assert_allclose(
        df.loc["R2", sd_col[0]], np.std([150.0, 350.0, 550.0]), atol=1e-4
    )


def test_single_seed_degrades_to_zero_spread():
    rows = [
        (0, 0, 0.0, [10.0, 100.0]), (0, 0, 1.0, [20.0, 200.0]),
    ]
    out = _run(rows, ["R1", "R2"], n_tp=1)
    df = _parse_tsv(out["data"]["tsv"])
    assert out["data"]["n_seeds"] == 1
    sd_col = [c for c in df.columns if c.endswith("_sd")][0]
    assert (df[sd_col] == 0).all()
    # mean is still that seed's mean.
    mean_col = [c for c in df.columns if not c.endswith("_sd")][0]
    np.testing.assert_allclose(df.loc["R1", mean_col], 15.0, atol=1e-6)


def test_cumulative_reconstruction_aligns_generations_across_seeds():
    # 2 seeds, 2 generations each, global_time RESETS to 0 each generation.
    # Cumulative: gen0 -> t 0,1 ; gen1 -> t (maxprev+1)+0.. so gen1 lands after
    # gen0. With n_tp=2 the two bins separate the two generations.
    rows = [
        # seed 0
        (0, 0, 0.0, [10.0]), (0, 0, 1.0, [10.0]),   # gen0 value 10
        (0, 1, 0.0, [90.0]), (0, 1, 1.0, [90.0]),   # gen1 value 90 (time RESET)
        # seed 1
        (1, 0, 0.0, [20.0]), (1, 0, 1.0, [20.0]),   # gen0 value 20
        (1, 1, 0.0, [80.0]), (1, 1, 1.0, [80.0]),   # gen1 value 80 (time RESET)
    ]
    out = _run(rows, ["R1"], n_tp=2)
    df = _parse_tsv(out["data"]["tsv"])
    mean_cols = [c for c in df.columns if not c.endswith("_sd")]
    assert len(mean_cols) == 2  # two bins = two generations, correctly separated
    # Bin 0 (gen0): pooled mean of {10,10,20,20} = 15. Bin 1 (gen1): {90,90,80,80}=85.
    # If cumulative reconstruction were MISSING, both generations would pool at
    # raw t in {0,1} and the two bins could not separate gen0 from gen1.
    np.testing.assert_allclose(df.loc["R1", mean_cols[0]], 15.0, atol=1e-6)
    np.testing.assert_allclose(df.loc["R1", mean_cols[1]], 85.0, atol=1e-6)
    # Spread in gen0 = std([10,20]) = 5 ; gen1 = std([90,80]) = 5.
    sd_cols = [c for c in df.columns if c.endswith("_sd")]
    np.testing.assert_allclose(df.loc["R1", sd_cols[0]], 5.0, atol=1e-4)
    np.testing.assert_allclose(df.loc["R1", sd_cols[1]], 5.0, atol=1e-4)


def test_generation_lower_bound_drops_early_generations():
    from v2ecoli.workflow.analyses.ptools_multiscale import PtoolsRxnsMultiseed
    # 1 seed, 3 generations (time resets), gen values 100 / 200 / 300.
    rows = [
        (0, 0, 0.0, [100.0]), (0, 0, 1.0, [100.0]),
        (0, 1, 0.0, [200.0]), (0, 1, 1.0, [200.0]),
        (0, 2, 0.0, [300.0]), (0, 2, 1.0, [300.0]),
    ]
    conn, history_sql = _make_history(rows)
    step = PtoolsRxnsMultiseed({}, core=allocate_core())
    out = step.analyze(
        conn=conn, history_sql=history_sql, sim_data=_fake_sim_data(["R1"]),
        variant_metadata={"n_tp": 1, "time_unit": "seconds",
                          "generation_lower_bound": 1},
    )
    df = _parse_tsv(out["data"]["tsv"])
    mean_col = [c for c in df.columns if not c.endswith("_sd")][0]
    # gen0 (100) dropped by the burn-in; mean of gens 1,2 = (200+200+300+300)/4 = 250.
    np.testing.assert_allclose(df.loc["R1", mean_col], 250.0, atol=1e-4)


def test_view_has_two_panels():
    rows = [
        (0, 0, 0.0, [10.0]), (1, 0, 0.0, [20.0]),
    ]
    out = _run(rows, ["R1"], n_tp=1)
    assert "cross-seed mean" in out["view"]
    assert "cross-seed spread" in out["view"]
