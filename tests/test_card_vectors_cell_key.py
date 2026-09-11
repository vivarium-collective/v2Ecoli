"""A cell is identified by every partition key the sweep carries (#776).

WHY THIS FILE EXISTS.

``extract_vectors`` keyed a cell as ``(lineage_seed, generation, agent_id)``. A
sweep whose variants -- or pooled experiments -- reuse a lineage seed therefore
merged every such cell into one: ``n_cells`` 1 where several exist, per-cell means
pooled across them, and nothing raised.

⚠ The ENSEMBLE MEAN of merged cells can equal the correct one (two equal-length
cells averaged together average to the same thing), so a test on ``vector`` alone
passes on the defect. These tests assert on ``n_cells`` and on ``per_cell``, which
is where the merge actually shows -- and on the derived-flux timestep, which the
same missing key mis-scales.
"""

from __future__ import annotations

import pytest

from v2ecoli.library import card_vectors

PFX = card_vectors._DMDT_PREFIX
DRY = card_vectors._DRY_MASS_COL


def _write(tmp_path, cells, *, dt=1.0, n_steps=20, counts=500.0, dry_mass=300.0):
    """One parquet per cell under a full hive layout.

    ``cells`` is ``[(experiment_id, variant, lineage_seed, agent_id, value, t0)]``;
    ``value`` fills a two-feature proteome vector, ``t0`` offsets the cell's clock.
    """
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet

    for exp, var, seed, agent, value, t0 in cells:
        d = (tmp_path / "sweep" / "history" / f"experiment_id={exp}" / f"variant={var}"
             / f"lineage_seed={seed}" / "generation=1" / f"agent_id={agent}")
        d.mkdir(parents=True)
        cols = {
            "global_time": [t0 + i * dt for i in range(n_steps)],
            "listeners__monomer_counts": [[value, value]] * n_steps,
            f"{PFX}ACET[p]": [counts] * n_steps,
            DRY: [dry_mass] * n_steps,
        }
        pa.parquet.write_table(pa.table(cols), d / "0.pq")
    return str(tmp_path / "sweep")


def test_variants_that_share_a_lineage_seed_are_separate_cells(tmp_path):
    sweep = _write(tmp_path, [("e", 0, 0, "0", 1.0, 0.0), ("e", 1, 0, "0", 3.0, 0.0)])
    node = card_vectors.extract_vectors(sweep)["omics"]["proteome"]
    assert node["n_cells"] == 2
    assert sorted(node["per_cell"]) == [[1.0, 1.0], [3.0, 3.0]]
    assert node["vector"] == [2.0, 2.0]


def test_experiments_that_share_a_variant_and_seed_are_separate_cells(tmp_path):
    sweep = _write(tmp_path, [("a", 0, 0, "0", 1.0, 0.0), ("b", 0, 0, "0", 5.0, 0.0)])
    node = card_vectors.extract_vectors(sweep)["omics"]["proteome"]
    assert node["n_cells"] == 2
    assert sorted(node["per_cell"]) == [[1.0, 1.0], [5.0, 5.0]]


def test_the_timestep_is_not_lagged_across_a_variant_boundary(tmp_path):
    """Two variants tick at 1 s, offset by half a second. Lagged across the
    boundary, the median delta is 0.5 s and every derived flux doubles."""
    sweep = _write(tmp_path, [("e", 0, 0, "0", 1.0, 0.0), ("e", 1, 0, "0", 1.0, 0.5)],
                   dt=1.0, counts=500.0, dry_mass=300.0)
    flux = card_vectors.extract_vectors(sweep)["fluxes"]["ACET[p]"]["vector"][0]
    expected = -500.0 / (300.0 * card_vectors._COUNTS_TO_MMOL_PER_GDCW_H * 1.0)
    assert flux == pytest.approx(expected, rel=1e-12)


def test_cells_of_one_variant_are_unchanged_by_the_wider_key(tmp_path):
    """The common case -- one variant, distinct seeds -- extracts as it always did."""
    sweep = _write(tmp_path, [("e", 0, 0, "0", 2.0, 0.0), ("e", 0, 1, "0", 4.0, 0.0)])
    node = card_vectors.extract_vectors(sweep)["omics"]["proteome"]
    assert node["n_cells"] == 2
    assert node["vector"] == [3.0, 3.0]


def test_the_key_includes_only_the_optional_partitions_a_sweep_has():
    assert card_vectors._cell_key(["lineage_seed", "generation", "agent_id", "x"]) == [
        "lineage_seed", "generation", "agent_id"]
    assert card_vectors._cell_key(["variant", "experiment_id", "lineage_seed",
                                   "generation", "agent_id"]) == [
        "experiment_id", "variant", "lineage_seed", "generation", "agent_id"]


def test_the_extractor_version_moved_past_the_merged_key():
    """A v5 envelope for a seed-sharing sweep holds merged cells; only the cache
    key can tell it apart from a correct one."""
    assert card_vectors.EXTRACTOR_VERSION >= 6
