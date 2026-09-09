"""Exchange fluxes derived from per-species dmdt counts.

WHY THIS FILE EXISTS.

``metabolism_redux`` does not write the classic ``external_exchange_fluxes``
array. It writes one scalar column per species, in molecule COUNTS per timestep,
under ``estimated_exchange_dmdt__<SPECIES>``. Before this change a redux sweep
produced NO ``fluxes`` group at all — the extraction silently returned omics
only, and a card asking for an exchange flux got nothing while the data sat in
the parquet it had just read.

⛔ The conversion is NOT a fitted constant. It is
``counts / N_A * 1000 / dry_mass / (dt/3600)`` — pure unit arithmetic. Measured
2026-09-09 against two independent runs that carry BOTH representations:
deriving the flux from the dmdt column reproduces the bespoke listener to a
worst-case relative error of 0.000000%, over >130k rows.

These tests pin the three things that would silently produce a WRONG NUMBER
rather than an error: the conversion itself, the sign, and reading ``dt`` from
the data instead of assuming 1 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.library import card_vectors

PFX = card_vectors._DMDT_PREFIX


def _write(tmp_path, *, counts, dry_mass, dt=1.0, n_steps=20, with_dry_mass=True):
    """A one-cell sweep carrying one dmdt species, at a fixed dt."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa, pyarrow.parquet

    d = tmp_path / "exp" / "history" / "experiment_id=e" / "part"
    d.mkdir(parents=True)
    cols = {
        "lineage_seed": [0] * n_steps,
        "generation": [2] * n_steps,
        "agent_id": ["0"] * n_steps,
        "global_time": [i * dt for i in range(n_steps)],
        f"{PFX}ACET[p]": [counts] * n_steps,
    }
    if with_dry_mass:
        cols[card_vectors._DRY_MASS_COL] = [dry_mass] * n_steps
    # one real vector column, so the sweep is extractable even with no fluxes
    cols["listeners__monomer_counts"] = [[1.0, 2.0]] * n_steps
    pa.parquet.write_table(pa.table(cols), d / "0.pq")
    return str(tmp_path / "exp")


def _expected(counts, dry_mass, dt):
    return -counts / (dry_mass * card_vectors._COUNTS_TO_MMOL_PER_GDCW_H * dt)


def test_dmdt_becomes_a_flux_node_with_the_species_name(tmp_path):
    sweep = _write(tmp_path, counts=-6600, dry_mass=380.0)
    out = card_vectors.extract_vectors(sweep)
    assert "ACET[p]" in out["fluxes"], (
        "the species token must come from the column name, so identity travels "
        "with the data instead of being enumerated in the library")
    node = out["fluxes"]["ACET[p]"]
    assert node["units"] == "mmol/gDCW/h"
    assert node["vector"] == pytest.approx([_expected(-6600, 380.0, 1.0)])
    # scalar observable -> a one-feature vector, so the per_cell matrix keeps
    # the same (n_cells x n_features) contract every other node has
    assert np.asarray(node["per_cell"]).shape == (1, 1)


def test_sign_is_flipped_so_secretion_is_positive(tmp_path):
    """The dmdt listener reports from the environment's side.

    ⛔ A sign error here is the defect this test exists for: it produces a
    perfectly plausible number with the biology inverted -- uptake reported as
    secretion -- and nothing downstream can detect it.
    """
    sweep = _write(tmp_path, counts=-6600, dry_mass=380.0)   # negative dmdt
    v = card_vectors.extract_vectors(sweep)["fluxes"]["ACET[p]"]["vector"][0]
    assert v > 0, "negative dmdt is SECRETION and must render positive"

    sweep2 = _write(tmp_path / "b", counts=+6600, dry_mass=380.0)
    v2 = card_vectors.extract_vectors(sweep2)["fluxes"]["ACET[p]"]["vector"][0]
    assert v2 < 0, "positive dmdt is UPTAKE and must render negative"


def test_timestep_is_read_from_the_data_not_assumed(tmp_path):
    """dt scales the flux inversely, so assuming 1 s mis-scales by exactly dt.

    Same counts, same dry mass, dt doubled -> half the flux. A hardcoded 1 s
    passes the first case and is wrong by 2x on the second.
    """
    a = card_vectors.extract_vectors(
        _write(tmp_path / "a", counts=-6600, dry_mass=380.0, dt=1.0)
    )["fluxes"]["ACET[p]"]["vector"][0]
    b = card_vectors.extract_vectors(
        _write(tmp_path / "b", counts=-6600, dry_mass=380.0, dt=2.0)
    )["fluxes"]["ACET[p]"]["vector"][0]
    assert b == pytest.approx(a / 2.0), "dt must come from global_time deltas"


def test_absent_dry_mass_omits_the_group_rather_than_guessing(tmp_path):
    """⛔ Absent means ABSENT. Without dry mass the conversion cannot be made,
    and emitting the raw counts under a mmol/gDCW/h label would be a wrong
    number wearing a correct unit -- worse than no node at all."""
    sweep = _write(tmp_path, counts=-6600, dry_mass=380.0, with_dry_mass=False)
    out = card_vectors.extract_vectors(sweep)
    assert "fluxes" not in out
    assert "proteome" in out["omics"], "the other groups must still extract"


def test_a_differently_cased_column_still_resolves(tmp_path):
    """Pins an ASSUMPTION this module now relies on: DuckDB resolves QUOTED
    identifiers case-insensitively.

    The SELECT quotes each column so a derived expression can sit beside a bare
    one. That is only safe because quoting does NOT make the identifier
    case-sensitive here — the opposite of Postgres. If that ever changed, or the
    engine were swapped, every sweep whose emitter cased a column differently
    from `_VECTOR_COLS` would silently lose that group. One key is already
    mixed-case (`...__mRNA_cistron_counts`), so the assumption is load-bearing.
    """
    pytest.importorskip("pyarrow")
    import pyarrow as pa, pyarrow.parquet
    d = tmp_path / "exp" / "history" / "experiment_id=e" / "part"
    d.mkdir(parents=True)
    # emit the proteome column in a case the dict key does NOT use
    odd = "LISTENERS__MONOMER_COUNTS"
    pa.parquet.write_table(pa.table({
        "lineage_seed": [0] * 4, "generation": [2] * 4, "agent_id": ["0"] * 4,
        "global_time": [0.0, 1.0, 2.0, 3.0],
        odd: [[1.0, 3.0]] * 4,
    }), d / "0.pq")
    out = card_vectors.extract_vectors(str(tmp_path / "exp"))
    assert out["omics"]["proteome"]["vector"] == pytest.approx([1.0, 3.0])
