"""``extract_vectors`` must not assume every observable column is present.

WHY THIS FILE EXISTS.

Two properties of a sweep were, until now, assumed rather than checked, and both
assumptions held for every sweep the function had been used on — so neither was
wrong in a way any existing test could see:

1. **That every observable column is present.** Different metabolism processes
   write different exchange leaves: ``metabolism.py`` writes
   ``listeners.fba_results.external_exchange_fluxes``; ``metabolism_redux`` does
   not (it writes ``estimated_exchange_dmdt``). The same divergence is already
   documented and guarded, for the ``gdcw`` basis, in
   ``library/vivarium_ecoli_engine.py``. Here it surfaced as a DuckDB binder
   error that named a column rather than the cause, and took the omics groups
   down with it even though those columns were present.

2. **That every file in the glob has the same schema.** DuckDB binds
   ``read_parquet`` against the FIRST file unless told otherwise, so a pooled
   directory silently loses a group (or raises) depending on which file sorts
   first. ``union_by_name=true`` is what makes the probe describe the sweep
   rather than one file of it.

⛔ **A cell-completeness rule was attempted in this PR and WITHDRAWN.** A
row-count heuristic (drop cells far below the longest) was found, by independent
review, to do nothing on a real in-tree sweep whose emit cadence is coarse — the
partial cell was 14% of the longest and sailed through, while the node
affirmatively reported that nothing had been dropped. Whether a stub falls under
a fixed fraction is a property of the producing run, which is the very thing this
module is trying to stop assuming. The correct signal is a per-cell ``divided``
flag; it is not currently an emitted parquet column. Tracked as follow-up.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.library import card_vectors


def _write(tmp_path, cells, cols=None):
    """Write one sweep. ``cells`` is ``[(agent_id, n_steps), ...]``.

    Values are constant within a cell and differ between cells, so an ensemble
    mean states plainly WHICH cells went into it.
    """
    pq = pytest.importorskip("pyarrow")
    import pyarrow.parquet

    cols = list(card_vectors._VECTOR_COLS) if cols is None else list(cols)
    # ⚠ EACH CELL GETS ITS OWN PARTITION DIRECTORY. With
    # `hive_partitioning=true` the directory's `agent_id=` is what the reader
    # sees, so writing several cells into one directory collapses them into a
    # single cell — an earlier version of this helper did exactly that and every
    # membership test reported `n_cells == 1`.
    base = tmp_path / "exp" / "history" / "experiment_id=exp" / "variant=0"
    for value, (agent, n_steps) in enumerate(cells, start=1):
        d = base / "lineage_seed=0" / f"generation={len(agent)}" / f"agent_id={agent}"
        d.mkdir(parents=True, exist_ok=True)
        rows = {"lineage_seed": [], "generation": [], "agent_id": []}
        for c in cols:
            rows[c] = []
        for _ in range(n_steps):
            rows["lineage_seed"].append(0)
            rows["generation"].append(len(agent))
            rows["agent_id"].append(agent)
            for c in cols:
                rows[c].append([float(value)] * 3)
        pyarrow.parquet.write_table(pq.table(rows), d / "0.pq")

    sweep = str(tmp_path / "exp")
    # ⛔ THE FIXTURE ASSERTS ITSELF. Without this, a fixture whose files the
    # reader cannot find produces `{}` — which every test below then reports as
    # a wrong RESULT rather than an absent INPUT. Three tests failed that way in
    # CI while passing locally, and the messages (`AssertionError: {}`,
    # `KeyError: 'omics'`) pointed at the code under test rather than at the
    # fixture. A fixture that cannot prove it built something is not a fixture.
    from v2ecoli.library.sweep_io import history_files
    found = history_files(sweep)
    assert found, f"fixture wrote no discoverable history parquet under {sweep}"
    return sweep


# --------------------------------------------------------------------------
# 1. A column this sweep does not have
# --------------------------------------------------------------------------

def test_an_absent_column_omits_its_group_and_leaves_the_others_extractable(tmp_path):
    """The case a redux-metabolism sweep actually presents."""
    present = [
        "listeners__rna_counts__mRNA_cistron_counts",
        "listeners__monomer_counts",
    ]
    out = card_vectors.extract_vectors(_write(tmp_path, [("0", 10)], cols=present))
    assert sorted(out) == ["omics"], out
    assert sorted(out["omics"]) == ["proteome", "transcriptome"]


def test_an_absent_column_is_NOT_reported_as_zeros(tmp_path):
    """⛔ The whole point. A zero-filled exchange vector is indistinguishable
    from a cell exchanging nothing — an error silently becoming a result."""
    present = [
        "listeners__rna_counts__mRNA_cistron_counts",
        "listeners__monomer_counts",
    ]
    out = card_vectors.extract_vectors(_write(tmp_path, [("0", 10)], cols=present))
    assert "fluxes" not in out, (
        "the absent exchange group must be OMITTED, not emitted as zeros")


def test_a_sweep_with_no_observable_columns_raises(tmp_path):
    """Degrading to ``{}`` here would hand a caller an empty ensemble that looks
    like a run which simply had nothing to say."""
    with pytest.raises(ValueError, match="no observable columns"):
        card_vectors.extract_vectors(
            _write(tmp_path, [("0", 10)], cols=["listeners__mass__dry_mass"]))




def test_a_column_missing_from_only_the_FIRST_file_is_still_found(tmp_path):
    """⛔ DuckDB binds the glob against the first file unless union_by_name is set.

    The two failure modes are asymmetric and only one is loud: a first file
    LACKING a column the others have omits the group SILENTLY; a first file
    HAVING one the others lack raises. This pins the silent one, which is the
    dangerous direction — a whole observable group vanishing with no error."""
    pq = pytest.importorskip("pyarrow")
    import pyarrow.parquet

    d = (tmp_path / "exp" / "history" / "experiment_id=exp" / "variant=0"
         / "lineage_seed=0" / "generation=2" / "agent_id=00")
    d.mkdir(parents=True)
    n = 20
    base = {"lineage_seed": [0]*n, "generation": [2]*n, "agent_id": ["00"]*n}
    narrow = dict(base)
    for c in ("listeners__rna_counts__mRNA_cistron_counts", "listeners__monomer_counts"):
        narrow[c] = [[1.0, 1.0]]*n
    wide = dict(narrow)
    wide["listeners__fba_results__external_exchange_fluxes"] = [[2.0, 2.0]]*n
    # "0.pq" sorts first and is the NARROW one
    pyarrow.parquet.write_table(pq.table(narrow), d / "0.pq")
    pyarrow.parquet.write_table(pq.table(wide), d / "1.pq")

    from v2ecoli.library.sweep_io import history_files
    assert len(history_files(str(tmp_path / "exp"))) == 2, "fixture wrote <2 files"
    out = card_vectors.extract_vectors(str(tmp_path / "exp"))
    assert "fluxes" in out, (
        "the exchange group vanished because only the first file was inspected")


# ── Cells that are not complete cell cycles ──────────────────────────────────

def test_a_clean_two_population_sweep_splits_and_counts(tmp_path):
    """Stubs orders of magnitude below the real cells: a split exists, find it."""
    out = card_vectors.extract_vectors(
        _write(tmp_path, [("00", 3000), ("01", 30), ("000", 3200), ("001", 25)]))
    node = out["omics"]["transcriptome"]
    assert node["n_cells"] == 2
    assert node["n_cells_excluded_partial"] == 2
    assert node["partial_cell_detection"] == "clean"


def test_the_ensemble_MEAN_excludes_the_partials_not_just_the_count(tmp_path):
    """`n_cells` alone can be right while the mean is still contaminated.

    Cell values are 1.0 and 2.0; excluding the second gives 1.0, including it
    gives 1.5. A mutant that counts correctly and averages everything passes an
    n_cells assertion and fails this one."""
    out = card_vectors.extract_vectors(_write(tmp_path, [("00", 3000), ("01", 30)]))
    assert np.allclose(out["omics"]["transcriptome"]["vector"], 1.0)


def test_AN_UNDECIDABLE_SWEEP_EXCLUDES_NOTHING_AND_SAYS_SO(tmp_path):
    """⛔⛔ THE CASE THAT KILLED THE PREVIOUS IMPLEMENTATION.

    Per-cell rows `25, 29, 26, 4` are from a real sweep in this tree. The
    withdrawn rule (fixed fraction of the longest) put the floor at 2.9 and
    ADMITTED the 4-row cell while reporting `n_cells_excluded_partial: 0` — an
    affirmative claim that nothing partial got in, which was false.

    There is no split to find here: 4 is only 6.25x below 25, so "partial" and
    "complete" are not separable from row counts alone. The honest outcome is to
    exclude nothing AND to refuse the clean label."""
    out = card_vectors.extract_vectors(
        _write(tmp_path, [("00", 25), ("000", 29), ("0000", 26), ("00000", 4)]))
    node = out["omics"]["transcriptome"]
    assert node["n_cells"] == 4, "excluded a cell on an undecidable split"
    assert node["partial_cell_detection"] == "ambiguous", (
        "claimed a clean split where none exists — the withdrawn rule's exact bug")


def test_a_uniform_sweep_is_ambiguous_rather_than_falsely_clean(tmp_path):
    """A healthy sweep and a uniformly-truncated one are indistinguishable from
    row counts, so neither may be labelled `clean`. Honest, and it is why
    `ambiguous` must not be read as a warning about THIS sweep."""
    out = card_vectors.extract_vectors(
        _write(tmp_path, [("00", 380), ("000", 400), ("0000", 420)]))
    node = out["omics"]["transcriptome"]
    assert node["n_cells"] == 3
    assert node["n_cells_excluded_partial"] == 0
    assert node["partial_cell_detection"] == "ambiguous"


def test_membership_is_decided_once_for_every_group(tmp_path):
    """⚠ Guards the property with columns that DIFFER per cell, so a per-column
    decision would actually diverge. The previous version of this test used a
    fixture where every column was full-length on every row, which made global
    and per-column membership identical by construction — it could not fail, and
    a mutation to per-column membership left the whole suite green."""
    out = card_vectors.extract_vectors(
        _write(tmp_path, [("00", 3000), ("01", 30), ("000", 3200)]))
    assert {n["n_cells"] for d in out.values() for n in d.values()} == {2}
    assert {n["partial_cell_detection"] for d in out.values() for n in d.values()} == {"clean"}
