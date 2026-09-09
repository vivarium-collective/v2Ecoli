"""Behavior tests for the flagella transcriptional-cascade Steps.

Ported (and adapted to the v2ecoli ``EcoliStep.update`` API) from the unit tests
Maya Abdalla wrote on the vEcoli ``biofilm`` branch. These exercise the K&A
SUM-gate math and the FlgM secretion gate in isolation; an end-to-end composite
test lives alongside the wired ``flagella_regulation`` feature.
"""

import numpy as np

from v2ecoli.library.schema import MetadataArray
from v2ecoli.processes.flagella_transcription_regulation import (
    FlagellaTranscriptionRegulation,
)
from v2ecoli.processes.flagella_flgm_secretion import FlagellaFlgMSecretion


CLASS_II_RNAS = [
    "EG10322_RNA", "EG11346_RNA", "EG11347_RNA",
    "G358_RNA", "G357_RNA", "G7028_RNA", "EG11355_RNA",
]

_PROMOTER_DTYPE = [
    ("_entryState", "i1"),
    ("TU_index", "<i8"),
    ("init_prob_override", "f8"),
    ("unique_index", "<i8"),
]

# Minimal nascent_flagellum dtype -- the fields FlagellaFlgMSecretion's
# update() actually reads (_entryState for the active-row mask, filament_length
# via attrs()), plus unique_index (required by MetadataArray itself). Real
# unique molecules carry more fields; not needed here.
_NASCENT_DTYPE = [
    ("_entryState", "i1"),
    ("filament_length", "<i8"),
    ("unique_index", "<i8"),
]


def _bulk(pairs):
    return np.array(pairs, dtype=[("id", "U40"), ("count", int)])


def _promoters(n):
    rows = [(1, tu, 0.0, tu) for tu in range(n)]
    return MetadataArray(np.array(rows, dtype=_PROMOTER_DTYPE), n)


def _nascent(n):
    """n active nascent_flagellum rows -- i.e. n completed hook-basal bodies."""
    rows = [(1, 0, i) for i in range(n)]
    return MetadataArray(np.array(rows, dtype=_NASCENT_DTYPE), n)


def _make_regulation():
    return FlagellaTranscriptionRegulation(parameters={
        "rna_ids": CLASS_II_RNAS,
        "flg_classII_rnaids": CLASS_II_RNAS,
        "flg_classIII_rnaids": [],
        "K_flhDC": 10.0,
        "K_fliA": 10.0,
        "seed": 0,
    })


# --------------------------------------------------------------------------
# FlagellaTranscriptionRegulation — the K&A SUM gate
# --------------------------------------------------------------------------

def test_sumgate_pi_in_unit_range():
    """p_i stays in [0, 1] across the activity range."""
    proc = _make_regulation()
    for X, Y in [(0.0, 0.0), (0.2, 0.8), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        p_i = (proc.beta * X + proc.beta_prime * Y) / (proc.beta + proc.beta_prime)
        assert np.all(p_i >= 0) and np.all(p_i <= 1.0), f"p_i out of range at X={X},Y={Y}: {p_i}"


def test_classII_overrides_positive_when_regulators_present():
    """With FlhDC and FliA present, every Class II promoter gets a positive override."""
    proc = _make_regulation()
    states = {
        "promoters": _promoters(7),
        "bulk": _bulk([("CPLX0-3930[c]", 50), ("EG11355-MONOMER[c]", 20)]),
        "timestep": 2.0,
        "next_update_time": 0.0,
        "global_time": 0.0,
    }
    out = proc.update(states)
    assert "set" in out["promoters"]
    overrides = out["promoters"]["set"]["init_prob_override"]
    assert np.all(overrides > 0), f"expected all positive overrides, got {overrides}"


def test_override_rises_with_free_fliA():
    """Higher free FliA (Y) raises the override for the FliA-dominated Class II gene."""
    proc_lo = _make_regulation()
    proc_hi = _make_regulation()
    base = dict(timestep=2.0, next_update_time=0.0, global_time=0.0)
    out_lo = proc_lo.update({
        "promoters": _promoters(7),
        "bulk": _bulk([("CPLX0-3930[c]", 50), ("EG11355-MONOMER[c]", 5)]), **base})
    out_hi = proc_hi.update({
        "promoters": _promoters(7),
        "bulk": _bulk([("CPLX0-3930[c]", 50), ("EG11355-MONOMER[c]", 200)]), **base})
    # EG11355 (fliA) is index 6: highest beta_prime/beta ratio -> most FliA-sensitive.
    lo = out_lo["promoters"]["set"]["init_prob_override"][6]
    hi = out_hi["promoters"]["set"]["init_prob_override"][6]
    assert hi > lo, f"fliA override should rise with free FliA: lo={lo}, hi={hi}"


def test_classIII_gated_by_fliA():
    """Class III override is ~0 when FliA is sequestered, positive when free."""
    rnas = CLASS_II_RNAS + ["EG10321_RNA"]  # a Class III gene
    def proc():
        return FlagellaTranscriptionRegulation(parameters={
            "rna_ids": rnas,
            "flg_classII_rnaids": CLASS_II_RNAS,
            "flg_classIII_rnaids": ["EG10321_RNA"],
            "K_flhDC": 10.0, "K_fliA": 10.0, "seed": 0,
        })
    base = dict(timestep=2.0, next_update_time=0.0, global_time=0.0)
    out_seq = proc().update({
        "promoters": _promoters(8),
        "bulk": _bulk([("CPLX0-3930[c]", 50), ("EG11355-MONOMER[c]", 0)]), **base})
    out_free = proc().update({
        "promoters": _promoters(8),
        "bulk": _bulk([("CPLX0-3930[c]", 50), ("EG11355-MONOMER[c]", 500)]), **base})
    # Class III TU is index 7.
    assert out_seq["promoters"]["set"]["init_prob_override"][7] == 0.0
    assert out_free["promoters"]["set"]["init_prob_override"][7] > 0.0


def test_regulation_advances_clock():
    proc = _make_regulation()
    out = proc.update({
        "promoters": _promoters(7),
        "bulk": _bulk([("CPLX0-3930[c]", 50), ("EG11355-MONOMER[c]", 20)]),
        "timestep": 2.0, "next_update_time": 10.0, "global_time": 10.0})
    assert out["next_update_time"] == 12.0


# --------------------------------------------------------------------------
# FlagellaFlgMSecretion — the Class II -> Class III timing gate
# --------------------------------------------------------------------------

# def _secretion(rate=0.1):  # old zero-order helper, superseded 2026-09-02
#     return FlagellaFlgMSecretion(parameters={"secretion_rate": rate})
def _secretion(rate=0.0015823):
    return FlagellaFlgMSecretion(parameters={"turnover_rate_per_s": rate})


def _sec_states(flgM, hbb, global_time=0.0, next_update_time=0.0):
    return {
        # "bulk": _bulk([("G369-MONOMER[c]", flgM), ("CPLX0-7452[j]", hbb)]),
        # ^ old trigger species (CPLX0-7452 bulk count), superseded 2026-08-27
        # -- the real trigger is nascent_flagellum, see flagella_flgm_secretion.py.
        "bulk": _bulk([("G369-MONOMER[c]", flgM)]),
        "nascent_flagellum": _nascent(hbb),
        "timestep": 2.0,
        "global_time": global_time,
        "next_update_time": next_update_time,
    }


def test_no_hbb_no_export():
    proc = _secretion()
    out = proc.update(_sec_states(flgM=50, hbb=0))
    assert dict(out["bulk"])[proc.flgM_idx] == 0


def test_no_flgm_no_export():
    proc = _secretion()
    out = proc.update(_sec_states(flgM=0, hbb=3))
    assert dict(out["bulk"])[proc.flgM_idx] == 0


# def test_export_scales_with_hbb():  # old zero-order behavior, no longer true
#     # -- superseded 2026-09-02, see flagella_flgm_secretion.py module docstring.
#     proc = _secretion(rate=1.0)
#     out1 = proc.update(_sec_states(flgM=100, hbb=1))
#     out2 = _secretion(rate=1.0).update(_sec_states(flgM=100, hbb=3))
#     d1 = abs(dict(out1["bulk"])[proc.flgM_idx])
#     d2 = abs(dict(out2["bulk"])[proc.flgM_idx])
#     assert d2 > d1, f"more HBBs should export more FlgM: {d1} vs {d2}"


def test_export_independent_of_hbb_count():
    """First-order fix (2026-09-02): once >=1 HBB is complete, export depends
    on the FlgM pool, not on how many HBBs there are -- replaces the old
    zero-order test_export_scales_with_hbb, whose premise this fix removed."""
    proc1 = _secretion(rate=1.0)
    proc2 = _secretion(rate=1.0)
    d1 = dict(proc1.update(_sec_states(flgM=100, hbb=1))["bulk"])[proc1.flgM_idx]
    d2 = dict(proc2.update(_sec_states(flgM=100, hbb=3))["bulk"])[proc2.flgM_idx]
    assert d1 == d2, f"export must not scale with HBB count once gated on: {d1} vs {d2}"


def test_export_scales_with_flgm_pool():
    """First-order fix (2026-09-02): export IS proportional to the current
    FlgM pool, at fixed rate and HBB-gate state."""
    proc_small = _secretion(rate=0.1)
    proc_large = _secretion(rate=0.1)
    d_small = abs(dict(proc_small.update(_sec_states(flgM=100, hbb=1))["bulk"])[proc_small.flgM_idx])
    d_large = abs(dict(proc_large.update(_sec_states(flgM=1000, hbb=1))["bulk"])[proc_large.flgM_idx])
    assert d_large > d_small, f"more FlgM should export more FlgM: {d_small} vs {d_large}"


def test_export_clamped_to_available():
    proc = _secretion(rate=100.0)
    out = proc.update(_sec_states(flgM=5, hbb=10))
    delta = dict(out["bulk"])[proc.flgM_idx]
    assert -5 <= delta <= 0, f"FlgM must not go negative: delta={delta}"


def test_secretion_gating_and_clock():
    proc = _secretion()
    # next_update_time in the future -> step is gated off
    assert proc.update_condition(2.0, _sec_states(50, 3, global_time=0.0, next_update_time=5.0)) is False
    out = proc.update(_sec_states(50, 1, global_time=10.0, next_update_time=10.0))
    assert out["next_update_time"] == 12.0
