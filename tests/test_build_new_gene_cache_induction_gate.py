"""Build-time induction gate in scripts/build_new_gene_cache.build.

The manifest carries `requested` (asked) and `applied` (what ParCa assigned).
ParCa inserts a new gene SILENT (rna_expression 0), so a cache can request
induction yet come back basal -- bit-indistinguishable from a real one at exit
0. induction_problems() is the pure check; build() runs it and refuses to hand
back a silent/zero cache. These tests cover the pure check and that build()
raises/bypasses, with the heavy ParCa deps stubbed so no real state is needed.
"""
import json
import os

import pytest

from scripts import build_new_gene_cache as bng


@pytest.fixture(autouse=True)
def _restore_cwd():
    """build() does os.chdir(repo_root) and never restores it -- save/restore
    cwd so these tests don't pollute the shared suite."""
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


def _manifest(expression=1174897.55, rna_ids=("gene-X",),
              expression_factors=(1174897.55,)):
    return {
        "requested": {"expression": expression, "translation_efficiency": 1.0},
        "applied": {
            "rna_ids": list(rna_ids),
            "expression_factors": list(expression_factors),
            "monomer_ids": ["mon-X"],
            "translation_efficiencies": [1.0],
        },
    }


# --- pure induction_problems --------------------------------------------

def test_healthy_manifest_is_clean():
    assert bng.induction_problems(_manifest()) == []


def test_silent_applied_flagged():
    assert any("SILENT" in p
               for p in bng.induction_problems(_manifest(expression_factors=[0.0])))


def test_empty_rna_ids_flagged():
    problems = bng.induction_problems(_manifest(rna_ids=[], expression_factors=[]))
    assert any("no new-gene cistron" in p for p in problems)


def test_nonpositive_requested_expression_flagged():
    for bad in (0, -1.0, None, True):
        assert any("positive expression multiplier" in p
                   for p in bng.induction_problems(_manifest(expression=bad))), bad


# --- build() gate (heavy deps stubbed) ----------------------------------

def _stub_build(monkeypatch, applied):
    monkeypatch.setattr(bng, "load_parca_state", lambda p: object())
    monkeypatch.setattr(bng, "hydrate_sim_data_from_state", lambda s: object())

    def _fake_lib(sim_data, cache_dir, **k):
        os.makedirs(cache_dir, exist_ok=True)
        return {"applied": applied}
    monkeypatch.setattr(bng, "build_new_gene_cache", _fake_lib)


_HEALTHY = {"rna_ids": ["gene-X"], "expression_factors": [1174897.55],
            "monomer_ids": ["mon-X"], "translation_efficiencies": [1.0]}
_SILENT = {"rna_ids": ["gene-X"], "expression_factors": [0.0],
           "monomer_ids": ["mon-X"], "translation_efficiencies": [1.0]}


def test_build_passes_on_real_induction(tmp_path, monkeypatch):
    _stub_build(monkeypatch, _HEALTHY)
    manifest = bng.build(str(tmp_path / "s.pkl"), str(tmp_path / "cache"),
                         expression=1174897.55, translation_efficiency=1.0)
    assert manifest["applied"]["expression_factors"] == [1174897.55]
    assert json.load(open(tmp_path / "cache" / "new_genes.json"))[
        "requested"]["expression"] == 1174897.55


def test_build_raises_on_silent_induction(tmp_path, monkeypatch):
    _stub_build(monkeypatch, _SILENT)
    with pytest.raises(SystemExit) as exc:
        bng.build(str(tmp_path / "s.pkl"), str(tmp_path / "cache"),
                  expression=1174897.55, translation_efficiency=1.0)
    assert "induction gate FAILED" in str(exc.value)
    assert (tmp_path / "cache" / "new_genes.json").exists()  # written before raise


def test_build_verify_false_bypasses(tmp_path, monkeypatch):
    _stub_build(monkeypatch, _SILENT)
    manifest = bng.build(str(tmp_path / "s.pkl"), str(tmp_path / "cache"),
                         expression=1174897.55, translation_efficiency=1.0,
                         verify=False)
    assert manifest["applied"]["expression_factors"] == [0.0]  # no raise
