"""``scripts/build_cache.py`` must stamp the strain it built for into the bundle.

The wrong-strain guard (P1-6, ``verify_cache_version(expected_build_params=...)``)
can only fire if the cache's own ``cache_version.json`` records which strain it
was built for. The in-process ``core.save_sim_input`` path already records that;
the CLI build path (``scripts/build_cache.py`` — the one the GovCloud ParCa job
runs) previously did not accept the strain at all, so a violacein cache built via
the CLI would be stamped wild-type and either verify clean against a wild-type
request (false pass) or, once a request carried the strain, fail against its own
cache (false fail). These tests pin the CLI's strain plumbing.

Hermetic: the heavy ParCa hydrate/save internals are mocked, so no fixture or
fit is required.
"""
from __future__ import annotations

import types

import pytest

import scripts.build_cache as bc


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        ("off", None),
        ("  off  ", None),
        ("   ", None),
        ("violacein", "violacein"),
        (" vioABCDE_MG1655_v2 ", "vioABCDE_MG1655_v2"),
    ],
)
def test_normalize_strain(raw, expected):
    """'off'/empty are the wild-type sentinel -> None (DEFAULT_BUILD_PARAMS);
    a real strain passes through, trimmed."""
    assert bc._normalize_strain(raw) == expected


def _patch_heavy(monkeypatch, captured):
    """Stub out fixture load / hydrate / save / read + filesystem side effects."""
    monkeypatch.setattr(bc, "load_parca_state", lambda fixture: object())
    monkeypatch.setattr(bc, "hydrate_sim_data_from_state", lambda state: object())

    def fake_save(sim_data, cache_dir, **kwargs):
        captured["cache_dir"] = cache_dir
        captured.update(kwargs)

    monkeypatch.setattr(bc, "save_sim_input", fake_save)
    monkeypatch.setattr(
        bc, "read_cache_version",
        lambda cache_dir: types.SimpleNamespace(inputs_hash="0123456789abcdef"),
    )
    monkeypatch.setattr(bc.os, "chdir", lambda path: None)
    monkeypatch.setattr(bc.os, "listdir", lambda path: [])


def test_build_cache_forwards_strain_to_save_sim_input(monkeypatch, tmp_path):
    """A real --new-genes lands in save_sim_input; 'off' bundle_overrides
    normalizes to None so the bundle stamps wild-type for that axis."""
    captured: dict = {}
    _patch_heavy(monkeypatch, captured)

    bc.build_cache(
        "fixture.pkl.gz", str(tmp_path),
        new_genes="violacein", bundle_overrides="off",
    )

    assert captured["new_genes"] == "violacein"
    assert captured["bundle_overrides"] is None


def test_build_cache_wildtype_by_default(monkeypatch, tmp_path):
    """No strain args -> save_sim_input sees None/None (unchanged wild-type build)."""
    captured: dict = {}
    _patch_heavy(monkeypatch, captured)

    bc.build_cache("fixture.pkl.gz", str(tmp_path))

    assert captured["new_genes"] is None
    assert captured["bundle_overrides"] is None
