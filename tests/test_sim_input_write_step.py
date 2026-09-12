"""Unit tests for SimInputWriteStep (N5) -- ParCa's optional persist step.

Verifies it reads the REAL sim_data off the ``sim_data_root`` port and calls
``save_sim_input`` with the bundle-identity params, and that it is registered
(so a document can reference it) but stays OUT of the default fit chain.
"""
from unittest.mock import patch

from v2ecoli.core import build_core
from v2ecoli.processes.parca.composite import STEP_ORDER
from v2ecoli.processes.parca.steps import ALL_STEP_CLASSES, SimInputWriteStep


def _make_step(config):
    return SimInputWriteStep(config, core=build_core())


def test_registered_but_not_in_default_chain():
    # Referenceable by name...
    assert ALL_STEP_CLASSES.get("SimInputWriteStep") is SimInputWriteStep
    # ...but NOT auto-added to the default ParCa fit chain (no behavior change).
    assert "SimInputWriteStep" not in STEP_ORDER


def test_update_calls_save_sim_input_and_returns_bundle_dir():
    sentinel_sim_data = object()
    step = _make_step({
        "cache_dir": "out/some_cache",
        "seed": 3,
        "new_genes": "violacein_MG1655_M5",
        "bundle_overrides": "models/parca/violacein_bundle_overrides.tsv",
    })
    with patch("v2ecoli.core.save_sim_input") as mock_save:
        out = step.update({"sim_data_root": sentinel_sim_data})
    mock_save.assert_called_once()
    args, kwargs = mock_save.call_args
    assert args[0] is sentinel_sim_data  # the REAL object, not a facade proxy
    assert kwargs["bundle_dir"] == "out/some_cache"
    assert kwargs["seed"] == 3
    assert kwargs["new_genes"] == "violacein_MG1655_M5"
    assert kwargs["bundle_overrides"] == "models/parca/violacein_bundle_overrides.tsv"
    assert out == {"bundle_dir": "out/some_cache"}


def test_empty_string_identity_params_become_none():
    step = _make_step({"cache_dir": "out/c"})  # condition/new_genes/etc default to ''
    with patch("v2ecoli.core.save_sim_input") as mock_save:
        step.update({"sim_data_root": object()})
    kwargs = mock_save.call_args.kwargs
    for k in ("condition", "fixed_media", "new_genes", "bundle_overrides", "bundle_manifest"):
        assert kwargs[k] is None  # '' -> None for save_sim_input's identity params
