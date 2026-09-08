"""Batch ecoli_baseline must thread per-cell biological build kwargs.

Regression tests for the pipeline-audit P0: baseline()'s batch path
(``n_seeds>1`` or ``n_generations>1``) used to forward only the seed/lineage
knobs to ``_build_batch_document``, silently DROPPING ``injected_processes``
(the metabolism-redux / violacein swap) plus ``features``, the four feature
toggles, ``exchange_fluxes``/``exchange_flux_basis`` and the two PDMP initiation
modes. The loss propagated all the way to ``meta_composite._lineage_node``,
which read ``config.get("injected_processes") or {}`` == ``{}`` and built every
generation cell BASAL — an injected batch degraded to basal FBA with no error.

These tests pin the full bridge:
    baseline() -> _build_batch_document -> runner_config
                -> dispatch_batch -> build_workflow_config
                -> build_meta_composite -> _lineage_node (per-generation cell)
and the fail-loud coverage guard that makes the class of bug structurally
impossible (a set-but-unforwarded batch-incompatible kwarg must raise, and every
WCM parameter must be explicitly classified).
"""

import pytest


SWAP = {"swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"}}


def test_batch_document_carries_injected_processes():
    """A batch build (n_seeds=2, n_generations=2) with injected_processes must
    put the injection (and the other per-cell build kwargs) into the
    BatchBaselineRunner config embedded in the batch-orchestrator document —
    NOT drop it (the audit defect)."""
    from v2ecoli.composites.ecoli_baseline import baseline

    doc = baseline(
        n_seeds=2, n_generations=2, injected_processes=SWAP,
        trna_attenuation=True, supercoiling=True,
        exchange_fluxes={"glucose_exchange": "GLC[p]"},
        exchange_flux_basis="gdcw",
        transcript_initiation_mode="poisson",
        features=["mass_conservation"])

    runner_config = doc["state"]["batch_runner"]["config"]
    assert (runner_config["injected_processes"]["swap_processes"]
            == {"ecoli-metabolism": "ecoli-metabolism-redux"})
    assert runner_config["trna_attenuation"] is True
    assert runner_config["supercoiling"] is True
    assert runner_config["exchange_fluxes"] == {"glucose_exchange": "GLC[p]"}
    assert runner_config["exchange_flux_basis"] == "gdcw"
    assert runner_config["transcript_initiation_mode"] == "poisson"
    assert runner_config["features"] == ["mass_conservation"]


def test_build_workflow_config_carries_injected_processes():
    """The runner's config translation must surface injected_processes (and the
    toggles/modes) in the workflow config the meta-composite consumes."""
    from v2ecoli.steps.batch_baseline_runner import build_workflow_config

    cfg = build_workflow_config(
        n_seeds=2, n_generations=2, analyses="none",
        injected_processes=SWAP, trna_attenuation=True,
        exchange_fluxes={"glucose_exchange": "GLC[p]"},
        exchange_flux_basis="gdcw", transcript_initiation_mode="poisson",
        polypeptide_initiation_mode="poisson", features=["mass_conservation"])

    assert (cfg["injected_processes"]["swap_processes"]
            == {"ecoli-metabolism": "ecoli-metabolism-redux"})
    assert cfg["trna_attenuation"] is True
    assert cfg["exchange_fluxes"] == {"glucose_exchange": "GLC[p]"}
    assert cfg["exchange_flux_basis"] == "gdcw"
    assert cfg["transcript_initiation_mode"] == "poisson"
    assert cfg["polypeptide_initiation_mode"] == "poisson"
    assert cfg["features"] == ["mass_conservation"]


def test_batch_chain_reaches_lineage_node():
    """End-to-end: the config produced from a batch dispatch must reach every
    per-(variant,seed) lineage node with the injection intact, so each
    generation's baseline() build engages metabolism-redux instead of basal FBA."""
    from v2ecoli.steps.batch_baseline_runner import BatchBaselineRunner, dispatch_batch
    from v2ecoli.workflow.meta_composite import build_meta_composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline

    # Build the batch document exactly as a caller would, then drive the runner's
    # own dispatch with a stub workflow so no simulation runs.
    doc = baseline(n_seeds=2, n_generations=2, injected_processes=SWAP,
                   trna_attenuation=True, transcript_initiation_mode="poisson",
                   exchange_fluxes={"glucose_exchange": "GLC[p]"})
    runner_config = doc["state"]["batch_runner"]["config"]
    runner = BatchBaselineRunner(runner_config, build_core())

    captured = {}

    def _stub_run_workflow(config):
        captured["config"] = config
        # Report a branch per seed, as the real workflow does: a workflow result
        # with NO branches is now a failed dispatch (dispatch_batch refuses to
        # record a batch in which no seed reported back as completed).
        return {"complete": True, "branches": {
            f"variant=0/seed={s}": {"complete": True, "summary": {"generations": []}}
            for s in range(int(config["lineage_seed"]),
                           int(config["lineage_seed"]) + int(config["n_init_sims"]))}}

    dispatch_batch(
        n_seeds=runner.n_seeds, n_generations=runner.n_generations,
        base_seed=runner.base_seed, single_daughters=runner.single_daughters,
        time_step=runner.time_step, max_duration=runner.max_duration,
        cache_dir=runner.cache_dir, out_dir="out/_test_batch_injected",
        experiment_id=runner.experiment_id, emitter=runner.emitter,
        parallel=None, variants=runner.variants, analyses="none",
        study=runner.study, base_config_overrides=runner.base_config_overrides,
        media=runner.media, injected_processes=runner.injected_processes,
        features=runner.features, ppgpp_regulation=runner.ppgpp_regulation,
        trna_attenuation=runner.trna_attenuation, supercoiling=runner.supercoiling,
        mass_conservation=runner.mass_conservation,
        exchange_fluxes=runner.exchange_fluxes,
        exchange_flux_basis=runner.exchange_flux_basis,
        transcript_initiation_mode=runner.transcript_initiation_mode,
        polypeptide_initiation_mode=runner.polypeptide_initiation_mode,
        run_workflow_fn=_stub_run_workflow)

    workflow_config = captured["config"]
    meta_doc = build_meta_composite(workflow_config)

    branches = list(meta_doc["state"]["branches"].values())
    assert branches, "expected at least one lineage branch"
    for branch in branches:
        node_cfg = branch["lineage"]["config"]
        # THE regression: pre-fix this was {} for every branch -> basal FBA.
        assert (node_cfg["injected_processes"]["swap_processes"]
                == {"ecoli-metabolism": "ecoli-metabolism-redux"})
        assert node_cfg["trna_attenuation"] is True
        assert node_cfg["transcript_initiation_mode"] == "poisson"
        assert node_cfg["exchange_fluxes"] == {"glucose_exchange": "GLC[p]"}


def test_batch_guard_raises_for_set_but_unforwarded_incompatible_key():
    """A single-cell-only parameter set to a non-default value in batch mode must
    RAISE rather than be silently dropped (mirrors the match_simdata guard)."""
    from v2ecoli.composites.ecoli_baseline import baseline

    with pytest.raises(ValueError, match="single-cell-only"):
        baseline(n_seeds=2, n_generations=1, emitter_out_dir="/tmp/somewhere")

    with pytest.raises(ValueError, match="single-cell-only"):
        baseline(n_seeds=1, n_generations=2, match_condition="with_aa")


def test_batch_parameter_coverage_is_total():
    """Structural guarantee: every WCM parameter is classified as either
    forwarded-to-batch or batch-incompatible, in exactly one set. This is what
    makes the next dropped kwarg fail loudly instead of silently no-op'ing."""
    from v2ecoli.composites.ecoli_baseline import (
        WCM_PARAMETERS, _BATCH_FORWARDED_PARAMETERS,
        _BATCH_INCOMPATIBLE_PARAMETERS, _assert_batch_parameter_coverage)

    classified = _BATCH_FORWARDED_PARAMETERS | _BATCH_INCOMPATIBLE_PARAMETERS
    assert set(WCM_PARAMETERS) == classified, (
        "unclassified WCM parameter(s): "
        f"{sorted(set(WCM_PARAMETERS) - classified)}")
    assert not (_BATCH_FORWARDED_PARAMETERS & _BATCH_INCOMPATIBLE_PARAMETERS)

    # A batch call with all batch-incompatible keys at their defaults passes.
    defaults = {k: WCM_PARAMETERS[k].get("default")
                for k in _BATCH_INCOMPATIBLE_PARAMETERS}
    _assert_batch_parameter_coverage(defaults)  # must not raise
