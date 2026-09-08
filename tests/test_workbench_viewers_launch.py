"""build_ptools_launch_url picks the right TSV to overlay.

A study's ``ptools/`` directory can hold more than the PTools EcoCyc exports —
e.g. a study that also ran the CD1 omics comparison writes ``cd1_*.tsv`` files
next to the ``ptools_*.tsv`` ones. Those ``cd1_*`` names sort BEFORE ``ptools_``
alphabetically, so a naive ``sorted(...)[0]`` would launch the wrong file. The
launcher must deterministically prefer the combined ``overview`` export.
"""
from __future__ import annotations

from pathlib import Path

from v2ecoli.workbench_viewers import build_ptools_launch_url

_TEMPLATE = "{server}/celOv.shtml?url={tsv_url}&class={cls}&column1={columns}"


def _write_tsv(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("frame_id\tt0\tt1\nEG10001\t1\t2\n", encoding="utf-8")


def _launch(study_dir: Path, ws_root: Path, **kw) -> dict:
    return build_ptools_launch_url(
        study_dir=study_dir,
        ws_root=ws_root,
        ptools_server_url="http://localhost:1555",
        ptools_omics_url_template=_TEMPLATE,
        public_base="http://localhost",
        **kw,
    )


def test_prefers_overview_over_alphabetically_earlier_tsv(tmp_path: Path) -> None:
    sd = tmp_path / "studies" / "cd2-api-final-mec"
    pt = sd / "ptools"
    # cd1_fluxomics sorts before ptools_* — the exact trap the fix closes.
    _write_tsv(pt / "cd1_fluxomics__variant=0.tsv")
    _write_tsv(pt / "ptools_overview_multigeneration__variant=0_seed=0.tsv")
    _write_tsv(pt / "ptools_rna_multigeneration__variant=0_seed=0.tsv")

    res = _launch(sd, tmp_path)
    assert "error" not in res, res
    assert res["tsv_url"].endswith(
        "ptools_overview_multigeneration__variant=0_seed=0.tsv"
    ), res["tsv_url"]
    # every TSV is still reported as available
    assert len(res["available"]) == 3


def test_falls_back_to_ptools_when_no_overview(tmp_path: Path) -> None:
    sd = tmp_path / "studies" / "cd2-x"
    pt = sd / "ptools"
    _write_tsv(pt / "cd1_metabolomics__variant=0.tsv")
    _write_tsv(pt / "ptools_rxns_multigeneration__variant=0_seed=0.tsv")

    res = _launch(sd, tmp_path)
    assert res["tsv_url"].endswith("ptools_rxns_multigeneration__variant=0_seed=0.tsv"), res["tsv_url"]


def test_explicit_analysis_filter_still_honored(tmp_path: Path) -> None:
    sd = tmp_path / "studies" / "cd2-y"
    pt = sd / "ptools"
    _write_tsv(pt / "ptools_overview_multigeneration__variant=0_seed=0.tsv")
    _write_tsv(pt / "ptools_rxns_multigeneration__variant=0_seed=0.tsv")

    # asking for rxns explicitly must win over the overview default
    res = _launch(sd, tmp_path, analysis="ptools_rxns_multigeneration")
    assert res["tsv_url"].endswith("ptools_rxns_multigeneration__variant=0_seed=0.tsv"), res["tsv_url"]


def test_no_tsvs_is_a_shaped_error(tmp_path: Path) -> None:
    sd = tmp_path / "studies" / "empty"
    sd.mkdir(parents=True)
    res = _launch(sd, tmp_path)
    assert res.get("available") == [] and "error" in res


def test_ptools_targets_include_landed_runs(tmp_path: Path) -> None:
    """The PTools run menu (`_ptools_targets`) must span local studies AND landed
    runs — the latter is how a GovCloud compose analysis (whose ptools TSVs the
    workbench land path copies into .pbg/runs/<id>/ptools/) shows up."""
    from v2ecoli.workbench_viewers import _ptools_targets

    _write_tsv(tmp_path / "studies" / "showcase" / "ptools" / "ptools_overview__v0.tsv")
    run_pt = tmp_path / ".pbg" / "runs" / "gov-run-1" / "ptools"
    _write_tsv(run_pt / "ptools_overview__v0_s0.tsv")
    _write_tsv(run_pt / "ptools_rxns__v0_s0.tsv")

    targets = _ptools_targets(tmp_path)
    studies = [t for t in targets if t.get("study")]
    runs = [t for t in targets if t.get("run")]
    assert any(t["study"] == "showcase" for t in studies)
    assert any(t["run"] == "gov-run-1" and "2 TSV" in t["detail"] for t in runs)


def test_launch_resolves_a_run_dir(tmp_path: Path) -> None:
    """`_launch` given a `run` (not a study) discovers TSVs under
    .pbg/runs/<run>/ and needs no study to exist."""
    from v2ecoli import workbench_viewers as wbv

    run_pt = tmp_path / ".pbg" / "runs" / "gov-run-2" / "ptools"
    _write_tsv(run_pt / "ptools_overview__v0_s0.tsv")
    # ptools_server_url must be configured for _launch to proceed; stub _ui_config.
    wbv._ui_config = lambda ws_root: {"ptools_server_url": "http://localhost:1555"}  # type: ignore[assignment]
    out = wbv._launch(tmp_path, study=None, run="gov-run-2", ctx={})
    assert "url" in out and "error" not in out, out
    missing = wbv._launch(tmp_path, study=None, run="does-not-exist", ctx={})
    assert missing.get("status") == 404
