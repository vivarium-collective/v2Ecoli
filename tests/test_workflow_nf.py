"""workflow_nf: the campaign DAG, rendered rather than run.

The properties worth pinning are the ones that would otherwise fail silently --
a variant whose strain inputs never reach ParCa still produces N runs that look
right and share one genotype.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from v2ecoli.composites.workflow_nf import (
    AnalysisTaskStep,
    ParcaTaskStep,
    build_workflow_nf,
)


@pytest.fixture(scope="module")
def core():
    from v2ecoli.core import build_core
    from v2ecoli.workflow.meta_composite import register_workflow_processes

    c = build_core()
    register_workflow_processes(c)
    return c


def _render(core, doc):
    from process_bigraph import Composite
    from process_bigraph.nextflow import render_composite

    return render_composite(Composite(doc, core=core), {"workflow_name": ""})


def test_shape_is_one_parca_per_variant_feeding_its_own_seeds() -> None:
    doc = build_workflow_nf(
        n_seeds=2, variants=[{"variant_name": "a"}, {"variant_name": "b"}]
    )
    state = doc["state"]
    assert sorted(state) == ["parca_v0", "parca_v1", "runs_v0", "runs_v1"]
    # each variant's sub-composite reads ITS OWN cache -- the point of a strain sweep
    assert state["runs_v0"]["inputs"]["cache"] == ["cache_v0"]
    assert state["runs_v1"]["inputs"]["cache"] == ["cache_v1"]
    # and the M lineages live inside it, wired to the take: port
    inner = state["runs_v0"]["config"]["state"]
    assert sorted(k for k in inner if k.startswith("lineage")) == [
        "lineage_v0_s0",
        "lineage_v0_s1",
    ]
    assert inner["lineage_v0_s1"]["inputs"]["cache_dir"] == ["cache"]


def test_strain_inputs_reach_PARCA_not_the_lineage() -> None:
    """The silent failure this guards: threading new_genes to the lineage instead
    (what lineage_ray_batch's variants collapse does today) gives ONE shared cache
    for the whole sweep -- N runs of the same genotype wearing different labels."""
    doc = build_workflow_nf(
        n_seeds=1,
        variants=[
            {
                "variant_name": "vio",
                "new_genes": "violacein_MG1655_M5",
                "bundle_overrides": "/m.json",
            }
        ],
    )
    parca_cfg = doc["state"]["parca_v0"]["config"]
    assert parca_cfg["new_genes"] == "violacein_MG1655_M5"
    assert parca_cfg["bundle_overrides"] == "/m.json"
    inner = doc["state"]["runs_v0"]["config"]["state"]
    assert "new_genes" not in inner["lineage_v0_s0"]["config"]


def test_no_variants_means_one_baseline_not_zero() -> None:
    """Zero variants would render an empty workflow that exits 0."""
    doc = build_workflow_nf(n_seeds=1, variants=None)
    assert "parca_v0" in doc["state"] and "runs_v0" in doc["state"]


def test_renders_the_two_level_scatter(core) -> None:
    doc = build_workflow_nf(
        n_seeds=2, variants=[{"variant_name": "a"}, {"variant_name": "b"}]
    )
    nf = _render(core, doc)
    # 2 parca + 2 lineage process blocks (one per node inside the sub-workflows)
    assert nf.count("workflow runs_v") == 2
    assert "ch_results_v0 = runs_v0(ch_cache_v0)" in nf
    assert "ch_results_v1 = runs_v1(ch_cache_v1)" in nf


def test_parca_script_carries_the_strain_flags_and_the_hydrate_step(core) -> None:
    """--new-genes/--bundle-overrides go to v2ecoli-parca ONLY: build_cache.py's CLI
    has neither (viva-api#410, a real crash). And the hydrate step is not optional --
    v2ecoli-parca emits only the raw parca_state.pkl."""
    step = ParcaTaskStep(
        config={"new_genes": "vio", "bundle_overrides": "/m.json"}, core=core
    )
    script = step.nextflow_script()
    parca_cmd, _, rest = script.partition("&&")
    assert "--new-genes vio" in parca_cmd and "--bundle-overrides /m.json" in parca_cmd
    assert "--new-genes" not in rest and "--bundle-overrides" not in rest
    assert "build_cache.py" in rest and "gzip" in rest


def test_parca_omits_the_off_sentinel(core) -> None:
    """'off' IS v2ecoli-parca's default for --new-genes, so passing it and omitting
    it are the same build. Omit, so the command stays byte-identical to a plain one."""
    assert (
        "--new-genes"
        not in ParcaTaskStep(config={"new_genes": "off"}, core=core).nextflow_script()
    )


def test_task_declarations_refuse_to_run_in_process(core) -> None:
    """These nodes carry no simulation logic. Running one would advance global_time
    and produce nothing while exiting 0 -- the silent-success class this plan is
    organised around."""
    with pytest.raises(RuntimeError, match="rendered"):
        ParcaTaskStep(config={}, core=core).update({})
    with pytest.raises(RuntimeError):
        AnalysisTaskStep(config={}, core=core).update({})


# --- the gather, and why it is gated ---------------------------------------


def test_gather_is_off_by_default() -> None:
    assert "analysis" not in build_workflow_nf(n_seeds=2)["state"]


def test_each_variant_collects_its_seeds_into_ONE_channel(core) -> None:
    """The fan-in, and why it needs nesting: M sibling channels cannot be wired to
    one port (a port maps to ONE store path -- the flat spelling raises
    `TypeError: unhashable type: 'list'` inside realize, before the renderer runs).
    A nested composite emits `take:`/`emit:` with CHAINED BINARY mixes, so M
    channels arrive at the parent as one."""
    nf = _render(core, build_workflow_nf(n_seeds=3, include_analysis=True))
    assert "workflow runs_v0 {" in nf
    assert "take:" in nf and "emit:" in nf
    assert nf.count("_merged = _merged.mix(") == 2  # 3 units -> 2 binary mixes
    assert "_merged.collect()" in nf


def test_analysis_takes_one_port_per_variant(core) -> None:
    """N named ports, each fed by a variant's already-collected channel. One port
    wired to N stores is what the document model cannot express."""
    nf = _render(
        core,
        build_workflow_nf(
            n_seeds=2,
            include_analysis=True,
            variants=[{"variant_name": "a"}, {"variant_name": "b"}],
        ),
    )
    assert "analysis(ch_results_v0, ch_results_v1" in nf


def test_analysis_task_argv_parses_against_the_real_cli(core) -> None:
    """The gather's emitted command must parse against the real v2ecoli-analyze
    CLI. It used to emit --experiment-id/--out-dir, which the CLI rejects with
    exit 2 (#722), so a --include-analysis campaign ran every lineage and then
    died at the gather. Parse the emitted argv against the one shared parser."""
    import shlex

    from v2ecoli.workflow.analysis_runner import build_analysis_arg_parser

    script = AnalysisTaskStep(
        config={"analysis_options": {"single": {"mass_fraction_summary": {}}}},
        core=core,
    ).nextflow_script()
    line = next(
        l for l in script.splitlines() if l.strip().startswith("v2ecoli-analyze")
    )
    args = build_analysis_arg_parser().parse_args(shlex.split(line)[1:])
    assert args.sweep_dir == "." and args.config == "analysis.config.json"


def test_analysis_options_reach_the_gather_config() -> None:
    """analysis_options ride in the staged node config (analysis.config.json), the
    only place the CLI reads them, and out_dir is task-local to match the declared
    `path "analysis"` output."""
    doc = build_workflow_nf(
        n_seeds=2,
        include_analysis=True,
        analysis_options={"single": {"mass_fraction_summary": {}}},
    )
    cfg = doc["state"]["analysis"]["config"]
    assert cfg["analysis_options"] == {"single": {"mass_fraction_summary": {}}}
    assert cfg["out_dir"] == "analysis"


def test_renders_at_run4_scale_without_hitting_the_255_wall(core) -> None:
    """go/no-go 4: 84 variants x 4 seeds = 336 lineages. The unrolled form died at
    256 arguments (`bad parameter count 257`); nesting keeps the parent call at one
    argument per VARIANT and the mixes binary."""
    variants = [{"variant_name": f"v{i}"} for i in range(84)]
    nf = _render(
        core, build_workflow_nf(n_seeds=4, include_analysis=True, variants=variants)
    )
    assert nf.count("workflow runs_v") == 84
    assert nf.count("= lineage_v") == 336
    # chained binary, never one n-ary call
    assert ".mix(" in nf and all(
        "," not in seg[: seg.index(")")] for seg in nf.split(".mix(")[1:]
    )


# --- the generator must stand on its own ------------------------------------


def _spec_for_workflow_nf():
    from process_bigraph.composite_spec import discover_specs
    from process_bigraph.composite_spec import get as get_spec

    key = "v2ecoli.composites.workflow_nf.workflow_nf"
    spec = get_spec(key)
    if spec is None:
        discover_specs()
        spec = get_spec(key)
    assert spec is not None, f"{key} is not registered"
    return spec


def test_generator_declares_the_core_extensions_its_document_needs() -> None:
    """Every node this generator emits is a `local:` address someone must register.

    The `core` fixture above calls register_workflow_processes BY HAND, so these
    tests passed while the real path -- resolving through the generator, which is
    what run_pbg and viva-api's render_nf do -- could not construct the document
    at all.
    """
    from v2ecoli.workflow.meta_composite import register_workflow_processes

    spec = _spec_for_workflow_nf()
    assert register_workflow_processes in (spec.core_extensions or [])


def test_document_realizes_against_a_core_built_ONLY_from_the_generator() -> None:
    """The effect check, and the one that actually failed on real infrastructure.

    Builds the core the way the dispatcher does -- build_core() plus the
    generator's declared extensions, nothing hand-added -- and constructs the
    document. Without core_extensions this raises

        Exception: no link found at address: {'protocol': 'local', 'data': 'composite'}

    which is the nested Composite, i.e. every sub-workflow this path exists for.
    """
    from process_bigraph import Composite
    from process_bigraph.composite_generator import apply_core_extensions

    from v2ecoli.core import build_core

    core = apply_core_extensions(_spec_for_workflow_nf(), build_core())
    doc = build_workflow_nf(n_seeds=1, n_generations=1)
    Composite(doc, core=core)  # must not raise


# --- what the emitted script assumes about the machine it runs on -----------


def _render_via_generator(**kwargs):
    from process_bigraph import Composite
    from process_bigraph.composite_generator import apply_core_extensions
    from process_bigraph.nextflow import render_composite

    from v2ecoli.core import build_core

    core = apply_core_extensions(_spec_for_workflow_nf(), build_core())
    doc = build_workflow_nf(**kwargs)
    return render_composite(Composite(doc, core=core), {"workflow_name": ""})


def test_build_cache_is_never_resolved_against_the_work_dir() -> None:
    """`scripts/` does not ship with the wheel, and a task's cwd is its work dir.

    So `python scripts/build_cache.py` from the work dir cannot work however the
    cwd is set. It is invoked from the CHECKOUT instead -- which is also what
    makes find_workspace_root() succeed (see the test below) -- while its
    inputs and outputs stay absolute paths into the work dir.
    """
    nf = _render_via_generator(n_seeds=1, n_generations=1)
    assert 'cd "/app/v2ecoli"' in nf
    assert "&& python scripts/build_cache.py" in nf
    # never the bare cwd-relative form that started this
    assert "&& python scripts/build_cache.py --fixture parca/" not in nf


def test_the_repo_root_is_not_a_shell_variable() -> None:
    """A `script:` block is a GROOVY string, so `${VAR:-default}` is interpolated
    by Groovy, not bash -- it dies at run time with
    `No signature of method: java.lang.String.negative()`. Measured, not
    hypothesised; see _repo_root."""
    nf = _render_via_generator(n_seeds=1, n_generations=1)
    assert "V2E_ROOT" not in nf


def test_every_task_carries_a_label() -> None:
    """`withLabel: lineage { cpus/memory/time }` in the executor profile matches
    NOTHING without these. Every task would take queue defaults and, in
    particular, no `time` -- the only bound on a runaway task."""
    nf = _render_via_generator(n_seeds=2, n_generations=1, include_analysis=True)
    assert nf.count("label 'lineage'") == 2
    assert "label 'parca'" in nf
    assert "label 'analysis'" in nf


def _pbg_quotes_script_overrides() -> bool:
    """process-bigraph#205: the renderer wraps a `nextflow_script()` override in a
    Groovy block. Without it ParcaTaskStep's command lands in `script:` as Groovy
    SOURCE and the file cannot compile -- which is a defect in the pinned
    process-bigraph, not in this repo, so the check below reports it as skipped
    rather than failing this suite for someone else's version."""
    from process_bigraph import nextflow as _nf

    return hasattr(_nf, "_as_script_block")


@pytest.mark.skipif(
    shutil.which("nextflow") is None, reason="nextflow binary not on PATH"
)
@pytest.mark.skipif(
    not _pbg_quotes_script_overrides(), reason="needs process-bigraph#205"
)
def test_the_rendered_workflow_actually_compiles(tmp_path) -> None:
    """The check that would have caught process-bigraph#205.

    A render can succeed, report a plausible summary and the right sub-workflow
    structure, and still emit a file Nextflow cannot parse -- an unquoted
    `script:` block is Groovy source. Only running it tells you.
    """
    (tmp_path / "main.nf").write_text(
        _render_via_generator(n_seeds=2, include_analysis=True)
    )
    (tmp_path / "nextflow.config").write_text(
        "profiles { local { process { executor='local' } } }\n"
    )
    proc = subprocess.run(
        ["nextflow", "run", "main.nf", "-profile", "local", "-stub-run"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        # Nextflow's banner is UTF-8; under a C/POSIX locale `text=True` decodes
        # as ascii and raises UnicodeDecodeError instead of reporting the run.
        encoding="utf-8",
        errors="replace",
    )
    combined = proc.stdout + proc.stderr
    assert "Script compilation error" not in combined, combined[:2000]
    assert "No signature of method" not in combined, combined[:2000]
    # It reaches EXECUTION: the only failure allowed here is the science CLI
    # being absent from the test machine.
    assert "executor >" in combined, combined[:2000]


@pytest.mark.skipif(
    shutil.which("nextflow") is None, reason="nextflow binary not on PATH"
)
@pytest.mark.skipif(
    not _pbg_quotes_script_overrides(), reason="needs process-bigraph#205"
)
def test_a_multi_variant_campaign_compiles(tmp_path) -> None:
    """The single-variant compile test above passes even when a >=2-variant
    render emits duplicate top-level process names: each variant's nested
    composite renders its lineages by leaf, so a bare `lineage_s{seed}` collides
    across variants ("Identifier lineage_s0 is already used"). Run 4's 84-genotype
    campaign and Run 2's grid are multi-variant, so render two variants and
    confirm Nextflow accepts the file."""
    (tmp_path / "main.nf").write_text(
        _render_via_generator(
            n_seeds=2,
            include_analysis=True,
            variants=[{"variant_name": "a"}, {"variant_name": "b"}],
        )
    )
    (tmp_path / "nextflow.config").write_text(
        "profiles { local { process { executor='local' } } }\n"
    )
    proc = subprocess.run(
        ["nextflow", "run", "main.nf", "-profile", "local", "-stub-run"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = proc.stdout + proc.stderr
    assert "Script compilation error" not in combined, combined[:2000]
    assert "is already used by another definition" not in combined, combined[:2000]
    assert "executor >" in combined, combined[:2000]


def test_build_cache_runs_from_the_checkout_but_writes_to_the_work_dir() -> None:
    """`build_cache.py` imports pbg_v2ecoli, whose apply_upstream_patches() calls
    find_workspace_root() -- a walk up from CWD for workspace.yaml. A Nextflow
    task's cwd is its work dir, which has none, so the import died with
    FileNotFoundError *after* ParCa had already run for 2.5 minutes.

    The workspace root must be the checkout (models/ lives there and is not
    shipped in site-packages), so that one command runs from there -- with
    absolute paths built from a shell $PWD captured first, so the declared
    output `cache` still lands where Nextflow looks for it.
    """
    nf = _render_via_generator(n_seeds=1, n_generations=1)
    assert 'WD="\\$PWD"' in nf, "must be ESCAPED: a bare $ is interpolated by Groovy"
    assert 'cd "/app/v2ecoli"' in nf
    assert '--cache "\\$WD/cache"' in nf
    # and it must come back, so the trailing cp writes into the work dir
    assert 'cd "\\$WD"' in nf


def test_an_explicitly_declared_repo_root_wins(monkeypatch, tmp_path) -> None:
    """A task scheduler gives each task its own cwd with no workspace above it.

    `find_workspace_root` walks up from CWD, so under Nextflow it raises, the
    `except` in candidate_repo_roots swallows it, and the only remaining root is
    site-packages -- where `models/` is not shipped. The failure then reads
    "INPUT_FILES entry does not exist ... tried roots: ['…/site-packages']",
    which looks like a renamed data file rather than an unresolved workspace.
    """
    import os

    from v2ecoli.library.cache_version import candidate_repo_roots

    declared = tmp_path / "checkout"
    declared.mkdir()
    monkeypatch.chdir(tmp_path)  # no workspace.yaml anywhere above
    monkeypatch.setenv("V2E_ROOT", str(declared))
    assert candidate_repo_roots()[0] == str(declared)

    # unset, and behaviour is exactly as before
    monkeypatch.delenv("V2E_ROOT")
    assert str(declared) not in candidate_repo_roots()

    # a declared-but-nonexistent root is ignored rather than poisoning the list
    monkeypatch.setenv("V2E_ROOT", str(tmp_path / "nope"))
    assert str(tmp_path / "nope") not in candidate_repo_roots()
    assert os.environ["V2E_ROOT"]


# --- publishDir: without it a successful campaign has no reachable output ----


def test_the_lineage_publishes_its_sweep(core) -> None:
    """Measured on a real run before this landed: 633 MB across 43 objects sat in
    the Nextflow WORK dir under a content hash, while the results prefix held
    78 KB of render artifacts and no science at all. A campaign that exits 0 with
    nowhere to read its output is the silent-success shape in its purest form."""
    rendered = _render(core, build_workflow_nf(n_seeds=1, n_generations=1))
    block = rendered.split("process lineage_v0_s0 {", 1)[1].split("}", 1)[0]
    assert "publishDir" in block


def test_the_analysis_publishes_its_report(core) -> None:
    """The gather's report IS the deliverable; an unpublished one is as
    unreachable as no report."""
    rendered = _render(core, build_workflow_nf(n_seeds=1, include_analysis=True))
    block = rendered.split("process analysis {", 1)[1].split("}", 1)[0]
    assert "publishDir" in block


def test_parca_does_not_publish_its_cache(core) -> None:
    """Deliberate. The cache is an INTERMEDIATE (~262 MB) and staging it
    task-to-task is what the work dir is for -- publishing would double every
    campaign's storage to no one's benefit."""
    rendered = _render(core, build_workflow_nf(n_seeds=1))
    block = rendered.split("process parca_v0 {", 1)[1].split("}", 1)[0]
    assert "publishDir" not in block


def test_the_publish_target_is_a_closure_over_a_param_with_a_local_fallback(
    core,
) -> None:
    """A literal cannot work: the destination is the RUN's own prefix, known only
    at dispatch. And the `?:` is not decoration -- without it a bare
    `nextflow run` dies on a missing param, which would make every local render
    unrunnable to buy nothing."""
    rendered = _render(core, build_workflow_nf(n_seeds=1))
    line = next(ln for ln in rendered.splitlines() if "publishDir" in ln)
    assert line.strip().startswith("publishDir {"), line
    assert "params.publish_dir" in line
    assert "?:" in line, "a missing param must fall back, not fail the run"
    assert 'mode: "copy"' in line


def test_every_lineage_publishes_to_the_same_target(core) -> None:
    """On purpose: the tree underneath is hive-partitioned
    (experiment_id/variant/lineage_seed/generation/agent_id), so copies interleave
    rather than collide. Identity lives in the partitions, not the directory name
    -- which is also why publishing per-task subdirs would BREAK the gather."""
    rendered = _render(core, build_workflow_nf(n_seeds=3, n_generations=1))
    lines = [ln.strip() for ln in rendered.splitlines() if "publishDir" in ln]
    assert len(lines) == 3
    assert len(set(lines)) == 1


# --- every kwarg must be DECLARED, or it is unreachable from a dispatch ------


def test_analysis_options_is_reachable_through_a_dispatch() -> None:
    """It was a function kwarg, threaded into the node config, and NOT declared
    in the generator's `parameters` block. Every dispatch goes through
    `CompositeSpec.to_document(overrides=...)`, whose `_merged_params` raises
    `KeyError: unknown override(s)` on anything undeclared -- so it could only be
    set by calling the builder in-process. Through the API it always fell back to
    {}, the gather had nothing to run, no `analysis/` was created, and the
    campaign failed its LAST node after running every lineage (simulation 475)."""
    from process_bigraph.composite_spec import discover_specs
    from process_bigraph.composite_spec import get as get_spec

    discover_specs()
    spec = get_spec("v2ecoli.composites.workflow_nf.workflow_nf")
    doc = spec.to_document(
        overrides={
            "analysis_options": {"multiseed": {}},
            "n_seeds": 2,
            "include_analysis": True,
        }
    )
    assert doc["state"]["analysis"]["config"]["analysis_options"] == {"multiseed": {}}


def test_every_builder_kwarg_is_a_declared_parameter() -> None:
    """The general form, so the next one is caught at test time rather than by a
    campaign failing on real infrastructure. A kwarg the builder accepts but the
    spec does not declare is dead on every path that matters."""
    import inspect

    from process_bigraph.composite_spec import discover_specs
    from process_bigraph.composite_spec import get as get_spec

    discover_specs()
    spec = get_spec("v2ecoli.composites.workflow_nf.workflow_nf")
    kwargs = {
        name
        for name, prm in inspect.signature(build_workflow_nf).parameters.items()
        if prm.kind is not inspect.Parameter.VAR_KEYWORD
    }
    undeclared = kwargs - set(spec.parameters)
    assert not undeclared, (
        f"accepted by the builder, unreachable via dispatch: {sorted(undeclared)}"
    )


# --- gate 1b: M seeds must be M founders, not M copies of one ---------------


def test_independent_founders_reaches_every_lineage() -> None:
    """v2ecoli#693: `_load_cache_bundle_cached` is memoised on `cache_dir` alone
    and returns the initial state BY REFERENCE, so M seeds off one cache share
    one founder cell. Measured on simulation 326: two seeds differed by 251 of
    16321 species while a single timestep of one seed changes 754 — i.e. the
    seeds differ by less than one tick. #712 added the opt-out in `baseline()`;
    until now no Nextflow campaign could reach it, because one ParCa per variant
    feeds every lineage."""
    from process_bigraph.composite_spec import discover_specs
    from process_bigraph.composite_spec import get as get_spec

    discover_specs()
    spec = get_spec("v2ecoli.composites.workflow_nf.workflow_nf")
    doc = spec.to_document(overrides={"n_seeds": 3, "independent_founders": True})
    inner = doc["state"]["runs_v0"]["config"]["state"]
    lineages = sorted(k for k in inner if k.startswith("lineage"))
    assert len(lineages) == 3
    for name in lineages:
        cfg = inner[name]["config"]
        assert cfg["independent_founders"] is True
        # task-local: `cache_dir` is staged as `path "cache"` and ParCa writes
        # simData.cPickle inside it
        assert cfg["founder_sim_data"] == "cache/simData.cPickle"
    # and each still draws from its OWN seed -- one founder per lineage_seed is
    # the entire point
    assert sorted(inner[n]["config"]["lineage_seed"] for n in lineages) == [0, 1, 2]


def test_founders_are_off_by_default_and_the_keys_are_absent() -> None:
    """Omitted, not False: absent means `baseline()` keeps its own default, and
    it keeps the cheap path cheap (re-drawing regenerates initial conditions per
    seed)."""
    from process_bigraph.composite_spec import discover_specs
    from process_bigraph.composite_spec import get as get_spec

    discover_specs()
    spec = get_spec("v2ecoli.composites.workflow_nf.workflow_nf")
    cfg = spec.to_document(overrides={"n_seeds": 2})["state"]["runs_v0"]["config"][
        "state"
    ]["lineage_v0_s0"]["config"]
    assert "independent_founders" not in cfg
    assert "founder_sim_data" not in cfg


def test_the_keys_survive_every_hop_to_baseline() -> None:
    """Three places drop config silently if a key is not declared there, and all
    three had to change: LineageStep's `_FORWARDED` whitelist, LineageProcess's
    `config_schema`, and the `_bio_kwargs` dict that calls `baseline()`. A key
    present in one and missing from the next is dropped with no error."""
    import inspect

    from v2ecoli.composites.ecoli_baseline import baseline
    from v2ecoli.workflow.lineage import LineageProcess
    from v2ecoli.workflow.lineage_step import _FORWARDED

    for key in ("independent_founders", "founder_sim_data"):
        assert key in _FORWARDED, f"LineageStep would drop {key}"
        assert key in LineageProcess.config_schema, f"LineageProcess would drop {key}"
        assert key in inspect.signature(baseline).parameters, (
            f"baseline() rejects {key}"
        )

    src = inspect.getsource(LineageProcess.update.__globals__["LineageProcess"])
    assert "independent_founders=" in src, "_bio_kwargs must pass it to baseline()"


# --- pre-built caches: recomputing one is a different experiment ------------


def _parca_block(core, **kw) -> str:
    rendered = _render(core, build_workflow_nf(n_seeds=2, **kw))
    return rendered.split("process parca_v0 {", 1)[1].split("\n}", 1)[0]


def test_a_cache_uri_makes_the_node_fetch_instead_of_compute(core) -> None:
    """CD2's payloads do not build their caches at dispatch: Run 1 uses ten
    pre-built per-seed K4 founder caches (staged at ray-parca-cache/9f84e6b/) and
    Run 2 the violacein bundle. A campaign that recomputes is running a different
    experiment, however green it looks."""
    block = _parca_block(core, cache_uri="s3://bucket/ray-parca-cache/9f84e6b/")
    assert "aws s3 cp --recursive s3://bucket/ray-parca-cache/9f84e6b/" in block
    assert "v2ecoli-parca" not in block, "must not also run a ParCa"


def test_the_dag_is_unchanged_by_reuse(core) -> None:
    """The node keeps its `path "cache"` output, so `take: cache` and every
    lineage's staged input are identical. Removing the node instead would leave
    the lineages wired to nothing -- a Nextflow input is fed by a channel, not a
    path."""
    with_uri = _render(core, build_workflow_nf(n_seeds=2, cache_uri="s3://b/c/"))
    without = _render(core, build_workflow_nf(n_seeds=2))
    names = lambda nf: [
        ln.split()[1] for ln in nf.splitlines() if ln.startswith("process ")
    ]
    assert names(with_uri) == names(without)
    assert 'path "cache"' in _parca_block(core, cache_uri="s3://b/c/")


def test_a_fetched_cache_is_checked_for_contents(core) -> None:
    """`aws s3 cp --recursive` on an empty or wrong prefix copies zero objects
    and exits 0, leaving a cache-shaped directory that is not a cache. Every
    lineage would then fail far from the cause."""
    block = _parca_block(core, cache_uri="s3://b/c/")
    assert "test -f cache/simData.cPickle" in block
    assert "test -f cache/sim_data_cache.dill" in block


def test_a_variants_own_cache_uri_wins(core) -> None:
    """Run 1 pairs a DIFFERENT founder cache per seed, so one campaign-wide value
    cannot express it; a sweep may also reuse some caches and build others."""
    doc = build_workflow_nf(
        n_seeds=1,
        cache_uri="s3://b/global/",
        variants=[
            {"variant_name": "a", "cache_uri": "s3://b/own/"},
            {"variant_name": "b"},
        ],
    )
    assert doc["state"]["parca_v0"]["config"]["cache_uri"] == "s3://b/own/"
    assert doc["state"]["parca_v1"]["config"]["cache_uri"] == "s3://b/global/"


def test_no_cache_uri_still_runs_parca(core) -> None:
    """The default path must be untouched -- this is additive."""
    block = _parca_block(core)
    assert "v2ecoli-parca" in block
    assert "aws s3 cp" not in block


# --- N seeds in ONE variant: the shape every earlier test missed -------------


def test_lineages_in_one_variant_emit_DISTINCT_output_names(core) -> None:
    """The gap that let the collision ship. Every earlier multi-* test used
    multiple VARIANTS with one seed each, so no two tasks ever shared an output
    name. Three seeds inside one variant is the first shape that collides:

        Process `analysis` input file name collision --
          There are multiple input files for each of the following file names: sweep

    Two failures, one cause -- `path sweep_v0` stages the gather's inputs under
    their OWN names, so N directories called `sweep` cannot be staged; and N
    concurrent publishDir copies to one destination name race (measured: only 2 of
    3 seeds published)."""
    doc = build_workflow_nf(n_seeds=3, n_generations=1)
    inner = doc["state"]["runs_v0"]["config"]["state"]
    out_dirs = [inner[k]["config"]["out_dir"] for k in inner if k.startswith("lineage")]
    assert len(out_dirs) == 3
    assert len(set(out_dirs)) == 3, f"names must differ across tasks: {out_dirs}"
    assert set(out_dirs) == {"sweep_v0_s0", "sweep_v0_s1", "sweep_v0_s2"}


def test_the_output_declaration_is_a_pattern_not_a_fixed_name(core) -> None:
    """`nextflow_port_decls` is read via `_class_annotation` ->
    `getattr(type(instance), ...)`, i.e. off the CLASS, so it cannot vary per
    instance -- a property would return the descriptor. A glob is what makes the
    per-lineage `out_dir` expressible at all."""
    rendered = _render(core, build_workflow_nf(n_seeds=2, n_generations=1))
    block = rendered.split("process lineage_v0_s0 {", 1)[1].split("\n}", 1)[0]
    assert 'path "sweep_*"' in block
    assert 'path "sweep"' not in block, "a fixed name collides across tasks"


def test_distinct_names_survive_across_variants_too(core) -> None:
    """Run 4 is 84 variants x N seeds; the name must be unique over BOTH axes."""
    doc = build_workflow_nf(
        n_seeds=2, variants=[{"variant_name": "a"}, {"variant_name": "b"}]
    )
    names = [
        doc["state"][f"runs_v{v}"]["config"]["state"][k]["config"]["out_dir"]
        for v in (0, 1)
        for k in doc["state"][f"runs_v{v}"]["config"]["state"]
        if k.startswith("lineage")
    ]
    assert len(names) == 4 and len(set(names)) == 4, names


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow not installed")
def test_a_multiseed_single_variant_render_compiles(core, tmp_path) -> None:
    """The render-shape assertions above cannot tell a valid script from one
    Nextflow rejects -- which is how this and process-bigraph#205 both shipped."""
    main_nf = tmp_path / "main.nf"
    main_nf.write_text(
        _render(
            core, build_workflow_nf(n_seeds=3, n_generations=1, include_analysis=True)
        )
    )
    r = subprocess.run(
        ["nextflow", "run", str(main_nf), "-preview"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr


# --- the structural check for a bug class that has now escaped four times -----
#
# `analysis_options` (#730), `independent_founders` (#731) and `cache_uri` (#732)
# were three separate fixes for ONE bug: a value LineageStep forwards, that the
# generator never declares, so a dispatch asking for it is silently ignored.
# `build_workflow_nf` ends in `**_ignored: Any`, which is what makes it silent --
# an undeclared parameter is swallowed, not rejected.
#
# Vigilance has now failed four times, so this pins the classification instead.
# Every `_FORWARDED` key must be in exactly one bucket, and adding a new one
# forces the choice rather than defaulting to "silently unreachable".

# Set per lineage node by the generator itself -- correctly NOT campaign knobs.
_DERIVED_BY_GENERATOR = {
    "seed",
    "lineage_seed",
    "generations",
    "out_dir",
    "variant_index",
    "variant_name",
    "founder_sim_data",
}

# Reachable per variant through `injected_processes`, via LineageProcess's
# `_feature_flag` (`_injected.get(k, self.config.get(k, default))`).
_REACHABLE_VIA_INJECTED_PROCESSES = {
    "features",
    "ppgpp_regulation",
    "trna_attenuation",
    "supercoiling",
    "mass_conservation",
    "exchange_fluxes",
    "exchange_flux_basis",
    "transcript_initiation_mode",
    "polypeptide_initiation_mode",
}

# Read straight off `self.config` with no escape hatch, and not declared: a
# campaign CANNOT set these. Documented rather than asserted away -- `media` and
# `time_step` are the ones that bite (CD2 Run 4's minimal-vs-tryptophan split is
# exactly a media choice), and `emit_paths` is the undeclared-emission hole
# behind viva-api#475's global_time-only parquet.
_KNOWN_UNREACHABLE = {
    "time_step",
    "media",
    "emitter",
    "emitter_arg",
    "single_daughters",
    "checkpoint_dir",
    "emit_paths",
}


def _declared_parameters() -> set[str]:
    import inspect

    from v2ecoli.workflow.lineage_step import LineageStep  # noqa: F401

    sig = inspect.signature(build_workflow_nf)
    return {
        n
        for n, p in sig.parameters.items()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def test_every_forwarded_key_is_classified() -> None:
    """Each `_FORWARDED` key is a declared parameter, derived by the generator,
    reachable via injected_processes, or a known gap -- never unclassified.

    A new forwarded key landing in none of these buckets is the exact shape of
    #730/#731/#732: expressible at every hop but the one that matters."""
    from v2ecoli.workflow.lineage_step import _FORWARDED

    classified = (
        _declared_parameters()
        | _DERIVED_BY_GENERATOR
        | _REACHABLE_VIA_INJECTED_PROCESSES
        | _KNOWN_UNREACHABLE
    )
    unclassified = set(_FORWARDED) - classified
    assert not unclassified, (
        f"{sorted(unclassified)} are forwarded by LineageStep but reach no hop of "
        "build_workflow_nf. Declare them as generator parameters, or add them to "
        "_KNOWN_UNREACHABLE with a reason. `**_ignored` will NOT raise on them."
    )


def test_the_known_gaps_have_not_silently_grown() -> None:
    """_KNOWN_UNREACHABLE is a debt list, not a dumping ground. Shrinking it is
    the goal; growing it should require editing this test deliberately."""
    from v2ecoli.workflow.lineage_step import _FORWARDED

    still_unreachable = {
        k
        for k in _KNOWN_UNREACHABLE
        if k not in _declared_parameters()
        and k not in _DERIVED_BY_GENERATOR
        and k not in _REACHABLE_VIA_INJECTED_PROCESSES
    }
    assert still_unreachable == _KNOWN_UNREACHABLE, (
        "these became reachable -- drop them from _KNOWN_UNREACHABLE: "
        f"{sorted(_KNOWN_UNREACHABLE - still_unreachable)}"
    )
    assert set(_FORWARDED) >= _KNOWN_UNREACHABLE, (
        "a key left _FORWARDED entirely; the gap list is stale: "
        f"{sorted(_KNOWN_UNREACHABLE - set(_FORWARDED))}"
    )
