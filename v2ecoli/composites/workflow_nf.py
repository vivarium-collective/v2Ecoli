"""``workflow_nf`` — the campaign shape, as a DAG a task scheduler can render.

The general campaign is a two-level scatter with a gather:

    ParCa(variant v)  ──►  cache_v
                             └─► for each seed m:  LineageStep(v, m)  ──┐
    ... for each variant ...                                            ├─►  analysis
                                                                        ┘

**This document is rendered, never ``run()`` in-process.** Its ParCa and analysis
nodes carry no simulation logic at all — only ports and a ``nextflow_script()``.

Two deliberate choices, both recorded because the obvious alternative is wrong:

* **ParCa uses the CLI, not ``--build``.** v2ecoli's registered ``parca`` generator
  is structural: it carries ``raw_data=None`` and does not set ``run_steps_on_init``,
  so ``Composite(doc).run(n)`` would advance ``global_time`` and **run nothing,
  exiting 0**. Emitting ``v2ecoli-parca`` as a script keeps the failure honest.

* **Analysis is not wrapped.** ``v2ecoli-analyze`` is already atomic and
  ``s3://``-capable; wrapping it would duplicate ``run_analyses``' own fan-out.

**Why each variant's lineages are NESTED rather than flat siblings.** A port maps to ONE
store path, so the obvious spelling of an N×M→1 fan-in — one ``sweep_dirs`` port wired to a
*list* of stores — cannot be constructed at all: it raises ``TypeError: unhashable type:
'list'`` inside ``Composite.__init__`` → ``core.realize`` → ``resolve``, **before the
renderer runs**. Same error process-bigraph#201 recorded for ``Mix``, same cause.

Nesting is what expresses it. Each variant's M lineages live in a sub-Composite, which #201
renders as a DSL2 sub-workflow::

    workflow runs_v0 {
        take: cache
        main:
        ch_sweep_v0_s0 = lineage_v0_s0(cache, ...)
        ch_sweep_v0_s1 = lineage_v0_s1(cache, ...)
        _merged = ch_sweep_s0
        _merged = _merged.mix(ch_sweep_s1)
        emit: _merged.collect()
    }

so M channels arrive at the parent as ONE. The analysis then takes one named port per
variant. Measured at Run 4 scale — 84 variants × 4 seeds = **336 lineages**: 421 process
blocks, 84 sub-workflows, rendered in 2.5 s, with the parent call at 84 arguments (under
Java's 255-parameter limit) and every mix binary.

The per-variant ParCa node is what makes a *strain* sweep real rather than nominal:
``new_genes`` / ``bundle_overrides`` reach **ParCa**, so each variant gets its own
cache. Threading them to the lineage instead — which is what today's
``lineage_ray_batch`` façade does by collapsing ``variants`` into
``config_overrides`` — gives one shared cache for the whole sweep, i.e. N runs of
the same genotype wearing different labels.
"""

from __future__ import annotations

import os
import shlex
from typing import Any

from process_bigraph import Step
from viva_superpowers.composite_generator import composite_generator
from v2ecoli.workflow.meta_composite import register_workflow_processes

# Mirrors viva-api's production _parca_command chain, which is the proven
# invocation. The order is load-bearing: v2ecoli-parca emits only the raw
# parca_state.pkl, so build_cache.py must hydrate it into the loadable bundle,
# and build_cache.py reads a GZIPPED fixture.
#
# --new-genes / --bundle-overrides go to v2ecoli-parca ONLY. build_cache.py's CLI
# has neither flag (confirmed against a real crash, viva-api#410); its own
# save_sim_input already writes a correct, strain-specific cache_version.json
# because ParCa received the flags one command earlier.
# The repo root, for artefacts that ship in the CHECKOUT rather than the wheel.
# `scripts/build_cache.py` is one: v2ecoli installs to site-packages and
# `scripts/` does not go with it, so the path has to be absolute. The default
# matches this image's WORKDIR; a deployment that puts the checkout elsewhere
# overrides V2E_ROOT.
_DEFAULT_ROOT = "/app/v2ecoli"


def _repo_root() -> str:
    """Absolute path to the checkout, resolved when the workflow is RENDERED.

    Deliberately not a shell variable in the emitted script: a `script:` block is
    a Groovy string, so `${V2E_ROOT:-/app/v2ecoli}` is interpolated by GROOVY, not
    bash -- it fails at run time with
    `No signature of method: java.lang.String.negative()`, which names nothing
    useful. Escaping it (`\${...}`) works but puts a Groovy-quoting subtlety in
    every Step author's hands, which is the mistake process-bigraph#205 exists to
    stop making.

    Baking the value in is correct here because the head that renders and the
    task that runs are the SAME IMAGE; if that ever stops being true, this is the
    line that has to change.
    """
    return os.environ.get("V2E_ROOT", _DEFAULT_ROOT)


# NOT cwd-relative, and NOT prefixed with `cd $V2E_ROOT`. A Nextflow task runs in
# its own work dir and its declared outputs (`path "cache"`) are resolved
# relative to that dir -- so cd-ing away would write the cache somewhere Nextflow
# never looks, and the task would fail with "Missing output file(s)" having done
# all the work. The chain/Ray path CAN cd because nothing reclaims its outputs by
# relative path (viva-api `_parca_command`); this one cannot.
_PARCA_CHAIN = (
    "v2ecoli-parca --mode {mode} --cpus {cpus} -o {simdata} --cache-dir {cache}{strain_flags}"
    " && gzip -f -k {simdata}/parca_state.pkl"
    # build_cache.py imports pbg_v2ecoli, whose apply_upstream_patches() calls
    # find_workspace_root() -- which walks up from CWD for a workspace.yaml. A
    # Nextflow task's cwd is its work dir, which has none, so the import dies
    # with FileNotFoundError AFTER ParCa has already done 2.5 minutes of work.
    # The workspace root must be the checkout (models/ lives there and is not
    # shipped in site-packages), so this ONE command runs from there.
    #
    # `\$WD` is a SHELL variable, escaped so Groovy emits a literal `$`. The
    # in/out paths are made absolute from it, so cd-ing does not move the
    # declared output `cache` out of the work dir where Nextflow looks for it.
    ' && WD="\$PWD" && cd "{root}"'
    " && python scripts/build_cache.py"
    ' --fixture "\$WD/{simdata}/parca_state.pkl.gz" --cache "\$WD/{cache}"'
    ' && cd "\$WD"'
    " && cp {simdata}/parca_state.pkl.gz {cache}/parca_state.pkl.gz"
)


class ParcaTaskStep(Step):
    """One ParCa build, as a task. Carries no simulation logic — ports + a script."""

    config_schema = {
        "variant_index": {"_type": "integer", "_default": 0},
        "variant_name": {"_type": "string", "_default": "baseline"},
        "new_genes": {"_type": "string", "_default": ""},
        "bundle_overrides": {"_type": "string", "_default": ""},
        "mode": {"_type": "string", "_default": "fast"},
        "cpus": {"_type": "integer", "_default": 8},
        "cache_dir": {"_type": "string", "_default": "out/cache"},
        "simdata_dir": {"_type": "string", "_default": "out/parca"},
    }

    nextflow_port_decls = {"cache_dir": 'path "cache"'}
    # Without a label, `withLabel: parca { cpus/memory/time }` in the executor
    # profile matches NOTHING -- every task silently takes the queue defaults,
    # and in particular gets NO `time`, which is the only bound on a runaway
    # task (plan-nextflow-dispatch §11.1).
    # Deliberately NOT published: the cache is an INTERMEDIATE, ~262 MB, and
    # staging it task-to-task is exactly what the work dir is for. Publishing it
    # would double the storage of every campaign to no one's benefit.
    nextflow_directives = {"label": "parca"}

    def inputs(self) -> dict[str, Any]:
        return {}

    def outputs(self) -> dict[str, Any]:
        return {"cache_dir": {"_type": "string", "_is_file": True}}

    def nextflow_script(self) -> str:
        flags = ""
        new_genes = str(self.config.get("new_genes") or "")
        overrides = str(self.config.get("bundle_overrides") or "")
        # "off" is v2ecoli-parca's own default for --new-genes; passing it
        # explicitly and omitting it are the same build, so omit.
        if new_genes and new_genes != "off":
            flags += f" --new-genes {shlex.quote(new_genes)}"
        if overrides:
            flags += f" --bundle-overrides {shlex.quote(overrides)}"
        return _PARCA_CHAIN.format(
            root=_repo_root(),
            mode=self.config.get("mode", "fast"),
            cpus=int(self.config.get("cpus", 8)),
            simdata=self.config.get("simdata_dir", "out/parca"),
            cache=self.config.get("cache_dir", "out/cache"),
            strain_flags=flags,
        )

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "ParcaTaskStep is a task declaration, not an in-process step: it is rendered "
            "and its nextflow_script() is what runs. Calling update() would silently "
            "produce no cache while reporting success."
        )


class AnalysisTaskStep(Step):
    """The gather: one analysis over every sweep the campaign produced."""

    config_schema = {
        "experiment_id": {"_type": "string", "_default": "default"},
        # Task-local output dir, matching the declared `path "analysis"`. The CLI
        # reads it from the staged config (not a flag) -- see analysis_runner.main.
        "out_dir": {"_type": "string", "_default": "analysis"},
        # The analyses to run, in the v2ecoli-analyze config shape
        # ({scale: {analysis_name: params}}); threaded from build_workflow_nf.
        "analysis_options": {"_type": "quote", "_default": {}},
        # One input port per variant. A port maps to ONE store path, so a single
        # port wired to a LIST of stores cannot be constructed at all
        # (TypeError: unhashable type: 'list', raised in core.realize before the
        # renderer runs). N named ports, each fed by one variant sub-workflow's
        # already-collected channel, is the shape the document model supports.
        "variant_indices": {"_type": "quote", "_default": [0]},
    }

    # Only the OUTPUT needs an override. The per-variant inputs are declared with
    # `_is_file: True`, which the renderer already turns into `path <name>` — and
    # this override is read off the CLASS (`_class_annotation` → `getattr(type(...))`),
    # so it cannot depend on config anyway.
    nextflow_port_decls = {"report": 'path "analysis"'}
    # Published for the same reason as LineageStep's sweep: the gather's report is
    # the deliverable, and an unpublished one is as unreachable as no report.
    nextflow_directives = {
        "label": "analysis",
        "publishDir": '{ params.publish_dir ?: "results" }, mode: "copy", overwrite: true',
    }

    def _variants(self) -> list[int]:
        return [int(i) for i in (self.config.get("variant_indices") or [0])]

    def inputs(self) -> dict[str, Any]:
        return {
            f"sweep_v{i}": {"_type": "string", "_is_file": True, "_cardinality": "many"}
            for i in self._variants()
        }

    def outputs(self) -> dict[str, Any]:
        return {"report": {"_type": "string", "_is_file": True}}

    def nextflow_script(self) -> str:
        # The v2ecoli-analyze CLI is `<sweep_dir> [--config CONFIG]`. This used to
        # emit --experiment-id/--out-dir, which the CLI rejects with exit 2 (#722).
        # The per-variant sweeps are staged into the task work dir, so sweep_dir is
        # "." -- history_files globs the hive tree recursively from there. The
        # analyses to run and the task-local out_dir ride in the staged node config
        # (analysis.config.json), so nothing else goes on the command line.
        return "v2ecoli-analyze . --config analysis.config.json"

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "AnalysisTaskStep is a task declaration, not an in-process step; see "
            "ParcaTaskStep.update."
        )


def _variant_specs(variants: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize the variant list. A campaign with no variants is ONE baseline
    variant, not zero -- zero would render an empty workflow that exits 0."""
    if not variants:
        return [{"variant_index": 0, "variant_name": "baseline"}]
    out = []
    for i, v in enumerate(variants):
        spec = dict(v)
        spec.setdefault("variant_index", i)
        spec.setdefault("variant_name", f"variant_{i}")
        out.append(spec)
    return out


@composite_generator(
    name="workflow_nf",
    # Without this the document cannot even be CONSTRUCTED: every node it emits
    # is a `local:` address this registers -- LineageStep, ParcaTaskStep,
    # AnalysisTaskStep -- and `composite`, the nested-Composite link that #201
    # renders as a sub-workflow. `build_core()` alone registers none of them, so
    # resolving through the generator (which is how run_pbg and render_nf do it)
    # failed with `no link found at address: {'protocol': 'local', 'data':
    # 'composite'}`.
    #
    # The tests below did not catch it because their fixture called
    # register_workflow_processes BY HAND, which is exactly the gap
    # `core_extensions` exists to close -- see lineage_ray_batch, which declares
    # `core_extensions=[register_ray_lineage]` for the same reason.
    core_extensions=[register_workflow_processes],
    description=(
        "The campaign DAG for task-granularity dispatch: one ParCa per variant, each feeding "
        "that variant's M LineageStep tasks, all N x M gathering into one analysis. Rendered, "
        "never run in-process -- the ParCa and analysis nodes are script declarations."
    ),
    parameters={
        "n_seeds": {
            "type": "integer",
            "default": 2,
            "description": "Seed-lineages per variant.",
        },
        "n_generations": {
            "type": "integer",
            "default": 1,
            "description": "Generations per lineage.",
        },
        "base_seed": {
            "type": "integer",
            "default": 0,
            "description": "First seed; seeds are contiguous per variant.",
        },
        "variants": {
            "type": "array",
            "default": None,
            "description": (
                "Per-variant strain inputs, each a dict that may carry variant_name, "
                "new_genes and bundle_overrides. These reach PARCA, giving each variant "
                "its own cache. None means a single baseline variant."
            ),
        },
        "experiment_id": {"type": "string", "default": "workflow_nf"},
        "out_dir": {"type": "string", "default": "out/workflow"},
        "max_duration_per_gen": {"type": "number", "default": 3600.0},
        "parca_mode": {"type": "string", "default": "fast"},
        "parca_cpus": {"type": "integer", "default": 8},
        "independent_founders": {
            "type": "boolean",
            "default": False,
            "description": (
                "Re-draw a founder cell per lineage_seed instead of every seed loading the "
                "ONE cached initial_state. Without it an M-seed campaign is not M replicates: "
                "_load_cache_bundle_cached is memoised on cache_dir alone and returns the "
                "initial state by reference, so the spread reflects downstream stochasticity "
                "only, not cell-to-cell founder variability (v2ecoli#693). Off by default "
                "because it re-generates initial conditions per seed and is slower."
            ),
        },
        "analysis_options": {
            "type": "object",
            "default": None,
            "description": (
                "Analyses the gather runs, staged into analysis.config.json and read back "
                "via --config. DECLARED here and not only accepted as a kwarg: every "
                "dispatch goes through CompositeSpec.to_document(overrides=...), whose "
                "_merged_params raises KeyError on any override missing from this block. "
                "Undeclared, it was unreachable from a dispatch, so the gather always got "
                "{} -- 'nothing to run', no analysis/ directory, and a campaign that ran "
                "every lineage and then failed its last node."
            ),
        },
        "include_analysis": {
            "type": "boolean",
            "default": False,
            "description": (
                "Append the N x M -> 1 gather node. DEFAULT FALSE, and that is "
                "a real limitation rather than a preference: see the module "
                "docstring. A flat port wired to many stores cannot be "
                "constructed at all."
            ),
        },
    },
)
def build_workflow_nf(
    n_seeds: int = 2,
    n_generations: int = 1,
    base_seed: int = 0,
    variants: list[dict[str, Any]] | None = None,
    experiment_id: str = "workflow_nf",
    out_dir: str = "out/workflow",
    max_duration_per_gen: float = 3600.0,
    parca_mode: str = "fast",
    parca_cpus: int = 8,
    include_analysis: bool = False,
    analysis_options: dict[str, Any] | None = None,
    independent_founders: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    sweep_paths: list[list[str]] = []

    for spec in _variant_specs(variants):
        vi = int(spec["variant_index"])
        vname = str(spec["variant_name"])
        parca_node = f"parca_v{vi}"
        cache_store = f"cache_v{vi}"

        state[parca_node] = {
            "_type": "step",
            "address": "local:ParcaTaskStep",
            "config": {
                "variant_index": vi,
                "variant_name": vname,
                "new_genes": spec.get("new_genes", ""),
                "bundle_overrides": spec.get("bundle_overrides", ""),
                "mode": parca_mode,
                "cpus": parca_cpus,
                # TASK-LOCAL RELATIVE PATHS, matching nextflow_port_decls. Nextflow
                # captures `path "cache"` by NAME in the task's own work dir: a
                # config writing to out/cache_v0 would leave nothing called
                # `cache`, and the task fails with "Missing output file(s)".
                # Per-variant identity lives in the config and the emitted
                # partitioning, not in the directory name. (@eagmon, review of #694.)
                "cache_dir": "cache",
                "simdata_dir": "parca",
            },
            "inputs": {},
            "outputs": {"cache_dir": [cache_store]},
        }

        # Each variant's M lineages live in a NESTED composite, which #201 renders
        # as a Nextflow sub-workflow: `take: cache` / `emit: <chained mixes>.collect()`.
        # That is what turns M sibling channels into ONE collected channel the
        # analysis can consume -- flat siblings cannot express the fan-in.
        inner_state: dict[str, Any] = {"cache": ""}
        for m in range(int(n_seeds)):
            seed = int(base_seed) + m
            # Namespace by variant AND seed. Each variant's lineages live in a
            # nested composite, but render_composite emits nested steps as
            # top-level Nextflow processes named by their leaf, so a bare
            # `lineage_s{seed}` collides across variants ("Identifier lineage_s0
            # is already used") and a >=1-variant campaign will not compile.
            node = f"lineage_v{vi}_s{seed}"
            sweep_store = f"sweep_v{vi}_s{seed}"
            # Everything that distinguishes this lineage lives in THIS config,
            # which is staged as its own file. No sibling shares it.
            config: dict[str, Any] = {
                "seed": seed,
                "lineage_seed": seed,
                "generations": int(n_generations),
                "max_duration_per_gen": float(max_duration_per_gen),
                "experiment_id": experiment_id,
                # Task-local, matching `path "sweep"` -- see the ParCa note above.
                # Identity is preserved by the hive partitioning the emitters write
                # (experiment_id/variant/lineage_seed/generation), not by this name.
                "out_dir": "sweep",
                "variant_index": vi,
                "variant_name": vname,
            }
            if independent_founders:
                # TASK-LOCAL, like every other path in this config: `cache_dir` is
                # staged by Nextflow as `path "cache"` and the ParCa task writes
                # simData.cPickle inside it, so this resolves against the task's
                # own work dir rather than any repo layout.
                config["independent_founders"] = True
                config["founder_sim_data"] = "cache/simData.cPickle"
            # Omitted, not empty -- see LineageStep for why the distinction matters.
            if spec.get("injected_processes"):
                config["injected_processes"] = spec["injected_processes"]
            if spec.get("config_overrides"):
                config["config_overrides"] = spec["config_overrides"]

            inner_state[sweep_store] = ""
            inner_state[node] = {
                "_type": "step",
                "address": "local:LineageStep",
                "config": config,
                # inside the sub-composite, `cache` is the take: port
                "inputs": {"cache_dir": ["cache"]},
                "outputs": {"sweep_dir": [sweep_store]},
            }

        results_store = f"results_v{vi}"
        sweep_paths.append([results_store])
        state[f"runs_v{vi}"] = {
            "_type": "process",
            "address": "local:composite",
            "config": {"state": inner_state},
            "inputs": {"cache": [cache_store]},
            "outputs": {"results": [results_store]},
        }

    if include_analysis:
        indices = [int(s["variant_index"]) for s in _variant_specs(variants)]
        state["analysis"] = {
            "_type": "step",
            "address": "local:AnalysisTaskStep",
            "config": {
                "experiment_id": experiment_id,
                # Task-local, matching the declared `path "analysis"` output --
                # NOT f"{out_dir}/analysis", which the task's work dir has no way
                # to produce (the gather runs in an isolated Nextflow work dir).
                "out_dir": "analysis",
                "analysis_options": analysis_options or {},
                "variant_indices": indices,
            },
            # one named port per variant, each fed by that variant's sub-workflow
            "inputs": {f"sweep_v{i}": [f"results_v{i}"] for i in indices},
            "outputs": {"report": ["report"]},
        }
    return {"state": state}
