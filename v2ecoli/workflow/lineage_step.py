"""``LineageStep`` — a whole lineage as ONE atomic task.

Why a Step wrapping the Process, rather than the Process itself:

``run_step`` calls ``instance.invoke(state)`` **exactly once**, while
``LineageProcess.update()`` advances **one generation per call**. Registering the
Process directly as a task node would therefore run generation 0, return, and
report success — a silent truncation, and the same class of defect as every other
one this pipeline has produced. This wrapper keeps the generation loop exactly
where it is and drives it to completion inside a single invocation.

What that buys, beyond correctness:

* ``cache_dir`` becomes a real **input wire** rather than config, so a DAG engine
  can stage the ParCa bundle into the task as a ``path``. ``LineageProcess.inputs()``
  is ``{}``, which leaves the producer→consumer edge invisible.
* One result port. ``LineageProcess`` emits ``summary`` + ``complete``; ``complete``
  is loop bookkeeping, not a result, and does not belong in a task's interface.
* No ``ray:`` address is involved, so a renderer sees an ordinary class. Ray is the
  addressing layer for running N lineages concurrently; a single lineage is strictly
  sequential (each generation depends on the previous daughter), so nothing about
  one lineage needs it.

**Separability.** Everything that distinguishes one lineage from another —  seed,
variant, and any process swap — lives in *this task's own config*. A task reads its
own file and nothing else: no shared store, no ordering constraint between siblings,
no barrier before the gather. An N×M sweep is expressed by which configs exist.
"""

from __future__ import annotations

import os
from typing import Any

from process_bigraph import Step

# Keys forwarded verbatim to LineageProcess when present. Deliberately a
# whitelist: a typo'd key should fail loudly here rather than be silently
# dropped into a Process that ignores unknown config.
_FORWARDED = (
    "seed",
    "lineage_seed",
    "generations",
    "max_duration_per_gen",
    "time_step",
    "media",
    "emitter",
    "emitter_arg",
    "experiment_id",
    "out_dir",
    "variant_index",
    "variant_name",
    "single_daughters",
    "checkpoint_dir",
    "emit_paths",
    "features",
    "ppgpp_regulation",
    "trna_attenuation",
    # Biology the ray: and single-cell paths already support. Omitting these
    # would drop them SILENTLY -- and exchange_fluxes/exchange_flux_basis is the
    # violacein-exchange KPI readout CD2 Run 2 reads, on the strain sweeps this
    # node exists to run. (@eagmon, review of #694.)
    "exchange_fluxes",
    "exchange_flux_basis",
    "supercoiling",
    "mass_conservation",
    "transcript_initiation_mode",
    "polypeptide_initiation_mode",
    # Opt-in independent founders (v2ecoli#712). Without these forwarded, a
    # Nextflow campaign cannot reach the capability at all: one ParCa per variant
    # feeds M lineages, so every seed loads the SAME cached initial_state.
    "independent_founders",
    "founder_sim_data",
)

# Forwarded ONLY when non-empty. "no swap requested" and "swap requested, empty"
# are different downstream: an empty injected_processes block is a config-less
# swap target, which v2ecoli#682 now (correctly) fails loud on, and an empty
# override block is the shape that replaced a config's flat fields in
# viva-api#401. Passing no key at all is the honest encoding of "nothing asked".
_FORWARDED_IF_SET = ("injected_processes", "config_overrides")


class LineageStep(Step):
    """Run a complete lineage — every generation — as one atomic invocation."""

    config_schema = {
        "seed": {"_type": "integer", "_default": 0},
        "lineage_seed": {"_type": "integer", "_default": 0},
        "generations": {"_type": "integer", "_default": 1},
        "max_duration_per_gen": {"_type": "float", "_default": 3600.0},
        "time_step": {"_type": "float", "_default": 1.0},
        "media": {"_type": "string", "_default": "minimal"},
        "emitter": {"_type": "string", "_default": "parquet"},
        "emitter_arg": {"_default": {}},
        "experiment_id": {"_type": "string", "_default": "default"},
        "out_dir": {"_type": "string", "_default": "out/workflow"},
        "cache_dir": {"_type": "string", "_default": ""},
        "variant_index": {"_type": "integer", "_default": 0},
        "variant_name": {"_type": "string", "_default": "baseline"},
        "single_daughters": {"_type": "boolean", "_default": True},
        "checkpoint_dir": {"_type": "string", "_default": ""},
        # `quote` for the same reason LineageProcess uses it: these are
        # heterogeneous config-shaped blocks whose nested lists are mangled if
        # bigraph-schema infers a schema from a `{}` default.
        "injected_processes": {"_type": "quote", "_default": {}},
        "config_overrides": {"_type": "quote", "_default": {}},
        "emit_paths": {"_type": "quote", "_default": []},
        "features": {"_default": []},
        "ppgpp_regulation": {"_type": "boolean", "_default": True},
        "trna_attenuation": {"_type": "boolean", "_default": False},
        "exchange_fluxes": {"_default": {}},
        "exchange_flux_basis": {"_type": "string", "_default": ""},
        "supercoiling": {"_type": "boolean", "_default": False},
        "mass_conservation": {"_type": "boolean", "_default": False},
        "transcript_initiation_mode": {"_type": "string", "_default": "discrete"},
        "polypeptide_initiation_mode": {"_type": "string", "_default": "discrete"},
        # A task that emits nothing must fail rather than report success
        # (plan-nextflow-dispatch.md, Phase 2). run_composite ships no such
        # guard, so adopting a DAG engine would otherwise drop the protection
        # viva-api#395 added on the Ray path.
        "require_output": {"_type": "boolean", "_default": True},
    }

    # How the renderer declares this task's ports in the emitted process block.
    # A GLOB, not a fixed name. Every lineage used to emit a directory named
    # literally `sweep` -- unique within a task, identical across tasks -- which
    # collides the moment more than one lineage is gathered or published:
    #
    #   Process `analysis` input file name collision --
    #     There are multiple input files for each of the following file names: sweep
    #   Failed to publish file: .../work/14/27c67.../sweep; to: .../vecoli-output/.../sweep [copy]
    #
    # Both halves of one cause. `path sweep_v0` stages N inputs under their own
    # names, so N directories called `sweep` cannot be staged; and N concurrent
    # publishDir copies to the same destination name race (measured: only 2 of 3
    # seeds published). The per-lineage name lives in `out_dir` config, which
    # workflow_nf sets to `sweep_v{vi}_s{seed}` -- this declaration cannot itself
    # vary, because `_class_annotation` reads it off the CLASS
    # (`getattr(type(instance), ...)`), so a pattern is what makes per-instance
    # names expressible at all.
    nextflow_port_decls = {
        "cache_dir": "path cache_dir",
        "sweep_dir": 'path "sweep_*"',
    }
    # A whole lineage is the long-running task on this path -- hours, not
    # minutes -- so it is the one that most needs the profile's `time` and
    # memory. Without a label, `withLabel: lineage` binds to nothing.
    # `publishDir` or the science is unreachable. Without it a campaign that exits
    # 0 leaves its output in the Nextflow WORK dir under a content hash --
    # measured on a real run: 633 MB across 43 objects at
    # `work/1d/8bdc05.../sweep/`, while the results prefix held 78 KB of render
    # artifacts and no data at all.
    #
    # A closure, not a literal: the destination is the RUN's own prefix and is
    # only known at dispatch. `params.publish_dir` is supplied by the caller
    # (viva-api passes the run's results URI); the `?:` keeps a bare
    # `nextflow run` working locally instead of failing on a missing param.
    #
    # Each lineage publishes its OWN `sweep_v{vi}_s{seed}` into the shared target.
    # An earlier version of this comment claimed they could all publish a
    # directory named `sweep` because "the copies interleave rather than collide
    # -- identity lives in the partitions, not in the directory name". That was
    # reasoning about the directory's CONTENTS while Nextflow reasons about its
    # NAME, and it held only while no two tasks shared a destination (1 seed per
    # variant). The first 3-seed campaign failed on it.
    #
    # So the published tree is N siblings rather than one merged directory. Every
    # consumer reaches the data through the recursive
    # `**/history/experiment_id=*/**/*.pq` glob in v2ecoli/library/sweep_io.py,
    # so N siblings and one merged tree are equivalent to readers.
    nextflow_directives = {
        "label": "lineage",
        "publishDir": '{ params.publish_dir ?: "results" }, mode: "copy", overwrite: true',
    }

    def inputs(self) -> dict[str, Any]:
        # _is_file makes ParCa -> lineage a staged edge rather than a shared path.
        return {"cache_dir": {"_type": "string", "_is_file": True}}

    def outputs(self) -> dict[str, Any]:
        return {"sweep_dir": {"_type": "string", "_is_file": True}}

    # --- helpers (overridden in unit tests to avoid running real biology) ---

    def _lineage_config(self, cache_dir: str) -> dict[str, Any]:
        config: dict[str, Any] = {"cache_dir": cache_dir}
        for key in _FORWARDED:
            if key in self.config:
                config[key] = self.config[key]
        for key in _FORWARDED_IF_SET:
            value = self.config.get(key)
            if value:
                config[key] = value
        return config

    def _run_lineage(self, config: dict[str, Any], interval: float) -> None:
        from process_bigraph import Composite

        from v2ecoli.core import build_core
        from v2ecoli.workflow.lineage import LineageProcess

        core = build_core()
        core.register_link("LineageProcess", LineageProcess)
        # `interval` is NOT optional. A process node without one defaults to 1.0,
        # so Composite.run(total) would call update() once per SIMULATED SECOND --
        # 7200 framework ticks for generations=2 x 3600s, each paying view/project/
        # apply, instead of one tick per generation. LineageProcess self-limits at
        # `generations`, so the result stays correct; the cost is pure overhead on
        # the production dispatch path. batch_lineage_ray sets the same value for
        # the same reason (see its docstring). (@eagmon, review of #694.)
        composite = Composite(
            {
                "state": {
                    "lineage": {
                        "_type": "process",
                        "address": "local:LineageProcess",
                        "config": config,
                        "interval": float(
                            self.config.get("max_duration_per_gen", 3600.0)
                        ),
                        "inputs": {},
                        "outputs": {"summary": ["summary"], "complete": ["complete"]},
                    }
                }
            },
            core=core,
        )
        # Composite.run takes TOTAL SIMULATED TIME, not a tick count. The
        # generation loop lives inside LineageProcess; this is simply a bound
        # generous enough for every generation to run.
        composite.run(interval)

    def _has_output(self, out_dir: str) -> bool:
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                if (
                    name.endswith((".pq", ".parquet"))
                    and os.path.getsize(os.path.join(root, name)) > 0
                ):
                    return True
                if name in (".zgroup", ".zarray", "zarr.json", ".zattrs"):
                    return True
        return False

    # --- the task itself --------------------------------------------------

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        cache_dir = state.get("cache_dir") or self.config.get("cache_dir") or ""
        if not cache_dir:
            raise ValueError(
                "LineageStep requires a cache_dir: it is the ParCa bundle this lineage "
                "reads. It arrives as a staged input wire; a task with neither the input "
                "nor a configured fallback has nothing to simulate."
            )

        generations = int(self.config.get("generations", 1))
        per_gen = float(self.config.get("max_duration_per_gen", 3600.0))
        interval = generations * per_gen

        self._run_lineage(self._lineage_config(str(cache_dir)), interval)

        out_dir = str(self.config.get("out_dir") or "")
        if (
            self.config.get("require_output", True)
            and out_dir
            and not self._has_output(out_dir)
        ):
            raise SystemExit(
                f"LineageStep produced no emitted output under {out_dir}. A lineage that "
                f"emits nothing is a failed task, not a successful one -- refusing to "
                f"report success (plan-nextflow-dispatch.md go/no-go 6)."
            )

        return {"sweep_dir": out_dir}
