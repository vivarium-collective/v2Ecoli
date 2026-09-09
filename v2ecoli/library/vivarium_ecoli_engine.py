"""Single-process, faithful-by-construction vEcoli engine for the comparison harness.

Instead of decomposing upstream vEcoli into ~50 individual process-bigraph steps and
reimplementing vivarium's Engine semantics (partition/allocation, update reconciliation,
division) in process-bigraph — which introduced a class of faithfulness bugs (mass
explosion from a missing bulk reconcile; a division-handoff crash from `_add` dropping
live process instances) — this module runs the GENUINE vEcoli on its OWN vivarium Engine
inside one wrapper, advancing it incrementally with ``Engine.run_for``.

Faithful by construction: partition, allocation, and per-tick update reconciliation are
all handled by vivarium-core exactly as upstream intends. Multi-generation is single-
lineage (matching v2ecoli's ``run_multigen_xarray``): each generation runs a genuine
vEcoli Engine to its division point, vEcoli's own ``divide_cell`` splits the state, and
the followed daughter seeds the next generation's Engine. No process-bigraph division.

Fork-robust: the vEcoli fork is selected by ``V2E_VECOLI_DIR`` (or the ``fork_dir`` arg),
imported through the same compiled-``wholecell``-pinning shim the step-decomposed wrapper
uses (:func:`v2ecoli.library.vecoli_pbg_upstream._ensure_upstream`). Swapping forks is a
one-knob change: point ``V2E_VECOLI_DIR`` at a different checkout + its matching ParCa
``sim_data``.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from v2ecoli.library.vecoli_pbg_upstream import _ensure_upstream
from v2ecoli.library.upstream_division import _n_chromosomes, _inc_to_fg
from v2ecoli.library.division import divide_cell


# Path to a FULL vEcoli config file the wrapped ``EcoliSim`` should load natively
# (add_processes / process_configs / topology / spatial_environment_config /
# variants). None => build the default baseline sim (``swap_processes``/``flow``
# only). Set for the duration of a run by ``run_vivarium_ecoli_pbg_multigen``
# (which restores it in a ``finally``), so every per-generation ``build_vivarium_
# ecoli`` rebuild picks it up without threading a param through 3 call sites, and
# a whole-config run cannot leak into a later default run in the same process.
_ECOLISIM_CONFIG_FILE: str | None = None


def set_ecolisim_config_file(path: str | None) -> None:
    """Point the wrapped single-node ``EcoliSim`` at a full vEcoli config file to
    load natively (or clear it with ``None``). Faithful-by-construction: vEcoli's
    own composer applies ``add_processes``/``process_configs``/``topology``/
    ``spatial_environment_config``/``variants``/etc. exactly as upstream, so a
    config whose model content can't be expressed as ``swap_processes``/``flow``
    (a multi-agent / spatial-environment model) still runs correctly as one pbg
    node — for ANY fork selected by ``$V2E_VECOLI_DIR``."""
    global _ECOLISIM_CONFIG_FILE
    _ECOLISIM_CONFIG_FILE = path


def _select_variant_params(variants_config: dict, variant_index: int):
    """Resolve a 1-based ``variant_index`` against a config ``variants`` block.

    Mirrors the fork's ``runscripts.create_variants`` convention: ``parse_variants``
    returns ``param_dicts`` and the fork maps ``param_dicts[i]`` to variant ``i+1``,
    reserving index 0 for the unperturbed baseline. Returns ``(None, None)`` for the
    baseline, else ``(variant_name, params_dict)``. Delegates the grid expansion
    (op prod/zip/add, value/linspace/arange) entirely to the fork.
    """
    if variant_index <= 0:
        return None, None                       # baseline: strict no-op
    if not variants_config:
        # variant_index >= 1 but nothing to select from → fail loud rather than
        # silently running the unperturbed baseline (Finding #3 in spirit).
        raise ValueError(
            f"variant index {variant_index} requested but config has no "
            f"'variants' block to select from")
    if len(variants_config) != 1:
        raise ValueError(
            f"expected exactly one variant in config, got {sorted(variants_config)}")
    (name, cfg), = variants_config.items()
    from runscripts.create_variants import parse_variants  # fork-bound
    param_dicts = parse_variants(cfg)
    idx = variant_index - 1
    if idx >= len(param_dicts):
        raise IndexError(
            f"variant index {variant_index} out of range: {len(param_dicts)} "
            f"grid point(s) (valid 1..{len(param_dicts)})")
    return name, param_dicts[idx]


def _apply_config_variant(sim_data, variants_config: dict, variant_index: int):
    """Apply the selected config variant to ``sim_data`` via the fork's own
    ``ecoli.variants.<name>.apply_variant``. Returns ``(sim_data, meta|None)``.
    ``sim_data`` must already be a fresh (non-shared) object."""
    name, params = _select_variant_params(variants_config, variant_index)
    if name is None:
        return sim_data, None
    import importlib
    mod = importlib.import_module(f"ecoli.variants.{name}")  # fork-bound
    sim_data = mod.apply_variant(sim_data, params)
    return sim_data, {"variant_name": name, "variant_index": int(variant_index),
                      "params": params}


# ---------------------------------------------------------------------------
# Build a genuine vEcoli vivarium Engine (fork-parameterized)
# ---------------------------------------------------------------------------

@dataclass
class EngineHandle:
    """A built genuine-vEcoli vivarium Engine plus the metadata needed to drive it."""
    engine: object
    sim: object
    dry_mass_inc_dict: dict = field(default_factory=dict)
    condition: str = "basal"
    media_id: str = "minimal"
    time_step: float = 1.0


def _is_tcs_modified_molecules_skew(exc: BaseException) -> bool:
    """True iff ``exc`` is the genuine-vEcoli fixture↔upstream skew: upstream
    reading ``TwoComponentSystem.modified_molecules`` off a sim_data fixture
    that predates that attribute. Kept narrow (both tokens present) so unrelated
    AttributeErrors are not misattributed."""
    msg = str(exc)
    return "modified_molecules" in msg and "TwoComponentSystem" in msg


def _fixture_skew_message(sim_data_path: str, fork_dir: str | None) -> str:
    """Actionable message for the fixture↔upstream skew — names the cause AND
    the two fixes (pin the checkout to match the fixture, or rebuild the
    fixture) instead of the bare ``AttributeError`` the composer would raise."""
    checkout = fork_dir or os.environ.get("V2E_VECOLI_DIR", "<V2E_VECOLI_DIR>")
    return (
        f"genuine-vEcoli build failed: the sim_data fixture at "
        f"{sim_data_path!r} predates upstream vEcoli's `modified_proteins` "
        f"feature (added 2026-05-01), but the vEcoli checkout at {checkout!r} "
        f"is newer and requires `TwoComponentSystem.modified_molecules`. "
        f"Resolve the version skew either by (a) pinning the vEcoli checkout "
        f"to a commit before 2026-05-01 (e.g. 7bf03433) so its source matches "
        f"this fixture, or (b) rebuilding an upstream-compatible fixture on "
        f"Linux via `scripts/build_upstream_parca.py` and pointing "
        f"`sim_data_path` at it."
    )


def build_vivarium_ecoli(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    swap_processes: dict | None = None,
    flow: dict | None = None,
    fork_dir: str | None = None,
    initial_overlay: dict | None = None,
    variant: int = 0,
    agent_id: str = "0",
) -> EngineHandle:
    """Build the genuine upstream vEcoli composite and wrap its vivarium Engine.

    ``fork_dir`` (or ``$V2E_VECOLI_DIR``) selects the vEcoli checkout; ``sim_data_path``
    is its matching upstream ParCa ``simData.cPickle``. ``initial_overlay`` (a daughter's
    divided ``bulk``/``unique``/``environment``/``boundary``) seeds a non-founder
    generation; ``None`` builds a fresh founder. ``variant`` selects a 1-based grid
    point from the loaded config's ``variants`` block (0 = baseline, no-op); only
    applies when a full config file (``set_ecolisim_config_file``) is in effect.
    ``agent_id`` is the lineage phylogeny key ("0" -> "00" -> ...); its LENGTH is
    the generation index the fork applies staged shifts on — see below.
    """
    if fork_dir:
        os.environ["V2E_VECOLI_DIR"] = fork_dir
    up = _ensure_upstream()
    EcoliSim = up["EcoliSim"]
    from vivarium.core.engine import Engine

    _cfgfile = _ECOLISIM_CONFIG_FILE
    _argv = sys.argv
    _cwd = os.getcwd()
    _fork = fork_dir or os.environ.get("V2E_VECOLI_DIR")
    _saved_sers: dict = {}
    if _cfgfile:
        # Fork-bind vivarium's serializers before ``EcoliSim`` deserializes the
        # config's ``!ParameterSerializer[...]`` / ``!units[...]`` tags. A
        # Serializer binds ``param_store`` at its module's import time; the
        # INSTALLED vEcoli (registered into vivarium's GLOBAL serializer_registry
        # first) lacks fork-only params -> "No parameter found at path". Replace
        # each registered serializer with a FRESH instance imported from the
        # fork's ``ecoli.library.serialize`` (now first on sys.path after
        # ``_ensure_upstream``), and RESTORE the originals afterward so this does
        # not leak into a later default run in the same process (determinism).
        try:
            import importlib as _il
            from vivarium.core.registry import serializer_registry as _sreg, Serializer as _Ser
            _fs = _il.import_module("ecoli.library.serialize")
            for _nm, _obj in list(_sreg.registry.items()):
                _forkcls = getattr(_fs, type(_obj).__name__, None)
                if isinstance(_forkcls, type) and issubclass(_forkcls, _Ser) and _forkcls is not _Ser:
                    _saved_sers[_nm] = _obj
                    _sreg.registry[_nm] = _forkcls()
        except Exception as _se:  # noqa: BLE001 — fall back to registry serializers
            print(f"[build_vivarium_ecoli] serializer fork-bind skipped "
                  f"({type(_se).__name__}: {_se})")
    # A full config file loads via ``--config`` (EcoliSim resolves its
    # ``inherit_from`` chain relative to the fork's own configs dir, so run from
    # cwd = the fork checkout). Empty argv => the default baseline sim.
    sys.argv = ([sys.argv[0], "--config", _cfgfile] if _cfgfile else sys.argv[:1])
    try:
        if _cfgfile and _fork and os.path.isdir(_fork):
            os.chdir(_fork)
        sim = EcoliSim.from_cli()
    finally:
        sys.argv = _argv
        os.chdir(_cwd)
        if _saved_sers:
            from vivarium.core.registry import serializer_registry as _sreg2
            _sreg2.registry.update(_saved_sers)

    sim.config["condition"] = condition
    sim.config["seed"] = int(seed)
    sim.config["sim_data_path"] = sim_data_path
    sim.config["time_step"] = float(time_step)
    # ⛔⛔ THE GENERATION INDEX *IS* THE AGENT ID, and the fork reads it off THIS
    # key. ``LoadSimData`` computes ``generation = len(kwargs["agent_id"])`` and
    # applies every ``sim_data.internal_shift_dict`` entry whose
    # ``shift_gen <= generation`` — the ONLY mechanism by which a config's staged
    # induction (``induction_gen``) ever fires. ``agent_id`` is an existing key in
    # the fork's own ``configs/default.json`` and flows
    # ``Ecoli(config) -> LoadSimData(**config)``, so setting it here is the whole
    # hop; but leaving it at the default "0" pins EVERY generation at ``len == 1``
    # and a scheduled shift silently never happens — the run completes as the
    # un-induced baseline with nothing in the output to say so. The caller
    # advances it per generation via ``daughter_phylogeny_id``.
    # ⇒ Same species as the variant discard this module already refuses: a
    # declared perturbation dropped one layer below where anyone was looking.
    sim.config["agent_id"] = str(agent_id)
    # ⭐ SAY WHICH GENERATION THIS BUILD THINKS IT IS. Nothing printed it before,
    # and that silence is precisely why a lineage pinned at generation 1 could run
    # to completion, emit every observable, and be graded — with the un-shifted
    # baseline in every generation and no line anywhere to contradict it.
    print(f"[build_vivarium_ecoli] agent_id={str(agent_id)!r} "
          f"-> generation {len(str(agent_id))}")
    # Apply the CONDITION's media. genuine vEcoli's LoadSimData defaults
    # media_timeline to ((0,'minimal'),) and `condition` alone never updates it
    # (the "have to change both" footgun), so without this the runner runs every
    # condition on 'minimal' media — e.g. no_oxygen ran AEROBIC instead of
    # minimal_minus_oxygen (O2=0). Set the condition's nutrients as fixed_media
    # (flows through Ecoli(config) -> LoadSimData(**config)). Hand the loaded
    # sim_data in too so EcoliSim reuses it (skips a 2nd ~300MB load).
    # Preload sim_data ONCE (reuse it so EcoliSim skips a 2nd ~300MB load). The
    # preload itself is best-effort — on failure EcoliSim just loads
    # sim_data_path natively — EXCEPT when a variant is requested: we must not
    # silently run the unperturbed baseline, so a preload failure there is loud.
    # ⛔⛔ REFUSE AT THE POINT OF DISCARD. Every `apply_variant` gate below is
    # `_cfgfile and int(variant)`, so a caller that requests a variant WITHOUT the
    # whole-config route had it silently dropped: threaded all the way here, then
    # ignored, and the run completed as the unperturbed baseline. That is the
    # exact substitution the variant machinery exists to prevent, reintroduced one
    # layer down from where anyone was looking.
    # ⚠ CORRECTED, and read this before deleting the refusal as dead code. It
    # WAS the default for a config declaring `variants` alongside `swap_processes`
    # and nothing else — `_needs_native` auto-enabled only on
    # `add_processes`/`spatial_environment_config`, so such a config took the swap
    # route and the gate could never pass (measured: variant=1, whole_config=None,
    # apply_variant never called). The caller now auto-enables the native route
    # whenever a config DECLARES variants, so that path is no longer reachable
    # through `run_comparison_ensemble`.
    # ⇒ The refusal stays because it defends the INVARIANT, not that one caller:
    # `build_vivarium_ecoli` has ~6 call sites and is importable directly. A
    # requested variant that cannot be applied must never degrade to baseline,
    # whoever asks. It is also what makes the `variant` stamp in `metadata_base`
    # below true by construction.
    # ⇒ A requested variant that cannot be applied is an ERROR, not a default.
    if int(variant) and not _cfgfile:
        raise ValueError(
            f"variant {int(variant)} was requested but no whole-config was loaded, "
            f"so the config's `variants` block is unreachable and the variant "
            f"would be silently discarded — the run would be the unperturbed "
            f"baseline. Pass the driving config as `whole_config` (the caller's "
            f"`--vecoli-whole-config on` forces it), or request variant 0.")

    _sd_obj = None
    _variant_simdata_tmp = None   # temp pickle holding variant-mutated sim_data
    try:
        import pickle as _pickle
        with open(sim_data_path, "rb") as _sdf:
            _sd_obj = _pickle.load(_sdf)
    except Exception as _loaderr:  # noqa: BLE001 — sim_data reuse is best-effort
        if _cfgfile and int(variant):
            raise        # a requested variant cannot be applied → fail loud
        print(f"[build_vivarium_ecoli] sim_data preload skipped: "
              f"{type(_loaderr).__name__} {_loaderr}")

    if _sd_obj is not None:
        # Variant application must fail LOUD for ALL exception types — it is
        # deliberately NOT inside the best-effort media block below. A swallowed
        # ImportError (fork off path) / KeyError / AttributeError from
        # parse_variants/apply_variant would silently run the UNPERTURBED
        # baseline while the caller believes variant=k applied.
        if _cfgfile and int(variant):
            _variants_cfg = sim.config.get("variants") or {}
            _sd_obj, _vmeta = _apply_config_variant(_sd_obj, _variants_cfg, int(variant))
            if _vmeta:
                print(f"[build_vivarium_ecoli] applied config variant "
                      f"'{_vmeta['variant_name']}' #{variant}: {_vmeta['params']}")
                # PERSIST the variant-mutated sim_data and repoint sim_data_path.
                # The composer's ``LoadSimData`` ALWAYS reloads sim_data from
                # ``sim_data_path`` and IGNORES a handed-in ``sim.config['sim_data']``
                # object, so an in-memory-only variant (e.g. a ``field_timeline``
                # that ``LoadSimData.get_field_timeline_config`` reads off
                # ``external_state``) is silently discarded and the run reverts to
                # the unperturbed baseline. Writing the mutated sim_data to a temp
                # pickle and pointing sim_data_path at it is what actually delivers
                # the variant to the composer.
                import tempfile as _tf
                _vfd, _variant_simdata_tmp = _tf.mkstemp(
                    prefix="v2e_variant_simdata_", suffix=".cPickle")
                os.close(_vfd)
                with open(_variant_simdata_tmp, "wb") as _vf:
                    _pickle.dump(_sd_obj, _vf)
                sim.config["sim_data_path"] = _variant_simdata_tmp
        # The condition's nutrients (fixed_media) IS best-effort — a lookup miss
        # just means EcoliSim keeps its default media; never fatal.
        try:
            _nutrients = (_sd_obj.conditions.get(condition, {}) or {}).get("nutrients")
            if _nutrients:
                sim.config["fixed_media"] = _nutrients
        except Exception as _mediaerr:  # noqa: BLE001 — media reuse is best-effort
            print(f"[build_vivarium_ecoli] media-from-condition skipped: "
                  f"{type(_mediaerr).__name__} {_mediaerr}")
        # Hand the loaded (possibly variant-mutated) sim_data to EcoliSim so it
        # reuses THIS object (must always run when the preload succeeded).
        sim.config["sim_data"] = _sd_obj
    # Division is handled by THIS module's single-lineage loop (genuine divide_cell
    # between generations), not vivarium's in-Engine Division roundtrip — so each
    # generation is one clean genuine-vEcoli Engine.
    sim.config["divide"] = False
    sim.config["d_period"] = False
    sim.config["generations"] = None
    sim.config["progress_bar"] = False
    sim.config["emit_paths"] = []
    if exclude_processes:
        existing = list(sim.config.get("exclude_processes", []) or [])
        sim.config["exclude_processes"] = existing + list(exclude_processes)
    # Process swap (e.g. FBA metabolism -> MetabolismRedux): EcoliSim applies
    # ``swap_processes`` natively at build_ecoli() the same way vEcoli's own
    # configs/metabolism_redux.json does, and ``flow`` reorders the swapped
    # process's dependents. Merge so a caller-supplied swap composes with any
    # config default rather than clobbering it.
    if swap_processes:
        merged_swap = dict(sim.config.get("swap_processes", {}) or {})
        merged_swap.update(swap_processes)
        sim.config["swap_processes"] = merged_swap
    if flow:
        merged_flow = dict(sim.config.get("flow", {}) or {})
        merged_flow.update(flow)
        sim.config["flow"] = merged_flow

    # Capture the per-media expectedDryMassIncreaseDict (drives the division
    # threshold) the same way upstream_division does — via the composer's sim_data.
    dry_mass_inc_dict: dict = {}
    import ecoli.composites.ecoli_master as _em
    _orig_init = _em.Ecoli.__init__
    _captured: dict = {}

    def _capturing_init(self, config, *a, **k):
        _orig_init(self, config, *a, **k)
        _captured["composer"] = self

    _em.Ecoli.__init__ = _capturing_init
    try:
        sim.build_ecoli()
    except KeyError as _ke:
        # ⛔ A saved fork state is stored under an ``agents/<id>`` envelope, and
        # the composer indexes it by THIS config's agent_id
        # (``ecoli_master.py``: ``full_initial_state["agents"][agent_id]``).
        # Those files are written by the founder, so they carry "0" — and a
        # non-founder generation asks for "00" and gets a bare ``KeyError: '00'``
        # from inside the composer, with nothing naming the cause.
        # ⇒ Name it. Only when the missing key IS our agent id, so an unrelated
        # KeyError still propagates untouched.
        if str(agent_id) != "0" and str(_ke).strip("'\"") == str(agent_id):
            raise KeyError(
                f"the initial state is stored under an 'agents' envelope that has "
                f"no key {str(agent_id)!r}. This generation's agent_id is "
                f"{str(agent_id)!r} (generation {len(str(agent_id))}), but a saved "
                f"state file is written by the founder and carries '0'. Seed "
                f"non-founder generations with `initial_overlay` (what the lineage "
                f"drivers do) rather than `initial_state_file`, or re-key the file."
            ) from _ke
        raise
    except AttributeError as _ae:
        # ⛔ Fixture ↔ upstream skew. Upstream vEcoli's
        # ``get_monomer_counts_listener_config`` reads
        # ``sim_data.process.two_component_system.modified_molecules`` — a derived
        # attribute the reconstruction only began emitting with the
        # ``modified_proteins.tsv`` feature (upstream master, 2026-05-01). A
        # ``simData.cPickle`` built BEFORE that predates the attribute, so the
        # unpickled ``TwoComponentSystem`` lacks it and the composer dies deep
        # inside ``sim_data.py`` with a bare ``AttributeError`` that names neither
        # the cause nor the fix. ⇒ Name both. Only when it IS this exact skew, so
        # an unrelated AttributeError still propagates untouched.
        if _is_tcs_modified_molecules_skew(_ae):
            raise RuntimeError(
                _fixture_skew_message(sim_data_path, fork_dir)) from _ae
        raise
    finally:
        _em.Ecoli.__init__ = _orig_init
        # The composer has now loaded sim_data from sim_data_path; the temp
        # pickle (if any) has served its purpose — remove it so per-generation
        # variant builds don't leak ~300MB files.
        if _variant_simdata_tmp:
            try:
                os.unlink(_variant_simdata_tmp)
            except OSError:
                pass
    composer = _captured.get("composer")
    if composer is not None:
        try:
            dry_mass_inc_dict = dict(
                composer.load_sim_data.sim_data.expectedDryMassIncreaseDict)
        except Exception:
            dry_mass_inc_dict = {}

    init_state = sim.generated_initial_state
    if initial_overlay:
        for key in ("bulk", "unique", "environment", "boundary"):
            if key in initial_overlay:
                init_state[key] = initial_overlay[key]

    engine = Engine(
        processes=sim.ecoli.processes,
        steps=sim.ecoli.steps,
        flow=sim.ecoli.flow,
        topology=sim.ecoli.topology,
        initial_state=init_state,
        emitter={"type": "null"},
        progress_bar=False,
    )

    media_id = "minimal"
    try:
        env = init_state.get("environment", {})
        media_id = env.get("media_id", "minimal") if isinstance(env, dict) else "minimal"
    except Exception:
        pass

    return EngineHandle(
        engine=engine, sim=sim, dry_mass_inc_dict=dry_mass_inc_dict,
        condition=condition, media_id=media_id, time_step=float(time_step))


# ---------------------------------------------------------------------------
# Read observables out of a live Engine state
# ---------------------------------------------------------------------------

def _state(engine):
    return engine.state.get_value()


# The 7 report-card observables (scripts/comparison_report_card.py OBSERVABLES), at the
# SAME paths v2ecoli emits (run_comparison_ensemble COMPARISON_PATHS) so both engines read
# identically: 5 in listeners.mass, 2 in listeners.unique_molecule_counts.
MASS_OBS = ("cell_mass", "dry_mass", "protein_mass", "rna_mass",
            "instantaneous_growth_rate")
COUNT_OBS = ("active_RNAP", "active_ribosome")
OBSERVABLES = MASS_OBS + COUNT_OBS


def cell_observables(engine) -> dict:
    """Pull the 7 comparison observables from the live Engine state. Single-cell, no
    agents wrapper (divide=False). Scalar axes + the raw bulk/unique for division.

    Spatial-aware: a whole-config run with a ``spatial_environment_config`` builds
    vEcoli's own colony composite whose top-level state is
    ``{agents:{<id>:cell}, multibody, reaction_diffusion, fields}`` — the cell's
    ``listeners``/``bulk``/``unique`` live under ``agents/<id>``, not the top level.
    Follow the first agent so the emit + the division gate read the real cell (else
    every observable reads 0 off an empty top level)."""
    st = _state(engine)
    _agents = st.get("agents")
    if isinstance(_agents, dict) and _agents:
        st = _agents.get("0") or next(iter(_agents.values()))
    listeners = st.get("listeners", {}) or {}
    mass = listeners.get("mass", {}) or {}
    umc = listeners.get("unique_molecule_counts", {}) or {}
    obs = {k: float(mass.get(k, 0.0) or 0.0) for k in MASS_OBS}
    obs.update({k: float(umc.get(k, 0.0) or 0.0) for k in COUNT_OBS})
    obs.update({
        "bulk": st.get("bulk"),
        "unique": st.get("unique"),
        "environment": st.get("environment", {}),
        "boundary": st.get("boundary", {}),
        "listeners": listeners,   # raw listener tree, for declared `observables`
    })
    return obs


def _select_observables(listeners: dict, paths) -> dict:
    """Read declared listener leaves out of the raw listener tree.

    ``paths`` are dotted ``"group.leaf"`` (or deeper) selectors relative to
    ``listeners`` — e.g. ``"rna_synth_prob.total_rna_init"``. Returns a nested
    dict mirroring that structure, with each leaf coerced to float; a missing
    path yields 0.0 so the emitted leaf stays a continuous trace. This is the
    general counterpart to the mass/count/exchange/bulk readers — it lets a
    study surface ANY genuine-vEcoli listener leaf without a code change here."""
    out: dict = {}
    for path in paths or []:
        parts = [p for p in str(path).replace("listeners.", "", 1).split(".") if p]
        if not parts:
            continue
        src = listeners or {}
        for k in parts[:-1]:
            src = src.get(k, {}) if isinstance(src, dict) else {}
        leaf = parts[-1]
        v = src.get(leaf) if isinstance(src, dict) else None
        cur = out
        for k in parts[:-1]:
            cur = cur.setdefault(k, {})
        try:
            cur[leaf] = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            cur[leaf] = 0.0
    return out


def _deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge ``src`` into ``dst`` (nested dicts merged, leaves set)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _merge_listener_schema(listeners_schema: dict, paths) -> None:
    """Add ``overwrite[float]`` leaves to a listeners output-schema dict for each
    dotted ``group.leaf`` observable path (mirrors _select_observables' nesting)."""
    for path in paths or []:
        parts = [p for p in str(path).replace("listeners.", "", 1).split(".") if p]
        if not parts:
            continue
        cur = listeners_schema
        for k in parts[:-1]:
            cur = cur.setdefault(k, {})
            if not isinstance(cur, dict):
                break
        else:
            cur[parts[-1]] = "overwrite[float]"


def _select_bulk_observables(obs_bulk, ids: list) -> dict:
    """Pick ``ids`` out of the inner cell's bulk store as floats.

    Handles BOTH representations the wrapped engine can hand us:
    - a numpy structured array (the REAL vEcoli ``bulk`` store: fields ``id``
      and ``count`` — see ``ecoli.library.schema`` / ``initial_conditions``),
      resolved id->count via a name->index map built once per call;
    - a plain ``Mapping`` (name->count), the shape the dict-based tests use.

    A requested id absent from the store yields ``0.0`` (a species absent this
    tick), never a KeyError/IndexError. Empty ``ids`` -> ``{}``. ``None`` or any
    unexpected value falls back to the mapping path -> all ``0.0`` (no crash).

    PURE: operates only on the passed array/dict — no fork import. Never uses
    truthiness on ``obs_bulk`` (a multi-element ndarray raises ValueError on
    ``bool``); emptiness is gated solely on ``ids``.
    """
    if not ids:
        return {}
    # numpy structured array (has a compound dtype with the expected fields)
    dtype = getattr(obs_bulk, "dtype", None)
    if dtype is not None and dtype.names is not None \
            and "id" in dtype.names and "count" in dtype.names:
        name_to_idx = {str(n): i for i, n in enumerate(obs_bulk["id"])}
        counts = obs_bulk["count"]
        return {i: (float(counts[name_to_idx[i]]) if i in name_to_idx else 0.0)
                for i in ids}
    # Mapping (dict-like) fallback: name->count, missing -> 0.0.
    from collections.abc import Mapping
    src = obs_bulk if isinstance(obs_bulk, Mapping) else {}
    return {i: float(src.get(i, 0.0)) for i in ids}


def _select_exchange_fluxes(environment, fluxes: dict, *, basis: str = "counts",
                            listeners: dict | None = None) -> dict:
    """Pick named metabolic exchange fluxes out of the wrapped cell, on ``basis``.

    ``fluxes`` maps ``leaf_name -> exchange_key`` (e.g.
    ``{"acetate_exchange": "AC[p]", "glucose_exchange": "GLC[p]"}``).
    Sign is preserved verbatim; consumers decide on ``abs``.

    ``basis`` selects WHICH QUANTITY the leaf carries, and therefore which store
    it is read from. The two are not interchangeable — see
    :mod:`v2ecoli.steps.derivers.exchange_flux_listener`, whose ``basis`` this
    mirrors so a study declares the quantity ONCE and both engines honour it:

    ``counts`` (default, unchanged behaviour)
        ``environment["exchange"]`` — a molecule-count RUNNING TOTAL, because the
        store accumulates (``state + update``) while metabolism writes a per-step
        delta. Lineage-cumulative: it does not reset at division.

    ``gdcw``
        ``listeners.fba_results.external_exchange_fluxes`` — the wrapped
        metabolism's own per-tick RATE in mmol/gDCW/h
        (``.asNumber(GDCW_BASIS)`` at the source), which is also the leaf genuine
        vEcoli's own bioproduction analyses read
        (``listeners__fba_results__external_exchange_fluxes``). ⚠ Those analyses
        index it POSITIONALLY, resolving names from emit metadata; this reads it
        by key, which works only for a metabolism that writes it as a mapping —
        see the TypeError below. ⚠ It is read
        rather than derived here deliberately: differencing the counts store to
        recover a rate would re-implement a conversion the wrapped process
        already performs, and the two would drift.

    ⚠ **The basis chooses the store, so a single declaration can never yield
    mixed quantities across leaves of one arm.** A per-leaf or fall-back-when-
    absent rule would: a molecule present in ``environment["exchange"]`` would
    report a cumulative count while one absent from it reported a rate, on two
    leaves of the same cell, with nothing to distinguish them downstream.

    A key absent this tick yields ``0.0`` so the leaf stays a continuous trace.
    ⚠ That also means an unresolvable key is indistinguishable from a true zero —
    callers declaring a molecule should confirm it is present on the chosen
    basis's store rather than reading 0.0 as a measurement.

    Deliberately generic: no molecule is special-cased here. GENERIC/pathway-
    agnostic by design — the flux map is supplied by config, so this stays out of
    the shared model's knowledge of any particular pathway."""
    from v2ecoli.steps.derivers.exchange_flux_listener import (
        BASIS_COUNTS, BASIS_GDCW, resolve_exchange_key)
    # Validated BEFORE the empty-map short-circuit, so a bad basis is refused on
    # every call rather than only when something is declared — the deriver
    # validates in initialize() regardless of its map, and the two must agree.
    basis = str(basis or BASIS_COUNTS)
    if basis not in (BASIS_COUNTS, BASIS_GDCW):
        # Refused rather than defaulted, matching the deriver: a silently
        # defaulted basis emits a running total under a rate's name.
        raise ValueError(
            f"exchange-flux basis {basis!r} unknown; expected "
            f"{BASIS_COUNTS!r} or {BASIS_GDCW!r}.")
    if not fluxes:
        return {}
    if basis == BASIS_GDCW:
        lst = listeners if isinstance(listeners, dict) else {}
        fba = lst.get("fba_results")
        source = (fba or {}).get("external_exchange_fluxes")
        # ⚠ NOT every metabolism writes this leaf the same way, and the two
        # shapes are not interchangeable. A process that keys it by metabolite id
        # (a dict) can be looked up here; one that writes a POSITIONAL ARRAY
        # cannot, because the id->index mapping lives in emit metadata and is not
        # in this store at all. Refused rather than treated as empty: falling
        # through to {} would emit 0.0 on every leaf of every tick — a flat zero
        # trace that reads exactly like a cell producing none of the molecule,
        # which is the failure this basis exists to remove.
        # ⚠ ABSENT is refused on the SAME footing as wrong-shaped, and that
        # symmetry is the point. Falling through to {} emitted 0.0 on every leaf
        # of every tick — a flat zero trace indistinguishable from a cell
        # producing none of the molecule. Measured: the public vEcoli's
        # `metabolism_redux` does not write this leaf at all (it writes
        # `estimated_exchange_dmdt` instead), so the configuration that failed
        # SILENTLY was the public one, while the loud TypeError below only ever
        # fired for stock `metabolism.py`. A leaf this basis is read from and
        # cannot find is a refusal, not a zero.
        if source is None:
            raise TypeError(
                "exchange-flux basis 'gdcw' reads "
                "listeners.fba_results.external_exchange_fluxes, and this run's "
                "metabolism does not write it. Refused rather than read as zero: "
                "an absent leaf would emit 0.0 on every molecule of every tick, "
                "which reads exactly like a cell producing none of them. Use "
                "basis 'counts', or a metabolism that writes the leaf keyed by "
                "metabolite id.")
        if not isinstance(source, dict):
            raise TypeError(
                "exchange-flux basis 'gdcw' needs "
                "listeners.fba_results.external_exchange_fluxes keyed by "
                f"metabolite id, but this run's metabolism writes a "
                f"{type(source).__name__}. Positional output cannot be resolved "
                "by key here (the id order is emit metadata, not store content). "
                "Use basis 'counts', or a metabolism that keys the leaf.")
    else:
        env = environment if isinstance(environment, dict) else {}
        source = env.get("exchange")
        source = source if isinstance(source, dict) else {}
    out = {}
    for leaf, key in fluxes.items():
        v = resolve_exchange_key(source, key)
        out[leaf] = float(v) if v is not None else 0.0
    return out


# ---------------------------------------------------------------------------
# pbg Process: genuine vEcoli as ONE process-bigraph node (vivarium Engine inside)
# ---------------------------------------------------------------------------

from process_bigraph import Process


class VivariumEcoliProcess(Process):
    """Genuine upstream vEcoli as a SINGLE process-bigraph node, with vivarium-core's
    own Engine running inside.

    This PRESERVES the process-bigraph design: v2ecoli and vEcoli compare on the SAME
    pbg runtime (one pbg node each at ``agents/0``), so the comparison stays a true
    pbg-vs-pbg test — but vivarium handles partition / update-reconciliation / division
    *inside* this node, so none of the Engine-reimplementation bugs (the bulk-reconcile
    mass explosion; the ``_add`` division crash) can arise.

    ``update(state, interval)`` advances the inner vivarium Engine by ``interval`` via
    ``Engine.run_for`` and writes the cell's mass observables back to the pbg store
    (``listeners.mass.*``), so the standard ``XArrayEmitter`` view reads them exactly as
    it does for v2ecoli. REST-ready: the in-process Engine can later be swapped for an
    out-of-process vivarium service behind this same port interface.
    """

    config_schema = {
        "sim_data_path": "string",
        "condition": {"_type": "string", "_default": "basal"},
        "seed": {"_type": "integer", "_default": 0},
        "time_step": {"_type": "float", "_default": 1.0},
        "exclude_processes": {"_type": "list[string]", "_default": []},
        "fork_dir": {"_type": "string", "_default": ""},
        "variant": {"_type": "integer", "_default": 0},
        # The lineage phylogeny key this cell is ("0" -> "00" -> ...). NOT
        # cosmetic: the wrapped fork derives its generation index from its LENGTH
        # (see build_vivarium_ecoli), so a config that omits it runs every
        # generation as the founder and no staged induction can fire.
        "agent_id": {"_type": "string", "_default": "0"},
        "observable_bulk_ids": {"_type": "list[string]", "_default": []},
        # {leaf_name: exchange_key} — metabolic exchange fluxes to emit under
        # listeners.exchange_flux.<leaf> (generic; the caller names the keys).
        "exchange_fluxes": {"_type": "map[string]", "_default": {}},
        # WHICH QUANTITY the exchange_flux leaves carry — "counts" (a
        # lineage-cumulative molecule total, read from environment.exchange) or
        # "gdcw" (a per-tick mmol/gDCW/h rate, read from the wrapped
        # metabolism's own listeners.fba_results.external_exchange_fluxes).
        # Mirrors the deriver's `basis` so a study declares it ONCE and both
        # engines honour it. See _select_exchange_fluxes.
        "exchange_flux_basis": {"_type": "string", "_default": "counts"},
        # Arbitrary genuine-vEcoli listener leaves to surface, as dotted
        # "group.leaf" paths under listeners (e.g. "rna_synth_prob.total_rna_init").
        # The fully-general measurement hook — any listener leaf, no code change.
        "observables": {"_type": "list[string]", "_default": []},
    }

    # Set by build_vivarium_ecoli_composite to inject a pre-built (possibly daughter-
    # overlaid) engine so the process doesn't rebuild EcoliSim. Consumed once at
    # construction. Single build is sequential; Ray seeds are separate processes.
    _PENDING_HANDLE = None

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        if VivariumEcoliProcess._PENDING_HANDLE is not None:
            self._handle = VivariumEcoliProcess._PENDING_HANDLE
            VivariumEcoliProcess._PENDING_HANDLE = None
            self._obs_bulk_ids = list(self.config.get("observable_bulk_ids") or [])
        else:
            self._handle = build_vivarium_ecoli(
                sim_data_path=self.config["sim_data_path"],
                condition=self.config["condition"],
                seed=int(self.config["seed"]),
                time_step=float(self.config["time_step"]),
                exclude_processes=list(self.config.get("exclude_processes") or []) or None,
                fork_dir=(self.config.get("fork_dir") or None),
                variant=int(self.config.get("variant") or 0),
                agent_id=str(self.config.get("agent_id") or "0"),
            )
            self._obs_bulk_ids = list(self.config.get("observable_bulk_ids") or [])
        self._exchange_fluxes = dict(self.config.get("exchange_fluxes") or {})
        self._exchange_flux_basis = str(
            self.config.get("exchange_flux_basis") or "counts")
        self._observables = list(self.config.get("observables") or [])

    def inputs(self):
        return {}

    def outputs(self):
        # Recomputed-absolute each tick → 'set' semantics (overwrite), matching
        # vivarium's listener _updater='set'.
        out = {"listeners": {
            "mass": {k: "overwrite[float]" for k in MASS_OBS},
            "unique_molecule_counts": {k: "overwrite[float]" for k in COUNT_OBS},
        }}
        if self._obs_bulk_ids:
            # Emit declared bulk counts under listeners.observable_bulk.<id> so the
            # candidate (v2ecoli) and reference (this) share ONE comparison path.
            out["listeners"]["observable_bulk"] = {
                i: "overwrite[float]" for i in self._obs_bulk_ids}
        if self._exchange_fluxes:
            out["listeners"]["exchange_flux"] = {
                leaf: "overwrite[float]" for leaf in self._exchange_fluxes}
        if self._observables:
            _merge_listener_schema(out["listeners"], self._observables)
        return out

    def update(self, state, interval):
        self._handle.engine.run_for(float(interval))
        obs = cell_observables(self._handle.engine)
        upd = {"listeners": {
            "mass": {k: obs[k] for k in MASS_OBS},
            "unique_molecule_counts": {k: obs[k] for k in COUNT_OBS},
        }}
        if self._exchange_fluxes:
            upd["listeners"]["exchange_flux"] = _select_exchange_fluxes(
                obs.get("environment"), self._exchange_fluxes,
                basis=self._exchange_flux_basis,
                listeners=obs.get("listeners"))
        if self._observables:
            _deep_merge(upd["listeners"],
                        _select_observables(obs.get("listeners", {}), self._observables))
        if self._obs_bulk_ids:
            upd["listeners"]["observable_bulk"] = _select_bulk_observables(
                obs.get("bulk", {}), self._obs_bulk_ids)
        return upd

    def divide(self) -> dict:
        """Split the inner cell with vEcoli's faithful ``divide_cell``; return
        daughter-0's overlay (bulk/unique/environment/boundary) to seed the next
        generation's Engine. The split is vivarium-native — no pbg ``_add``."""
        obs = cell_observables(self._handle.engine)
        d1, _d2 = divide_cell({
            "bulk": obs["bulk"], "unique": obs["unique"],
            "environment": obs["environment"], "boundary": obs["boundary"]})
        return d1

    def division_signals(self) -> tuple:
        """``(dry_mass_fg, n_chromosomes)`` for the lineage driver's division gate."""
        obs = cell_observables(self._handle.engine)
        return obs["dry_mass"], _n_chromosomes(obs["unique"])


def build_vivarium_ecoli_composite(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    swap_processes: dict | None = None,
    flow: dict | None = None,
    fork_dir: str | None = None,
    core=None,
    agent_id: str = "0",
    initial_overlay: dict | None = None,
    variant: int = 0,
    observable_bulk_ids: list | None = None,
    exchange_fluxes: dict | None = None,
    exchange_flux_basis: str = "counts",
    observables: list | None = None,
):
    """Wrap a single :class:`VivariumEcoliProcess` as a one-node pbg Composite under
    ``agents/<agent_id>`` — the genuine-vEcoli analogue of the v2ecoli agent composite,
    so the SAME ``run_multigen_xarray`` / ``XArrayEmitter`` path serves both engines.

    ``initial_overlay`` (a daughter's divided bulk/unique/env/boundary) seeds a non-
    founder generation, and ``agent_id`` is that generation's phylogeny key — the
    wrapped fork derives its generation index from its LENGTH, so a non-founder
    generation built with "0" runs as the founder. Returns ``(composite, info)``. The process writes
    ``listeners.mass.*`` (overwrite/set semantics) into the agent store each tick.
    """
    from process_bigraph import Composite
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    # Build the (optionally daughter-seeded) engine once and inject it so the process
    # doesn't rebuild EcoliSim.
    VivariumEcoliProcess._PENDING_HANDLE = build_vivarium_ecoli(
        sim_data_path=sim_data_path, condition=condition, seed=int(seed),
        time_step=float(time_step), exclude_processes=list(exclude_processes or []) or None,
        swap_processes=swap_processes or None, flow=flow or None,
        fork_dir=fork_dir or None, initial_overlay=initial_overlay, variant=int(variant),
        agent_id=str(agent_id))
    proc = VivariumEcoliProcess(config={
        "sim_data_path": sim_data_path, "condition": condition, "seed": int(seed),
        "time_step": float(time_step),
        "exclude_processes": list(exclude_processes or []),
        "fork_dir": fork_dir or "",
        "variant": int(variant),
        "agent_id": str(agent_id),
        "observable_bulk_ids": list(observable_bulk_ids or []),
        "exchange_fluxes": dict(exchange_fluxes or {}),
        "exchange_flux_basis": str(exchange_flux_basis or "counts"),
        "observables": list(observables or []),
    }, core=core)
    iface = proc.interface()

    # Wire the process's output ports to the agent's stores. The process only
    # DECLARES a ``bulk`` output port when observable ids are configured, so only
    # then do we wire ``bulk`` -> ``["bulk"]`` (agents/<id>/bulk); otherwise pbg
    # would drop an unmapped-but-declared port and the observables never land.
    _outputs = {"listeners": ["listeners"]}
    if list(observable_bulk_ids or []):
        _outputs["bulk"] = ["bulk"]
    cell_state = {
        "vivarium_ecoli": {
            "_type": "process",
            "instance": proc,
            "_inputs": iface.get("inputs", {}),
            "_outputs": iface.get("outputs", {}),
            "inputs": {},
            "outputs": _outputs,
            "interval": float(time_step),
        }
    }
    state = {"agents": {agent_id: cell_state}, "global_time": 0.0}
    composite = Composite(dict(schema=dict(), state=state), core=core)
    return composite, {"core": core, "agent_root": "agents",
                       "agent_id": agent_id, "process": proc}


def _dperiod_should_divide(handle) -> tuple[bool, int]:
    """Genuine vEcoli's DEFAULT division criterion (``d_period=True``), faithfully:
    the wcEcoli D-period mechanism (``ecoli.processes.cell_division.MarkDPeriod``).

    Divide as soon as the cell's time reaches a full chromosome's ``division_time``
    — an attribute set DURING replication (= replication-complete + D_period, where
    D_period comes from sim_data) — for a chromosome that has not yet triggered
    division, once there are >= 2 full chromosomes. The dry-mass threshold is NOT
    used (real vEcoli ignores it under d_period=True; v2's MarkDPeriod is identical).

    Returns ``(should_divide, n_full_chromosomes)``. Reads the genuine-vEcoli inner
    Engine directly, so ``division_time`` flows straight from sim_data → vEcoli's
    replication → here — no re-derivation, no mass approximation.
    """
    eng = handle.engine
    gt = float(getattr(eng, "global_time", 0.0) or 0.0)
    obs = cell_observables(eng)
    u = obs.get("unique") or {}
    fc = u.get("full_chromosome") if isinstance(u, dict) else None
    if fc is None or not hasattr(fc, "dtype") or fc.dtype.names is None:
        return False, 0
    names = fc.dtype.names
    active = (fc["_entryState"].view(np.bool_) if "_entryState" in names
              else np.ones(len(fc), dtype=bool))
    nchrom = int(active.sum())
    if nchrom < 2 or "division_time" not in names:
        return False, nchrom
    dt = np.asarray(fc["division_time"])[active]
    htd = (np.asarray(fc["has_triggered_division"]).astype(bool)[active]
           if "has_triggered_division" in names else np.zeros(nchrom, dtype=bool))
    untriggered = dt[~htd]
    if untriggered.size == 0:
        return False, nchrom
    return bool(gt >= float(untriggered.min())), nchrom


def _vecoli_config_summary(handle, *, condition: str, seed: int,
                           time_step: float, exclude_processes) -> dict:
    """JSON-able summary of the resolved vEcoli config the run actually used.

    Pulls the process/step NAMES the genuine-vEcoli Engine built plus a sanitized
    copy of ``EcoliSim.config`` (scalars/lists/small dicts only — the ~300MB
    ``sim_data`` object and other non-serializable values are dropped). Written
    next to the vEcoli zarr as ``vecoli_build_config.json`` so the comparison
    report shows vEcoli's OWN full config alongside v2ecoli's. Best-effort."""
    sim = getattr(handle, "sim", None)
    ecoli = getattr(sim, "ecoli", None) if sim is not None else None
    processes = sorted((getattr(ecoli, "processes", {}) or {}).keys()) if ecoli else []
    steps = sorted((getattr(ecoli, "steps", {}) or {}).keys()) if ecoli else []
    safe: dict = {}
    for k, v in (getattr(sim, "config", {}) or {}).items():
        if k in ("sim_data",):                 # huge object — never serialize
            continue
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 64:
            try:
                json.dumps(list(v))
                safe[k] = list(v)
            except Exception:  # noqa: BLE001
                safe[k] = f"<{type(v).__name__}[{len(v)}]>"
        elif isinstance(v, dict) and len(v) <= 32:
            try:
                json.dumps(v)
                safe[k] = v
            except Exception:  # noqa: BLE001
                safe[k] = f"<dict[{len(v)} keys]>"
    return {
        "engine": "vecoli", "source": "EcoliSim.config",
        "condition": condition, "seed": int(seed), "time_step": float(time_step),
        "media_id": getattr(handle, "media_id", None),
        "n_processes": len(processes), "processes": processes, "steps": steps,
        "exclude_processes": list(exclude_processes or []),
        "config": safe,
    }


def run_vivarium_ecoli_pbg_multigen(
    *,
    store_path,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    max_generations: int = 2,
    max_steps_per_gen: int = 9000,
    time_step: float = 1.0,
    chunk: int = 20,
    exclude_processes: list | None = None,
    swap_processes: dict | None = None,
    flow: dict | None = None,
    fork_dir: str | None = None,
    mass_multiplier: float = 1.0,
    core=None,
    experiment_id: str = "vecoli",
    variant: int = 0,
    lineage_seed: int = 0,
    whole_config: str | None = None,
    exchange_fluxes: dict | None = None,
    exchange_flux_basis: str = "counts",
    observable_bulk_ids: list | None = None,
    observables: list | None = None,
) -> dict:
    """Single-lineage multigen for the vEcoli **pbg node**, emitting the v2ecoli-format zarr.

    ``whole_config`` (a full vEcoli config-file path) makes the wrapped ``EcoliSim``
    load that config NATIVELY instead of the default baseline — so a config whose
    model content can't be expressed as ``swap_processes``/``flow`` (one declaring
    ``add_processes`` and/or a ``spatial_environment_config``) runs faithfully as
    one node. ⊕ A config declaring a ``variants`` block also needs this route,
    because ``apply_variant`` runs only when a whole-config was loaded.
    ⚠ The module-level config file is set here and reset by a bare trailing
    statement AFTER the generation loop — **not** in a ``finally``, despite what
    an earlier version of this docstring claimed. An exception out of this
    function therefore leaves it set process-wide. Tolerable today only because
    no in-process caller consumes it afterwards (the sequential seed path has no
    per-seed ``except``, and Ray gives each seed its own worker) — **not**
    because the isolation is real.

    Each generation is a one-node pbg ``Composite`` (``VivariumEcoliProcess``) driven by
    ``composite.run``; a per-generation ``XArrayEmitter`` writes a ``generation=N``
    partition into the shared store (the SAME emitter v2ecoli uses). At the division
    criterion (dry_mass ≥ birth + expectedDryMassIncrease AND ≥2 chromosomes) vEcoli's own
    ``divide_cell`` splits the inner cell and the followed daughter seeds the next
    generation's Composite. No pbg ``_add`` — the division-handoff crash cannot occur.
    """
    import shutil
    from pathlib import Path
    from v2ecoli.library.xarray_run import _build_emitter, _filter_agent_state
    from v2ecoli.library.upstream_division import daughter_phylogeny_id

    # Scope the whole-config selection to this run so every per-generation
    # ``build_vivarium_ecoli`` rebuild sees it and a later default run is unaffected.
    set_ecolisim_config_file(whole_config)

    if core is None:
        from v2ecoli.core import build_core
        core = build_core()
    store_path = str(store_path)
    if Path(store_path).exists():
        shutil.rmtree(store_path)

    exchange_fluxes = dict(exchange_fluxes or {})
    observable_bulk_ids = list(observable_bulk_ids or [])
    observables = list(observables or [])
    _view_vars = {
        "mass": {k: [{"path": k, "dtype": "<f8"}] for k in MASS_OBS},
        "unique_molecule_counts": {k: [{"path": k, "dtype": "<f8"}] for k in COUNT_OBS},
    }
    if exchange_fluxes:
        _view_vars["exchange_flux"] = {
            leaf: [{"path": leaf, "dtype": "<f8"}] for leaf in exchange_fluxes}
    # Declared arbitrary listener leaves ("group.leaf" under listeners): add each
    # to the nested listeners view so _filter_agent_state emits it.
    for path in observables:
        parts = [p for p in str(path).replace("listeners.", "", 1).split(".") if p]
        if len(parts) < 2:
            continue
        cur = _view_vars
        for k in parts[:-1]:
            cur = cur.setdefault(k, {})
        cur[parts[-1]] = [{"path": parts[-1], "dtype": "<f8"}]
    # Declared bulk observables ride under listeners.observable_bulk.<id> (the
    # process emits them there), so BOTH engines compare on one path.
    if observable_bulk_ids:
        _view_vars["observable_bulk"] = {
            i: [{"path": i, "dtype": "<f8"}] for i in observable_bulk_ids}
    view = [{"root": ("listeners",), "variables": _view_vars}]
    # ⚠ `variant` here is a PROVENANCE CLAIM stamped into every zarr partition,
    # and it is true only because `build_vivarium_ecoli` REFUSES a variant it
    # cannot apply. Before that refusal existed, a variant could be threaded this
    # far, recorded here, and then silently discarded at the `_cfgfile and
    # int(variant)` gate — so the store asserted a perturbation that never ran,
    # and every downstream reader (report card, sidecar, published artifact)
    # would have repeated the claim.
    # ⛔ If that refusal is ever weakened, this line starts lying again. They are
    # one invariant in two places; do not separate them.
    metadata_base = {
        # ⭐ THE ROUTE IS PROVENANCE. It changes the PROCESS SET — the native path
        # carries `exclude_processes: ['exchange_data']`, so metabolism's uptake
        # bounds are set by a Step that runs on one route and not the other — and
        # until now nothing in the store recorded which route produced it. Two
        # zarrs graded against each other could differ by route with no way to
        # detect it. One key makes that confound visible.
        "whole_config_route": bool(whole_config),
        "experiment_id": experiment_id, "variant": int(variant),
        "lineage_seed": int(lineage_seed), "time_step": float(time_step),
        "max_duration": float(max_generations * max_steps_per_gen),
    }

    overlay = None
    # ⛔⛔ BOTH OF THESE ARE PHYLOGENY KEYS AND BOTH MUST ADVANCE — they are kept
    # as two names because they are consumed by two different things, and that is
    # exactly how one of them came to be pinned at "0" for the whole lineage:
    #   · ``partition_agent_id`` — the EMITTER's key, distinct per generation so
    #     each writes its own zarr partition (avoids a same-store collision).
    #   · ``composite_agent_id`` — the inner cell's key in the pbg agents map AND,
    #     via ``build_vivarium_ecoli``, the wrapped fork's GENERATION INDEX
    #     (``LoadSimData`` reads ``len(agent_id)``). Pinned at "0" it reports
    #     generation 1 forever, so a config's staged induction never fires and the
    #     whole lineage runs as the un-induced baseline, silently.
    # ⇒ v2ecoli's own native lineage already advances its agent id
    #   (``v2ecoli/workflow/lineage.py``); only this wrapped-fork driver did not.
    composite_agent_id = "0"
    partition_agent_id = "0"
    done_global = 0
    divisions = 0
    gens_done = 0
    final_cell_mass = None
    build_config = None
    em = None  # ONE lineage-scoped emitter, opened at gen 0, advanced per division

    for gen in range(max_generations):
        # gen 0 is a fresh founder (overlay=None); later generations seed the inner
        # Engine from the previous generation's daughter (overlay set below).
        comp, info = build_vivarium_ecoli_composite(
            sim_data_path=sim_data_path, condition=condition, seed=seed + gen,
            time_step=time_step, exclude_processes=exclude_processes,
            swap_processes=swap_processes, flow=flow,
            fork_dir=fork_dir, core=core, agent_id=composite_agent_id,
            initial_overlay=overlay, variant=variant,
            exchange_fluxes=exchange_fluxes,
            exchange_flux_basis=exchange_flux_basis,
            observable_bulk_ids=observable_bulk_ids, observables=observables)
        proc = info["process"]
        comp.run(1)  # warm-up tick so listeners materialise
        if gen == 0:                       # capture vEcoli's OWN resolved config once
            try:
                build_config = _vecoli_config_summary(
                    proc._handle, condition=condition, seed=seed,
                    time_step=time_step, exclude_processes=exclude_processes)
            except Exception as _cfgerr:  # noqa: BLE001 — never block the run
                print(f"[vecoli-config] summary skipped: "
                      f"{type(_cfgerr).__name__} {_cfgerr}")
        if em is None:
            # ONE emitter drives the whole lineage: generation 1 is opened here,
            # and each division advances it in place via advance_generation()
            # below. This replaces the old build-a-fresh-emitter-per-generation
            # flow, whose per-generation close() was wrapped in
            # `except AssertionError: pass` — a swallow that, when a
            # (sub-buffer_size) generation failed to persist, left the zarr store
            # as bare group skeletons and crashed the next generation's
            # _check_group. advance_generation() guarantees each generation is
            # flushed + consolidated on disk before the next one opens.
            em = _build_emitter(
                core=core, store_path=store_path, view=view,
                metadata_base=metadata_base,
                generation=1,
                # Inherit build_emitter_config's buffer_size default (600): flush
                # a handful of times per generation, not every few steps.
                agent_id=partition_agent_id, output_metadata={})

        steps = 1
        divided = False
        while steps < max_steps_per_gen:
            comp.run(chunk)
            steps += chunk
            done_global += chunk
            # ⛔ ONE WALK, TWO CONSUMERS — a silent divergence between them is the
            # exact shape of the defect this loop was fixed for. Assert it rather
            # than trusting that the two assignments below stay together.
            assert composite_agent_id == partition_agent_id, (
                f"the cell's own key {composite_agent_id!r} and the emitter's "
                f"partition key {partition_agent_id!r} diverged — the partition "
                f"would be labelled with a different cell's generation")
            agent_state = comp.state["agents"][composite_agent_id]
            payload = _filter_agent_state(agent_state, view)
            # Relabel the payload to the emitter's phylogeny key (the emitter strips
            # the agent prefix via get_in(data, ("agents", partition_agent_id))).
            em.update({"time": float(done_global), "global_time": float(done_global),
                       "agents": {partition_agent_id: payload}})
            mass = agent_state["listeners"]["mass"]
            final_cell_mass = float(mass.get("cell_mass", 0.0) or 0.0)
            # Divide by genuine vEcoli's D-period criterion (the d_period=True
            # default), NOT a dry-mass threshold. The old mass-based rule diverged
            # from real vEcoli on fast-growth media (e.g. with_aa, multifork
            # replication): D-period fires when replication+D_period elapses, well
            # before the mass threshold, so the mass rule divided ~40% too late.
            should_divide, _nchrom = _dperiod_should_divide(proc._handle)
            if should_divide:
                divided = True
                break

        gens_done += 1
        if not divided:
            break
        overlay = proc.divide()
        # One phylogeny walk, both keys — see the block where they are declared.
        # ⚠ If these ever diverge, the emitter's generation label and the fork's
        # internal generation index stop describing the same cell.
        daughter_id = daughter_phylogeny_id(partition_agent_id)[0]
        partition_agent_id = daughter_id
        composite_agent_id = daughter_id
        divisions += 1
        if gen < max_generations - 1:
            # Finalize this generation (flush trailing buffer + mark the division
            # event + consolidate) and open the daughter's partition in the same
            # store — guaranteed, never swallowed. The last generation is
            # finalized by the terminal close() after the loop.
            em.advance_generation(agent_id=partition_agent_id, success=True)

    if em is not None:
        # Terminal finalize of the final generation (flush + mark division +
        # consolidate). Not swallowed: a real persistence failure must surface
        # rather than silently leave a skeleton store.
        em.close(success=True)
    set_ecolisim_config_file(None)  # reset for the next run (deterministic isolation)
    return {"generations": gens_done, "divisions": divisions,
            "store": store_path, "final_cell_mass": final_cell_mass,
            "build_config": build_config}


# ---------------------------------------------------------------------------
# Single-lineage multi-generation driver (standalone, non-pbg — local utility)
# ---------------------------------------------------------------------------

def run_vivarium_ecoli_multigen(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    max_generations: int = 2,
    max_steps_per_gen: int = 9000,
    time_step: float = 1.0,
    chunk: int = 20,
    exclude_processes: list | None = None,
    fork_dir: str | None = None,
    mass_multiplier: float = 1.0,
    on_emit=None,
) -> dict:
    """Drive genuine vEcoli single-lineage for ``max_generations`` generations.

    Each generation: build a genuine-vEcoli Engine (seeded with the previous daughter's
    divided state), advance with ``run_for(chunk)`` until the division criterion is met
    (dry_mass >= birth_dry_mass + expectedDryMassIncrease AND >=2 chromosomes), then split
    with vEcoli's ``divide_cell`` and carry one daughter into the next generation.

    ``on_emit(gen, t_global, obs)`` is called once per chunk with the live observables
    (the caller writes them wherever it likes — e.g. a matched-timepoint zarr).

    Returns a summary dict ``{generations, divisions, steps_per_gen, final_cell_mass}``.
    """
    overlay = None
    divisions = 0
    steps_per_gen: list[int] = []
    t_global = 0.0
    last_obs = None
    # The fork derives its generation index from ``len(agent_id)``; walk the same
    # phylogeny the pbg driver does, so a generation reports itself honestly.
    # ⚠ This driver takes neither ``variant`` nor ``whole_config``, so it cannot
    # CREATE an ``internal_shift_dict`` (only ``apply_variant`` does). A shift
    # fires here only if the sim_data pickle handed in already carries one.
    from v2ecoli.library.upstream_division import daughter_phylogeny_id
    agent_id = "0"

    for gen in range(max_generations):
        h = build_vivarium_ecoli(
            sim_data_path=sim_data_path, condition=condition, seed=seed + gen,
            time_step=time_step, exclude_processes=exclude_processes,
            fork_dir=fork_dir, initial_overlay=overlay, agent_id=agent_id)
        engine = h.engine
        threshold = None
        gen_steps = 0
        divided = False

        while gen_steps < max_steps_per_gen:
            engine.run_for(float(chunk))
            gen_steps += chunk
            obs = cell_observables(engine)
            last_obs = obs
            if on_emit is not None:
                on_emit(gen, t_global + gen_steps, obs)

            if threshold is None and obs["dry_mass"] > 0:
                inc = _inc_to_fg(h.dry_mass_inc_dict.get(h.media_id))
                if inc is None:
                    inc = _inc_to_fg(h.dry_mass_inc_dict.get("minimal"))
                if inc is None:
                    inc = obs["dry_mass"]  # fallback: mass doubling
                threshold = obs["dry_mass"] + inc * mass_multiplier

            n_chrom = _n_chromosomes(obs["unique"])
            if threshold is not None and obs["dry_mass"] >= threshold and n_chrom >= 2:
                divided = True
                break

        steps_per_gen.append(gen_steps)
        t_global += gen_steps

        if not divided:
            # ran out of steps before dividing — stop the lineage here
            break
        # vEcoli's faithful split; follow daughter 0 into the next generation
        d1, _d2 = divide_cell({
            "bulk": last_obs["bulk"], "unique": last_obs["unique"],
            "environment": last_obs["environment"], "boundary": last_obs["boundary"]})
        overlay = d1
        agent_id = daughter_phylogeny_id(agent_id)[0]
        divisions += 1

    return {
        "generations": len(steps_per_gen),
        "divisions": divisions,
        "steps_per_gen": steps_per_gen,
        "final_cell_mass": (last_obs or {}).get("cell_mass"),
    }
