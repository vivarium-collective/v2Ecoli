"""Resolve, classify, translate, and inject add/swap processes.

Wheel-shipped package resolver. This module lives INSIDE the ``v2ecoli`` package
(imported absolutely as ``v2ecoli.library.inject``) so ``ecoli_baseline`` no
longer resolves it via a bare ``from scripts._compare.inject import ...`` after a
``sys.path.insert`` — a bare import that picked up whichever repo's ``scripts/``
happened to be on ``sys.path`` (on the GovCloud pod, a vendored copy), silently
shadowing the intended resolver. The absolute package import kills that shadow.

Native process classes stay DOWNSTREAM (they are not shipped in this wheel):
resolve them through the native-injection registry (:func:`register_native_injection`,
populated downstream, e.g. by sms-ecoli). The resolver fails LOUD by construction
— a registered name whose class will not import raises, and a built config missing
a declared required key raises — rather than falling back to an empty/default
config (the root of the one-tick "empty config" collapse this design fixes).

Runs in the v2ecoli sim subprocess (where vivarium-core + any fork repo are
importable). The ``__main__`` below obtains the resolved specs as JSON for the
report + early fail-fast.

Single-fork constraint: this module imports ``ecoli.processes`` once per
process class resolution; a given process lifetime must use exactly one fork
repo (the harness invokes one ``--vecoli-repo`` per run).
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import os
import re
import sys
from typing import Any


class InjectionError(RuntimeError):
    """A fork process cannot be injected (unsupported / unresolved)."""


@contextlib.contextmanager
def _idempotent_registration():
    """Make vivarium's ``Registry.register`` idempotent during a fork import.

    A real vEcoli fork's ``ecoli/__init__.py`` re-registers emitters/processes
    into vivarium's global singleton registries (already populated by the
    installed ``ecoli``); the stock ``register`` raises on a duplicate key whose
    value differs (a re-imported module yields new class objects). Within this
    context we skip keys that already exist and add only new ones, restoring the
    original method afterwards. If vivarium is not importable (the duck-typed
    fixture fork), this is a no-op.
    """
    try:
        from vivarium.core import registry as _vreg
    except Exception:  # noqa: BLE001 — fixture fork has no vivarium registries
        yield
        return
    orig = _vreg.Registry.register

    def _idempotent(self, key, item, alternate_keys=tuple()):
        for rk in [key, *alternate_keys]:
            if rk not in self.registry:
                self.registry[rk] = item
        self.main_keys.append(key)

    _vreg.Registry.register = _idempotent
    try:
        yield
    finally:
        _vreg.Registry.register = orig


# Cache populated by resolve_injections: (module, qualname) -> class.
# Allows _import_class to find fork classes without requiring the fork's
# ecoli.* modules to remain in sys.modules during apply_injected_processes.
_fork_class_cache: dict[tuple[str, str], type] = {}

# Memoization cache for resolve_injections: stable key -> list of spec dicts.
# Prevents re-importing the fork's ecoli.* package on every generation call
# (baseline() invokes resolve_injections once per generation; without this
# cache a real fork whose ecoli/__init__.py registers vivarium singleton
# entries would fail on duplicate registration at generation 2).
_RESOLVE_CACHE: dict[str, list] = {}


def classify_process(cls) -> str:
    """Return 'partitioned' | 'pbg_native' | 'vivarium_1' for a process class."""
    if hasattr(cls, "calculate_request") or hasattr(cls, "evolve_state"):
        return "partitioned"
    if hasattr(cls, "inputs") and hasattr(cls, "outputs"):
        return "pbg_native"
    if hasattr(cls, "ports_schema") and (
            hasattr(cls, "next_update") or hasattr(cls, "update")):
        return "vivarium_1"
    raise InjectionError(
        f"{cls.__name__}: not a recognizable process (no ports_schema/inputs).")


def _should_inject_as_step(cls) -> bool:
    """Whether a fork process must be injected as a pbg STEP (immediate,
    cascade-applied at interval -1.0) rather than an interval-scheduled process.

    A vivarium ``Step`` (deriver, ``update_condition``-gated — e.g.
    ``MetabolismRedux``) MUST be a step: injected as an interval process its
    ``next_update_time`` output (read by ``GlobalClock``) is deferred to its
    front-time ``global_time + interval`` and so applies a tick late. On the
    second tick ``GlobalClock.calculate_timestep`` then sees
    ``next_update_time == global_time`` for that process, returns
    ``full_step == 0``, and simulation time never advances — the run spins in
    ``_run_inner`` forever (the metabolism_redux tick-2 hang). As a step the
    deriver applies within the same tick, so ``next_update_time`` advances before
    the clock recomputes. Steps (``PartitionedProcess``) are already rejected by
    :func:`classify_process` as ``partitioned``, so the injectable set is plain
    vivarium Processes (stay processes) plus vivarium Steps (become steps).

    An explicit ``_force_step`` attribute always wins. Detection is
    signature-based (``update_condition``) so it does NOT depend on ``vivarium``
    being importable: a deriver is ``update_condition``-gated regardless of its
    base class, and — critically — a v2ecoli-NATIVE deriver (e.g. the native
    ``ecoli-metabolism-redux``) is a pbg-native ``EcoliProcess``, NOT a
    ``vivarium.Step`` subclass, so the old ``issubclass(cls, vivarium.Step)``
    test returned False for it and injected it as an interval process. That is
    exactly the tick-2 non-advancing-``global_clock`` collapse this guard exists
    to prevent — and on a native/vecoli-free image (no ``vivarium`` installed)
    the import-failure fallback misclassified EVERY deriver. The ``vivarium.Step``
    subclass check is kept as a fallback for a fork Step that somehow lacks the
    method.
    """
    if bool(getattr(cls, "_force_step", False)):
        return True
    # A pbg-NATIVE deriver — a process exposing inputs()/outputs() that is also
    # update_condition-gated (see ``update_condition`` on metabolism /
    # tf_binding / chromosome_structure / …). Native derivers are NOT
    # vivarium.Step subclasses, so the issubclass check below misses them, yet
    # they must be steps. Requiring inputs()+outputs() keeps a plain vivarium
    # Process out (it exposes ports_schema, not inputs/outputs — and inherits a
    # base update_condition), and needs no vivarium import.
    if (hasattr(cls, "update_condition")
            and hasattr(cls, "inputs") and hasattr(cls, "outputs")):
        return True
    try:
        from vivarium.core.process import Step
    except Exception:  # noqa: BLE001 — fixture fork has no vivarium
        return False
    return isinstance(cls, type) and issubclass(cls, Step)


def _fork_registry(fork_repo: str):
    """Import the fork's ``ecoli.processes.process_registry`` and return it.

    Uses a save/restore pattern around ecoli.* in sys.modules so that:
    - The fork's ecoli.processes is loaded fresh (for registry access).
    - The installed vEcoli's ecoli.* modules are restored afterwards, preventing
      duplicate class-object registrations in vivarium singleton registries.

    Real-fork duplicate registrations: a REAL vEcoli fork's ``ecoli/__init__.py``
    re-registers emitters/processes into vivarium's GLOBAL singleton registries
    (already populated by the installed ``ecoli``), so re-importing the fork
    would raise "registry already contains an entry for ...". We make vivarium's
    ``Registry.register`` idempotent (skip keys that already exist, add new ones)
    for the duration of the fork import only, then restore it — see
    :func:`_idempotent_registration`. The fork's NEW process names (those in
    ``add_processes``) still register; names shared with the installed ecoli keep
    the installed class, which is irrelevant since we only resolve the
    added/swapped classes. The duck-typed fixture fork has no vivarium registries,
    so the context manager is a no-op there.
    """
    fork_abs = os.path.abspath(fork_repo)
    if fork_repo not in sys.path:
        sys.path.insert(0, fork_repo)

    # Partition current ecoli.* entries: save the real (non-fork) ones; evict all.
    saved_real: dict[str, object] = {}
    for k in [k for k in sys.modules if k == "ecoli" or k.startswith("ecoli.")]:
        mod = sys.modules.pop(k)
        mod_file = getattr(mod, "__file__", None) or ""
        if not os.path.abspath(mod_file).startswith(fork_abs):
            saved_real[k] = mod  # keep for restore

    try:
        with _idempotent_registration():
            fork_mod = importlib.import_module("ecoli.processes")
    except Exception as exc:  # noqa: BLE001
        _restore_ecoli(saved_real, fork_repo)
        raise InjectionError(
            f"could not import 'ecoli.processes' from fork {fork_repo!r}: {exc}")

    registry = getattr(fork_mod, "process_registry", None)
    if registry is None or not hasattr(registry, "access"):
        _restore_ecoli(saved_real, fork_repo)
        raise InjectionError(
            f"fork {fork_repo!r} ecoli.processes has no process_registry.access")

    # Done with the fork's ecoli.*; restore the real vEcoli modules.
    # Class objects from the fork survive via the registry handle (and later via
    # _fork_class_cache populated in resolve_injections).
    _restore_ecoli(saved_real, fork_repo)
    return registry


def _restore_ecoli(saved_real: dict, fork_repo: str) -> None:
    """Evict fork ecoli.* from sys.modules, restore real ones, remove fork from path."""
    for k in [k for k in sys.modules if k == "ecoli" or k.startswith("ecoli.")]:
        del sys.modules[k]
    sys.modules.update(saved_real)
    try:
        sys.path.remove(fork_repo)
    except ValueError:
        pass


def _force_fork_class(fork_repo: str, cls: type) -> type:
    """Return the FORK's version of a class, defeating the installed-vEcoli shadow.

    ``registry.access(name)`` keeps the INSTALLED class for names shared with the
    installed vEcoli (``_fork_registry``'s idempotent registration skips re-adding
    the fork's). For a process that exists in BOTH — e.g. the antibiotic subsystem
    (``antibiotic-transport-odeint``, ``permeability``, ...) — the installed class
    can carry a DIFFERENT store structure and crash at runtime. Re-import the class
    from the fork's own module and return that. No-op if ``cls`` is already the
    fork's; falls back to ``cls`` if the fork copy can't be resolved."""
    fork_abs = os.path.abspath(os.path.expanduser(fork_repo))
    try:
        if os.path.abspath(inspect.getfile(cls)).startswith(fork_abs):
            return cls  # already the fork's
    except Exception:  # noqa: BLE001 — builtins / no source file
        pass
    module, qualname = cls.__module__, cls.__qualname__
    if fork_repo not in sys.path:
        sys.path.insert(0, fork_repo)
    saved_real: dict[str, object] = {}
    for k in [k for k in sys.modules if k == "ecoli" or k.startswith("ecoli.")]:
        mod = sys.modules.pop(k)
        if not os.path.abspath(getattr(mod, "__file__", "") or "").startswith(fork_abs):
            saved_real[k] = mod
    try:
        with _idempotent_registration():
            fork_mod = importlib.import_module(module)
        obj = fork_mod
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    except Exception:  # noqa: BLE001 — fall back to the shadowed class
        return cls
    finally:
        _restore_ecoli(saved_real, fork_repo)


@contextlib.contextmanager
def _fork_module_shadow(fork_repo: str):
    """Import ``ecoli.*`` from the FORK for the duration of the block.

    The module-level counterpart to :func:`_force_fork_class`, which defeats the
    same installed-vEcoli shadow for a class. ``_fork_registry`` restores the
    installed ``ecoli.*`` as soon as it has the registry handle, so by the time
    :func:`resolve_injections` runs, a bare ``import ecoli.library.sim_data``
    resolves to site-packages (``vecoli``), NOT to ``fork_repo``.
    """
    fork_abs = os.path.abspath(os.path.expanduser(fork_repo))
    if fork_repo not in sys.path:
        sys.path.insert(0, fork_repo)
    saved_real: dict[str, object] = {}
    for k in [k for k in sys.modules if k == "ecoli" or k.startswith("ecoli.")]:
        mod = sys.modules.pop(k)
        if not os.path.abspath(getattr(mod, "__file__", "") or "").startswith(fork_abs):
            saved_real[k] = mod
    try:
        with _idempotent_registration():
            yield
    finally:
        _restore_ecoli(saved_real, fork_repo)


def build_fork_config(fork_repo: str, sim_data_path: str, name: str) -> dict:
    """Build a fork process's config from the FORK's own ``LoadSimData``.

    The faithful, complete config source for a converted/swapped vEcoli process:
    vEcoli's ``ecoli.library.sim_data.LoadSimData(sim_data_path).get_config_by_name``
    supplies every parameter the real process needs (where v2ecoli's reimplemented
    getter can drift). Raises if the fork has no config-getter for ``name``.

    ⚠ The import MUST happen under :func:`_fork_module_shadow`. Without it the
    name ``ecoli.library.sim_data`` resolves to the INSTALLED vEcoli, so a config
    getter that the fork has extended silently yields the installed vEcoli's
    smaller dict — every fork-only key is absent and the process falls back to its
    own class default. That failure is silent: the process still builds, still
    runs, and produces a plausible-looking result computed with the wrong config.
    """
    import importlib
    fork_abs = os.path.abspath(os.path.expanduser(fork_repo))
    # Does this fork ship a config source AT ALL? Decide that from the fork's own
    # files, BEFORE importing, so the outcome does not depend on whether an
    # unrelated vEcoli happens to be installed in the environment. Without this
    # check a fork with no ``sim_data`` module behaves two different ways: with a
    # vEcoli installed the import succeeds, resolves outside the fork and the
    # guard below kills the run; with none it raises ModuleNotFoundError and the
    # caller falls back to the default config. Same fork, same call, opposite
    # outcomes.
    has_module = any(
        os.path.exists(os.path.join(fork_abs, "ecoli", "library", leaf))
        for leaf in ("sim_data.py", "sim_data"))
    if not has_module:
        raise ModuleNotFoundError(
            f"fork {fork_repo!r} has no ecoli/library/sim_data module; it cannot "
            "configure processes. Falling back to the default config.")
    with _fork_module_shadow(fork_repo):
        sim_data_mod = importlib.import_module("ecoli.library.sim_data")
        mod_file = os.path.abspath(getattr(sim_data_mod, "__file__", "") or "")
        if not mod_file.startswith(fork_abs):
            raise InjectionError(
                f"{name!r}: ecoli.library.sim_data resolved to {mod_file!r}, "
                f"outside fork {fork_repo!r}; the config would be built from the "
                "installed vEcoli and silently omit fork-only keys.")
        loader = sim_data_mod.LoadSimData(sim_data_path=sim_data_path)
        return dict(loader.get_config_by_name(name))


def build_native_redux_config(sim_data_path: str) -> dict:
    """Build the ``ecoli-metabolism-redux`` config from the v2ecoli-NATIVE bundle,
    with NO dependency on the vEcoli-private fork — the fork-free twin of
    ``build_fork_config(fork_repo, fork_sim_data, "ecoli-metabolism-redux")``.

    ``build_fork_config`` sources the redux config from the FORK's own
    ``LoadSimData.get_metabolism_redux_config`` (Unum scalars + sim_data-bound
    callables), which ``adapt_metabolism_redux_config`` then resolves into the
    native ``MetabolismReduxClassic`` surface (bare floats + a
    ``homeostatic_concentrations`` dict + an ``initial_exchange_molecules`` list).
    A vEcoli-free config carries no ``fork_sim_data`` and must never touch the
    fork, so this reads v2ecoli's OWN ``LoadSimData.get_metabolism_redux_config``
    from the native violacein bundle instead.

    ⚠ v2ecoli's native getter SERIALISES the three callables the adapter needs
    (``exchange_data_from_media`` / ``concentration_updates`` /
    ``get_biomass_as_concentrations``) into ``{"_function","_data"}`` dicts, so it
    re-injects the LIVE native objects here — the same objects the fork exposes,
    read off the native sim_data — so ``adapt_metabolism_redux_config`` (the
    validated surface bridge) runs unchanged and fork-free. Scalars come back a
    Unum/pint MIX; ``_asnumber`` handles both.
    """
    import pbg_v2ecoli  # noqa: F401 — data-root redirect for the native bundle
    from v2ecoli.library.sim_data import LoadSimData
    loader = LoadSimData(sim_data_path=sim_data_path)
    sd = loader.sim_data
    cfg = dict(loader.get_metabolism_redux_config())
    # Replace the serialised (_function/_data) callables with the LIVE native
    # objects adapt_metabolism_redux_config resolves against. These are the same
    # methods the fork's redux config carries, taken off v2ecoli's own sim_data.
    cfg["exchange_data_from_media"] = sd.external_state.exchange_data_from_media
    cfg["concentration_updates"] = sd.process.metabolism.concentration_updates
    cfg["get_biomass_as_concentrations"] = sd.mass.getBiomassAsConcentrations
    # Kinetic constraints (kcat capacity bounds): re-inject the LIVE native
    # get_kinetic_constraints callable + its reaction/enzyme/substrate id lists
    # off v2ecoli's OWN sim_data (fork-free), so adapt_metabolism_redux_config can
    # BRIDGE them instead of deferring kinetics off. Without kcat bounds the native
    # LP over-produces general flux, skewing the enzyme/TRP trajectory that feeds
    # the violacein ODE (the KPI ran ~2-3x high on the kinetics-off arm).
    _m = sd.process.metabolism
    cfg["get_kinetic_constraints"] = _m.get_kinetic_constraints
    cfg["kinetic_constraint_reactions"] = list(_m.kinetic_constraint_reactions)
    cfg["kinetic_constraint_enzymes"] = list(_m.kinetic_constraint_enzymes)
    cfg["kinetic_constraint_substrates"] = list(_m.kinetic_constraint_substrates)
    return cfg


def _asnumber(value, unit=None):
    """Bare-float magnitude of a possibly-Unum / possibly-pint value.

    A vEcoli ``Unum`` exposes ``.asNumber([unit])`` (magnitude in its own units
    when called bare, or converted to ``unit``). A ``pint`` Quantity exposes
    ``.magnitude`` (and ``.to(unit)``). v2ecoli's native ``get_metabolism_redux_config``
    returns a MIX — most quantities stay Unum (the vendored ``wholecell.utils.units``),
    but a few top-level constants (``avogadro``, ``cell_density``) are pint via
    ``unum_to_pint`` — so the fork-free path must de-unit both. A plain float
    passes through.
    """
    if hasattr(value, "asNumber"):  # Unum (fork + most native quantities)
        return value.asNumber() if unit is None else value.asNumber(unit)
    if hasattr(value, "magnitude"):  # pint Quantity (native top-level constants)
        # ``unit`` here is only ever a fork Unum unit (the mM conversion inside
        # adapt_metabolism_redux_config), and the pint values this branch sees
        # (avogadro, cell_density) are de-unit'd WITHOUT a target unit, so a
        # bare-magnitude read in the Quantity's own (native) units is correct.
        return float(value.magnitude)
    return value


def adapt_metabolism_redux_config(config_dict: dict) -> dict:
    """Translate the FORK's ``get_metabolism_redux_config`` output into the plain
    config the NATIVE ``MetabolismReduxClassic`` process declares.

    ``build_fork_config`` returns vEcoli's OWN redux config, whose surface is the
    fork ``MetabolismReduxClassic.__init__`` contract: ``Unum`` scalars
    (``avogadro`` [1/mol], ``cell_density`` [g/L], ``dark_atp`` [mmol/g],
    ``non_growth_associated_maintenance`` [mmol/g/h]) and sim_data-bound CALLABLES
    (``concentration_updates`` + ``get_biomass_as_concentrations`` for the
    homeostatic objective; ``exchange_data_from_media`` for the initial exchange
    set). The native port instead declares a de-unit'd, pre-resolved surface
    (bare floats; a ``homeostatic_concentrations`` dict; an
    ``initial_exchange_molecules`` list) — see its module docstring's "Sim_data-
    driven callables replaced by plain config" adaptation.

    This adapter resolves that surface by REPRODUCING the fork ``__init__``
    verbatim (``ecoli/processes/metabolism_redux_classic.py`` lines ~224-240):

        exchanges  = exchange_data_from_media(media_id)["externalExchangeMolecules"]
        conc_dict  = concentration_updates.concentrations_based_on_nutrients(media_id)
        conc_dict.update(get_biomass_as_concentrations(doubling_time))
        homeostatic_metabolites, homeostatic_concs = conc_dict keys / .asNumber(mM)

    Values already plain (a synthetic/test config, or a config with no fork
    getters) pass through unchanged. Kinetics are DEFERRED (see below).

    ⚠ DEFERRED — kinetic constraints. The fork's ``get_kinetic_constraints`` is a
    sim_data-bound callable that both CONSUMES and RETURNS ``Unum`` quantities,
    while the native port passes/expects bare mM floats. Faithfully bridging it
    needs a two-sided unit wrap (attach units on input, strip on output) that is
    error-prone and orthogonal to the violacein readout (the violacein pathway
    reactions are NOT among ``kinetic_constraint_reactions`` — verified — so
    kinetics never touches VIOLACEIN[c]). Rather than risk wrong kinetic numbers,
    kinetics is turned OFF here via the native port's own documented off-switch
    (``get_kinetic_constraints=None`` + empty kinetic reaction/enzyme/substrate
    lists → the ``USE_KINETICS`` off path). This lowers general-flux fidelity vs.
    the fork (kcat capacity bounds are dropped) but is honest and leaves the
    homeostatic + maintenance + steady-state LP — the part that governs whether
    any metabolite (violacein included) is produced — intact.
    """
    cfg = dict(config_dict)
    # De-unit the scalar parameters to the native port's fixed-float convention
    # (its config_schema declares each as a bare float in exactly these units).
    for key in ("avogadro", "cell_density", "dark_atp",
                "non_growth_associated_maintenance", "cell_dry_mass_fraction"):
        if key in cfg:
            cfg[key] = _asnumber(cfg[key])

    media_id = cfg.get("media_id")
    # initial_exchange_molecules <- exchange_data_from_media(media_id), plus the
    # per-tick UPTAKE set the excluded `exchange_data` step would have written to
    # environment.exchange_data (see _exchange_data_seed below).
    edm = cfg.get("exchange_data_from_media")
    if callable(edm) and media_id is not None:
        ex = edm(media_id)
        if "initial_exchange_molecules" not in cfg:
            cfg["initial_exchange_molecules"] = list(ex["externalExchangeMolecules"])
        # The config excludes the `exchange_data` step (matching the fork), so
        # nothing populates environment.exchange_data — leaving the redux LP with
        # NO allowed nutrient uptake and rendering it infeasible. For a fixed
        # medium that step's output is constant, so seed the store with its t=0
        # value: the media's unconstrained + constrained importers. The native LP
        # uses only the KEYS (the allowed-uptake set), so the constrained rate
        # magnitudes are de-unit'd to bare floats for a clean map[float] store.
        unconstrained = [str(m) for m in ex["importUnconstrainedExchangeMolecules"]]
        constrained = {
            str(m): float(_asnumber(v))
            for m, v in dict(ex["importConstrainedExchangeMolecules"]).items()}
        cfg["_exchange_data_seed"] = {
            ("environment", "exchange_data", "unconstrained"): unconstrained,
            ("environment", "exchange_data", "constrained"): constrained,
        }

    # homeostatic_concentrations <- concentrations_based_on_nutrients(media_id)
    #                               U get_biomass_as_concentrations(doubling_time)
    if "homeostatic_concentrations" not in cfg:
        cu = cfg.get("concentration_updates")
        gbac = cfg.get("get_biomass_as_concentrations")
        doubling_time = cfg.get("doubling_time")
        if cu is not None and callable(getattr(cu, "concentrations_based_on_nutrients", None)):
            conc_dict = dict(cu.concentrations_based_on_nutrients(media_id))
            if callable(gbac) and doubling_time is not None:
                conc_dict.update(gbac(doubling_time))
            # mM = mmol/L; convert each Unum concentration to a bare mM float.
            try:
                from wholecell.utils import units as _u  # fork units
                mM = _u.mmol / _u.L
                cfg["homeostatic_concentrations"] = {
                    met: _asnumber(conc, mM) for met, conc in conc_dict.items()}
            except Exception:  # noqa: BLE001 — units unavailable: bare magnitudes
                cfg["homeostatic_concentrations"] = {
                    met: _asnumber(conc) for met, conc in conc_dict.items()}

    # Kinetics ON — BRIDGE the native get_kinetic_constraints (Unum in/out) onto
    # the native port's bare-float surface (supersedes the former off-switch
    # deferral). The port's kinetics block passes bare concentrations
    # (counts_to_molar * counts, in mmol/L = CONC_UNITS) and forms
    # (timestep_s * result); v2ecoli's callable expects Unum concentrations and
    # returns Unum (KINETIC_CONSTRAINT_CONC_UNITS/s = umol/L/s). So attach mmol/L
    # on input and strip mmol/L/s on output — reproducing the fork's
    # metabolism_redux_classic unit handling exactly (gated by an exact-match test).
    _native_gkc = cfg.get("get_kinetic_constraints")
    if callable(_native_gkc):
        try:
            from wholecell.utils import units as _u
            _CONC = _u.mmol / _u.L
            _RATE = _CONC / _u.s

            def _bridged_get_kinetic_constraints(enzyme_conc, substrate_conc,
                                                 _gkc=_native_gkc, _c=_CONC, _r=_RATE):
                # enzyme_conc/substrate_conc arrive as bare-float arrays already in
                # mmol/L; _c * arr attaches the unit (Unum on the LEFT so its
                # __mul__ wraps the array — ndarray*Unum lets numpy win and drops
                # the unit). The callable's own .asNumber(umol/L) rescales.
                out = _gkc(_c * enzyme_conc, _c * substrate_conc)
                return out.asNumber(_r)

            cfg["get_kinetic_constraints"] = _bridged_get_kinetic_constraints
        except Exception:  # noqa: BLE001 — units unavailable: safe fall back to off
            cfg["get_kinetic_constraints"] = None
            cfg["kinetic_constraint_reactions"] = []
            cfg["kinetic_constraint_enzymes"] = []
            cfg["kinetic_constraint_substrates"] = []
    else:
        cfg["kinetic_constraint_reactions"] = []
        cfg["kinetic_constraint_enzymes"] = []
        cfg["kinetic_constraint_substrates"] = []

    # Violacein pathway ODE + export-flux pin (native port of the fork's
    # violacein mechanism). Enable it by DATASET MEMBERSHIP -- exactly as the
    # fork gates the same machinery: the violacein new-gene enzymes register as
    # "NG-"-prefixed catalysts only when the violacein dataset is loaded, so a
    # plain (non-violacein) bundle leaves include_violacein_reactions False and
    # the native redux is byte-for-byte untouched. On the violacein bundle this
    # activates the pathway ODE and the VIOLACEIN[c]-export pin so the
    # native-candidate arm produces violacein faithfully.
    if "include_violacein_reactions" not in cfg:
        cfg["include_violacein_reactions"] = any(
            str(c).startswith("NG-") for c in cfg.get("catalyst_ids", [])
        )

    # Seed a VIOLACEIN key into the environment.exchange store when the pathway
    # is active. environment.exchange is a map[float] store initialised from the
    # cache's initial_state with ONLY the media's ~87 external (periplasmic)
    # molecules; a bare-float map leaf ACCUMULATES (state + update), which
    # updates existing keys but does not add the redux process's brand-new
    # cytoplasmic VIOLACEIN[c] secretion key -- so its export flux never reaches
    # environment.exchange and the ExchangeFluxListener's violacein_exchange leaf
    # stays 0. Seeding the (compartment-stripped, matching the redux writer)
    # "VIOLACEIN" key to 0.0 lets the per-tick accumulate land, so the KPI leaf
    # carries the real secreted-violacein flux. Gated on the pathway being active
    # so non-violacein runs are untouched.
    if cfg.get("include_violacein_reactions"):
        seed = cfg.get("_exchange_data_seed") or {}
        seed[("environment", "exchange", "VIOLACEIN")] = 0.0
        cfg["_exchange_data_seed"] = seed
    return cfg


def _compose_store_path(base: list, rel) -> list:
    """Resolve a vivarium sub-path ``rel`` against an accumulated ``base``.

    Applies vivarium path semantics for a SINGLE-CELL mount (the harness injects
    into the top-level ``ecoli_baseline``, not a spatial ``agents/<id>``
    compartment):
      - ``".."`` pops one segment off ``base``; when ``base`` is already at root
        it is the *phantom agent-compartment hop* the spatial config assumes but
        the single-cell composite lacks — consumed as a no-op so the leaf lands
        root-relative (e.g. ``species.bulk: ["..","bulk"] -> ["bulk"]``).
      - ``"null"`` / ``None`` are grouping markers that contribute no store
        segment (they appear as ``_path: ["null"]``) — skipped.
      - any other segment is appended.
    """
    path = list(base)
    for seg in list(rel):
        if seg == "..":
            if path:
                path.pop()
        elif seg in ("null", None):
            continue
        else:
            path.append(seg)
    return path


def _has_scatter(node) -> bool:
    """True if a topology subtree wires leaves ACROSS stores (``..``-relative
    paths or a ``["null"]`` grouping ``_path``) rather than all under one base.

    A non-scattered nested port (metabolism's ``environment``:
    ``{"_path": ("environment",), "exchange": ("exchange",)}``) mounts its WHOLE
    port subtree at the base store, so subports the topology doesn't name (e.g.
    ``environment.media_id`` / ``environment.exchange_data``, read but unmapped)
    still resolve. A scattered port (vEcoli's antibiotic subsystem) needs each
    named leaf wired individually."""
    if isinstance(node, dict):
        p = node.get("_path")
        if p is not None and any(s in ("null", None) for s in list(p)):
            return True
        return any(_has_scatter(v) for k, v in node.items() if k != "_path")
    return ".." in list(node)


def translate_vivarium_topology(topo: dict, _base: list | None = None) -> dict:
    """Translate a vivarium-1.0 topology to a process-bigraph wires tree.

    A flat entry ``port: (a, b)`` becomes ``port: [a, b]``. A nested port with a
    real ``_path`` and NO cross-store scatter collapses to that base (so the
    whole port subtree mounts there — preserving vivarium's convention that
    unmapped subports resolve under the base; this is metabolism's ``environment``
    and matches the pre-existing behavior exactly). A *scattered* nested port —
    vEcoli's antibiotic subsystem, whose sub-ports fan out across stores via
    ``..``-relative paths (e.g. ``mecillinam.species.bulk -> ["..","bulk"]``,
    ``mecillinam.reaction_parameters.decay.kf ->
    ["..","kinetic_parameters","mecillinam","decay_kf"]``) — is preserved as a
    nested wires tree so ``make_edge``/``list_paths`` wires each leaf to its real
    store. Collapsing a scattered port to its ``_path`` base silently dropped
    every leaf (the bulk store never reached the process → ``bulk["id"]`` on a
    bare list). Each leaf resolves to a root-relative path via
    :func:`_compose_store_path` (the single-cell mount consumes the phantom
    agent-compartment ``..``).
    """
    base = list(_base or [])
    out: dict = {}
    for port, path in dict(topo).items():
        if port == "_path":
            continue
        if isinstance(path, dict):
            real_path = None
            if "_path" in path and not any(s in ("null", None) for s in list(path["_path"])):
                real_path = path["_path"]
            if real_path is not None and not _has_scatter(path):
                # Simple subtree (metabolism): collapse to the base store.
                out[port] = _compose_store_path(base, real_path)
            else:
                # Scattered subsystem (antibiotic): wire each leaf individually.
                sub_base = _compose_store_path(base, real_path) if real_path is not None else base
                out[port] = translate_vivarium_topology(path, _base=sub_base)
        else:
            out[port] = _compose_store_path(base, path)
    return out


def _iter_leaf_paths(topology):
    """Yield ``(port_key_path, store_path)`` for every leaf in a (possibly
    nested) translated topology. ``store_path`` is a list; ``port_key_path`` is
    the tuple of nested port names leading to it. Flat entries yield one leaf."""
    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, prefix + (k,))
        else:
            yield prefix, node
    yield from walk(topology, ())


def _topology_store_roots(topology) -> set:
    """Root store names touched by any leaf of a translated topology."""
    return {p[0] for _, p in _iter_leaf_paths(topology) if p}


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; ``over`` wins. Returns a new dict."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config_initial_state(fork_repo: str, config: dict) -> dict:
    """Resolve a vEcoli config's initial state for INJECTED stores.

    Honors the same two knobs ``EcoliSim`` does — ``initial_state`` (inline) and
    ``initial_state_overrides`` (names of JSON files under the fork's
    ``data/``) — merged in the same order (overrides on top of inline). This is
    what gives an injected subsystem its real starting values (e.g. the
    cell-wall model's ``murein_state`` counts) instead of bare schema defaults,
    so the subsystem's own first-update logic can build the rest (e.g. PBPBinding
    samples the murein lattice when ``wall_state.lattice`` is None). The big
    ``initial_state_file`` (bulk/unique) is NOT loaded here — matched bulk is
    handled by --match-initial-state; this resolves only the small,
    subsystem-specific stores the config declares. Best-effort: a missing
    override file is logged and skipped, never fatal."""
    merged: dict = dict(config.get("initial_state") or {})
    for name in (config.get("initial_state_overrides") or []):
        rel = name if name.endswith(".json") else f"{name}.json"
        candidates = [
            os.path.join(fork_repo, "data", rel),
            os.path.join(fork_repo, "data", "overrides", os.path.basename(rel)),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            print(f"[inject] initial_state override {name!r} not found under "
                  f"{fork_repo}/data — skipping")
            continue
        try:
            with open(path) as fh:
                merged = _deep_merge(merged, json.load(fh))
        except Exception as e:  # noqa: BLE001
            print(f"[inject] initial_state override {name!r} unreadable "
                  f"({type(e).__name__}: {e}) — skipping")
    return merged


def _resolve_param_store_seeds(fork_repo: str, mapping: dict) -> dict:
    """Resolve fork ``param_store`` values into ``store_path_tuple -> pint.
    Quantity`` seeds, for a config-declared process's initial values the
    single-cell candidate has no upstream process to compute (e.g. a spatial
    shape process's periplasm/cytoplasm volume — without a shape process the
    scaffolded default reads 0 and a process dividing by it raises).
    ``mapping`` is ``{dotted store path: dotted param_store key}`` (both
    ``.``-joined — JSON-safe; split here). Config-driven: this function knows
    nothing about any particular fork's molecules or processes, only how to
    read its ``param_store`` by an arbitrary key path. Activates the fork the
    same way :func:`_force_fork_class` does (evict installed ecoli, fork
    first on path, idempotent registration, restore)."""
    if not mapping:
        return {}
    fork_abs = os.path.abspath(os.path.expanduser(fork_repo))
    if fork_repo not in sys.path:
        sys.path.insert(0, fork_repo)
    saved_real: dict[str, object] = {}
    for k in [k for k in sys.modules if k == "ecoli" or k.startswith("ecoli.")]:
        mod = sys.modules.pop(k)
        if not os.path.abspath(getattr(mod, "__file__", "") or "").startswith(fork_abs):
            saved_real[k] = mod
    seeds: dict = {}
    try:
        with _idempotent_registration():
            from ecoli.library.parameters import param_store
        for store_path, ps_key in mapping.items():
            try:
                seeds[tuple(store_path.split("."))] = param_store.get(
                    tuple(ps_key.split(".")))
            except Exception as e:  # noqa: BLE001 — skip a param the fork lacks
                print(f"[inject] shape_seed_param_store: {store_path!r} <- "
                      f"{ps_key!r} unavailable ({type(e).__name__}: {e})")
    except Exception as e:  # noqa: BLE001 — never block injection on this
        print(f"[inject] shape_seed_param_store unavailable ({type(e).__name__}: {e})")
    finally:
        _restore_ecoli(saved_real, fork_repo)
    return seeds


def _resolve_literal_seeds(mapping: dict) -> dict:
    """Resolve ``{dotted store path: {"magnitude": num, "units": "expr"}}``
    into ``store_path_tuple -> pint.Quantity`` seeds, for a config-declared
    literal value (e.g. a rate constant the config's own unit tag can't
    resolve). Uses vivarium's shared unit registry — pint expression parsing
    is process-wide, not fork-specific, so (unlike
    :func:`_resolve_param_store_seeds`) no fork activation is needed."""
    if not mapping:
        return {}
    seeds: dict = {}
    try:
        from vivarium.library.units import units
    except Exception as e:  # noqa: BLE001
        print(f"[inject] shape_seed_literal unavailable ({type(e).__name__}: {e})")
        return seeds
    for store_path, spec in mapping.items():
        try:
            seeds[tuple(store_path.split("."))] = spec["magnitude"] * units(spec["units"])
        except Exception as e:  # noqa: BLE001
            print(f"[inject] shape_seed_literal: {store_path!r} unresolvable "
                  f"({type(e).__name__}: {e})")
    return seeds


# Matches a vEcoli-serialized-value TAG, e.g. ``!ParameterSerializer[path]`` or
# ``!units[0 count]`` (see ``ecoli.library.serialize`` in the vEcoli fork).
_SERIALIZER_TAG_RE = re.compile(r"^!\w+\[.*\]$", re.S)


def _fork_repo_from_env() -> str:
    """Best-effort vEcoli fork checkout for resolving ``!ParameterSerializer``
    tags on the NATIVE path, read from the same env vars a comparison run
    already sets (``$V2E_VECOLI_DIR`` / ``$VECOLI_REPO``). Empty string when
    neither is set."""
    return os.environ.get("V2E_VECOLI_DIR") or os.environ.get("VECOLI_REPO", "")


def _iter_serializer_tags(obj):
    """Yield every serializer-tag string (``!Name[...]``) anywhere in ``obj``."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_serializer_tags(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_serializer_tags(v)
    elif isinstance(obj, str) and _SERIALIZER_TAG_RE.match(obj):
        yield obj


def _deserialize_config_values(obj, fork_repo: str):
    """Deserialize vEcoli-serialized config values so a pbg_native injection
    process gets real values, on BOTH the fork-wrapped AND the native path:
    ``!ParameterSerializer[path]`` -> the FORK ``param_store`` Quantity,
    ``!units[...]`` -> a pint Quantity (via vivarium's ``deserialize_value``
    dispatcher). Non-tagged values pass through unchanged.

    THE GAP THIS CLOSES. A fork-wrapped (``vivarium_1``) process has its
    serializer tags resolved by the bridge's ``initial_state()`` overlay, so it
    never sees a raw tag. A ``pbg_native`` process does NOT go through that
    overlay -- it reads ``spec["config"]`` directly. When a genuine (non-baked)
    vEcoli config is translated straight onto native ``ecoli_baseline`` (e.g.
    ``--from-vecoli-config ... --composite ecoli_baseline``), its
    ``process_configs`` still carry raw ``!ParameterSerializer[...]`` tag
    strings, which then survive into the native process (e.g. a native gillespie
    port's ``kf > 0`` raises ``TypeError: '>' not supported between instances of
    'str' and 'int'``; a native transport port's ``_param_magnitude`` never
    handles a raw tag). The fix belongs in this SHARED gate (not a per-process
    band-aid), applied to every pbg_native config below in
    :func:`resolve_injections`.

    A ``!ParameterSerializer[...]`` tag resolves ONLY against the fork's own
    ``ecoli.library.serialize`` param_store -- the fork provides the calibrated
    VALUES while the native process provides the LOGIC; this is a legitimate
    native run. The fork is located via ``fork_repo`` (else ``$V2E_VECOLI_DIR``
    / ``$VECOLI_REPO``) and activated the same way
    :func:`_resolve_param_store_seeds` does (evict installed ecoli, fork first on
    path, idempotent registration, restore) so the FORK's param_store -- not an
    installed ``ecoli`` shadow -- answers the lookup. ``!units[...]`` (and other
    non-parameter tags) resolve via vivarium's own dispatcher and need no fork.

    RAISES a clear ``ValueError`` -- naming the offending tag(s) and how to fix
    it -- when a ``!ParameterSerializer[...]`` tag is present but no fork is
    reachable (or the fork cannot resolve it), instead of letting a raw tag
    string reach a downstream numeric comparison. A config carrying NO serializer
    tags is a true no-op and never touches the fork."""
    tags = list(_iter_serializer_tags(obj))
    if not tags:
        # Pre-baked config (the common native case): nothing to resolve, and the
        # fork must never be touched.
        return obj

    param_tags = sorted({t for t in tags if t.startswith("!ParameterSerializer[")})
    effective_fork = fork_repo or _fork_repo_from_env()

    if param_tags and not effective_fork:
        raise ValueError(
            f"native injection config carries unresolved vEcoli serializer "
            f"tag(s) {param_tags!r}, and no vEcoli fork checkout is available "
            f"to resolve them. A !ParameterSerializer[...] tag is resolved "
            f"against the fork's ecoli.library param_store (the fork supplies "
            f"the calibrated VALUES; the native process supplies the LOGIC). "
            f"Either set $V2E_VECOLI_DIR / $VECOLI_REPO to a vEcoli fork "
            f"checkout, or pre-bake the config to plain floats/Quantities (as "
            f"the repo's *_vecoli_free.json configs do).")

    if not effective_fork:
        # Only fork-independent tags (e.g. !units[...]) remain: vivarium's own
        # dispatcher resolves them without any fork.
        from vivarium.core.serialize import deserialize_value

        def _walk_nofork(o):
            if isinstance(o, dict):
                return {k: _walk_nofork(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_walk_nofork(v) for v in o]
            if isinstance(o, str) and _SERIALIZER_TAG_RE.match(o):
                try:
                    return deserialize_value(o)
                except Exception:  # noqa: BLE001 — non-fatal: keep raw units tag
                    return o
            return o

        return _walk_nofork(obj)

    # A fork IS reachable: activate it (evict installed ecoli, fork first on
    # path, restore after) so ``ecoli.library.serialize`` resolves against the
    # FORK param_store, then walk + deserialize every tag.
    fork_abs = os.path.abspath(os.path.expanduser(effective_fork))
    if effective_fork not in sys.path:
        sys.path.insert(0, effective_fork)
    saved_real: dict[str, object] = {}
    for k in [k for k in sys.modules if k == "ecoli" or k.startswith("ecoli.")]:
        mod = sys.modules.pop(k)
        if not os.path.abspath(getattr(mod, "__file__", "") or "").startswith(fork_abs):
            saved_real[k] = mod
    try:
        with _idempotent_registration():
            import ecoli.library.serialize as _fs  # FRESHLY imported FORK serializers
            from vivarium.core.registry import Serializer
            from vivarium.core.serialize import deserialize_value
        # Instantiate the FORK's own Serializer subclasses (their module-level
        # param_store is now the fork's). Dispatch !-values through these FIRST so
        # !ParameterSerializer[...] resolves against the FORK param_store, NOT the
        # installed ecoli's (whose global serializer is what vivarium's
        # deserialize_value would otherwise use). deserialize_value is the
        # fallback for the rest (e.g. !units[...], which is fork-independent).
        fork_serializers = []
        for _name in dir(_fs):
            _obj = getattr(_fs, _name)
            if isinstance(_obj, type) and issubclass(_obj, Serializer) and _obj is not Serializer:
                try:
                    fork_serializers.append(_obj())
                except Exception:  # noqa: BLE001 — skip a serializer that won't instantiate
                    pass

        def _deser(v):
            for s in fork_serializers:
                try:
                    if s.can_deserialize(v):
                        return s.deserialize(v)
                except Exception:  # noqa: BLE001 — try the next serializer
                    continue
            try:
                return deserialize_value(v)
            except Exception:  # noqa: BLE001 — vivarium can't parse it either
                pass
            if v.startswith("!ParameterSerializer["):
                # A param tag the fork could NOT resolve (e.g. a param_store key
                # this checkout lacks). Never pass it downstream as a raw string.
                raise ValueError(
                    f"native injection config: {v!r} could not be resolved "
                    f"against the vEcoli fork at {effective_fork!r} (its "
                    f"ParameterSerializer/param_store has no such entry). It "
                    f"must not reach a native process as a raw tag string; "
                    f"check the fork checkout or pre-bake the value.")
            return v  # a non-param tag vivarium couldn't parse: leave as-is

        def _walk(o):
            if isinstance(o, dict):
                return {k: _walk(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_walk(v) for v in o]
            if isinstance(o, str) and _SERIALIZER_TAG_RE.match(o):
                return _deser(o)
            return o

        return _walk(obj)
    except ValueError:
        raise  # a clear, actionable error — must NOT be swallowed
    except Exception as e:  # noqa: BLE001 — never block injection on an optional deserialize
        print(f"[inject] config deserialize unavailable ({type(e).__name__}: {e})")
        return obj
    finally:
        _restore_ecoli(saved_real, effective_fork)


# ---------------------------------------------------------------------------
# Native-injection registry (Arm 3 seam).
#
# Replaces the former hardcoded native-class map (name -> (module, qualname)).
# v2ecoli DEFINES the registry and ships it EMPTY: the native process classes
# live DOWNSTREAM (sms-ecoli), which populates this registry at import time via
# :func:`register_native_injection`. Keeping the classes out of the wheel is the
# whole point — the wheel-shipped resolver never imports the downstream process
# package, so it can no longer be shadowed by whichever copy sits on sys.path
# (the root of the empty-config one-tick collapse).
#
# "Fail loud by construction" (Fable): a name that is REGISTERED but whose class
# fails to import RAISES (:func:`_resolve_native_injection`), and a built config
# missing any ``required_config_keys`` RAISES (in :func:`resolve_injections`) —
# there is no silent fallback to an empty/default config.
# ---------------------------------------------------------------------------
class NativeInjection:
    """One registry entry: how to resolve + configure a native injected process."""
    __slots__ = ("module_path", "class_name", "topology",
                 "required_config_keys", "config_builder")

    def __init__(self, module_path, class_name, topology=None,
                 required_config_keys=(), config_builder=None):
        self.module_path = module_path
        self.class_name = class_name
        # Topology: either a static wires dict, or a callable
        # ``(name, config_dict) -> wires dict`` (e.g. a drug-parametric native
        # topology). Applied in resolve_injections; WINS over the config topology
        # for a native class (whose flat ports differ from the fork's nested ones).
        self.topology = topology
        # Keys the built config MUST carry non-empty for this process to run.
        self.required_config_keys = tuple(required_config_keys)
        # ``(native_sim_data_path) -> config dict``: builds the native config off
        # this run's own bundle sim_data when the config would otherwise be empty
        # (the fork-free swap case). May stash an ``_exchange_data_seed`` mapping.
        self.config_builder = config_builder


_NATIVE_INJECTION_REGISTRY: dict[str, NativeInjection] = {}


def register_native_injection(name, module_path, class_name, *, topology=None,
                              required_config_keys=(), config_builder=None):
    """Register a native process port for a vEcoli add/swap process ``name``.

    Populated DOWNSTREAM (sms-ecoli) so the native process classes stay out of
    the v2ecoli wheel. ``module_path``/``class_name`` are imported LAZILY (only
    when a config actually injects ``name``, via absolute import in
    :func:`_import_class`), so registering a process whose module is absent in
    this environment costs nothing until it is used — and then fails loud rather
    than silently falling back. Re-registering a ``name`` replaces its entry.
    """
    _NATIVE_INJECTION_REGISTRY[name] = NativeInjection(
        module_path, class_name, topology=topology,
        required_config_keys=required_config_keys, config_builder=config_builder)


def _native_injection_entry(name):
    """The registry entry for ``name``, or ``None`` if ``name`` is not registered."""
    return _NATIVE_INJECTION_REGISTRY.get(name)


def _resolve_native_injection(name):
    """Resolve the native process CLASS for ``name`` from the registry.

    Returns ``None`` when ``name`` is NOT registered (the caller then falls
    through to the fork path). When ``name`` IS registered but
    ``module_path:class_name`` cannot be imported, RAISES :class:`InjectionError`
    — the fail-loud contract. The former hardcoded map swallowed this import error
    and fell back to an empty/default config, which is exactly the tick-0 collapse
    this replaces.
    """
    entry = _NATIVE_INJECTION_REGISTRY.get(name)
    if entry is None:
        return None
    try:
        return _import_class(entry.module_path, entry.class_name)
    except Exception as exc:  # noqa: BLE001 — registered means it MUST resolve
        raise InjectionError(
            f"native injection {name!r} is registered to "
            f"{entry.module_path}:{entry.class_name}, but that class could not be "
            f"imported ({type(exc).__name__}: {exc}). A registered native process "
            f"must resolve — refusing to fall back to a fork/empty config."
        ) from exc


def resolve_injections(fork_repo: str, config: dict) -> list[dict[str, Any]]:
    """Resolve add_processes/swap_processes -> a list of InjectionSpec dicts.

    Raises InjectionError on partitioned processes, sim_data process_configs,
    unknown names, or fork import failure.

    Results are memoized by (fork_repo, relevant config subset) so the fork's
    ecoli.* package is imported only ONCE per subprocess lifetime.  Callers
    receive a shallow copy of each cached spec dict; fail-fast InjectionErrors
    still raise normally on a cache miss (only successful results are cached).

    EXECUTION ORDER, and its limits.
    The returned order IS the execution order: ``apply_injected_processes``
    assigns strictly descending priorities in this sequence, so a companion
    process declared before the process that reads what it writes will run
    first. Two limits a caller has to know:

    * The sequence is ``add_processes`` first, then ``swap_processes`` targets —
      built below, and NOT the caller's interleaving. A swap target therefore
      always runs after every added process. That happens to be the order a
      companion listener needs; it is not a general way to express "B before A".
    * Ordering applies to STEP edges only. A process edge is scheduled by
      ``interval``, and priority is not consulted for it, so declaration order
      says nothing about processes.

    ⚠ Neither limit is validated, and this does not restore FORK order: a fork
    config can declare a ``flow`` block placing a swapped process mid-run (e.g.
    metabolism-redux after chromosome-structure). That key is consumed only on
    the vEcoli reference side and is not carried into ``injected_processes``, so
    injected steps run at the END of the tick regardless of what the fork says.
    """
    key = json.dumps({
        "fork_repo": fork_repo,
        "add_processes": config.get("add_processes") or [],
        "swap_processes": config.get("swap_processes") or {},
        "process_configs": config.get("process_configs") or {},
        "topology": config.get("topology") or {},
        "time_step": config.get("time_step", 1.0),
        "output_ports": config.get("output_ports") or {},
        "strip_pint_ports": config.get("strip_pint_ports") or {},
        "defer_ports": config.get("defer_ports") or {},
        "attach_pint_ports": config.get("attach_pint_ports") or {},
        "initial_state": config.get("initial_state") or {},
        "initial_state_overrides": config.get("initial_state_overrides") or [],
    }, sort_keys=True, default=str)  # default=str: process_configs may hold
    # sim_data-derived numpy arrays (e.g. a swapped metabolism config) that are
    # not natively JSON-serializable; stringifying them keeps the memo key stable.
    if key in _RESOLVE_CACHE:
        return [dict(s) for s in _RESOLVE_CACHE[key]]

    interval = float(config.get("time_step", 1.0))
    process_configs = config.get("process_configs") or {}
    topologies = config.get("topology") or {}
    # Resolve the config's initial_state + initial_state_overrides ONCE; each
    # spec carries the slice for its own topology roots (below).
    config_initial_state = resolve_config_initial_state(fork_repo, config)

    names = list(config.get("add_processes") or [])
    names += list((config.get("swap_processes") or {}).values())
    # Swap TARGETS (the process that replaces a baseline one) must end up with a
    # real config; a bare add_process may legitimately run on config_schema
    # defaults. Used by the config-less-swap guard below.
    swap_targets = set((config.get("swap_processes") or {}).values())

    # Lazy fork registry: a fully-native config (every add/swap name registered
    # via register_native_injection) NEVER imports the fork's ecoli.* package —
    # the fork registry is loaded only when a non-native name actually needs it.
    # A non-native name with no fork_repo is a hard error (nothing resolves it).
    _registry_box: list = []

    def _get_registry():
        if not _registry_box:
            if not fork_repo:
                raise InjectionError(
                    "no fork_repo (vEcoli-free config) but a process is not "
                    "native (not registered via register_native_injection); "
                    "cannot resolve it without the fork.")
            _registry_box.append(_fork_registry(fork_repo))
        return _registry_box[0]

    specs: list[dict[str, Any]] = []
    # Store seeds a native config_builder/adapter computes for shared stores (e.g.
    # the redux adapter's environment.exchange_data uptake set); merged into
    # shape_seed after the loop and applied by apply_injected_processes.
    _pending_shape_seeds: dict = {}
    for name in names:
        # NATIVE-FIRST (Arm 3): when a native v2 process is REGISTERED for this
        # vEcoli name, inject it DIRECTLY instead of fork-wrapping. A registered
        # name whose class fails to import RAISES (fail loud — see
        # _resolve_native_injection). classify_process then returns 'pbg_native'
        # and apply_injected_processes places it without the vivarium bridge. Falls
        # through to the fork path only when the name is NOT registered.
        native_entry = _native_injection_entry(name)
        native_cls = _resolve_native_injection(name)
        if native_cls is not None:
            cls = native_cls
        else:
            try:
                cls = _get_registry().access(name)
            except KeyError:
                raise InjectionError(f"add/swap process {name!r} not in fork registry.")
            # Defeat the installed-vEcoli shadow: for names shared with the installed
            # ecoli, `access` returns the INSTALLED class (whose store layout may
            # differ). Force the fork's own class so the transferred code actually runs.
            cls = _force_fork_class(fork_repo, cls)
        kind = classify_process(cls)
        if kind == "partitioned":
            raise InjectionError(
                f"{name!r} is a partitioned process (calculate_request/"
                "evolve_state); not supported in v1. Extension point: wrap as "
                "PartitionedProcess (v2ecoli/steps/partition.py).")

        pcfg = process_configs.get(name, "default")
        if pcfg == "sim_data":
            raise InjectionError(
                f"{name!r}: process_configs 'sim_data' is unsupported for new "
                "processes (no ParCa entry). Provide an explicit dict or 'default'.")
        config_dict = None if pcfg in ("default", None) else dict(pcfg)
        # A 'default' config + a fork_sim_data path → auto-build the FULL, faithful
        # config from the FORK's own LoadSimData (vEcoli configures its own
        # process), instead of an empty config. Falls back to default if the fork
        # has no config-getter for this process (e.g. a brand-new add_process).
        if config_dict is None and config.get("fork_sim_data"):
            try:
                config_dict = build_fork_config(
                    fork_repo, config["fork_sim_data"], name)
            except InjectionError:
                # The fork-resolution guard. NEVER downgrade this to the default
                # config: a config built from the wrong vEcoli is silently wrong
                # (fork-only keys absent -> class defaults), which is the exact
                # failure this guard exists to make loud.
                raise
            except Exception as e:  # noqa: BLE001 — not fork-configurable; use default
                print(f"[inject] fork config for {name!r} unavailable "
                      f"({type(e).__name__}); using default. {e}")
                config_dict = None

        # FORK-FREE native config builder (#683). A native swap/add whose config
        # would otherwise be empty builds it from THIS run's own bundle sim_data
        # via the registry's config_builder (e.g. a composition of
        # build_native_redux_config + adapt_metabolism_redux_config for
        # metabolism-redux), reading config["cache_dir"]/simData.cPickle. The
        # builder is v2ecoli-clean (imports only v2ecoli/pbg_v2ecoli); it may stash
        # a store seed under _exchange_data_seed, which is popped here and carried
        # as a shape_seed so the process constructor never sees a non-schema key.
        if (config_dict is None and native_entry is not None
                and native_entry.config_builder is not None
                and config.get("cache_dir")):
            native_sd = os.path.join(config["cache_dir"], "simData.cPickle")
            if not os.path.isfile(native_sd):
                raise InjectionError(
                    f"{name!r}: native injection config needs this run's bundle "
                    f"sim_data at {native_sd!r}, which is missing.")
            built = native_entry.config_builder(native_sd)
            if isinstance(built, dict):
                store_seed = built.pop("_exchange_data_seed", None)
                if store_seed:
                    _pending_shape_seeds.update(store_seed)
            config_dict = built

        # FAIL LOUD (sms-ecoli#210 Gate 0 / #375 §3d, v2ecoli#667): a SWAP TARGET
        # that reaches here with no config on the NATIVE (fork-free) path would run
        # on config_schema defaults. For a swapped metabolism (ecoli-metabolism ->
        # ecoli-metabolism-redux) those defaults are an empty stoichiometry and zero
        # homeostatic targets, so the process does nothing, the generation collapses
        # after one tick, and the run STILL reports success (the exact silent-failure
        # this exists to stop). The native config builder is now available
        # (build_native_redux_config / adapt_metabolism_redux_config in this module)
        # and is wired through a registry config_builder above — so a swap target
        # that STILL has no config here has no explicit dict, no fork_sim_data, and
        # no registered config_builder (or no cache_dir). Refuse loudly rather than
        # silently run wild-type.
        if (config_dict is None
                and name in swap_targets
                and not config.get("fork_sim_data")):
            raise InjectionError(
                f"{name!r} is a swap target but has NO config on the native "
                f"(fork-free) path: process_configs names no explicit dict for it, "
                f"there is no fork_sim_data to build from, and no native "
                f"config_builder is registered for it (or this run supplies no "
                f"cache_dir bundle). Running it now would fall back to config_schema "
                f"defaults (for metabolism-redux: an empty stoichiometry / 0 "
                f"homeostatic targets), which silently collapses the generation to "
                f"one tick and reports success. Fix: provide process_configs[{name!r}] "
                f"explicitly, or register a native config_builder for {name!r} via "
                f"register_native_injection and build the config from this run's "
                f"cache_dir bundle (sms-ecoli#210 Gate 0 / v2ecoli#667).")

        # NATIVE-FIRST: deserialize vEcoli-serialized process_config values —
        # !ParameterSerializer[path] -> param_store Quantity, !units[...] ->
        # pint Quantity — before a pbg_native process ever reads its config.
        # A fork-wrapped (vivarium_1) process has these resolved by the bridge's
        # initial_state() overlay, but a pbg_native process reads spec["config"]
        # directly, so a config translated from a genuine (non-baked) vEcoli
        # config would otherwise hand it a raw tag string. A no-op for a
        # pre-baked config (no tags, fork untouched); raises a clear error if a
        # !ParameterSerializer[...] tag survives with no fork reachable. See
        # _deserialize_config_values.
        if kind == "pbg_native" and config_dict is not None:
            config_dict = _deserialize_config_values(config_dict, fork_repo)

        # FAIL LOUD by required key (Fable): after the config is built for a
        # REGISTERED native process, RAISE if any declared required key is absent
        # or empty. A config-less injected process must not run — this is the
        # per-process backstop against the same empty-config/tick-0 collapse the
        # #682 swap-target guard catches, but keyed on the SPECIFIC fields a
        # process needs (e.g. metabolism-redux's homeostatic_concentrations),
        # applying to add_processes too (not only swap targets).
        if native_entry is not None and native_entry.required_config_keys:
            _cfg = config_dict or {}
            missing = [k for k in native_entry.required_config_keys if not _cfg.get(k)]
            if missing:
                raise InjectionError(
                    f"{name!r}: native injection config is missing or empty for "
                    f"required key(s) {missing!r}. A config-less injected process "
                    f"would fall back to schema defaults and silently collapse the "
                    f"generation — refusing to run it. Provide these via "
                    f"process_configs[{name!r}] or a registered config_builder.")

        # Topology. A native class's registered topology (a static wires dict, or a
        # callable (name, config_dict) -> wires) WINS: the config's topology is
        # written for the FORK's (nested) ports and is WRONG for the native class's
        # restructured (flat) ports. Falls back to the config topology / class
        # default when no native topology is registered. Fork-wrapped (vivarium_1)
        # processes keep the config's fork topology unchanged.
        topo = None
        if native_entry is not None and native_entry.topology is not None:
            ntopo = native_entry.topology
            topo = ntopo(name, config_dict) if callable(ntopo) else ntopo
        if topo is None:
            topo = topologies.get(name)
        if topo is None:
            topo = getattr(cls, "topology", getattr(cls, "TOPOLOGY", {}))
        topo = translate_vivarium_topology(topo)

        # Cache class for apply step (survives sys.modules restore in _fork_registry).
        _fork_class_cache[(cls.__module__, cls.__qualname__)] = cls

        # Slice the config's resolved initial_state to THIS process's topology
        # roots, so each process seeds only the stores it actually wires.
        roots = _topology_store_roots(topo)
        proc_initial = {r: config_initial_state[r]
                        for r in roots if r in config_initial_state}

        specs.append({
            "name": name,
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "kind": kind,
            "as_step": _should_inject_as_step(cls),
            "config": config_dict,
            "topology": topo,
            "interval": interval,
            # Restrict the bridge's write surface to these ports (the bridge
            # over-declares every port as both input AND output by default; a
            # swapped process must not re-type stores another process owns, e.g.
            # the mass deriver's listeners.mass). None = default (all ports).
            "output_ports": (config.get("output_ports") or {}).get(name),
            # Per-port pint→raw stripping for ports the process attaches its own
            # (unum) units to, e.g. metabolism_redux's listeners.mass.cell_mass.
            "strip_pint_ports": (config.get("strip_pint_ports") or {}).get(name),
            # Explicit defer ports (declare as {_type:node}, deferring to the
            # composite's store type). Overrides the auto-defer-all-shared default
            # in apply — so ports that need their real type (e.g. boundary's pint
            # for .to("mM")) are NOT flattened. None = auto.
            "defer_ports": (config.get("defer_ports") or {}).get(name),
            # {port: unit} to wrap raw magnitudes as pint for pint-reading ports.
            "attach_pint_ports": (config.get("attach_pint_ports") or {}).get(name),
            # Config-resolved initial values for this process's stores
            # (initial_state + initial_state_overrides), applied at injection.
            "initial_state": proc_initial,
        })
    # Config-declared candidate-side ParCa-cache gaps: bulk species the
    # candidate's ParCa cache never applies (e.g. a fork sim_data flag that
    # adds molecules at runtime), and initial values a config-declared
    # process needs but the single-cell candidate has no upstream process to
    # compute (e.g. a spatial shape process's volumes/areas). Both are
    # entirely config-driven — this function has no fork- or process-
    # specific knowledge, only how to read the two declared mappings.
    # Stashed on the first spec; apply gathers them across all specs.
    if specs:
        extra_species = config.get("extra_bulk_species") or []
        if extra_species:
            specs[0]["bulk_species_add"] = list(extra_species)
        seed: dict = {}
        seed.update(_resolve_param_store_seeds(
            fork_repo, config.get("shape_seed_param_store") or {}))
        seed.update(_resolve_literal_seeds(config.get("shape_seed_literal") or {}))
        seed.update(_pending_shape_seeds)  # native config_builder/adapter store seeds
        if seed:
            specs[0]["shape_seed"] = seed
    # Seed reaction TYPE strings into kinetic_parameters from any injected
    # process's OWN config. The process's ports_schema computes every
    # reaction_parameters default as ``0 * <config-value>`` — which for the
    # STRING ``type`` yields '' — so the reaction type never reaches its
    # store (``kinetic_parameters.<name>.<rxn>.reaction_type``) and a
    # process reading it raises "Unknown reaction type". These are static
    # config values; seed them explicitly, keyed by the topology store path.
    # Generic: reads whichever injected process's own config declares
    # ``initial_reaction_parameters``, not any particular fork or process.
    rtype_seed: dict = {}
    for sp in specs:
        irp = (sp.get("config") or {}).get("initial_reaction_parameters") or {}
        for group, reactions in irp.items():
            for rxn, params in (reactions or {}).items():
                t = params.get("type") if isinstance(params, dict) else None
                if isinstance(t, str) and t:
                    rtype_seed[("kinetic_parameters", group, rxn, "reaction_type")] = t
    if rtype_seed and specs:
        specs[0].setdefault("shape_seed", {}).update(rtype_seed)
    _RESOLVE_CACHE[key] = specs
    return [dict(s) for s in specs]


def _import_class(module: str, qualname: str):
    # Check the fork class cache first (populated by resolve_injections).
    # This allows fork classes to be retrieved even after ecoli.* sys.modules
    # has been restored to the real vEcoli package.
    cached = _fork_class_cache.get((module, qualname))
    if cached is not None:
        return cached
    mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _schema_defaults(schema: dict) -> dict:
    """Recursively pull ``{key: _default}`` out of a vivarium ports_schema subtree.

    Leaf = a dict carrying ``_default``; branches recurse. Ports/keys without a
    default are skipped (nothing to materialize)."""
    out: dict = {}
    for k, v in (schema or {}).items():
        if not isinstance(v, dict):
            continue
        if "_default" in v:
            out[k] = v["_default"]
        else:
            sub = _schema_defaults(v)
            if sub:
                out[k] = sub
    return out


def _merge_missing(dst: dict, src: dict) -> None:
    """Set keys from ``src`` into ``dst`` ONLY where absent (recursive). Never
    overwrites an existing value — the composite's real state always wins."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_missing(dst[k], v)
        elif k not in dst:
            dst[k] = v


def _overlay_config_initial_state(cell_state: dict, initial_state: dict | None,
                                  protected: set) -> list:
    """Deep-merge a config-resolved ``initial_state`` onto ``cell_state`` (config
    wins over schema defaults / declared store types), skipping ``protected``
    baseline roots (whose representation differs from a config's plain-dict
    counts, and whose bulk is seeded separately by --match-initial-state).
    Returns the list of skipped (protected) roots.

    Shared by the ``vivarium_1`` (:func:`_materialize_declared_state`) and
    ``pbg_native`` (:func:`_materialize_native_declared_state`) injection paths so
    a config-declared ``initial_state`` seeds an injected NEW store the SAME way
    regardless of the process kind."""
    skipped = []
    for root, value in (initial_state or {}).items():
        if root in protected:
            skipped.append(root)
            continue
        if isinstance(value, dict) and isinstance(cell_state.get(root), dict):
            cell_state[root] = _deep_merge(cell_state[root], value)
        else:
            cell_state[root] = value
    return skipped


def _native_port_schemas(cls, config: dict | None, core) -> dict:
    """Best-effort: instantiate a pbg-native process and return its merged
    ``inputs()`` + ``outputs()`` port schemas, OUTPUTS winning at each leaf.

    An OUTPUT port's declared type is the one that must compose FORWARD onto the
    store it wires (a downstream reader then sees the written value), so it takes
    precedence over a same-named input port's type. Returns ``{}`` if the process
    cannot be probed — typing is then skipped and the store falls back to the
    pre-fix create-empty behavior (no worse than before)."""
    inst = None
    try:
        inst = cls(config or {}, core=core)
    except Exception:  # noqa: BLE001 — try the core-less signature
        try:
            inst = cls(config or {})
        except Exception as e:  # noqa: BLE001 — never block injection on the probe
            print(f"[inject] native port-schema probe skipped "
                  f"({type(e).__name__}: {e})")
            return {}
    schemas: dict = {}
    for meth in ("inputs", "outputs"):  # outputs merged 2nd -> win at the leaf
        fn = getattr(inst, meth, None)
        if fn is None:
            continue
        try:
            got = fn() or {}
        except Exception:  # noqa: BLE001
            got = {}
        if isinstance(got, dict):
            schemas = _deep_merge(schemas, got)
    return schemas


def _leaf_port_type(schemas: dict, port_keys: tuple):
    """Walk a (nested) port-schema by ``port_keys`` and return the leaf's pbg type
    (a bare-string type like ``map[overwrite[array[float]]]`` or a ``{_type: ...}``
    entry's ``_type``), or None when the leaf declares no concrete type."""
    node = schemas
    for k in port_keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    if isinstance(node, str):
        return node
    if isinstance(node, dict) and "_type" in node:
        return node["_type"]
    return None


def _materialize_native_declared_state(cell_state: dict, cls, config: dict | None,
                                       topology: dict, name: str, core,
                                       initial_state: dict | None = None,
                                       protected_roots: set | None = None) -> None:
    """pbg_native analog of :func:`_materialize_declared_state`.

    Two jobs the pre-fix ``cell_state[root] = {}`` branch did NOT do:

    1. **Type each injected store from the process's own declared port.** A bare
       ``{}`` pre-seed makes bigraph-schema infer a plain ADDITIVE map for the new
       store, which then WINS over a wrapped Step's ``overwrite[...]`` /
       ``map[overwrite[...]]`` output — so the write never composes forward. Here
       each FRESH root store is created carrying the ``_type`` its wiring port
       declares (respecting the same guard's intent as the vivarium_1 path: give
       the overwrite store its real type instead of letting a plain-value pre-seed
       clobber it). This is what makes the environmental field chain
       (``field_timeline`` -> ``fields`` -> ``well_mixed_field`` ->
       ``boundary.external``) actually deliver a dose: ``field_timeline``'s
       ``fields: map[overwrite[array[float]]]`` output now composes to
       ``well_mixed_field``'s read instead of being dropped.
    2. **Overlay the config's per-process ``initial_state``.** The native branch
       previously consumed no ``initial_state`` at all, silently dropping a
       config-declared seed (e.g. a static ``fields`` dose). Now the same overlay
       the vivarium_1 path uses applies here, config winning.

    Baseline roots (``protected_roots`` — e.g. the structured ``bulk`` array and
    the shared ``boundary`` store) are never re-typed or clobbered: they keep
    their real composite-owned type, and a config ``initial_state`` targeting them
    is skipped (logged), exactly as on the vivarium_1 path."""
    protected = protected_roots or set()
    schemas = _native_port_schemas(cls, config, core)
    typed: list[str] = []
    for port_keys, path in _iter_leaf_paths(topology):
        if not path:
            continue
        root = path[0]
        # Ensure every wired root store exists (the pre-fix behavior).
        store = cell_state.setdefault(root, {})
        # Type only a FRESH, single-level root store from its port's declared
        # type. A baseline root keeps its composite-owned type; a leaf nested
        # BELOW a fresh root keeps the plain create-empty behavior (none arise in
        # the native antibiotic chain, whose fresh stores — `fields`,
        # `<drug>_env`, `<drug>_exchange` — are all single-level roots).
        if root in protected or len(path) != 1 or not isinstance(store, dict):
            continue
        leaf_type = _leaf_port_type(schemas, port_keys)
        if leaf_type and "_type" not in store:
            store["_type"] = leaf_type
            typed.append(f"{root}: {leaf_type}")
    skipped = _overlay_config_initial_state(cell_state, initial_state, protected)
    seeded = sorted(r for r in (initial_state or {}) if r not in protected)
    if typed:
        print(f"[inject] {name}: native store(s) typed ({'; '.join(typed)})"
              + (f"; config initial_state → {', '.join(seeded)}" if seeded else ""))
    if skipped:
        print(f"[inject] {name}: config initial_state for baseline store(s) "
              f"{', '.join(sorted(skipped))} skipped (owned by v2 / --match-initial-state)")


def _materialize_declared_state(cell_state: dict, cls, config: dict | None,
                                topology: dict, name: str,
                                initial_state: dict | None = None,
                                protected_roots: set | None = None) -> None:
    """Fill the state a vivarium-1.0 process declares (ports_schema defaults)
    into ``cell_state`` along its topology, creating missing stores/fields only,
    then overlay the config's resolved ``initial_state`` for this process's roots.

    This is what lets a SURPRISE fork process/subsystem inject + run unattended:
    its private stores are created with schema defaults (so reads never KeyError),
    and the config's ``initial_state`` / ``initial_state_overrides`` seed the real
    starting values (e.g. the cell-wall ``murein_state`` counts) — config wins
    over the bare schema default. Values the composite ALREADY owns from v2's
    baseline are never overwritten by a schema default (only by an explicit
    config initial_state targeting that store)."""
    try:
        v1 = cls(config or {})
        pschema = v1.ports_schema()
    except Exception as e:  # noqa: BLE001 — never block injection on schema probe
        print(f"[inject] {name}: ports_schema probe skipped ({type(e).__name__}: {e})")
        return
    for port_keys, path in _iter_leaf_paths(topology):
        if not path:
            continue
        # Follow the nested port-key path into the (nested) ports_schema to find
        # this leaf's declared default(s); a scattered antibiotic sub-port like
        # ``mecillinam.species.bulk`` seeds only the store it actually wires.
        schema_node = pschema if isinstance(pschema, dict) else None
        for k in port_keys:
            schema_node = schema_node.get(k) if isinstance(schema_node, dict) else None
        if isinstance(schema_node, dict) and "_default" in schema_node:
            # A LEAF port (e.g. ``volumes.cytoplasm`` -> ``0 * units.fL``): seed
            # the store itself with the default VALUE, not an empty ``{}``.
            # Leaving ``{}`` makes pbg realize a ``quantity``-typed store against a
            # dict with no ``magnitude`` field -> ``KeyError: 'magnitude'`` at
            # build. Only fill if the store slot is still absent (composite wins).
            default_val = schema_node["_default"]
            # SKIP empty-string ports_schema placeholders (vEcoli's antibiotic
            # declares ``reaction_parameters`` leaves as ``""`` — real values are
            # serializer tags resolved by the bridge's initial_state() overlay,
            # #489). Seeding "" into a ``quantity``-typed store (``kinetic_parameters``
            # rate constants, written by ``permeability``) makes pbg realize an
            # empty-string pint magnitude at build. Leaving the slot absent lets
            # the wrapped process's overlaid port default (the deserialized
            # quantity) flow in instead.
            if isinstance(default_val, str) and default_val == "":
                continue
            parent = cell_state
            for seg in path[:-1]:
                parent = parent.setdefault(seg, {})
            if isinstance(parent, dict) and path[-1] not in parent:
                parent[path[-1]] = default_val
        else:
            defaults = _schema_defaults(schema_node)
            node = cell_state
            for seg in path:
                node = node.setdefault(seg, {})
            if defaults and isinstance(node, dict):
                _merge_missing(node, defaults)
    # Overlay the config-resolved initial_state (config wins over schema
    # defaults) onto the NEW stores this injection introduced. Roots that v2's
    # baseline already owns (``protected_roots`` — e.g. the structured ``bulk``
    # array, ``boundary``) are NEVER touched here: their representation differs
    # from a config's plain-dict counts, and matched bulk is seeded separately by
    # --match-initial-state. So a config bulk override is skipped (logged), while
    # the subsystem's own stores (murein_state / wall_state / pbp_state) seed.
    protected = protected_roots or set()
    skipped = _overlay_config_initial_state(cell_state, initial_state, protected)
    # Surface what top-level stores this process introduced / seeded.
    intro = sorted({r for r in _topology_store_roots(topology) if r in cell_state})
    seeded = sorted(r for r in (initial_state or {}) if r not in protected)
    if skipped:
        print(f"[inject] {name}: config initial_state for baseline store(s) "
              f"{', '.join(sorted(skipped))} skipped (owned by v2 / --match-initial-state)")
    if intro:
        print(f"[inject] {name}: declared-state materialized (roots: "
              f"{', '.join(intro)})"
              + (f"; config initial_state → {', '.join(seeded)}" if seeded else ""))


def _augment_bulk_species(cell_state: dict, names: list[str]) -> None:
    """Append molecule ``names`` absent from the candidate bulk store, in place.

    v2ecoli's runtime ``bulk`` store is a structured array ``(id, count, *submass)``
    that carries mass inline. A fork can add species to its own sim_data at
    runtime (e.g. via a sim_data flag) that the candidate's ParCa cache never
    applies, so those ids are missing and index lookups against ``bulk['id']``
    fail. Append the missing ids with count 0 and zero submass — mass only
    matters once the count grows (typically needs an environment exposure the
    candidate doesn't have), and appending at the END leaves every existing
    molecule's index unchanged, so other processes (which resolve their
    indices against this same ``bulk['id']`` at t=0) are unaffected."""
    import numpy as np
    bulk = cell_state.get("bulk")
    if bulk is None or not hasattr(bulk, "dtype"):
        return
    existing = set(np.asarray(bulk["id"]).tolist())
    new = [n for n in names if n not in existing]
    if not new:
        return
    rows = np.zeros(len(new), dtype=bulk.dtype)  # count + every submass = 0
    rows["id"] = new
    cell_state["bulk"] = np.append(bulk, rows)
    print(f"[inject] bulk store: appended {len(new)} injected species "
          f"(count 0): {', '.join(new)}")


def apply_injected_processes(cell_state: dict, flow_order: list, core,
                             specs: list[dict]) -> list[str]:
    """Add each resolved spec to ``cell_state`` + ``flow_order`` (in place)."""
    from v2ecoli.library.vivarium_bridge import wrap_vivarium_process
    from v2ecoli.composites._helpers import make_edge

    # Append injected-subsystem bulk species absent from v2ecoli's ParCa bulk
    # store (config-declared via extra_bulk_species) BEFORE materializing
    # processes, so bulk-index lookups resolve at the first update.
    add_species: list[str] = []
    for spec in specs:
        for nm in (spec.get("bulk_species_add") or []):
            if nm not in add_species:
                add_species.append(nm)
    if add_species:
        _augment_bulk_species(cell_state, add_species)

    # Roots v2's baseline already owns BEFORE any injection — config
    # initial_state must never clobber these (e.g. the structured bulk array).
    baseline_roots = set(cell_state)
    added: list[str] = []
    for spec in specs:
        cls = _import_class(spec["module"], spec["qualname"])
        if spec["kind"] == "vivarium_1":
            # Wire EVERY ports_schema port: an explicit topology entry wins, but a
            # port the process DECLARES yet leaves unmapped defaults to a
            # same-named top-level store — exactly vivarium-1.0's convention
            # (e.g. cell-wall's pbp_state, read in next_update but absent from the
            # registered TOPOLOGY). Without this such a port KeyErrors on tick 0.
            try:
                _pschema = cls(spec["config"] or {}).ports_schema()
                for _p in (_pschema or {}):
                    spec["topology"].setdefault(_p, (_p,))
            except Exception as e:  # noqa: BLE001 — never block on the probe
                print(f"[inject] {spec['name']}: topology auto-port skipped "
                      f"({type(e).__name__}: {e})")

            # Materialize declared state + config initial_state BEFORE deferring
            # ports, so this process's NEW root stores already exist in cell_state
            # and are picked up by the auto-defer below. vivarium's Engine does
            # this at build; without it a process that introduces its own stores
            # (cell-wall's murein_state / wall_state / pbp_state) or reads a field
            # absent from a shared store (boundary.volume before ecoli-shape's
            # first write) crashes on tick 0. Fills only missing stores/fields,
            # then overlays the config's initial_state (config wins).
            _materialize_declared_state(cell_state, cls, spec["config"],
                                        spec["topology"], spec["name"],
                                        initial_state=spec.get("initial_state"),
                                        protected_roots=baseline_roots)

            # Defer ports to the composite's store type ({_type:node}) instead of
            # the process's inferred type. Auto-default: defer every port whose
            # root store now exists in cell_state — which (after materialization)
            # covers BOTH shared stores (avoids the float-vs-quantity subtype
            # clash on stores like listeners.mass) AND this process's own new
            # stores (so a `_default: None` field like wall_state.lattice is held
            # as a node that accepts None now and the model's sampled ndarray
            # later, instead of failing to type). An explicit spec defer_ports
            # overrides this — so ports that need their real type (e.g. boundary's
            # pint for .to("mM")) keep it.
            if spec.get("defer_ports") is not None:
                defer_ports = list(spec["defer_ports"])
            else:
                # Auto-defer only FLAT top-level ports whose root store existed
                # BEFORE this apply() call — deferring sets a port to
                # {_type:node}, which for a nested, *scattered* port would
                # erase the ``bulk_array`` typing its ``species.bulk`` leaf
                # needs. Scattered ports keep their translate_ports typing
                # (bulk stays bulk_array); per-leaf store-type deferral, where
                # needed, is handled by the nested wiring landing on the
                # composite's own typed stores.
                #
                # Scoped to baseline_roots (captured pre-injection), NOT the
                # live cell_state — a store an EARLIER spec in this same
                # apply() call just introduced (e.g. a process's own new
                # top-level store) must NOT be auto-deferred to a generic
                # node just because it now exists; that erases its real
                # overwrite[quantity] typing and a later update on it raises
                # NotFoundLookupError (apply(None, <Quantity>, <Quantity>, ...)).
                defer_ports = [p for p, path in spec["topology"].items()
                               if isinstance(path, list) and path and path[0] in baseline_roots]
            wrapped = wrap_vivarium_process(cls, name=spec["name"],
                                            as_step=spec["as_step"],
                                            output_ports=spec.get("output_ports"),
                                            defer_ports=defer_ports,
                                            strip_pint_ports=spec.get("strip_pint_ports"),
                                            attach_pint_ports=spec.get("attach_pint_ports"))
        else:  # pbg_native
            wrapped = cls
            # Create + TYPE this native process's injected stores from its own
            # declared ports, and overlay the config's initial_state (config
            # wins). Without the typing, a freshly-created store (e.g. `fields`)
            # is inferred as a plain additive map that clobbers a Step's
            # overwrite output, so the field-delivery chain never propagates a
            # dose; without the overlay, a config-declared seed was dropped. Both
            # done here, before make_edge, so the store exists + is typed when the
            # edge wires — the pbg_native counterpart to the vivarium_1 branch's
            # _materialize_declared_state call above.
            _materialize_native_declared_state(
                cell_state, cls, spec["config"], spec["topology"], spec["name"],
                core, initial_state=spec.get("initial_state"),
                protected_roots=baseline_roots)
        core.register_link(spec["name"], wrapped)
        # ALSO register under the exact address make_edge() will stamp on this
        # edge (f'{type(instance).__module__}.{type(instance).__qualname__}').
        # At division, process-bigraph re-realizes the daughter subtree
        # (Composite._realize_structural_subtrees -> bigraph_schema's
        # realize_link) for the daughter's copy of this edge, which has no
        # live `instance` -- it resolves the process class purely from the
        # edge's `address` via local_lookup_registry(core, data) ==
        # core.link_registry.get(data), NOT by spec['name']. A dynamically
        # wrapped vivarium_1 class (wrap_vivarium_process) keeps
        # __module__ == 'v2ecoli.library.vivarium_bridge' and gets
        # __qualname__ == f'{v1_cls.__name__}Bridge' -- a string that is
        # neither importable nor equal to spec['name'], so without this extra
        # registration division crashes with "no link found at address:
        # {'protocol': 'local', 'data': 'v2ecoli.library.vivarium_bridge.
        # <X>Bridge'}". Harmless (and not strictly needed) for pbg_native
        # classes, whose real module.qualname IS importable; applied for all
        # injected specs since this is the one shared registration point.
        core.register_link(f"{wrapped.__module__}.{wrapped.__qualname__}", wrapped)

        instance = wrapped(spec["config"] or {}, core=core)
        edge_type = "step" if spec["kind"] == "pbg_native" and spec["as_step"] \
            else ("step" if spec["as_step"] else "process")
        cell_state[spec["name"]] = make_edge(
            instance, spec["topology"], edge_type=edge_type,
            config=spec["config"] or {})
        flow_order.append(spec["name"])
        added.append(spec["name"])
    # Break the priority tie among the processes just injected.
    #
    # `make_edge` gives every Step the DEFAULT priority 1.0, and
    # `inject_flow_dependencies` — which replaces that with distinct descending
    # values — has already run by the time we get here (it is called before
    # injection in the baseline generator). So without this, every injected step
    # carries priority 1.0: tied with each other, and with no way for a companion
    # process to run before the process that reads what it writes.
    #
    # The consequence is INTERMITTENT rather than a clean failure: whichever of
    # two tied steps the scheduler happens to run first decides whether a
    # consumer sees a populated store or an empty one, so the same build can
    # succeed and then fail on a later run. Measured on a real injected pair:
    # the same script passed and failed across repeat runs with nothing else
    # changed.
    #
    # Priorities descend in the order processes were injected, and start below
    # the lowest baseline priority so injected steps still run after the
    # baseline (which is where appending them to flow_order already put them).
    # Declaration order therefore expresses the dependency — the same
    # explicit-not-inferred contract as the `--inject-process` surface itself.
    if added:
        baseline_priorities = [
            e["priority"] for name, e in cell_state.items()
            if name not in added and isinstance(e, dict)
            and isinstance(e.get("priority"), (int, float))
        ]
        base = min(baseline_priorities) if baseline_priorities else 1.0
        # STEP edges only. `make_edge` writes `priority` for steps and `interval`
        # for processes, and the scheduler reads `priority` only when ordering
        # steps — so writing it onto a process edge would look like an ordering
        # guarantee while changing nothing. Skipping them keeps the inert case
        # visible instead of silently pretending to have ordered it.
        offset = 0
        for name in added:
            edge = cell_state.get(name)
            # Check the edge TYPE directly rather than inferring it from the
            # presence of a `priority` key. They agree today only because
            # `make_edge` always writes that key for a step -- so the proxy would
            # silently skip a step edge that ever arrived without it, which is
            # the exact failure this branch exists to prevent.
            if not (isinstance(edge, dict) and edge.get("_type") == "step"):
                continue                       # a process edge: ordered by interval
            offset += 1
            edge["priority"] = float(base) - offset
    # Overwrite the scaffolded shape stores (materialized to 0 from a
    # process's ports_schema defaults) with config-declared real initial
    # values (shape_seed_param_store / shape_seed_literal, resolved by
    # resolve_injections), AFTER every process materialized — so the
    # non-zero value wins over the 0-default port that also wires there.
    # Without this a process dividing by e.g. a volume store gets zero.
    shape_seed: dict = {}
    for spec in specs:
        shape_seed.update(spec.get("shape_seed") or {})
    for store_path, value in shape_seed.items():
        node = cell_state
        for seg in store_path[:-1]:
            node = node.setdefault(seg, {})
        if isinstance(node, dict):
            node[store_path[-1]] = value
    if shape_seed:
        print(f"[inject] seeded {len(shape_seed)} shape store(s): "
              f"{', '.join('.'.join(map(str, p)) for p in shape_seed)}")
    # Re-normalize boundary.external to PLAIN mM floats after seeding. The
    # baseline runs _normalize_boundary_units at build time, but injection (and
    # its shape_seed_literal pass) runs AFTER that — so a shape_seed writing a
    # per-drug dose to boundary.external.<drug> lands as a pint Quantity
    # (magnitude * units, from _resolve_literal_seeds). boundary.external is
    # `map[overwrite[float[mM]]]` — it holds PLAIN floats (metabolism/
    # well_mixed_field write bare mM floats; the unit lives in the SCHEMA, not the
    # value). A pint Quantity in that store silently realizes to None at composite
    # build (directly verified), so the antibiotic transport reads
    # boundary.external[<drug>] == None -> NaN -> solve_ivp "y0 must be finite" at
    # the first tick. Normalizing here (the same contract the baseline applies)
    # keeps the STATIC shape_seed dose finite, mirroring the DYNAMIC field-chain
    # fix above. Only boundary.external is touched — Quantity-typed shape stores
    # (volumes, kinetic_parameters, potential) are untouched and keep their pint
    # values, which their (quantity-typed) stores accept.
    if isinstance(cell_state.get("boundary"), dict):
        try:
            from v2ecoli.composites._helpers import _normalize_boundary_units
            _normalize_boundary_units(cell_state)
        except Exception as e:  # noqa: BLE001 — never block injection on normalize
            print(f"[inject] boundary.external normalize skipped "
                  f"({type(e).__name__}: {e})")
    return added


def remove_processes(cell_state: dict, flow_order: list, names) -> list[str]:
    """Remove named processes/steps from a cell-state tree + flow order in place.

    The 'remove' half of a swap: a ``swap_processes`` mapping {old: new} adds the
    converted ``new`` (via :func:`apply_injected_processes`) and removes ``old``
    here; a config's ``exclude_processes`` list is removed the same way. Names not
    present are ignored (returns only the names actually removed from cell_state).
    """
    removed: list[str] = []
    for name in names:
        if name in cell_state:
            del cell_state[name]
            removed.append(name)
        while name in flow_order:
            flow_order.remove(name)
    return removed


if __name__ == "__main__":
    # argv: <fork_repo> <config_json_path>  -> prints specs JSON to stdout
    fork_repo, cfg_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    json.dump(resolve_injections(fork_repo, cfg), sys.stdout)
