"""Registered composite for Maya's Aim 2B — rule-based flagellar assembly (pbg-nfsim).

Wraps a production+complexation document (``MonomerProduction`` feeds monomers;
``NFSimProcess`` assembles them through the flagella BNGL: free monomers ->
export apparatus -> motor/basal body -> hook -> complete flagellum) as a
REGISTERED ``@composite_generator``. This is what ``flagella-04-nfsim-assembly``
runs — so the run is a real registered composite in the Simulations DB and opens
in the Composite Explorer, instead of the ad-hoc in-code ``Composite({"state":
doc})`` that ``run_nfsim_assembly.py`` used to build.

pbg-nfsim is the OPTIONAL ``nfsim`` extra, so every import is lazy: this module is
safe to import in venvs without it (``build_core`` discovers every
``composites/*.py``); only *building* the composite needs pbg-nfsim installed.

MODEL SOURCE (2026-08-12): the document (BNGL model file + monomer production
rates) is built from v2ecoli's OWN copy of the generator --
workspace/investigations/flagella-cascade/studies/flagella-04-complexation-nfsim/
models/generate_flagella_bngl.py -- not from pbg_nfsim's bundled example. pbg-nfsim
ships a generic demo flagella model; this investigation's corrected, cited
stoichiometry (matching v2ecoli's own reaction network -- see that module's
docstring for full provenance) needs to evolve alongside v2ecoli's own flagella
Steps, and eventually couple to them directly, so it's owned here rather than
living inside the external tool's example folder. pbg_nfsim itself supplies only
the generic runtime engine (NFSimProcess, MonomerProduction) -- genuinely
reusable tooling, kept as the external dependency.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any

# from pbg_superpowers.composite_generator import composite_generator  # old name, pbg->viva rebrand complete 2026-09-04
from viva_superpowers.composite_generator import composite_generator

_LOCAL_MODEL_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "workspace", "investigations", "flagella-cascade", "studies",
    "flagella-04-complexation-nfsim", "models", "generate_flagella_bngl.py",
)


def _import_local_model():
    """Load v2ecoli's own generate_flagella_bngl.py by file path (it lives in
    an investigation folder, not an installed package)."""
    spec = importlib.util.spec_from_file_location(
        "flagella_cascade_nfsim_model", _LOCAL_MODEL_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_pbg_nfsim():
    # bionetgen 0.8.6 does `from pkg_resources import packaging` (removed on
    # py3.12); shim it from the standalone `packaging` before importing pbg_nfsim.
    import pkg_resources
    import packaging as _packaging
    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = _packaging
    import pbg_nfsim
    return pbg_nfsim


def _register_nfsim_links(core: Any) -> Any:
    """Register the processes the assembly document wires (idempotent)."""
    pbg_nfsim = _import_pbg_nfsim()
    from process_bigraph.emitter import RAMEmitter
    core.register_link("nfsim", pbg_nfsim.NFSimProcess)
    core.register_link("monomer-production", pbg_nfsim.MonomerProduction)
    core.register_link("ram-emitter", RAMEmitter)
    return core


@composite_generator(
    name="flagella_nfsim_assembly",
    description=(
        "Rule-based flagellar assembly (Maya's Aim 2B): pbg-nfsim MonomerProduction "
        "feeds monomers into NFSimProcess, which assembles ~30 flagellar proteins "
        "through the 237-rule BNGL (export apparatus -> motor/basal body -> hook -> "
        "complete flagellum). Standalone stochastic rule-based assembly; not "
        "WCM-coupled."
    ),
    parameters={
        "n_steps": {
            "type": "integer", "default": 20,
            "description": "Number of complexation ticks to run.",
        },
        "complexation_interval": {
            "type": "float", "default": 120.0,
            "description": "NFsim complexation update interval (s).",
        },
        "production_interval": {
            "type": "float", "default": 1.0,
            "description": "MonomerProduction update interval (s).",
        },
        "production_rate_scale": {
            "type": "float", "default": 1.0,
            "description": "Scalar applied to the monomer production rates.",
        },
    },
    core_extensions=[_register_nfsim_links],
)
def flagella_nfsim_assembly(
    core: Any = None,
    *,
    n_steps: int = 20,
    complexation_interval: float = 120.0,
    production_interval: float = 1.0,
    production_rate_scale: float = 1.0,
) -> dict:
    """Build the production+complexation document (the assembly composite),
    using v2ecoli's own model + rates (see module docstring)."""
    _import_pbg_nfsim()  # ensures pbg_nfsim + its shim are importable before running
    model = _import_local_model()
    return model.make_production_document(
        n_steps=n_steps,
        complexation_interval=float(complexation_interval),
        production_interval=float(production_interval),
        production_rate_scale=float(production_rate_scale),
    )
