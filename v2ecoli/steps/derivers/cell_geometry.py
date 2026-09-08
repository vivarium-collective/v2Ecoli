"""
=============
Cell Geometry
=============

Native cell-shape geometry deriver, ported from vEcoli's
``ecoli/processes/shape.py``.

Splits the whole-cell volume (from the mass listener) into periplasm and
cytoplasm compartments, and derives the outer surface area via 3D capsule
geometry (a cylinder capped with hemispheres). A general, drug-agnostic feature:
it exists for any injected subsystem that needs the geometry split — e.g. an
injected ``antibiotic_transport_odeint`` chain that divides
periplasmic/cytoplasmic molecule counts by the periplasm/cytoplasm volumes and
reads the outer surface area. It writes the SAME store paths vEcoli's
``ecoli-shape`` writes (``periplasm.global.volume`` / ``cytoplasm.global.volume``
/ ``boundary.outer_surface_area``), so a single config can serve both a reference
arm (native ``ecoli-shape``) and this candidate step against the identical
topology the config's transport/permeability/concentrations_deriver wiring reads
(``["..","periplasm","global","volume"]`` etc). None of these are otherwise
populated in the single-cell candidate, since nothing else in v2ecoli writes them.
Enable by name (``features=['cell_geometry']`` or an injected subsystem's
``requires_features: ['cell_geometry']``).

**Bridge-quirk guard**: vEcoli's ``Shape`` wires its ``cell_global`` port
directly to the agent's ``boundary`` store. In the candidate's vivarium
bridge, that would hand the WHOLE ``boundary`` store to the step — including
the media ``external`` dict of plain floats — which trips ``Shape``'s
all-items-are-``pint.Quantity`` assert (``shape.py`` ports_schema/next_update).
This step sidesteps that entirely: it reads the cell volume from THIS
candidate's OWN mass listener (``listeners.mass.volume``), never from
``boundary``, and only ever WRITES a single scalar (``outer_surface_area``)
into ``boundary`` — it never reads ``boundary`` at all.
"""

import math

from bigraph_schema.contract import ProcessContract
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.types.quantity import ureg as units
# The injected antibiotic_transport_odeint / permeability / gillespie /
# concentrations_deriver are vEcoli processes that do their unit arithmetic in
# VIVARIUM's pint registry (e.g. ``units.mol / (volume * N_A)``). v2ecoli sets
# its OWN ``bigraph_schema.units`` registry as pint's application registry, so a
# volume written as a v2ecoli ``quantity[float,L]`` deserialises back into the
# v2ecoli registry and pint refuses to operate across the two ("Cannot operate
# with Quantity of different registries"). This step therefore WRITES its outputs
# as vivarium-registry Quantities into ``any``-typed stores (pbg keeps the raw
# object, exactly as the harness's vivarium-unit shape-seeds do), so the vEcoli
# consumers receive quantities in their own registry. Inputs (the v2ecoli mass
# listener) are read via ``as_quantity`` and reduced to a plain magnitude first,
# so the input registry never matters.
from vivarium.library.units import units as viv_units

NAME = "cell-geometry"
TOPOLOGY = {
    "listeners": ("listeners",),
    "periplasm": ("periplasm",),
    "cytoplasm": ("cytoplasm",),
    "boundary": ("boundary",),
}

PI = math.pi

# vEcoli ecoli/processes/shape.py Shape.defaults["periplasm_fraction"].
PERIPLASM_FRACTION = 0.2

# 1 um**3 == 1e-15 L (equivalently, 1 um**3 == 1 fL) — used to convert the
# plain-float cell volume (L) into um**3 for the length/surface-area math,
# which vEcoli's shape.py performs in micron-based units.
L_PER_UM3 = 1e-15


def length_from_volume(volume_um3, width_um):
    """Cell length (um) from volume (um**3) and width (um), via 3D capsule
    geometry: V = (4/3)*pi*r**3 + pi*r**2*a, l = a + 2*r.

    Ported verbatim (modulo units) from vEcoli
    ``ecoli/processes/shape.py:length_from_volume``.
    """
    radius = width_um / 2
    cylinder_length = (volume_um3 - (4 / 3) * PI * radius**3) / (PI * radius**2)
    return cylinder_length + 2 * radius


def surface_area_from_length(length_um, width_um):
    """Outer surface area (um**2) from length + width, via 3D capsule
    geometry: SA = 4*pi*r**2 + 2*pi*r*a.

    Ported verbatim (modulo units) from vEcoli
    ``ecoli/processes/shape.py:surface_area_from_length``.
    """
    radius = width_um / 2
    cylinder_length = length_um - width_um
    return 4 * PI * radius**2 + 2 * PI * radius * cylinder_length


class CellGeometry(Step):
    """Split whole-cell volume into periplasm/cytoplasm; derive outer surface
    area from a fixed cell width.

    Ports:
      * ``listeners.mass.volume`` (in): whole-cell volume, pint
        Quantity[fL] — this candidate's OWN mass listener, never the shared
        ``boundary`` store.
      * ``periplasm.global.volume`` / ``cytoplasm.global.volume`` (out): pint
        Quantity[L] — the vEcoli ``ecoli-shape`` store paths the injected
        ``antibiotic_transport_odeint`` chain divides molecule counts by.
      * ``boundary.outer_surface_area`` (out): pint Quantity[um**2].
    """

    contract = ProcessContract(
        summary=(
            "Splits whole-cell volume into periplasm/cytoplasm compartments "
            "at a fixed periplasm fraction, and derives cell length and outer "
            "surface area from volume + a fixed width by treating the cell as "
            "a 3D capsule (a cylinder capped by two hemispheres)."
        ),
        math=[
            "V_peri = f_p · V  ;  V_cyto = (1 − f_p) · V",
            "V = (4/3)·π·r³ + π·r²·a   (r = w/2, a = cylinder segment length)  ⇒  a = (V − (4/3)·π·r³) / (π·r²)",
            "L = a + 2·r = a + w",
            "A = 4·π·r² + 2·π·r·a   (a = L − w)",
        ],
        symbols={
            "V": "whole-cell volume, from listeners.mass.volume (fL, converted to L then um**3 for the geometry math)",
            "f_p": "periplasm fraction, fixed constant PERIPLASM_FRACTION = 0.2 (ported from vEcoli shape.py Shape.defaults)",
            "V_peri": "periplasm volume = f_p · V (L)",
            "V_cyto": "cytoplasm volume = (1 − f_p) · V (L)",
            "w": "cell width (um); config parameter width_um, default 1.0",
            "r": "cell radius = w/2 (um)",
            "a": "cylinder (non-hemispherical) segment length (um)",
            "L": "total cell length (um) = a + w",
            "A": "outer surface area (um**2)",
        },
        inputs={
            "listeners": "Reads listeners.mass.volume — this Step's OWN whole-cell volume (pint Quantity[fL]), never the shared boundary store (see module docstring bridge-quirk guard).",
        },
        outputs={
            "periplasm": "Writes periplasm.global.volume = f_p · V (pint Quantity[L], vivarium registry).",
            "cytoplasm": "Writes cytoplasm.global.volume = (1 − f_p) · V (pint Quantity[L], vivarium registry).",
            "boundary": "Writes boundary.outer_surface_area (pint Quantity[um**2], vivarium registry), derived from V and width_um via the capsule geometry.",
        },
        config={
            "width_um": "Fixed cell width (um) used in the capsule geometry; a config PARAMETER, not a port (default 1.0, ported from vEcoli shape.py Shape.defaults['width']).",
        },
        assumptions=[
            "Cell shape is a 3D capsule: a cylinder of length a = L − w capped by two hemispheres of radius r = w/2.",
            "Cell width is fixed (config parameter width_um), not itself derived from volume or dynamics.",
            "Periplasm fraction is a fixed constant (0.2), not derived from volume or composition.",
            "Outputs use overwrite (SET) semantics — no step-wise accumulation.",
        ],
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # Fixed cell width (um), ported from shape.py Shape.defaults["width"]
        # (1.0 um). A config PARAMETER, not a port: deliberately not wired
        # to `boundary` (see module docstring on the bridge quirk).
        "width_um": {"_type": "float", "_default": 1.0},
    }

    def inputs(self):
        return {
            "listeners": {
                "mass": {
                    "volume": {"_type": "quantity[float,fL]", "_default": 0.0},
                },
            },
        }

    def outputs(self):
        # ``quantity[...]`` leaves so pbg can APPLY the per-tick update — but the
        # VALUES written are vivarium-registry Quantities (see the module-level
        # import comment). v2ecoli's ``Quantity.realize`` returns an incoming
        # ``pint.Quantity`` UNCHANGED (it only rebuilds via the app registry for
        # bare dict/scalar encodings), so the vivarium object survives intact and
        # the vEcoli consumers read it in their own registry.
        return {
            "periplasm": {
                "global": {
                    "volume": {
                        "_type": "overwrite[quantity[float,L]]", "_default": 0.0},
                },
            },
            "cytoplasm": {
                "global": {
                    "volume": {
                        "_type": "overwrite[quantity[float,L]]", "_default": 0.0},
                },
            },
            "boundary": {
                "outer_surface_area": {
                    "_type": "overwrite[quantity[float,um**2]]", "_default": 0.0},
            },
        }

    def initialize(self, config):
        self.width_um = self.parameters.get("width_um", 1.0)

    def compute(self, cell_volume_L, width_um=None):
        """Pure geometry math (plain floats, no pint) — testable in
        isolation. ``cell_volume_L``: whole-cell volume in liters. Returns
        ``{"periplasm": <L>, "cytoplasm": <L>, "outer_surface_area": <um**2>}``.
        """
        if width_um is None:
            width_um = self.width_um
        periplasm = cell_volume_L * PERIPLASM_FRACTION
        cytoplasm = cell_volume_L * (1 - PERIPLASM_FRACTION)
        volume_um3 = cell_volume_L / L_PER_UM3
        length_um = length_from_volume(volume_um3, width_um)
        outer_surface_area = surface_area_from_length(length_um, width_um)
        return {
            "periplasm": periplasm,
            "cytoplasm": cytoplasm,
            "outer_surface_area": outer_surface_area,
        }

    def update(self, states, interval=None):
        cell_volume = as_quantity(states["listeners"]["mass"]["volume"], units.fL)
        cell_volume_L = cell_volume.to(units.L).magnitude
        geometry = self.compute(cell_volume_L)
        # Emit in VIVARIUM's registry so the downstream vEcoli processes can do
        # their unit arithmetic (see the module-level import comment).
        return {
            "periplasm": {
                "global": {"volume": geometry["periplasm"] * viv_units.L},
            },
            "cytoplasm": {
                "global": {"volume": geometry["cytoplasm"] * viv_units.L},
            },
            "boundary": {
                "outer_surface_area": geometry["outer_surface_area"] * viv_units.um**2,
            },
        }
