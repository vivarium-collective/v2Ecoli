"""Shape step — capsule cell geometry from mass.

A process-bigraph ``Step`` implementing the **Shape** process of Skalnik et al.
2023, *Whole-cell modeling of E. coli colonies* (PLoS Comput. Biol.; SM §3.1).
The cell is a **capsule** (cylinder capped by hemispheres) of fixed width ``w``
and fixed density; it grows purely by elongation, so **length derives from
volume** (= mass / density):

    v   = mass / density                                  (density = 1.1 g/mL)
    l   = (v − 4/3·π(w/2)³) / (π(w/2)²) + w               (cylinder + 2 caps)
    a_o = 4π(w/2)² + 2π(w/2)(l − w)                       (outer membrane area)
    a_i = a_o · (1 − f_p)^(2/3)                           (inner membrane area)
    v_p = v · f_p ,   v_c = v · (1 − f_p)                 (periplasm / cytoplasm)

It returns the capsule geometry (in Å) as plain-float coordinates so downstream
consumers can build representations directly: the cell envelope grows with mass,
and the pre-division (max-mass) state is the elongated, about-to-divide cell.
"""
from __future__ import annotations

import math

from process_bigraph import Step

from bigraph_schema.contract import ProcessContract

# Unit bridge: 1 g/mL == 1000 fg/fL  (fg = 1e-15 g, fL = 1e-15 L); fL ≈ µm³.
_G_PER_ML_TO_FG_PER_FL = 1000.0


def shape_from_mass(mass_fg: float, width_um: float = 1.0,
                    density_g_per_ml: float = 1.1,
                    periplasm_fraction: float = 0.2) -> dict:
    """Capsule cell shape for a cell of ``mass_fg`` (Skalnik et al. 2023 §3.1).

    Width, density and periplasm fraction are fixed; everything else derives from
    volume = mass/density. Lengths/areas/volumes in µm / µm² / fL(µm³); capsule
    coordinates are returned in Å as plain floats.
    """
    v = max(0.0, float(mass_fg) / (density_g_per_ml * _G_PER_ML_TO_FG_PER_FL))  # fL = µm³
    w = float(width_um)
    r = w / 2.0                                                    # cap radius, µm
    lcyl = max(0.0, (v - (4.0 / 3.0) * math.pi * r ** 3) / (math.pi * r ** 2))  # µm
    length = lcyl + w                                             # tip-to-tip, µm
    outer_sa = 4.0 * math.pi * r ** 2 + 2.0 * math.pi * r * (length - w)  # µm²
    inner_sa = outer_sa * (1.0 - periplasm_fraction) ** (2.0 / 3.0)
    radius_A = r * 1e4                        # µm → Å
    half_len_A = (lcyl / 2.0) * 1e4
    s = (1.0 - periplasm_fraction) ** (1.0 / 3.0)
    inner_radius_A = radius_A * s
    inner_half_len_A = half_len_A * s
    return {
        "mass_fg": float(mass_fg),
        "density_g_per_ml": density_g_per_ml,
        "width_um": w,
        "volume_fl": v,
        "length_um": length,
        "outer_sa_um2": outer_sa,
        "inner_sa_um2": inner_sa,
        "periplasm_fraction": periplasm_fraction,
        "periplasm_vol_fl": v * periplasm_fraction,
        "cytoplasm_vol_fl": v * (1.0 - periplasm_fraction),
        "radius_A": radius_A,
        "half_len_A": half_len_A,
        "inner_radius_A": inner_radius_A,
        "inner_half_len_A": inner_half_len_A,
        "envelope": {
            "outer_radius_A": radius_A,
            "outer_half_len_A": half_len_A,
            "inner_radius_A": inner_radius_A,
            "inner_half_len_A": inner_half_len_A,
            "periplasm_fraction": periplasm_fraction,
            "periplasm_vol_fl": v * periplasm_fraction,
            "cytoplasm_vol_fl": v * (1.0 - periplasm_fraction),
            "outer_sa_um2": outer_sa,
            "inner_sa_um2": inner_sa,
        },
    }


# Backwards-compatible alias (the step originally exposed only the capsule).
capsule_from_mass = shape_from_mass

# Emitted shape keys (the shape_from_mass dict minus the Capsule object). A
# map[float] store only merges updates onto keys that already exist, so the
# 'shape' store must be pre-seeded with these (see zero_shape()).
SHAPE_KEYS = (
    "mass_fg", "density_g_per_ml", "width_um", "volume_fl", "length_um",
    "outer_sa_um2", "inner_sa_um2", "periplasm_fraction", "periplasm_vol_fl",
    "cytoplasm_vol_fl", "radius_A", "half_len_A", "inner_radius_A",
    "inner_half_len_A",
)


def zero_shape():
    """Zero-filled shape dict for seeding a map[float] 'shape' store."""
    return {k: 0.0 for k in SHAPE_KEYS}


class ShapeStep(Step):
    """Compute capsule cell shape from mass (fixed width, density, periplasm frac)."""

    contract = ProcessContract(
        summary=(
            "Derive capsule cell geometry (Skalnik et al. 2023 §3.1) from "
            "cell_mass: fixed width and density fix the cross-section, so "
            "volume comes straight from mass and length derives purely from "
            "volume by elongation. Reports length, envelope surface areas, "
            "and periplasm/cytoplasm volumes each step (overwrite, not "
            "accumulated)."
        ),
        math=[
            "v = mass / (density · 1000)                              (fg / (g/mL·1000) → fL = µm³)",
            "r = w / 2",
            "l_cyl = max(0, (v − 4/3·π·r³) / (π·r²))                    (cylindrical section length)",
            "length = l_cyl + w                                        (tip-to-tip capsule length)",
            "a_o = 4π·r² + 2π·r·l_cyl                                   (outer membrane surface area)",
            "a_i = a_o · (1 − f_p)^(2/3)                                (inner membrane surface area)",
            "v_p = v · f_p ,   v_c = v · (1 − f_p)                      (periplasm / cytoplasm volumes)",
            "radius_A = r · 1e4 ,  half_len_A = (l_cyl/2) · 1e4         (µm → Å)",
            "inner_radius_A = radius_A · (1 − f_p)^(1/3) ,  inner_half_len_A = half_len_A · (1 − f_p)^(1/3)",
        ],
        symbols={
            "mass": "cell mass (fg), read from listeners.mass.cell_mass",
            "density": "fixed cell density (g/mL), config 'density_g_per_ml' (default 1.1)",
            "v": "total cell volume (fL = µm³), = mass/density",
            "w": "fixed cell width/diameter (µm), config 'width_um' (default 1.0)",
            "r": "capsule cap radius = w/2 (µm)",
            "l_cyl": "length of the cylindrical (non-capped) section (µm)",
            "length": "tip-to-tip capsule length = l_cyl + w (µm)",
            "a_o": "outer membrane surface area (µm²)",
            "a_i": "inner membrane surface area (µm²)",
            "f_p": "periplasm volume fraction, config 'periplasm_fraction' (default 0.2)",
            "v_p": "periplasm volume (fL)",
            "v_c": "cytoplasm volume (fL)",
            "radius_A": "outer cap radius in Å (r · 1e4)",
            "half_len_A": "half the cylindrical section length in Å",
            "inner_radius_A": "inner-membrane cap radius in Å, scaled by (1-f_p)^(1/3)",
            "inner_half_len_A": "half the inner-membrane cylindrical length in Å",
        },
        inputs={
            "mass": "Reads the nested cell_mass sub-store (listeners.mass.cell_mass) as the sole driver of shape; accepts a pint Quantity[fg] or a plain float.",
        },
        outputs={
            "shape": "Overwrites the flat shape dict each step (map[overwrite[float]]): mass_fg, density_g_per_ml, width_um, volume_fl, length_um, outer_sa_um2, inner_sa_um2, periplasm_fraction, periplasm_vol_fl, cytoplasm_vol_fl, radius_A, half_len_A, inner_radius_A, inner_half_len_A.",
        },
        assumptions=[
            "Cell shape is a capsule (cylinder capped by two hemispheres) of fixed width; growth is purely by elongation, so length derives from volume.",
            "Width, density, and periplasm fraction are fixed config constants, not simulated quantities.",
            "Density default is 1.1 g/mL; periplasm fraction default is 0.2 of total volume.",
            "Output is overwritten (SET), not accumulated, each step — 'shape' reports current geometry, not a running sum.",
        ],
    )

    config_schema = {
        "width_um": {"_type": "float", "_default": 1.0},          # fixed cell width (diameter)
        "density_g_per_ml": {"_type": "float", "_default": 1.1},   # fixed cell density
        "periplasm_fraction": {"_type": "float", "_default": 0.2}, # f_p
    }

    def inputs(self):
        # Wire the whole listeners.mass sub-store (schema realization needs a
        # nested dict schema with a dict-form leaf — a bare "float" string can't
        # be annotated onto the cell_mass store). We only read cell_mass.
        return {"mass": {"cell_mass": {"_type": "quantity[float,fg]",
                                       "_default": 0.0}}}

    def outputs(self):
        # map[overwrite[float]] (a container type) resolves to a dict schema, and
        # overwrite makes each tick SET the value rather than accumulate — the
        # step reports the current geometry each step, it doesn't add to it. A
        # plain map[float] would sum the per-tick shapes; a bare "any" resolves to
        # a string that schema realization can't annotate in the full composite.
        return {"shape": "map[overwrite[float]]"}

    def update(self, state, interval=None):
        # cell_mass arrives as a pint Quantity (live baseline listeners) nested
        # under the mass sub-store, or as a plain float / {"mass_fg": ...} in
        # tests and npz snapshots — accept both shapes.
        m = state.get("mass")
        if isinstance(m, dict):
            raw = m.get("cell_mass")
        elif m is not None:
            raw = m
        else:
            raw = state.get("mass_fg")
        mass = float(getattr(raw, "magnitude", raw) or 0.0)
        shape = shape_from_mass(
            mass, self.config["width_um"], self.config["density_g_per_ml"],
            self.config["periplasm_fraction"])
        # Drop the Capsule objects + the envelope sub-dict so the emitted shape
        # stays a flat map[float]; their numbers are retained as radius_A /
        # half_len_A / inner_radius_A / inner_half_len_A.
        for k in ("capsule", "inner_capsule", "envelope"):
            shape.pop(k, None)
        return {"shape": shape}
