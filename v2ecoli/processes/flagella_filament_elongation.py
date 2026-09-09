"""Flagellar filament elongation — incremental FliC polymerization.

Added 2026-08-06, part of Maya Abdalla's flagella-cascade investigation.
Grows each nascent_flagellum's filament_length one subunit-batch at a
time per tick (mirrors polypeptide_elongation.py's treatment of
translation), instead of one giant Gillespie complexation event --
20,000 copies of the same molecule in one reaction blows up SSA
propensity calculations combinatorially, and isn't how real export works
anyway (FliC is added incrementally at the distal tip).

Rate law: dL/dt = a / (b + L)  [subunits/s]
Citation: Renault et al. 2017, eLife 6:e23136, "Bacterial flagella grow
through an injection-diffusion mechanism."
Current: rate_a=15,556, rate_b=575 subunits.
CORRECTED (2026-09-01): rate_a was 26,450, which didn't match Renault's
own fitted k_on for either of their two datasets (33.35/s Fig 2,
27.09/s Fig 3). rate_b=575 already closely matched Fig 3's derived value
(~574) -- Fig 3 is the stronger dataset (six-color labeling vs Fig 2's
three, 291 filaments / 1,276 data points, wider dynamic range). Re-derived
rate_a=15,556 to match Fig 3 self-consistently (implied k_on =
15,556/575 = 27.05/s, vs Fig 3's own 27.09/s). Old value kept per
standing preserve-old-code rule -- see config_schema below.

target_length = 5,000 subunits. Real range is 20,000-40,000 (PMC7696725);
cut for practical single-generation simulation windows (completion time
scales ~L^2/a). Kept in sync with CPLX0-7452_RXN's FliC coefficient in
complexation_reactions_modified.tsv.

Multiple simultaneous filaments fair-share the same combined FliC pool
(free + FLIS-FLIC-CPLX), scaled down proportionally if demand exceeds
supply -- no draw-order bias.

On completion: consumes 5x FliD, deletes the nascent_flagellum, adds +1
real CPLX0-7452.

FliS chaperone recycling: elongation draws preferentially from
FLIS-FLIC-CPLX (protected pool) before free FliC, releasing FliS back on
consumption -- Sci Rep 8:11115 (2018), "FliW and FliS are released during
flagellin export... recycled." FliS binds FliC as a homodimer (Auvray,
Thomas, Fraser & Hughes 2001, J Mol Biol 308:221-229: "FliS homodimers
bind to FliC monomers"), so each unit of complex consumed releases 2 free
FliS monomers, not 1. Binding affinity: Muskotal et al. 2006, FEBS Lett
580:3916, Kd=5.26e-8 M.
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import NASCENT_FLAGELLUM_ARRAY


NAME = "ecoli-flagella-filament-elongation"
TOPOLOGY = {
    "bulk": ("bulk",),
    "nascent_flagellum": ("unique", "nascent_flagellum"),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_filament_elongation"),
    "global_time": ("global_time",),
}

# TARGET_LENGTH = 10000  # kept per standing preserve-old-code rule
# TARGET_LENGTH = 20000  # kept per standing preserve-old-code rule
TARGET_LENGTH = 5000   # subunits; matches CPLX0-7452_RXN's FliC coefficient


class FlagellaFilamentElongation(Step):
    """Incremental FliC polymerization onto nascent flagella."""

    description = (
        "FlagellaFilamentElongation — length-dependent incremental FliC addition.\n\n"
        "    rate(L) = a / (b + L)   [subunits/s], a~=15556, b~=575 (Renault et al. 2017, Fig 3)\n"
        "  Grows each nascent_flagellum's filament_length; fair-shares free FliC across\n"
        "  simultaneous filaments; on reaching target_length (5,000 subunits), consumes\n"
        "  5x FliD and converts the unique molecule into +1 real CPLX0-7452 bulk count."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "fliC_id": {"_type": "string", "_default": "EG10321-MONOMER[e]"},
        "fliD_id": {"_type": "string", "_default": "EG10841-MONOMER[e]"},
        "flagellum_id": {"_type": "string", "_default": "CPLX0-7452[j]"},
        "fliD_per_completion": {"_type": "integer", "_default": 5},
        "target_length": {"_type": "integer", "_default": TARGET_LENGTH},
        # Renault et al. 2017 (eLife 6:e23136), Fig 3 dataset, converted to
        # subunit-count units -- see module docstring's CORRECTED note
        # (2026-09-01). Old value kept per standing preserve-old-code rule:
        # "rate_a": {"_type": "float", "_default": 26450.0},
        "rate_a": {"_type": "float", "_default": 15556.0},
        "rate_b": {"_type": "float", "_default": 575.0},
        # FliS chaperone recycling (added 2026-08-21, see module docstring).
        "fliS_id": {"_type": "string", "_default": "EG11388-MONOMER[c]"},
        "flis_flic_cplx_id": {"_type": "string", "_default": "FLIS-FLIC-CPLX[e]"},
    }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "nascent_flagellum": {"_type": NASCENT_FLAGELLUM_ARRAY, "_default": []},
            "timestep": {"_type": "float[s]", "_default": 2.0},
            "next_update_time": {"_type": "overwrite[float[s]]", "_default": 0.0},
            "global_time": {"_type": "float[s]", "_default": 0.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "nascent_flagellum": NASCENT_FLAGELLUM_ARRAY,
            "next_update_time": "overwrite[float[s]]",
        }

    def initialize(self, config):
        self.fliD_per_completion = self.parameters["fliD_per_completion"]
        self.target_length = self.parameters["target_length"]
        self.rate_a = self.parameters["rate_a"]
        self.rate_b = self.parameters["rate_b"]
        self.fliC_idx = None
        self.fliD_idx = None
        self.flagellum_idx = None
        self.fliS_idx = None
        self.flis_flic_cplx_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.fliC_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.fliC_idx = bulk_name_to_idx(self.parameters["fliC_id"], bulk_ids)
            self.fliD_idx = bulk_name_to_idx(self.parameters["fliD_id"], bulk_ids)
            self.flagellum_idx = bulk_name_to_idx(
                self.parameters["flagellum_id"], bulk_ids
            )
            self.fliS_idx = bulk_name_to_idx(self.parameters["fliS_id"], bulk_ids)
            self.flis_flic_cplx_idx = bulk_name_to_idx(
                self.parameters["flis_flic_cplx_id"], bulk_ids
            )

        next_update = {"next_update_time": states["global_time"] + states["timestep"]}

        nascent = states["nascent_flagellum"]
        (lengths,) = attrs(nascent, ["filament_length"])
        n_active = lengths.size
        if n_active == 0:
            return next_update

        dt = states["timestep"]
        desired = np.round(self.rate_a / (self.rate_b + lengths) * dt).astype(np.int64)
        desired = np.maximum(desired, 0)

        # Old (pre-FliS-recycling) version, free FliC only, kept per
        # standing preserve-old-code rule:
        # fliC_available = counts(states["bulk"], self.fliC_idx)
        # total_desired = int(desired.sum())
        # if total_desired > fliC_available and total_desired > 0:
        #     # Fair-share: multiple simultaneous filaments compete for the
        #     # same free FliC pool -- scale everyone down proportionally
        #     # rather than letting draw order create bias.
        #     scale = fliC_available / total_desired
        #     desired = np.floor(desired * scale).astype(np.int64)

        # FliS chaperone recycling (2026-08-21, see module docstring): the
        # real, protected supply is FLIS-FLIC-CPLX + free FliC combined --
        # fair-share against the COMBINED pool, not free FliC alone.
        fliC_available_free = counts(states["bulk"], self.fliC_idx)
        fliC_available_protected = counts(states["bulk"], self.flis_flic_cplx_idx)

        # Removed 2026-08-28: a 0.3 draw-fraction cap used to sit here,
        # added to protect the shared equilibrium Step's legacy ODE solver
        # from overshooting negative on a large single-tick complex draw.
        # That solver no longer handles FLIS-FLIC-CPLX_RXN at all (see
        # flagella_flis_flic_equilibrium.py) -- its exact closed-form solve
        # can't overshoot negative by construction, so the cap's original
        # reason is gone. Removing it produced a single clean step-response
        # per division instead of repeated crash-and-recover cycling. Old
        # line kept per standing preserve-old-code rule:
        # MAX_COMPLEX_DRAW_FRACTION = 0.3
        # complex_draw_cap = int(fliC_available_protected * MAX_COMPLEX_DRAW_FRACTION)
        complex_draw_cap = int(fliC_available_protected)

        fliC_available_total = fliC_available_free + complex_draw_cap
        total_desired = int(desired.sum())
        if total_desired > fliC_available_total and total_desired > 0:
            # Fair-share: multiple simultaneous filaments compete for the
            # same combined FliC pool -- scale everyone down proportionally
            # rather than letting draw order create bias.
            scale = fliC_available_total / total_desired
            desired = np.floor(desired * scale).astype(np.int64)

        new_lengths = lengths + desired
        did_complete = new_lengths >= self.target_length
        n_complete = int(did_complete.sum())

        fliC_consumed = int(desired.sum())
        (protein_mass,) = attrs(nascent, ["massDiff_protein"])
        fliC_subunit_mass = states["bulk"]["protein_submass"][self.fliC_idx]
        new_protein_mass = protein_mass + desired * fliC_subunit_mass

        # Draw preferentially from the protected (FliS-escorted) pool
        # first, up to the numerical-stability cap -- exporting/
        # incorporating that FliC releases its chaperone back to the free
        # FliS pool, matching the real escort-cycle mechanism. Only fall
        # back to free (unprotected) FliC for any remaining demand.
        from_complex = min(fliC_consumed, complex_draw_cap)
        from_free = fliC_consumed - from_complex

        bulk_updates = []
        if from_complex > 0:
            bulk_updates.append((self.flis_flic_cplx_idx, -from_complex))
            # 2x: FliS binds as a homodimer (Auvray et al. 2001), so each
            # unit of complex consumed releases 2 free FliS monomers, not
            # 1 -- see module docstring's 2026-08-31 STOICHIOMETRY
            # CORRECTION. Old line kept per standing preserve-old-code
            # rule: bulk_updates.append((self.fliS_idx, from_complex))
            bulk_updates.append((self.fliS_idx, 2 * from_complex))
        if from_free > 0:
            bulk_updates.append((self.fliC_idx, -from_free))
        if n_complete > 0:
            bulk_updates.append(
                (self.fliD_idx, -n_complete * self.fliD_per_completion)
            )
            bulk_updates.append((self.flagellum_idx, n_complete))
        if bulk_updates:
            next_update["bulk"] = bulk_updates

        next_update["nascent_flagellum"] = {
            "delete": np.where(did_complete)[0],
            "set": {
                "filament_length": new_lengths,
                "massDiff_protein": new_protein_mass,
            },
        }
        return next_update
