"""Flagellar filament nucleation — the rare, rate-limited start of a new
flagellum, distinct from (fast) filament growth.

Added 2026-08-06, part of Maya Abdalla's flagella-cascade investigation.
Companion to flagella_filament_elongation.py -- see that module's docstring
for the full story of why filament growth had to be split into a separate
nucleation + elongation pair of Steps (in short: a real ~20,000-subunit FliC
stoichiometry crammed into a single Gillespie complexation reaction makes
the SSA's combinatorial propensity calculation blow up numerically).

Mechanism
---------
Consumes the same real stoichiometry CPLX0-7452_RXN already requires for
everything EXCEPT the filament itself and its cap: 1x FLAGELLAR-MOTOR-
COMPLEX, 120x FlgE (hook), 11x FlgK, 11x FlgL (hook-filament junction) --
see complexation_reactions_modified.tsv. When enough of each is available,
creates ONE new `nascent_flagellum` unique molecule (filament_length=0),
which flagella_filament_elongation.py then grows.

Rate: this is deliberately rate-limited, not "as fast as material allows" --
grounded in Chang, Sung & Hong (2025) Biochem Biophys Reports 42:102051,
"Intrinsic clustering of flagellar basal body proteins in E. coli," which
found that excess FliF/FlhA protein preferentially accumulates into
PRE-EXISTING basal-body clusters rather than nucleating new ones (a
diffusion-capture, autocatalytic mechanism) -- i.e. real E. coli doesn't
turn a flood of subunit into a flood of new flagella; existing structures
absorb it instead. That paper gives the qualitative mechanism but not a
specific rate; the rate constant used here is a DERIVED ESTIMATE (not a
direct literature citation), back-calculated from Sisti et al. 2017 (Sci
Rep 7:41189)'s observed ~2-8 flagella/cell reached over one generation:
reaching ~5 flagella over a ~50 min generation implies roughly one new
nucleation event every ~10 min (rate ~= 1/600s ~= 0.00167/s). Flagged
plainly as an estimate to revisit, not a measured constant -- same standard
as the FlhDC degradation rate and other estimated constants in this
investigation.

Ordered in the composite flow: before ecoli-flagella-filament-elongation
(new nascent flagella should be visible to elongation the same tick they're
created, matching the FlgM-secretion -> transcript_initiation ordering
tolerance already established elsewhere in flagella_regulation).

BUG FOUND AND FIXED (2026-08-06, same day): the first implementation
computed n_events = round(nucleation_rate * timestep) EVERY tick. With
nucleation_rate ~= 0.00167/s and a ~2s timestep, that's round(0.0033) --
which rounds to exactly zero on every single tick, forever, regardless of
how much simulated time passes (a direct test confirmed zero nascent
flagella after a full 2400s run despite ample available material). This
differs from other rate-based Steps in this codebase (e.g.
flagella_flgm_secretion.py) that multiply their rate by a large COUNT, not a
fixed small expected-event number -- those cross the 0.5 rounding
threshold easily; a single-event-per-interval process never does. Fixed by
switching from a per-tick probabilistic rate to a fixed-interval trigger:
next_update_time is scheduled ~1/nucleation_rate seconds ahead (not just
one timestep), and each time the Step actually fires, it deterministically
creates exactly one nascent flagellum (if material allows) rather than
trying to accumulate sub-1 probability across many small ticks.

SECOND BUG FOUND AND FIXED (2026-08-11): next_update_time's schema default
is 0.0 (same shared bigraph pattern every flagella Step uses), and
global_time also starts at/near 0.0 -- so update_condition's
next_update_time <= global_time is trivially true on the very first tick,
firing nucleation immediately at simulation start instead of waiting the
intended ~1/nucleation_rate seconds like every SUBSEQUENT event correctly
does. This gives every single-generation run one "free" nucleation event
that skips the rate limit meant to make this Step rare. Harmless for the
other flagella Steps (they're deliberately meant to fire immediately, every
tick -- see flagella_motor_switch_assembly.py), but wrong specifically for
this one. Fixed in update(): the very first call (detected the same way
bulk indices are lazily resolved, self.idx is None) only starts the clock
(schedules next_update_time one interval out) and returns without creating
a flagellum, so the first nucleation waits its full turn exactly like every
one after it.
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.schema_types import NASCENT_FLAGELLUM_ARRAY


NAME = "ecoli-flagella-filament-nucleation"
TOPOLOGY = {
    "bulk": ("bulk",),
    "nascent_flagellum": ("unique", "nascent_flagellum"),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_filament_nucleation"),
    "global_time": ("global_time",),
}

# Real stoichiometry for everything a nascent flagellum needs BEFORE filament
# growth begins -- matches CPLX0-7452_RXN's non-filament components exactly
# (complexation_reactions_modified.tsv).
_NUCLEATION_REQUIREMENTS = {
    "FLAGELLAR-MOTOR-COMPLEX[j]": 1,
    "G361-MONOMER[c]": 120,   # FlgE, hook
    "EG11967-MONOMER[e]": 11,  # FlgK
    "EG11545-MONOMER[e]": 11,  # FlgL
}


class FlagellaFilamentNucleation(Step):
    """Rate-limited creation of new nascent_flagellum unique molecules."""

    description = (
        "FlagellaFilamentNucleation — rare start of a new flagellum.\n\n"
        "    fires once every ~1/nucleation_rate seconds (fixed-interval trigger,\n"
        "    NOT a per-tick probability -- see module docstring for the rounding\n"
        "    bug this replaced), creating 1 new nascent_flagellum if material allows.\n"
        "  Consumes 1x motor complex + 120x FlgE + 11x FlgK + 11x FlgL per event.\n"
        "  Deliberately slow -- real basal-body clustering favors growing existing\n"
        "  structures over nucleating new ones (Chang/Sung/Hong 2025)."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "flhdc_motor_complex_id": {"_type": "string", "_default": "FLAGELLAR-MOTOR-COMPLEX[j]"},
        "flgE_id": {"_type": "string", "_default": "G361-MONOMER[c]"},
        "flgK_id": {"_type": "string", "_default": "EG11967-MONOMER[e]"},
        "flgL_id": {"_type": "string", "_default": "EG11545-MONOMER[e]"},
        # DERIVED ESTIMATE, not a literature-measured rate -- see module
        # docstring. ~1 nucleation event per ~10 min.
        "nucleation_rate": {"_type": "float", "_default": 0.00167},
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
        self.nucleation_rate = self.parameters["nucleation_rate"]
        # Fixed-interval trigger, not a per-tick probability -- see module
        # docstring "BUG FOUND AND FIXED" for why round(rate*timestep) never
        # fires for a slow, single-event process. ~599s for the default rate.
        self.nucleation_interval = 1.0 / self.nucleation_rate
        self._ids = [
            self.parameters["flhdc_motor_complex_id"],
            self.parameters["flgE_id"],
            self.parameters["flgK_id"],
            self.parameters["flgL_id"],
        ]
        self._per_event = np.array([1, 120, 11, 11])
        self.idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        first_call = self.idx is None
        if self.idx is None:
            bulk_ids = states["bulk"]["id"]
            self.idx = bulk_name_to_idx(self._ids, bulk_ids)

        if first_call:
            # BUG FIX (2026-08-11): next_update_time defaults to 0.0, same as
            # global_time at simulation start, so the very first call would
            # otherwise fire immediately -- see module docstring "SECOND BUG
            # FOUND AND FIXED". Treat the first call as purely starting the
            # clock: schedule the first real opportunity one full interval
            # out, without creating a flagellum.
            return {"next_update_time": states["global_time"] + self.nucleation_interval}

        available = counts(states["bulk"], self.idx)
        material_limited_max = int(np.min(available // self._per_event))
        n_events = max(0, min(1, material_limited_max))

        update = {
            "next_update_time": states["global_time"] + self.nucleation_interval,
        }
        if n_events == 0:
            return update

        update["bulk"] = [(self.idx, -n_events * self._per_event)]
        update["nascent_flagellum"] = {
            "add": {
                "filament_length": np.zeros(n_events, dtype=np.int64),
            }
        }
        return update
