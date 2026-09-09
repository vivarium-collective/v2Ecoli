"""Flagellar motor switch complex (C-ring) assembly — moved out of Gillespie
SSA for the same numerical reason as filament elongation, but modeled as
fast/immediate, not rate-limited.

Added 2026-08-06, part of Maya Abdalla's flagella-cascade investigation.

Why this isn't a Gillespie complexation reaction
--------------------------------------------------
CPLX0-7450_RXN (FliG(-34) + FliM(-34) + FliN(-111) -> CPLX0-7450) was added
this session as a real, previously-missing staged intermediate (see
complexation_reactions_added.tsv). On its own it's not obviously worse than
other large-coefficient reactions already safe in this model (e.g. the
pre-existing, unmodified LUMAZINESYN-CPLX_RXN has a coefficient of 60) --
but combined with FLAGELLAR-MOTOR-COMPLEX_RXN's own several simultaneously
large coefficients, ParCa's step_05_fit_condition.py (calculateBulkDistributions,
which runs the WHOLE complexation network through Gillespie SSA with an
enormous 2**31 s time step from raw expression-derived counts) reproducibly
hit "failed simulation: total propensity is NaN". Direct diagnosis (excluding
CPLX0-7452_RXN alone, then also CPLX0-7450_RXN alone) confirmed neither
exclusion alone was sufficient -- FLAGELLAR-MOTOR-COMPLEX_RXN's own combination
of FIVE simultaneous double-digit coefficients (FlgH:26, MotA:55, MotB:22,
FlgG:24, FliF:34) is implicated too, likely because propensity for a
multi-reactant reaction is a PRODUCT of per-reactant combinatorial terms --
several large-but-individually-survivable coefficients multiplied together
can still overflow even when no single one does on its own.

Both CPLX0-7450_RXN and FLAGELLAR-MOTOR-COMPLEX_RXN are excluded from
sim_data.process.complexation (see complexation.py's RUNTIME_EXCLUDED_REACTIONS)
and replaced by this Step and flagella_motor_complex_assembly.py.

Unlike filament elongation (genuinely slow, incremental biology -- Renault
et al. 2017) or nucleation (genuinely rare -- Chang/Sung/Hong 2025), ordinary
complex assembly from already-available subunits IS supposed to be fast --
the ONLY reason this moved out of Gillespie is numerical, not biological.
So this Step is deliberately NOT rate-limited: it converts as much material
as is available every tick, matching the character of ordinary fast
complexation (uniform sim_data.constants.complexation_rate), just computed
deterministically instead of stochastically.

Real stoichiometry (all from cryo-EM structural studies, see
complexation_reactions_added.tsv for full citations): FliG=34 (C-ring
34-fold symmetry), FliM=34, FliN=111+/-13 ("Precise Measurement of the
Stoichiometry of the Adaptive Bacterial Flagellar Switch," PMC10128058).

MS-RING ORDERING FIX (2026-08-11): FliF (MS-ring, x34) added as a fourth
reactant. Real assembly order (Minamino & Namba 2008 Nature; Chevance &
Hughes 2008 Nat Rev Microbiol) has the MS-ring form FIRST, then the
C-ring assembles onto its cytoplasmic face -- before this fix, FliF was
consumed much later, lumped into flagella_motor_complex_assembly.py
alongside unrelated stator/rod parts. CPLX0-7450 now represents the
merged "MS-ring + C-ring" structure (a reasonable single-stage merge --
many structural papers describe these as forming one continuous ring
assembly, not always two independently-isolable intermediates).
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "ecoli-flagella-motor-switch-assembly"
TOPOLOGY = {
    "bulk": ("bulk",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_motor_switch_assembly"),
    "global_time": ("global_time",),
}

# Old requirements (FliG/FliM/FliN only), kept per standing preserve-old-code
# rule:
# _REQUIREMENTS = {
#     "FLIG-FLAGELLAR-SWITCH-PROTEIN[i]": 34,
#     "FLIM-FLAGELLAR-C-RING-SWITCH[i]": 34,
#     "FLIN-FLAGELLAR-C-RING-SWITCH[m]": 111,
# }
_REQUIREMENTS = {
    "FLIF-FLAGELLAR-MS-RING[i]": 34,        # MS-ring -- MS-RING ORDERING FIX, 2026-08-11
    "FLIG-FLAGELLAR-SWITCH-PROTEIN[i]": 34,
    "FLIM-FLAGELLAR-C-RING-SWITCH[i]": 34,
    "FLIN-FLAGELLAR-C-RING-SWITCH[m]": 111,
}


class FlagellaMotorSwitchAssembly(Step):
    """Fast, deterministic assembly of CPLX0-7450 from FliF/FliG/FliM/FliN."""

    description = (
        "FlagellaMotorSwitchAssembly — FliF/FliG/FliM/FliN -> CPLX0-7450 (MS-ring + C-ring).\n\n"
        "    n_formed = min(available // per_unit)\n"
        "  Deterministic, not rate-limited -- moved out of Gillespie SSA purely for\n"
        "  numerical reasons (combinatorial propensity overflow), not biological ones."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "fliF_id": {"_type": "string", "_default": "FLIF-FLAGELLAR-MS-RING[i]"},
        "fliG_id": {"_type": "string", "_default": "FLIG-FLAGELLAR-SWITCH-PROTEIN[i]"},
        "fliM_id": {"_type": "string", "_default": "FLIM-FLAGELLAR-C-RING-SWITCH[i]"},
        "fliN_id": {"_type": "string", "_default": "FLIN-FLAGELLAR-C-RING-SWITCH[m]"},
        "product_id": {"_type": "string", "_default": "CPLX0-7450[i]"},
    }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "timestep": {"_type": "float[s]", "_default": 2.0},
            "next_update_time": {"_type": "overwrite[float[s]]", "_default": 0.0},
            "global_time": {"_type": "float[s]", "_default": 0.0},
        }

    def outputs(self):
        return {
            "bulk": "bulk_array",
            "next_update_time": "overwrite[float[s]]",
        }

    def initialize(self, config):
        self._reactant_ids = [
            self.parameters["fliF_id"],
            self.parameters["fliG_id"],
            self.parameters["fliM_id"],
            self.parameters["fliN_id"],
        ]
        self._per_unit = np.array([34, 34, 34, 111])
        self.product_id = self.parameters["product_id"]
        self.reactant_idx = None
        self.product_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.reactant_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.reactant_idx = bulk_name_to_idx(self._reactant_ids, bulk_ids)
            self.product_idx = bulk_name_to_idx(self.product_id, bulk_ids)

        available = counts(states["bulk"], self.reactant_idx)
        n_formed = int(np.min(available // self._per_unit))

        update = {"next_update_time": states["global_time"] + states["timestep"]}
        if n_formed <= 0:
            return update

        all_idx = np.concatenate((self.reactant_idx, [self.product_idx]))
        deltas = np.concatenate((-n_formed * self._per_unit, [n_formed]))
        update["bulk"] = [(all_idx, deltas)]
        return update
