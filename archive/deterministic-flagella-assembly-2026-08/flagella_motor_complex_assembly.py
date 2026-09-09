"""Flagellar motor complex (basal body) assembly — moved out of Gillespie
SSA, same numerical reason as flagella_motor_switch_assembly.py.

Added 2026-08-06, part of Maya Abdalla's flagella-cascade investigation.
See flagella_motor_switch_assembly.py's docstring for the full diagnostic
story (why FLAGELLAR-MOTOR-COMPLEX_RXN -- not just CPLX0-7452_RXN or
CPLX0-7450_RXN -- had to be excluded from
sim_data.process.complexation: five simultaneous double-digit coefficients
multiplying together in one reaction's propensity calculation, confirmed by
direct testing that excluding either of the other two alone was not
sufficient).

Deliberately NOT rate-limited, same reasoning as the motor-switch-complex
Step: this is ordinary fast complex assembly in real biology, just moved
out of Gillespie SSA for numerical reasons.

Two real structural gaps fixed together with this reaction's real
stoichiometry back in complexation_reactions_modified.tsv, both preserved
here:
  (1) The export apparatus (CPLX0-7451) was previously built by its own
      reaction but never consumed by anything downstream -- wired in here
      as a genuine reactant.
  (2) FliG/FliM/FliN are no longer consumed directly -- they're consumed by
      flagella_motor_switch_assembly.py to form CPLX0-7450 first, which
      THIS Step then consumes as a single unit.

Real stoichiometry (cryo-EM structural studies -- see
complexation_reactions_modified.tsv for full citations): FlgH/FlgI=26
(L-ring/P-ring), FliE=6/FlgB=5/FlgC=6/FlgF=5 (proximal rod), FlgG=24
(distal rod), MotA~55/MotB~22 (stator, derived estimate), FliL=2 (per
Maya's own "Master Flagella Info" spreadsheet).

HIERARCHY FIX (2026-08-11): CPLX0-7450 (C-ring) removed as a direct
requirement here. Real assembly order (Minamino & Namba 2008 Nature;
Chevance & Hughes 2008 Nat Rev Microbiol) has the export apparatus insert
into the central pore of a PRE-FORMED MS-C-ring, not assemble independently
and merge here later -- so CPLX0-7450 is now consumed one step upstream, by
CPLX0-7451_RXN itself (see complexation_reactions_modified.tsv), and
CPLX0-7451 arriving here already represents "export apparatus inserted into
ring." Consuming CPLX0-7450 again at this step would double-count the
C-ring for one motor complex.

MS-RING ORDERING FIX (2026-08-11, same investigation): FliF (MS-ring, x34)
also removed as a direct requirement here, for the identical reason one
level further upstream -- the MS-ring forms before the C-ring, not
alongside the stator/rod parts at the very last stage. FliF is now
consumed by CPLX0-7450_RXN itself (see complexation_reactions_added.tsv),
so it arrives here already folded into CPLX0-7451 (via the CPLX0-7450
dependency chained through CPLX0-7451_RXN).
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "ecoli-flagella-motor-complex-assembly"
TOPOLOGY = {
    "bulk": ("bulk",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_motor_complex_assembly"),
    "global_time": ("global_time",),
}

_REQUIREMENTS = {
    # "CPLX0-7450[i]": 1,   # REMOVED 2026-08-11: C-ring now consumed one step
    #                       # upstream, by CPLX0-7451_RXN itself -- see
    #                       # complexation_reactions_modified.tsv HIERARCHY FIX.
    #                       # Kept here per standing preserve-old-code rule.
    # "FLIF-FLAGELLAR-MS-RING[i]": 34,   # REMOVED 2026-08-11: MS-ring now
    #                       # consumed two steps upstream, folded into
    #                       # CPLX0-7450_RXN itself -- see
    #                       # complexation_reactions_added.tsv MS-RING
    #                       # ORDERING FIX. Kept here per standing
    #                       # preserve-old-code rule.
    "CPLX0-7451[j]": 1,                                # export apparatus (now transitively includes MS-ring + C-ring, post both hierarchy fixes)
    "FLGH-FLAGELLAR-L-RING[j]": 26,                     # L-ring
    "MOTA-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]": 55,       # stator
    "MOTB-FLAGELLAR-MOTOR-STATOR-PROTEIN[i]": 22,       # stator
    "FLGB-FLAGELLAR-MOTOR-ROD-PROTEIN[j]": 5,           # proximal rod
    "FLGC-FLAGELLAR-MOTOR-ROD-PROTEIN[j]": 6,           # proximal rod
    "FLGF-FLAGELLAR-MOTOR-ROD-PROTEIN[j]": 5,           # proximal rod
    "FLGG-FLAGELLAR-MOTOR-ROD-PROTEIN[o]": 24,          # distal rod
    "EG10322-MONOMER[j]": 2,                            # FliL
    "EG11346-MONOMER[p]": 6,                            # FliE
}


class FlagellaMotorComplexAssembly(Step):
    """Fast, deterministic assembly of FLAGELLAR-MOTOR-COMPLEX."""

    description = (
        "FlagellaMotorComplexAssembly — basal-body assembly into FLAGELLAR-MOTOR-COMPLEX.\n\n"
        "    n_formed = min(available // per_unit)\n"
        "  Deterministic, not rate-limited -- moved out of Gillespie SSA purely for\n"
        "  numerical reasons. Wires in CPLX0-7451 (export apparatus, previously\n"
        "  orphaned) and CPLX0-7450 (motor switch complex) as real reactants."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "product_id": {"_type": "string", "_default": "FLAGELLAR-MOTOR-COMPLEX[j]"},
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
        self._reactant_ids = list(_REQUIREMENTS.keys())
        self._per_unit = np.array(list(_REQUIREMENTS.values()))
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
