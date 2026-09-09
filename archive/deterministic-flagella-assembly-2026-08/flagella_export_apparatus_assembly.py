"""Flagellar export apparatus (CPLX0-7451) assembly — moved out of Gillespie
SSA for architectural consistency, not a numerical necessity like the other
flagella Steps.

Added 2026-08-11, part of Maya Abdalla's flagella-cascade investigation.

Why this moved out of ecoli-complexation
------------------------------------------
Unlike CPLX0-7450_RXN, FLAGELLAR-MOTOR-COMPLEX_RXN, and CPLX0-7452_RXN
(excluded from sim_data.process.complexation because their real
stoichiometry -- FliN x111, FliC x5,000-20,000 -- blows up Gillespie SSA's
combinatorial propensity term), CPLX0-7451_RXN's largest coefficient
(FlhA x9) was never a numerical problem on its own; it stayed in the
ordinary SSA complexation pool.

The HIERARCHY FIX (2026-08-11, complexation_reactions_modified.tsv) added
CPLX0-7450 (C-ring) as a genuine reactant here -- real assembly order
(Minamino & Namba 2008 Nature; Chevance & Hughes 2008 Nat Rev Microbiol)
has the export apparatus insert into a PRE-FORMED C-ring, not assemble
independently. But CPLX0-7450 only ever exists transiently inside
flagella_motor_switch_assembly.py's single deterministic tick -- nothing
else touches it before or after. Leaving CPLX0-7451_RXN in SSA meant a
molecule produced by one mechanism (deterministic, once per tick) had to be
consumed by a completely different one (stochastic propensity, its own
timing within the tick) -- confirmed by direct testing to visibly stall
ongoing motor-complex replenishment (motor-complex pool drained 6->3 over
2400s instead of the stable 4-6 oscillation seen before the hierarchy fix).

Moving this reaction into a Step removes that cross-mechanism race. It does
NOT remove the real FlhA bottleneck -- FlhA's own low, translation-limited
copy number (observed staying in single digits against a /9 requirement)
still caps n_formed via the same min(available // per_unit) formula every
other flagella Step uses; this Step only changes HOW the reaction fires
(deterministically, the instant reactants clear threshold) not what
resource limits it.

Real stoichiometry (cryo-EM structural studies, see
complexation_reactions_modified.tsv for full citations): FlhA=9 (homo-
nonamer), FlhB=1, FliP:FliQ:FliR=5:4:1 (Kuhlen et al. 2018 Nat Struct Mol
Biol), FliH=12/FliI=6/FliJ=1 (independently confirmed literature match),
FliO=1 (OPEN QUESTION -- transient assembly scaffold, not necessarily part
of the mature complex; left as-is, see the TSV comment), CPLX0-7450=1 (the
new hierarchy-fix dependency).
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "ecoli-flagella-export-apparatus-assembly"
TOPOLOGY = {
    "bulk": ("bulk",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_export_apparatus_assembly"),
    "global_time": ("global_time",),
}

_REQUIREMENTS = {
    "CPLX0-7450[i]": 1,          # motor switch complex (C-ring) -- hierarchy fix, 2026-08-11
    "G370-MONOMER[i]": 9,        # FlhA (homo-nonamer)
    "G7028-MONOMER[i]": 1,       # FlhB
    "EG11224-MONOMER[j]": 1,     # FliO -- open question, see module docstring
    "EG11975-MONOMER[i]": 5,     # FliP
    "EG11976-MONOMER[j]": 4,     # FliQ
    "EG11977-MONOMER[i]": 1,     # FliR
    "EG11656-MONOMER[c]": 12,    # FliH
    "G377-MONOMER[c]": 6,        # FliI
    "G378-MONOMER[c]": 1,        # FliJ
}


class FlagellaExportApparatusAssembly(Step):
    """Fast, deterministic assembly of CPLX0-7451 from CPLX0-7450 + export-gate monomers."""

    description = (
        "FlagellaExportApparatusAssembly — CPLX0-7450 + FlhA/FlhB/FliO/FliP/FliQ/"
        "FliR/FliH/FliI/FliJ -> CPLX0-7451 (export apparatus).\n\n"
        "    n_formed = min(available // per_unit)\n"
        "  Deterministic, not rate-limited -- moved out of Gillespie SSA 2026-08-11 for\n"
        "  architectural consistency (removes a cross-mechanism race with the\n"
        "  deterministic C-ring Step), not because its own stoichiometry was too large."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "product_id": {"_type": "string", "_default": "CPLX0-7451[j]"},
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
