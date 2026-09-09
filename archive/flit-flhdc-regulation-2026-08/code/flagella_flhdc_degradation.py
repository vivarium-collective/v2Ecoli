"""FlhD4C2 (CPLX0-3930) degradation — ClpXP-mediated master-regulator turnover.

Added 2026-08-05 as part of Maya Abdalla's flagella-cascade investigation, to
address the flagella-count runaway documented there (flagella-02-flagella-
count-unbounded-runaway / flagella-02-missing-flhdc-complex-degradation in
workspace/investigations/flagella-cascade/studies/flagella-02-transcription-
regulation/study.yaml).

Mechanism
---------
The standard v2ecoli protein_degradation process only degrades individual
monomers, never assembled complexes (see protein_degradation.py's TODO at the
top of ProteinDegradation) -- so once FlhD and FlhC monomers assemble into
CPLX0-3930 (FlhD4C2), there is no decay sink for the complex at all; it can
only be consumed by reverse complexation. Real E. coli/Salmonella actively
degrade the ASSEMBLED complex specifically via the ATP-dependent ClpXP
protease -- Tomoyasu et al. 2003 (Mol Microbiol 48:443) show ClpXP recognizes
FlhD2C2 in complex form and does not appreciably degrade the free individual
subunits, and that FlhC's half-life is ~5x longer in a clpXP-null mutant. This
growth-rate-coupled turnover is the primary known control point on flagellar
NUMBER (Sisti et al. 2017, Sci Rep 7:41189: mean 7.8 flagella/cell at fast
growth vs 2.4/cell at slow growth, tracing to flhDC transcript/regulator
level, not a geometric/nucleation-site limit).

v2ecoli has neither half of this pathway: no ClpXP process exists anywhere in
the repo, and complexes are categorically excluded from the standard
degradation process. This Step is a minimal, targeted addition -- first-order
decay of CPLX0-3930 only -- not a full ClpXP mechanistic model.

Rate constant
-------------
LITERATURE-ANCHORED ESTIMATE, not a directly-measured E. coli/CPLX0-3930-
specific rate: Tomoyasu et al. 2003 (the direct Salmonella/E. coli source) is
paywalled and its exact quantitative half-life was not accessible. Claret &
Hughes 2000 (J Bacteriol 182:833-836) measured FlhD/FlhC half-lives directly
in the closely related Proteus mirabilis system (same conserved master
regulators): ~5-6 min during early swarm differentiation, dropping to ~2 min
after peak differentiation. This Step uses a ~4 min half-life midpoint as its
default (k = ln(2) / 240s ~= 0.00289 /s). Treat this as a documented starting
estimate to refine if better E. coli-specific kinetics become available or if
empirical calibration against known flagella-count data suggests otherwise --
NOT a precise literature-measured constant.

Ordered in the composite flow: inserted first within the flagella_regulation
feature, before ecoli-flagella-flgm-secretion and ecoli-flagella-transcription-
regulation, so the SUM-gate's X = FlhDC/(K_flhDC + FlhDC) term reads the
post-degradation FlhDC count each tick (degradation net of that tick's
complexation-driven synthesis, mirroring continuous real-cell turnover).

Refs: Tomoyasu T et al. (2003) Mol Microbiol 48:443; Claret L & Hughes KT
(2000) J Bacteriol 182:833; Sisti F, Ha DG, O'Toole GA, Hozbor D, Fujiwara T
et al. (2017) Sci Rep 7:41189.
"""


from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "ecoli-flhdc-degradation"
TOPOLOGY = {
    "bulk": ("bulk",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flhdc_degradation"),
    "global_time": ("global_time",),
}


class FlhDCDegradation(Step):
    """First-order degradation of the assembled FlhD4C2 (CPLX0-3930) complex."""

    description = (
        "FlhDCDegradation — ClpXP-mediated turnover of the FlhD4C2 master regulator.\n\n"
        "    degraded = round(flhDC_count * degradation_rate * timestep)\n"
        "  Complexes have no decay sink in the standard protein_degradation process;\n"
        "  real ClpXP degrades assembled FlhD4C2 specifically, the primary growth-rate\n"
        "  control on flagellar number (Sisti et al. 2017; Tomoyasu et al. 2003)."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "flhdc_id": {"_type": "string", "_default": "CPLX0-3930[c]"},
        # First-order rate constant, /s -- see module docstring for full
        # provenance/caveats. ~4 min half-life midpoint from Claret & Hughes
        # 2000 (Proteus FlhD/FlhC half-lives, ~2-6 min): ln(2)/240s ~= 0.00289.
        "degradation_rate": {"_type": "float", "_default": 0.00289},
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
        self.degradation_rate = self.parameters["degradation_rate"]
        # Bulk index resolved lazily against the live bulk array.
        self.flhdc_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.flhdc_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.flhdc_idx = bulk_name_to_idx(self.parameters["flhdc_id"], bulk_ids)

        flhdc_count = counts(states["bulk"], self.flhdc_idx)

        degraded = 0
        if flhdc_count > 0:
            degraded = min(
                int(flhdc_count),
                int(round(flhdc_count * self.degradation_rate * states["timestep"])),
            )

        return {
            "bulk": [(self.flhdc_idx, -degraded)],
            "next_update_time": states["global_time"] + states["timestep"],
        }
