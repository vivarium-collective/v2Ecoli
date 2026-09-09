"""FliT-mediated FlhD4C2 checkpoint — the real negative-feedback brake on the
flagellar master regulator, triggered by completed flagellar export.

Added 2026-08-06 as part of Maya Abdalla's flagella-cascade investigation, as
the biology-grounded mechanism for the flagella-count-runaway problem. An
earlier ad hoc flagella nucleation cap (flagella_nucleation_cap.py, a
hard-coded ceiling on complete-flagellum count) was tried first, then removed
2026-08-06 at Maya's explicit instruction -- see
feedback_biology_first_no_quick_fixes in the investigation's standing notes:
she rejected an arbitrary hard ceiling on flagella count in favor of a real,
citable regulatory mechanism, this Step. There is no longer any hard-coded
ceiling anywhere in the flagella_regulation feature; whatever limit on
flagella count emerges now falls out of the titration dynamics below (or
does not -- that is an open empirical question, not something assumed here).

Mechanism
---------
FliT (EG11389-MONOMER) is a dedicated FliD export chaperone. It exists as a
homodimer (FLIT-DIMER, see complexation_reactions_added.tsv) and binds free
FliD 1:1 (FLIT-FLID-CPLX_RXN, equilibrium_reactions.tsv) at the measured 2:1
FliT:FliD stoichiometry (Yamamoto & Kutsukake 2006, J Bacteriol 188:5124).
While FliD is being actively exported through a growing flagellum, FliT stays
bound and sequestered. Once a flagellum completes (CPLX0-7452_RXN consumes the
free-FliD pool directly, unmodified from the vendored reaction), the
FliT:FliD equilibrium shifts toward dissociation to replenish that pool,
releasing free FliT-dimer as a direct consequence of shared-reactant
mass-action chemistry.

That newly-freed FliT then does double duty as a checkpoint signal: free FliT
binds FlhD4C2 via FlhC and (a) sterically blocks FlhD4C2 from Class II
promoters and (b) markedly enhances its ClpXP-mediated degradation --
catalytically, at sub-stoichiometric FliT concentrations (Yakhnin et al.,
PMC4239645: "0.02 uM FliT2 sufficiently enhanced degradation of a 100-fold
molar excess of FlhC"). FliT itself is recycled, not consumed, by this
reaction (Yamamoto & Kutsukake 2006; Tomoyasu et al. 2003, Mol Microbiol
48:443 -- ClpXP degrades FlhD4C2 specifically in complex form, not FliT).

This is the real, literature-grounded feedback loop that ties flagellum
completion back to shutting off flhDC/Class III transcription: more complete
flagella -> more free FliT -> more FlhD4C2 degraded -> less Class II/III
transcription -> assembly slows. There is no hard-coded ceiling anywhere in
this Step; the limit on flagella number is an EMERGENT property of the
titration dynamics below, to be observed empirically rather than assumed.

Fast-equilibrium reduction (Utsey & Keener 2020, PLOS Comput Biol 16:e1007689)
-------------------------------------------------------------------------
Utsey & Keener's full ODE model tracks free FliT (v) and FliT-bound-FlhD4C2
(c2) as separate state variables:
    dc2/dt = kappa2 * x * v - (delta2 + gamma7) * c2
where x = FlhD4C2. Assuming c2 equilibrates fast relative to x's own
dynamics (their own reduced-model step, not an approximation we invented),
c2 tracks its quasi-steady-state value continuously:
    c2_ss = kappa2 * x * v / (delta2 + gamma7)
Writing u = v + c2 for the total FliT-dimer pool available to this binding
(the current free-FliT-dimer bulk count -- c2 is never stored as a separate
species here, only used transiently each tick to compute a degradation flux):
    v  = u / (1 + kappa2 * x / (delta2 + gamma7))
    c2 = u - v
FlhD4C2 is then degraded at rate delta2 * c2 per unit time (the ClpXP term);
free FliT (v) is left as-is in the bulk pool -- catalytic, recycled, exactly
as the biology requires.

Parameter provenance -- what is literature vs. what is an estimate
-------------------------------------------------------------------
delta2 (ClpXP degradation rate of FlhD4C2 while bound to FliT) is taken
directly from Utsey & Keener's Table 2: delta2 = 3 /min = 0.05 /s. This is a
rate constant (units of inverse time only), so it converts cleanly and is
used here as reported.

kappa2/(delta2+gamma7) (Utsey & Keener Table 2: kappa2=60, gamma7=0.3 /min)
is NOT used directly -- their model is explicitly nondimensionalized to
arbitrary concentration units for qualitative dynamical-systems analysis, not
calibrated to real E. coli molecule counts, so plugging their raw ratio in
against actual bulk counts here would be meaningless (their "concentration
units" and this model's molecule counts are not the same scale). What IS
taken from the paper is the FUNCTIONAL FORM of the titration -- a
competitive, saturating binding curve, v = u / (1 + x / K) -- not its
absolute numeric scale.

K_half (the FlhD4C2 count at which FliT-dimer becomes half-titrated into the
bound/degrading state) has no direct literature source in molecule-count
terms, and reusing K_flhDC=50 from the Kalir & Alon SUM-gate Step
(flagella_transcription_regulation.py / get_flagella_transcription_regulation_
config) is NOT itself an independent biological corroboration -- that 50 is
not literature-derived either; per git history (commit 606dc939) it was
carried over as-is from Maya's own earlier hand-tuned SUM-gate calibration on
her vEcoli `biofilm` branch, with no cited source for the specific value.
Reusing it here is a choice for internal CONSISTENCY (the same "count scale
at which FlhD4C2 matters" recurring in two mechanisms, rather than a third,
unrelated guess) -- not a second confirmation of the number. K_half=50
remains an open, unvalidated calibration parameter on both ends and should
be revisited with either (a) real E. coli FlhD4C2 copy-number literature, if
found, or (b) empirical sensitivity testing once the mechanism can be
observed running -- flag prominently, do not treat as settled.

Ordered in the composite flow: after ecoli-flhdc-degradation (so it reads the
post-basal-decay FlhD4C2 count) and before ecoli-flagella-flgm-secretion /
ecoli-flagella-transcription-regulation (so the SUM-gate reads the
post-checkpoint FlhD4C2 count each tick).

Refs: Utsey B & Keener JP (2020) PLOS Comput Biol 16:e1007689; Yamamoto S &
Kutsukake K (2006) J Bacteriol 188:5124; Yakhnin H et al. PMC4239645;
Tomoyasu T et al. (2003) Mol Microbiol 48:443.
"""


from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts


NAME = "ecoli-flit-flhdc-checkpoint"
TOPOLOGY = {
    "bulk": ("bulk",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flit_flhdc_checkpoint"),
    "global_time": ("global_time",),
}


class FliTFlhDCCheckpoint(Step):
    """FliT-titration checkpoint: enhanced ClpXP degradation of FlhD4C2 when
    bound by free FliT-dimer, released upon flagellum completion."""

    description = (
        "FliTFlhDCCheckpoint — FliT-mediated negative feedback on FlhD4C2.\n\n"
        "    v = u / (1 + x / K_half); c2 = u - v\n"
        "    degraded = round(delta2 * c2 * timestep)\n"
        "  x = FlhD4C2 count, u = free FliT-dimer count (fast pre-equilibrium\n"
        "  partition into free (v) vs FlhD4C2-bound (c2) FliT, Utsey & Keener 2020).\n"
        "  FliT is recycled, not consumed -- only FlhD4C2 is degraded."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "flhdc_id": {"_type": "string", "_default": "CPLX0-3930[c]"},
        "flit_dimer_id": {"_type": "string", "_default": "FLIT-DIMER[c]"},
        # delta2, Utsey & Keener 2020 Table 2 (3/min -> /s). ClpXP degradation
        # rate of FlhD4C2 specifically while bound to free FliT-dimer.
        "bound_degradation_rate": {"_type": "float", "_default": 0.05},
        # K_half: FlhD4C2 count at which FliT-dimer titration half-saturates.
        # NOT a literature constant -- reuses the SUM-gate's K_flhDC=50 count
        # scale; see module docstring "Parameter provenance" for why.
        "k_half": {"_type": "float", "_default": 50.0},
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
        self.bound_degradation_rate = self.parameters["bound_degradation_rate"]
        self.k_half = self.parameters["k_half"]
        # Bulk indices resolved lazily against the live bulk array.
        self.flhdc_idx = None
        self.flit_dimer_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.flhdc_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.flhdc_idx = bulk_name_to_idx(self.parameters["flhdc_id"], bulk_ids)
            self.flit_dimer_idx = bulk_name_to_idx(
                self.parameters["flit_dimer_id"], bulk_ids
            )

        x = counts(states["bulk"], self.flhdc_idx)
        u = counts(states["bulk"], self.flit_dimer_idx)

        degraded = 0
        if x > 0 and u > 0:
            v = u / (1.0 + x / self.k_half)
            c2 = u - v
            degraded = min(
                int(x),
                int(round(self.bound_degradation_rate * c2 * states["timestep"])),
            )

        return {
            "bulk": [(self.flhdc_idx, -degraded)],
            "next_update_time": states["global_time"] + states["timestep"],
        }
