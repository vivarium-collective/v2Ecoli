"""FliS:FliC equilibrium binding -- exact closed-form solve, real Step.

Added 2026-08-28, part of Maya Abdalla's flagella-cascade investigation.

Why this exists, separate from the shared ecoli-equilibrium Step
------------------------------------------------------------------
FLIS-FLIC-CPLX_RXN used to live inside the shared ~150-reaction equilibrium
system (v2ecoli/processes/equilibrium.py), solved every tick by a general
numerical ODE solver (scipy solve_ivp) that has to handle all ~150
reactions together, converging to t -> infinity.

That solver crashed repeatedly at real division events (confirmed twice,
via direct diagnostic printout, that FliS and FliC are the exact species
involved both times). Root cause, confirmed directly: scipy's solve_ivp
default absolute error tolerance (atol=1e-6, in mol/L) is enormous
relative to this reaction's real scale -- at the second crash, free FliS
was 278 copies and free FliC was 626 copies in a ~1fL cell, i.e.
concentrations of ~3e-7 and ~6e-7 M. A single molecule in that cell is
~1.6e-9 M. The default atol (1e-6 M) is over 600x the concentration of
ONE MOLECULE. The solver is allowed to be wrong by an amount bigger than
hundreds of real molecules and still call itself converged. Slowing the
reaction's rate constants down (tried twice, 1000x then another 100x,
Kd preserved both times) never fixed this, because it never touches this
tolerance mismatch at all -- it was the wrong knob.

FLIS-FLIC-CPLX_RXN, confirmed (2026-08-28) to touch no other reaction in
the whole shared equilibrium system, and no other LIVE reaction anywhere
else in the model (FliC's other historical appearance, CPLX0-7452_RXN's
-20000/-5000 coefficient in complexation_reactions_modified.tsv, is
confirmed excluded from ever running at runtime -- see
RUNTIME_EXCLUDED_REACTIONS in reconstruction/ecoli/dataclasses/process/
complexation.py). A binding reaction this simple has an exact,
closed-form algebraic solution for its true equilibrium point -- no
numerical integration needed at all, so no tolerance to get wrong, no
overshoot possible, ever, by construction.

This Step is that exact solve, done directly, every firing. It replaces
FLIS-FLIC-CPLX_RXN's role from the shared equilibrium system entirely.
The shared system's own copy of this reaction is neutralized (both rates
set to 0 in sim_data.process.equilibrium, NOT deleted -- deleting a row
would require rebuilding the shared system's stoichiometry matrix,
symbolic derivatives, and every other reaction's array index alongside
it, a much bigger and riskier change than zeroing two numbers for a
reaction that already has its own dedicated Step) so it no longer
participates in that solve at all, while every one of the other ~150
real reactions in that shared system is completely untouched.

STOICHIOMETRY CORRECTION (2026-08-31): this reaction is NOT simple 1:1:1
FliS:FliC binding. Confirmed directly against the primary literature:
Auvray, Thomas, Fraser & Hughes (2001), J Mol Biol 308:221-229, in
Salmonella -- "FliS homodimers bind to FliC monomers." FliS's own native
solution state is a stable homodimer (also consistent with Sajo et al.
2016, FEBS Lett 590:1103, characterizing Salmonella FliS's structural
plasticity without ever describing a loose monomer/dimer equilibrium --
the usual signature of tight, fast self-association for a small,
structured protein interface). The real reaction is therefore two
sequential steps: 2 FliS(monomer) <-> FliS2(dimer) (fast, tight,
uncharacterized Kd), then FliS2(dimer) + FliC <-> Complex (the step
Muskotal et al. 2006's ITC actually measured, Kd=5.26e-8 M, on a
DIMER:FliC basis -- their reported "1:1" stoichiometry is 1 dimer : 1
FliC, not 1 raw FliS protein : 1 FliC).

Modeled here via the standard fast-pre-equilibrium approximation:
assume FliS dimerization is fast/tight enough that essentially all
uncomplexed FliS exists as dimer, so "dimer count" can be read directly
off the raw monomer bulk count (divide by 2) without tracking a
separate free-dimer species. This is a real approximation (no published
FliS self-dimerization Kd exists to solve the two steps exactly --
searched directly, 2026-08-31, not found), but it gets the MECHANISM
right (dimer is the substrate-binding species, matching the literature)
where the alternative -- a single lumped 2-FliS-monomer + FliC -> Complex
elementary reaction -- would not: real chaperone-substrate binding does
not happen via 3-body collisions, and a lumped reaction like that would
need its own from-scratch M^2-unit constant that isn't derivable from
what's published either. Old 1:1:1 assumption kept per standing
preserve-old-code rule -- see commented-out lines in update() below.

The math
--------
Let A = free FliS *monomer* count, D = free FliS *dimer* count
(=A // 2, fast-pre-equilibrium approximation above), B = free FliC,
C = FLIS-FLIC-CPLX, with conserved totals D_tot = D + C and
B_tot = B + C (nothing else in the live model creates or destroys these
species except this one binding reaction and the downstream
filament-elongation Step that consumes C directly -- both totals are
exactly conserved from this Step's point of view within one firing, and
elongation's own chaperone-release side was updated in lockstep, see
flagella_filament_elongation.py). At equilibrium:

    Kd = D * B / C = (D_tot - C)(B_tot - C) / C

Rearranged into a standard quadratic in C (identical form to the old
1:1:1 case, just with D_tot substituted for the old A_tot):

    C^2 - (D_tot + B_tot + Kd) * C + D_tot * B_tot = 0

    C = [(D_tot + B_tot + Kd) - sqrt((D_tot + B_tot + Kd)^2 - 4*D_tot*B_tot)] / 2

(the smaller root is the physical one -- the larger root exceeds
min(D_tot, B_tot), which is impossible). This is solved directly, once,
every firing -- no iteration, no ODE, no tolerance setting anywhere in
this calculation. The only change from the 1:1:1 version is bookkeeping:
every unit of C formed or dissociated now moves 2 raw FliS monomers, not
1 (see the bulk update at the end of update()) -- mathematically
guaranteed never to overdraw free FliS, since 2*(C - cplx) <= A by
construction whenever C > cplx (D_tot - cplx = A // 2, so
2*(D_tot - cplx) <= A always).

Kd itself is unchanged from before: 5.26e-8 M, Muskotal et al. 2006
(FEBS Lett 580:3916, isothermal titration calorimetry, Ka=1.9e7/M) --
the same real, cited number used throughout this investigation, now
understood to be on a dimer:FliC basis rather than raw-monomer:FliC.
Converted from molar to a molecule-count basis using the cell's real,
current volume each firing (same cell_mass / cell_density -> volume
conversion the shared equilibrium Step already uses), since this Step
operates on raw bulk counts directly, not concentrations.

Ordered in the composite flow: right before ecoli-flagella-filament-
elongation (the Step that consumes FLIS-FLIC-CPLX as the "protected"
FliC pool), so elongation always sees an up-to-date equilibrium each
tick -- matching where the shared equilibrium Step used to sit relative
to elongation's own read of this same complex.
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.types.quantity import ureg as units


NAME = "ecoli-flagella-flis-flic-equilibrium"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_flis_flic_equilibrium"),
    "global_time": ("global_time",),
}


class FlagellaFliSFliCEquilibrium(Step):
    """Exact closed-form FliS:FliC equilibrium -- see module docstring."""

    description = (
        "FlagellaFliSFliCEquilibrium — exact FliS-dimer:FliC binding equilibrium, no ODE.\n\n"
        "    D_tot = free_FliS // 2 + FLIS-FLIC-CPLX   (dimer units, fast pre-equilibrium)\n"
        "    B_tot = free_FliC + FLIS-FLIC-CPLX\n"
        "    C = [(D_tot+B_tot+Kd) - sqrt((D_tot+B_tot+Kd)^2 - 4*D_tot*B_tot)] / 2\n"
        "  Sets FLIS-FLIC-CPLX to the true equilibrium point directly, every firing.\n"
        "  Each unit of C moves 2 raw FliS monomers (Auvray et al. 2001: FliS binds as\n"
        "  a homodimer), not 1 -- see module docstring's STOICHIOMETRY CORRECTION."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "fliS_id": {"_type": "string", "_default": "EG11388-MONOMER[c]"},
        "fliC_id": {"_type": "string", "_default": "EG10321-MONOMER[e]"},
        "flis_flic_cplx_id": {"_type": "string", "_default": "FLIS-FLIC-CPLX[e]"},
        # Real, cited value (Muskotal et al. 2006, FEBS Lett 580:3916, ITC,
        # Ka=1.9e7/M) -- unchanged from what the shared equilibrium system
        # used before this Step existed. This IS the biological number;
        # nothing about moving this reaction into its own Step changes it.
        "kd_molar": {"_type": "float", "_default": 5.26e-8},
        "cell_density": {"_type": "float", "_default": 1100.0},  # g/L, same
        # real constant used throughout this codebase (see e.g.
        # ecoli_baseline.py's dnaa-box-binding wiring).
        "n_avogadro": {"_type": "float", "_default": 6.02214076e23},  # /mol
        "timestep": {"_type": "float", "_default": 2.0},
    }

    def inputs(self):
        return {
            "bulk": {"_type": "bulk_array", "_default": []},
            "listeners": {
                "mass": {
                    "cell_mass": {"_type": "quantity[float,fg]", "_default": 0},
                },
            },
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
        self.kd_molar = self.parameters["kd_molar"]
        self.cell_density = self.parameters["cell_density"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.fliS_idx = None
        self.fliC_idx = None
        self.cplx_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.fliS_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.fliS_idx = bulk_name_to_idx(self.parameters["fliS_id"], bulk_ids)
            self.fliC_idx = bulk_name_to_idx(self.parameters["fliC_id"], bulk_ids)
            self.cplx_idx = bulk_name_to_idx(
                self.parameters["flis_flic_cplx_id"], bulk_ids)

        next_update = {"next_update_time": states["global_time"] + states["timestep"]}

        free_fliS = int(counts(states["bulk"], self.fliS_idx))
        free_fliC = int(counts(states["bulk"], self.fliC_idx))
        cplx = int(counts(states["bulk"], self.cplx_idx))

        # Old 1:1:1 assumption, kept per standing preserve-old-code rule
        # (see module docstring's 2026-08-31 STOICHIOMETRY CORRECTION):
        # a_tot = free_fliS + cplx
        # FliS binds FliC as a homodimer (Auvray et al. 2001), not as a raw
        # monomer -- fast-pre-equilibrium approximation: free dimer count
        # = free_fliS // 2 (integer division -- a lone unpaired monomer
        # can't form a partial dimer this tick).
        d_tot = free_fliS // 2 + cplx
        b_tot = free_fliC + cplx
        if d_tot == 0 or b_tot == 0:
            # Nothing to bind at all -- no-op, matches the physical answer
            # (C=0) exactly, no need to run the quadratic.
            return next_update

        # cell_volume = cell_mass / cell_density  [g / (g/L) = L], same
        # conversion the shared equilibrium Step uses.
        cell_mass_g = (
            as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        ).to(units.g).magnitude
        cell_volume_L = cell_mass_g / self.cell_density

        # Kd in molecule-count terms, for THIS cell's current real volume.
        kd_counts = self.kd_molar * self.n_avogadro * cell_volume_L

        s = d_tot + b_tot + kd_counts
        discriminant = s * s - 4.0 * d_tot * b_tot
        # Guard against tiny negative values from floating-point roundoff
        # right at the d_tot==b_tot edge case -- the true discriminant is
        # never negative for real, non-negative d_tot/b_tot/kd_counts.
        discriminant = max(discriminant, 0.0)
        c_star = (s - np.sqrt(discriminant)) / 2.0
        # c_star is mathematically guaranteed <= min(d_tot, b_tot); clip
        # only to guard against floating-point overshoot at that boundary.
        c_star = min(max(c_star, 0.0), d_tot, b_tot)

        delta = int(round(c_star)) - cplx
        if delta == 0:
            return next_update

        # delta > 0: more complex forms, consuming free FliS (2 monomers
        # per unit -- a homodimer, Auvray et al. 2001) and free FliC (1
        # per unit). delta < 0: complex dissociates, releasing both back.
        # Exact and mass-conserving by construction -- 2*delta can never
        # overdraw free_fliS, since d_tot - cplx = free_fliS // 2 bounds
        # how many additional dimers (delta, when positive) can form.
        next_update["bulk"] = [
            (self.fliS_idx, -2 * delta),
            (self.fliC_idx, -delta),
            (self.cplx_idx, delta),
        ]
        return next_update
