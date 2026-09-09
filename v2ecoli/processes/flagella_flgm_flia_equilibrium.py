"""FlgM:FliA equilibrium binding -- exact closed-form solve, real Step.

Added 2026-09-01, part of Maya Abdalla's flagella-cascade investigation.

Why this exists, separate from the shared ecoli-equilibrium Step
------------------------------------------------------------------
FLGM-FLIA-CPLX_RXN used to live inside the shared ~150-reaction
equilibrium system (v2ecoli/processes/equilibrium.py), solved every tick
by a general numerical ODE solver (scipy solve_ivp), the same solver
that crashed repeatedly on FLIS-FLIC-CPLX_RXN (see
flagella_flis_flic_equilibrium.py) for the identical reason: its default
absolute error tolerance (atol=1e-6 M) is enormous relative to a real
molecule's concentration in this cell (~1.6e-9 M). FlgM:FliA's REAL Kd
(~1.8e-10 M, Chadsey et al. 1998) is roughly 290x TIGHTER than FliS:FliC's
(5.26e-8 M) -- an even worse fit for that solver's tolerance, not a
better one.

Rather than fix that directly, this reaction was worked around by
DELIBERATELY WEAKENING the model's own Kd to 2e-7 M -- about 1000x
weaker than the real value -- purely to keep the shared solver's answer
far enough from zero to avoid crashing. This was never a biological
number; it was a stability patch. Confirmed directly (2026-09-01) that
the crash risk was always about the SOLVER, not the Kd itself: the exact
closed-form solve used here cannot overshoot negative regardless of how
tight Kd is, by the same construction already proven for FliS:FliC.
That removes the only reason the Kd was ever weakened, so this Step uses
the real, cited value instead.

This Step is that exact solve, done directly, every firing. It replaces
FLGM-FLIA-CPLX_RXN's role from the shared equilibrium system entirely.
The shared system's own copy of this reaction is neutralized (both rates
set to 0 in sim_data.process.equilibrium, NOT deleted -- see
flagella_flis_flic_equilibrium.py's docstring for why zeroing rather than
deleting the row is the safer change) so it no longer participates in
that solve at all, while every one of the other ~150 real reactions in
that shared system is completely untouched.

The math
--------
Simple 1:1:1 binding (FlgM + FliA <-> FLGM-FLIA-CPLX -- no stoichiometry
surprise here the way FliS:FliC had; Chadsey et al. 1998's SPR
measurement reports a direct 1:1 Kd, no oligomeric-state complication
found). Let A = free FlgM, B = free FliA, C = FLGM-FLIA-CPLX, with
conserved totals A_tot = A + C, B_tot = B + C (nothing else in the live
model creates or destroys these three species except this one binding
reaction and flagella_flgm_secretion.py's direct draw on free FlgM --
that Step never touches the complex or FliA, so both totals are exactly
conserved from THIS Step's point of view within one firing). At
equilibrium:

    Kd = A * B / C = (A_tot - C)(B_tot - C) / C

Rearranged into a standard quadratic in C (identical form/derivation to
flagella_flis_flic_equilibrium.py's, see that file for the full algebra):

    C^2 - (A_tot + B_tot + Kd) * C + A_tot * B_tot = 0

    C = [(A_tot + B_tot + Kd) - sqrt((A_tot + B_tot + Kd)^2 - 4*A_tot*B_tot)] / 2

(the smaller root is the physical one). Solved directly, once, every
firing -- no iteration, no ODE, no tolerance setting anywhere in this
calculation, so no overshoot possible, ever, by construction.

Kd: 1.8e-10 M, Chadsey, Karlinsey & Hughes (1998), Genes Dev 12:3123
(Salmonella, SPR: ka=8.9e5 /M/s, kd=1.6e-4 /s -- kd/ka=1.8e-10 M,
self-consistent with the paper's own separately-reported Kd~2e-10 M).
This is the REAL value -- no longer relaxed. Converted from molar to a
molecule-count basis using the cell's real, current volume each firing
(same cell_mass / cell_density -> volume conversion the shared
equilibrium Step and flagella_flis_flic_equilibrium.py both already use).

Relaxation-timescale check (2026-09-01, done before building this):
confirmed the real dissociation half-life alone (ln(2)/kd = ~72 min)
is NOT the right number to judge instant-equilibrium-per-tick against --
that number describes an isolated complex with nothing around to rebind
it. The real relaxation time, accounting for BOTH directions
(tau = 1/(ka*(A_eq+B_eq) + kd)), works out to ~1-3 seconds at this
cell's real FlgM/FliA concentration scale -- comparable to this
codebase's own 2s tick, not 72 minutes. Solving to exact equilibrium
every firing is a reasonable approximation on that basis.

Ordered in the composite flow: right after ecoli-flagella-flgm-secretion
(the Step that drains free FlgM), so this Step's re-solve always reflects
that same tick's fresh FlgM level rather than the previous tick's --
matching secretion's own docstring ("as cytoplasmic FlgM drops, the
FLGM-FLIA-CPLX equilibrium shifts toward releasing free FliA") as a
same-tick response rather than the shared equilibrium Step's previous,
much-earlier position in the tick (layer 2, well before secretion runs
at all). Confirmed (2026-09-01) this ordering difference is real but
very unlikely to matter in practice -- FlgM changes by only a small
amount per 2s tick, so one tick's lag barely changes the equilibrium
Step's own answer either way; the same-tick placement is chosen for
consistency with the biology's own description, not because the 2-second
difference is expected to visibly change any output.
"""


import numpy as np

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.types.quantity import ureg as units


NAME = "ecoli-flagella-flgm-flia-equilibrium"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "flagella_flgm_flia_equilibrium"),
    "global_time": ("global_time",),
}


class FlagellaFlgMFliAEquilibrium(Step):
    """Exact closed-form FlgM:FliA equilibrium -- see module docstring."""

    description = (
        "FlagellaFlgMFliAEquilibrium — exact 1:1:1 binding equilibrium, no ODE.\n\n"
        "    A_tot = free_FlgM + FLGM-FLIA-CPLX\n"
        "    B_tot = free_FliA + FLGM-FLIA-CPLX\n"
        "    C = [(A_tot+B_tot+Kd) - sqrt((A_tot+B_tot+Kd)^2 - 4*A_tot*B_tot)] / 2\n"
        "  Sets FLGM-FLIA-CPLX to the true equilibrium point directly, every firing,\n"
        "  using the real Kd (1.8e-10 M, Chadsey et al. 1998) instead of the shared\n"
        "  solver's deliberately-relaxed 2e-7 M."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        "flgM_id": {"_type": "string", "_default": "G369-MONOMER[c]"},
        "fliA_id": {"_type": "string", "_default": "EG11355-MONOMER[c]"},
        "flgm_flia_cplx_id": {"_type": "string", "_default": "FLGM-FLIA-CPLX[c]"},
        # Real, cited value (Chadsey, Karlinsey & Hughes 1998, Genes Dev
        # 12:3123, SPR, Salmonella, kd/ka=1.8e-10 M) -- the shared
        # equilibrium system used a deliberately relaxed 2e-7 M instead,
        # purely to avoid that solver's crash mode. This Step's exact
        # solve can't overshoot regardless of Kd, so the real value is
        # used directly. See module docstring.
        "kd_molar": {"_type": "float", "_default": 1.8e-10},
        "cell_density": {"_type": "float", "_default": 1100.0},  # g/L, same
        # real constant used throughout this codebase.
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
        self.flgM_idx = None
        self.fliA_idx = None
        self.cplx_idx = None

    def update_condition(self, timestep, states):
        return states["next_update_time"] <= states["global_time"]

    def update(self, states, interval=None):
        if self.flgM_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.flgM_idx = bulk_name_to_idx(self.parameters["flgM_id"], bulk_ids)
            self.fliA_idx = bulk_name_to_idx(self.parameters["fliA_id"], bulk_ids)
            self.cplx_idx = bulk_name_to_idx(
                self.parameters["flgm_flia_cplx_id"], bulk_ids)

        next_update = {"next_update_time": states["global_time"] + states["timestep"]}

        free_flgM = int(counts(states["bulk"], self.flgM_idx))
        free_fliA = int(counts(states["bulk"], self.fliA_idx))
        cplx = int(counts(states["bulk"], self.cplx_idx))

        a_tot = free_flgM + cplx
        b_tot = free_fliA + cplx
        if a_tot == 0 or b_tot == 0:
            # Nothing to bind at all -- no-op, matches the physical answer
            # (C=0) exactly, no need to run the quadratic.
            return next_update

        # cell_volume = cell_mass / cell_density  [g / (g/L) = L], same
        # conversion the shared equilibrium Step and
        # flagella_flis_flic_equilibrium.py both already use.
        cell_mass_g = (
            as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        ).to(units.g).magnitude
        cell_volume_L = cell_mass_g / self.cell_density

        # Kd in molecule-count terms, for THIS cell's current real volume.
        kd_counts = self.kd_molar * self.n_avogadro * cell_volume_L

        s = a_tot + b_tot + kd_counts
        discriminant = s * s - 4.0 * a_tot * b_tot
        # Guard against tiny negative values from floating-point roundoff
        # right at the a_tot==b_tot edge case -- the true discriminant is
        # never negative for real, non-negative a_tot/b_tot/kd_counts.
        discriminant = max(discriminant, 0.0)
        c_star = (s - np.sqrt(discriminant)) / 2.0
        # c_star is mathematically guaranteed <= min(a_tot, b_tot); clip
        # only to guard against floating-point overshoot at that boundary.
        c_star = min(max(c_star, 0.0), a_tot, b_tot)

        delta = int(round(c_star)) - cplx
        if delta == 0:
            return next_update

        # delta > 0: more complex forms, consuming free FlgM and free
        # FliA. delta < 0: complex dissociates, releasing both back.
        # Exact and mass-conserving by construction -- the same delta
        # applied to all three, opposite signs.
        next_update["bulk"] = [
            (self.flgM_idx, -delta),
            (self.fliA_idx, -delta),
            (self.cplx_idx, delta),
        ]
        return next_update
