"""
===========
Equilibrium
===========

This process models how ligands are bound to or unbound from their
transcription factor binding partners in a fashion that maintains
equilibrium.

Mathematical Model
------------------
The equilibrium binding/unbinding is computed in two phases:

1. **Steady-state ODE solve**: Given current molecule counts x, cell volume V,
   and Avogadro's number N_A, solve the reaction ODE system to steady state
   to obtain integer reaction fluxes nu_j (net number of times each reaction
   fires):

       x_ss, nu = fluxesAndMoleculesToSS(x, V, N_A)

2. **Greedy flux correction**: If the allocator provides fewer molecules than
   the ODE solution requires, reaction fluxes are iteratively reduced to
   avoid driving any metabolite count negative. The algorithm decrements
   fluxes one unit at a time for reactions consuming limiting metabolites:

       while any (S @ nu + x_allocated < 0):
           reduce offending forward fluxes by 1 (clamp at 0)
           reduce offending reverse fluxes by 1 (clamp at 0)

   Convergence: each iteration reduces at least one flux by 1, so the
   loop terminates in at most sum(|nu|) iterations.

   The final molecule count change is:

       delta_x = S @ nu_corrected

   where S is the stoichiometry matrix (molecules x reactions).
"""

import numpy as np
from scipy.integrate import solve_ivp

from bigraph_schema.contract import ProcessContract
from v2ecoli.library.schema import bulk_name_to_idx, counts, attrs
from v2ecoli.library.schema_types import DNAA_BOX_ARRAY
from v2ecoli.library.ecoli_step import EcoliStep as Step
from wholecell.utils.random import stochasticRound
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity


# Register default topology for this process, associating it with process name
NAME = "ecoli-equilibrium"
TOPOLOGY = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
    # dnaa-3 Phase 2b: optional read-only port for the DnaA-box unique store.
    # Read to count rows currently bound by DnaA-ATP (DnaA_bound_form == 1) so
    # the DNAA-INTRINSIC-HYDROLYSIS-RXN flux covers BOTH the free pool
    # (bulk[MONOMER0-160]) AND the bound pool. Without this, the dnaa_box_binding
    # step writes Pi/PROTON/WATER directly to bulk, bypassing this step's
    # stoichMatrix machinery → metabolism over-allocates biomass synthesis.
    "DnaA_boxes": ("unique", "DnaA_box"),
    # dnaa-3 Phase 2c: shared bound-hydrolysis count from dnaa_box_binding.
    # Read instead of re-sampling so the byproduct count matches the form
    # swaps exactly. (Falls back to local resampling if the port is unwired.)
    "dnaa_hydrolysis": ("process_state", "dnaa_hydrolysis"),
}

# DnaA molecule and reaction IDs needed for the bound-pool hydrolysis trick.
# Resolved at first update against self.moleculeNames and self.reaction_ids.
_DNAA_ATP_ID = "MONOMER0-160[c]"
_DNAA_ADP_ID = "MONOMER0-4565[c]"
_DNAA_HYDROLYSIS_RXN_ID = "DNAA-INTRINSIC-HYDROLYSIS-RXN"
# Bound-form enum (mirrors dnaa_box_binding.py: 0 = free, 1 = bound-ATP, 2 = bound-ADP).
_FORM_BOUND_ATP = 1
# Per-spec pseudo-first-order rate constant (Sekimizu 1987 / Stage 1 PDF):
# k = 0.046 / min ≈ 7.667e-4 / s. The dnaa_box_binding step uses the same rate
# (independently) for the in-place bound ATP→ADP form swap.
_HYDROLYSIS_RATE_PER_SEC = 0.046 / 60.0


class Equilibrium(Step):
    """Equilibrium Step

    Models TF-ligand binding/unbinding. Ligands are TF-specific and not
    consumed by any other process, so this process does not compete for
    resources and runs as a plain Step (no request/allocate/evolve cycle).
    """

    description = (
        "Equilibrium — ligand–TF binding driven to steady state.\n\n"
        "1. ODE solve for integer reaction fluxes ν:  x_ss, ν = fluxesAndMoleculesToSS(x, V, N_A).\n"
        "2. Greedy correction under allocation: while any(S·ν + x_alloc < 0),\n"
        "   decrement offending forward/reverse fluxes (clamp ≥ 0).\n"
        "Δx = S·ν;  S = stoichiometry matrix (molecules × reactions)."
    )

    name = NAME
    topology = TOPOLOGY

    contract = ProcessContract(
        summary=(
            "Bind/unbind ligands to their transcription-factor partners by solving the reaction "
            "system to steady state each step, rounding the continuous solution to integer reaction "
            "fluxes, and greedily correcting those fluxes so no allocated metabolite goes negative."
        ),
        symbols={
            "x": "vector of current molecule counts for the equilibrium species (counts)",
            "x_ss": "steady-state molecule counts returned by the ODE solve (counts)",
            "x_alloc": "molecule counts actually allocated to this process for the greedy check (counts)",
            "V": "cell volume, computed as cell_mass / cell_density (litres, L)",
            "N_A": "Avogadro's number, counts per mole (1/mol)",
            "ν": "integer reaction-flux vector: net number of times each reaction fires (counts)",
            "S": "stoichiometry matrix, molecules × reactions (dimensionless integers)",
            "Δx": "net molecule-count change applied to bulk this step, Δx = S·ν (counts)",
        },
        inputs={
            "bulk": (
                "Reads counts x of the equilibrium molecules (ligands, TFs, complexes) as the ODE "
                "initial condition and as the non-negativity bound in the greedy correction."
            ),
            "listeners": (
                "Reads listeners.mass.cell_mass and divides by cell_density to get the volume V used "
                "to convert counts to concentrations for the ODE."
            ),
            "timestep": "Reads dt: the kinetic-phase integration horizon and the divisor for the reaction_rates listener.",
            "DnaA_boxes": (
                "Optional read-only port: counts DnaA-box rows in the bound-ATP form to add extra "
                "DNAA-INTRINSIC-HYDROLYSIS-RXN flux covering the bound pool (empty ⇒ free-pool-only behaviour)."
            ),
            "dnaa_hydrolysis": (
                "Optional port carrying dnaa_box_binding's bound-hydrolysis event count so byproducts are "
                "sampled from the same event; a -1 sentinel means unwired and the step resamples locally."
            ),
        },
        outputs={
            "bulk": (
                "Writes Δx = S·ν, the net binding/unbinding change per molecule, with the bound-pool DnaA-ATP/ADP "
                "portion reverted (bound ATP/ADP are box-bound, not in bulk)."
            ),
            "listeners": (
                "Writes equilibrium_listener.reaction_rates: the per-product molecule change divided by the "
                "timestep (1/s)."
            ),
        },
        config={
            "stoichMatrix": "Stoichiometry matrix S (molecules × reactions) mapping fluxes ν to count changes Δx.",
            "moleculeNames": "Bulk ids of the equilibrium species, in the row order of S.",
            "reaction_ids": "Reaction ids in the column order of S (used to locate the DnaA hydrolysis reaction).",
            "complex_ids": "Bulk ids of the TF–ligand complex products formed by binding (stored at init; unused in this build's update path).",
            "cell_density": "Cell density dividing cell_mass to obtain the volume V for count↔concentration conversion.",
            "n_avogadro": "Avogadro's number N_A converting between counts and molar concentrations.",
            "fluxesAndMoleculesToSS": "Legacy opaque solver returning steady-state counts and fluxes when rates_fn is absent.",
            "rates_fn": "Inline ODE rate function r(t, y, kf, kr); if provided it supersedes fluxesAndMoleculesToSS.",
            "rates_jac_fn": "Jacobian of rates_fn for the stiff solver.",
            "rates_fwd": "Forward rate constants kf per reaction.",
            "rates_rev": "Reverse rate constants kr per reaction.",
            "mets_to_rxn_fluxes": "Matrix mapping continuous molecule-count deltas back to reaction fluxes ν.",
            "Rp": "Reactant partition matrix: molecules consumed by forward fluxes.",
            "Pp": "Product partition matrix: molecules consumed by reverse fluxes.",
        },
        assumptions=[
            "Ligands are TF-specific and not consumed by other processes, so this runs as a plain Step with no request/allocate/evolve cycle.",
            "Fast reactions reach equilibrium within a step (steady-state solve to t≈1e20); reactions flagged in integrate_dt_mask are instead integrated over dt in a kinetic phase.",
            "Continuous ODE concentration changes are converted to counts and stochastically rounded to integer fluxes ν.",
            "The greedy correction decrements offending forward/reverse fluxes one unit at a time (clamped at 0) until no allocated metabolite is negative, terminating within sum(|ν|) iterations.",
            "The DnaA bound-pool hydrolysis extension is a no-op when its ids are absent from the cache or no DnaA boxes are bound, collapsing to free-pool-only flux.",
        ],
        references=[
            "Sekimizu (1987) — DnaA intrinsic ATP-hydrolysis rate (~0.046/min) used for the bound-pool hydrolysis flux.",
        ],
    )

    config_schema = {
        'cell_density': {'_type': 'float[g/L]', '_default': 0.0},
        'complex_ids': {'_type': 'list[string]', '_default': []},
        'fluxesAndMoleculesToSS': {'_type': 'method', '_default': None},  # legacy: opaque ODE solver (used if rates_fn not provided)
        'jit': {'_type': 'boolean', '_default': False},
        'moleculeNames': {'_type': 'list[string]', '_default': []},
        'n_avogadro': {'_type': 'float[1/mol]', '_default': 0.0},
        'reaction_ids': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'stoichMatrix': {'_type': 'array[integer]', '_default': []},  # (molecules x reactions) stoichiometry matrix S
        # ODE components (for inline solver -- if provided, fluxesAndMoleculesToSS is ignored)
        'rates_fn': {'_type': 'method', '_default': None},        # callable(t, y, kf, kr) -> rate vector
        'rates_jac_fn': {'_type': 'method', '_default': None},    # callable(t, y, kf, kr) -> jacobian
        'rates_fwd': {'_type': 'array[float]', '_default': []},   # forward rate constants
        'rates_rev': {'_type': 'array[float]', '_default': []},   # reverse rate constants
        'mets_to_rxn_fluxes': {'_type': 'array[float]', '_default': []},  # maps delta_molecules -> reaction_fluxes
        'Rp': {'_type': 'array[float]', '_default': []},          # reactant partition matrix
        'Pp': {'_type': 'array[float]', '_default': []},          # product partition matrix
    }

    def initialize(self, config):
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Stoichiometry matrix S: (n_molecules x n_reactions)
        self.stoichMatrix = self.parameters["stoichMatrix"]
        self.product_indices = [
            idx for idx in np.where(np.any(self.stoichMatrix > 0, axis=1))[0]
        ]

        self.moleculeNames = self.parameters["moleculeNames"]
        self.molecule_idx = None

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.complex_ids = self.parameters["complex_ids"]
        self.reaction_ids = self.parameters["reaction_ids"]

        # dnaa-3 Phase 2b: indices for the DnaA bound-pool hydrolysis injection.
        # Resolved lazily on first update so this remains backwards compatible
        # with caches that don't include MONOMER0-160 / MONOMER0-4565 in
        # moleculeNames or DNAA-INTRINSIC-HYDROLYSIS-RXN in reaction_ids.
        # (All current caches do include them, but the lazy resolve keeps the
        # change defensive — the bound-pool extension is a no-op if any index
        # is missing.)
        self._hydrolysis_rxn_idx = None
        self._dnaa_atp_local_idx = None  # position within self.molecule_idx / moleculeNames
        self._dnaa_adp_local_idx = None
        self._dnaa_resolve_attempted = False

        # ODE solver components: if rates_fn is provided, use inline solver;
        # otherwise fall back to the legacy opaque callable
        self.rates_fn = self.parameters.get("rates_fn")
        if self.rates_fn is not None:
            self.rates_jac_fn = self.parameters["rates_jac_fn"]
            self.rates_fwd = np.asarray(self.parameters["rates_fwd"])
            self.rates_rev = np.asarray(self.parameters["rates_rev"])
            self.mets_to_rxn_fluxes = np.asarray(self.parameters["mets_to_rxn_fluxes"])
            self.Rp = np.asarray(self.parameters["Rp"])
            self.Pp = np.asarray(self.parameters["Pp"])
            # Per-reaction kinetic flag. True = integrate over the simulation
            # timestep; False = solve to steady state. Falls back to all-False
            # if the cache was built before this column existed.
            mask = self.parameters.get("integrate_dt_mask")
            if mask is None:
                mask = [False] * len(self.rates_fwd)
            self.integrate_dt_mask = np.asarray(mask, dtype=bool)
        else:
            self.fluxesAndMoleculesToSS = self.parameters["fluxesAndMoleculesToSS"]
            self.jit = self.parameters.get("jit", False)

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0},
                },
            },
            'timestep': {'_type': 'integer[s]', '_default': 1},
            # Optional read-only port — defaults to an empty array so older
            # composites without the DnaA_box unique store still resolve.
            'DnaA_boxes': {'_type': DNAA_BOX_ARRAY, '_default': []},
            # dnaa-3 Phase 2c: shared bound-hydrolysis count from
            # dnaa_box_binding. -1 sentinel = port unwired → fall back to
            # local resampling for older composites.
            'dnaa_hydrolysis': {
                'bound_count': {'_type': 'integer', '_default': -1},
            },
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'equilibrium_listener': {
                    'reaction_rates': {'_type': 'overwrite[array[float[1/s]]]', '_default': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                },
            },
        }

    def _solve_ode_to_steady_state(self, molecule_counts, cell_volume, dt):
        """Solve the reaction ODE system.

        Two-phase integration handles slow reactions that violate the
        fast-equilibrium assumption:
          1. Kinetic phase: integrate the full system over the simulation
             timestep ``dt``. Reactions flagged ``integrate_dt_mask=True``
             only contribute through this phase.
          2. Equilibrium phase: continue integrating to ``t = 1e20``
             with the kinetic-only reactions' rates set to zero, so the
             remaining fast equilibria settle to steady state given the
             phase-1 molecule pool.

        If no reactions are flagged, the equilibrium phase alone produces
        the original behaviour.

        Returns:
            reaction_fluxes: integer reaction flux vector (n_reactions,)
            molecules_needed: molecules required for these fluxes
        """
        y_init = molecule_counts / (cell_volume * self.n_avogadro)
        has_kinetic = bool(self.integrate_dt_mask.any())

        # Phase-1 (kinetic) derivative uses the full rates.
        def deriv_kinetic(t, y):
            return self.stoichMatrix @ self.rates_fn(
                t, y, self.rates_fwd, self.rates_rev)

        def jac_kinetic(t, y):
            return self.stoichMatrix @ self.rates_jac_fn(
                t, y, self.rates_fwd, self.rates_rev)

        # Phase-2 (SS) zeroes out the kinetic-only reactions so the
        # remaining equilibria can settle.
        rates_fwd_ss = np.where(
            self.integrate_dt_mask, 0.0, self.rates_fwd)
        rates_rev_ss = np.where(
            self.integrate_dt_mask, 0.0, self.rates_rev)

        def deriv_ss(t, y):
            return self.stoichMatrix @ self.rates_fn(
                t, y, rates_fwd_ss, rates_rev_ss)

        def jac_ss(t, y):
            return self.stoichMatrix @ self.rates_jac_fn(
                t, y, rates_fwd_ss, rates_rev_ss)

        # Phase 1: kinetic integration over dt (only if any reaction is
        # flagged; otherwise skip straight to the SS solve).
        if has_kinetic:
            for method in ["LSODA", "BDF"]:
                try:
                    sol1 = solve_ivp(deriv_kinetic, [0, dt], y_init,
                                     method=method, t_eval=[0, dt],
                                     jac=jac_kinetic)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(
                    "Could not solve equilibrium kinetic-phase ODE.")
            y_after_dt = sol1.y[:, -1]
            y_after_dt[y_after_dt < 0] = 0
        else:
            y_after_dt = y_init

        # Phase 2: steady-state solve from the phase-1 endpoint, with
        # kinetic-only reactions disabled.
        for method in ["LSODA", "BDF"]:
            try:
                sol = solve_ivp(deriv_ss, [0, 1e20], y_after_dt,
                                method=method, t_eval=[0, 1e20],
                                jac=jac_ss)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("Could not solve equilibrium ODE to steady state.")

        # Convert concentration changes back to molecule count changes
        y = sol.y.T
        y[y < 0] = 0
        y_molecules = y * (cell_volume * self.n_avogadro)
        # Total delta = (phase-1 endpoint - initial) + (phase-2 endpoint -
        # phase-1 endpoint) = phase-2 endpoint - initial.
        delta_molecules = y_molecules[-1] - y_init * (cell_volume * self.n_avogadro)

        # Round continuous molecule changes to integer reaction fluxes
        for _ in range(100):
            reaction_fluxes = stochasticRound(
                self.random_state, self.mets_to_rxn_fluxes @ delta_molecules)
            if np.all(molecule_counts + self.stoichMatrix @ reaction_fluxes >= 0):
                break
        else:
            raise ValueError("Negative counts in equilibrium steady state.")

        # Compute molecules needed: partition into reactants consumed
        fwd_flux = np.clip(reaction_fluxes, 0, None)
        rev_flux = np.clip(-reaction_fluxes, 0, None)
        molecules_needed = self.Rp @ fwd_flux + self.Pp @ rev_flux

        return reaction_fluxes, molecules_needed

    def update(self, states, interval=None):
        # At t=0, convert molecule name strings to bulk array indices
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.moleculeNames, states["bulk"]["id"]
            )

        # First-tick resolve of the DnaA hydrolysis indices.
        if not self._dnaa_resolve_attempted:
            self._dnaa_resolve_attempted = True
            try:
                self._hydrolysis_rxn_idx = list(self.reaction_ids).index(
                    _DNAA_HYDROLYSIS_RXN_ID)
                self._dnaa_atp_local_idx = list(self.moleculeNames).index(_DNAA_ATP_ID)
                self._dnaa_adp_local_idx = list(self.moleculeNames).index(_DNAA_ADP_ID)
            except ValueError:
                # Missing from this cache — fall back to free-pool-only behaviour.
                self._hydrolysis_rxn_idx = None
                self._dnaa_atp_local_idx = None
                self._dnaa_adp_local_idx = None

        molecule_counts = counts(states["bulk"], self.molecule_idx)

        # dnaa-3 Phase 2b: count bound-ATP rows on the DnaA-box unique store.
        # If the port is empty (older composite) or no DnaA_box rows have been
        # seeded yet (pre-chromosome), n_bound_atp = 0 and the behaviour
        # collapses to the original free-pool-only flux.
        n_bound_atp = 0
        dnaa_boxes = states.get("DnaA_boxes")
        if (dnaa_boxes is not None
                and self._hydrolysis_rxn_idx is not None
                and getattr(dnaa_boxes, "dtype", None) is not None
                and dnaa_boxes.dtype.names is not None
                and "_entryState" in dnaa_boxes.dtype.names
                and "DnaA_bound_form" in dnaa_boxes.dtype.names
                and len(dnaa_boxes) > 0):
            # attrs() handles the _entryState mask internally and returns only
            # the active rows.
            (bound_form,) = attrs(dnaa_boxes, ["DnaA_bound_form"])
            n_bound_atp = int(np.count_nonzero(bound_form == _FORM_BOUND_ATP))

        # cell_volume = cell_mass / cell_density  [g / (g/L) = L]
        cell_mass_g = (
            as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        ).to(units.g).magnitude
        cell_volume = cell_mass_g / self.cell_density

        dt = float(states.get("timestep", 1))

        # Solve ODE to steady state -> reaction fluxes nu
        if self.rates_fn is not None:
            reaction_fluxes, _ = self._solve_ode_to_steady_state(
                molecule_counts, cell_volume, dt)
        else:
            reaction_fluxes, _ = self.fluxesAndMoleculesToSS(
                molecule_counts, cell_volume, self.n_avogadro,
                self.random_state, jit=self.jit, timestep=dt)

        # dnaa-3 Phase 2b: inject extra DNAA-INTRINSIC-HYDROLYSIS-RXN flux to
        # cover the bound-pool hydrolysis events. Without this, the bound
        # ATP→ADP form swap (owned by dnaa_box_binding) would still happen but
        # the Pi/PROTON/WATER byproducts would either (a) be missing entirely
        # or (b) be written directly to bulk by dnaa_box_binding, bypassing
        # FBA's accounting → metabolism over-allocates biomass synthesis.
        # Routing the byproducts through this step's stoichMatrix is the fix.
        #
        # dnaa-3 Phase 2c: read the bound-hydrolysis count from
        # dnaa_box_binding's process_state port so the byproducts are sampled
        # from the SAME stochastic event as the form swaps. Falls back to
        # local resampling if the port is unwired (sentinel = -1) for older
        # composites.
        extra_h_molecules = 0
        shared_count = int(states.get("dnaa_hydrolysis", {}).get(
            "bound_count", -1))
        if shared_count >= 0 and self._hydrolysis_rxn_idx is not None:
            extra_h_molecules = min(shared_count, n_bound_atp)
        elif self._hydrolysis_rxn_idx is not None and n_bound_atp > 0:
            # Fallback: local resampling (legacy Phase 2b behaviour).
            raw_extra = stochasticRound(
                self.random_state,
                np.asarray([_HYDROLYSIS_RATE_PER_SEC * n_bound_atp * dt]))
            extra_h_molecules = int(raw_extra[0])
            if extra_h_molecules > n_bound_atp:
                extra_h_molecules = n_bound_atp

        # Greedy flux correction: the steady-state solution may exceed
        # available counts (stochastic rounding overshoot). Reduce fluxes
        # one at a time for reactions that drive any metabolite negative.
        # Converges in at most sum(|nu|) iterations.
        reaction_fluxes = reaction_fluxes.copy()
        # Snapshot the pre-injection flux at the hydrolysis reaction. Used
        # below to figure out what fraction of the post-greedy flux is the
        # bound-pool extra vs. the free-pool baseline (greedy may have
        # reduced either one if a shared byproduct like WATER was limiting).
        pre_inject_h_flux = (
            int(reaction_fluxes[self._hydrolysis_rxn_idx])
            if self._hydrolysis_rxn_idx is not None else 0)

        # Temporarily inflate molecule_counts at the DnaA-ATP slot so the
        # greedy loop "sees" the bound DnaA-ATP as available headroom for
        # the extra flux. n_bound_atp is the right headroom (bound pool size,
        # not just extra_h_molecules) so the loop never rejects the injected
        # flux for ATP-availability reasons. Restored after the loop.
        molecule_counts_for_greedy = molecule_counts.copy()
        if extra_h_molecules > 0:
            reaction_fluxes[self._hydrolysis_rxn_idx] += extra_h_molecules
            molecule_counts_for_greedy[self._dnaa_atp_local_idx] += n_bound_atp

        max_iterations = int(np.abs(reaction_fluxes).sum()) + 1
        for _ in range(max_iterations):
            negative_idxs = np.where(
                np.dot(self.stoichMatrix, reaction_fluxes) + molecule_counts_for_greedy < 0
            )[0]
            if len(negative_idxs) == 0:
                break
            limited_stoich = self.stoichMatrix[negative_idxs, :]
            fwd_rxn_idxs = np.where(
                np.logical_and(limited_stoich < 0, reaction_fluxes > 0)
            )[1]
            rev_rxn_idxs = np.where(
                np.logical_and(limited_stoich > 0, reaction_fluxes < 0)
            )[1]
            reaction_fluxes[fwd_rxn_idxs] -= 1
            reaction_fluxes[rev_rxn_idxs] += 1
            reaction_fluxes[fwd_rxn_idxs] = np.fmax(0, reaction_fluxes[fwd_rxn_idxs])
            reaction_fluxes[rev_rxn_idxs] = np.fmin(0, reaction_fluxes[rev_rxn_idxs])
        else:
            raise ValueError(
                "Could not get positive counts in equilibrium with allocated molecules."
            )

        # delta_x = S @ nu_corrected
        delta_molecules = np.dot(self.stoichMatrix, reaction_fluxes).astype(int)

        # Revert the bound portion from the DnaA-ATP / DnaA-ADP bulk delta.
        # bf8b82e's stoichMatrix says DnaA-ATP + WATER → DnaA-ADP + Pi + PROTON.
        # The free-pool baseline (pre_inject_h_flux) operates on bulk DnaA-ATP
        # → bulk DnaA-ADP. The extra injected fires from the bound pool: the
        # consumed ATP was bound (not in bulk), and the produced ADP becomes
        # box-bound ADP (the in-place form swap is owned by dnaa_box_binding),
        # not bulk ADP. So for the bound portion only:
        #     delta_bulk[DnaA-ATP] += bound_fired   (revert false consumption)
        #     delta_bulk[DnaA-ADP] -= bound_fired   (revert false production)
        # WATER / Pi / PROTON come from bulk in both cases — correct as-is.
        if extra_h_molecules > 0:
            post_h_flux = int(reaction_fluxes[self._hydrolysis_rxn_idx])
            # Attribute greedy reductions to the bound pool first (free pool
            # gets the leftover). If greedy didn't reduce, bound_fired = extra.
            bound_fired = min(extra_h_molecules,
                              max(0, post_h_flux - pre_inject_h_flux))
            if bound_fired > 0:
                delta_molecules[self._dnaa_atp_local_idx] += bound_fired
                delta_molecules[self._dnaa_adp_local_idx] -= bound_fired

        return {
            "bulk": [(self.molecule_idx, delta_molecules)],
            "listeners": {
                "equilibrium_listener": {
                    "reaction_rates": delta_molecules[self.product_indices]
                    / states["timestep"]
                }
            },
        }


def test_equilibrium_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    listeners = sim.query()["agents"]["0"]["listeners"]
    assert isinstance(listeners["equilibrium_listener"]["reaction_rates"][0], list)
    assert isinstance(listeners["equilibrium_listener"]["reaction_rates"][1], list)


if __name__ == "__main__":
    test_equilibrium_listener()
