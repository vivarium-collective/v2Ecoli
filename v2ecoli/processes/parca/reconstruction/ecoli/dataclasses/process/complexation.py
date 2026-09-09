"""
SimulationData for the Complexation process
"""

import numpy as np
from v2ecoli.processes.parca.wholecell.utils import units

# The Cython extension ``mc_complexation`` is only needed when
# ``Complexation.__init__`` actually runs.  Tests that merely import
# this module (e.g. to pull ``STORE_PATH`` out of the parca composite
# for wiring checks) shouldn't trip over a missing .so.  Defer the
# import to the one use site; raise only when the functionality is
# invoked.  This keeps ``test_parca_ports_and_wiring.py`` collectable
# on CI without the Cython build step, while preserving the original
# error for real parca runs.
try:
    from v2ecoli.processes.parca.wholecell.utils.mc_complexation import (
        mccBuildMatrices,
    )
    _MCC_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    mccBuildMatrices = None  # type: ignore[assignment]
    _MCC_IMPORT_ERROR = exc


class ComplexationError(Exception):
    pass


class MoleculeNotFoundError(ComplexationError):
    pass


class Complexation(object):
    # Reactions present in raw_data.complexation_reactions (and so used for
    # mass/compartment auto-derivation elsewhere -- getter_functions.py's
    # _build_protein_complex_masses and molecule_groups.py's
    # _build_molecule_groups both read raw_data directly, independent of
    # this class) but excluded from THIS class's own reaction set, and so
    # from every downstream consumer of sim_data.process.complexation
    # (both the runtime Gillespie config AND ParCa's own internal fitting,
    # e.g. step_05_fit_condition.py's calculateBulkDistributions -- both
    # read sim_data.process.complexation directly).
    #
    # CPLX0-7452_RXN (flagella-cascade investigation, 2026-08-06): its real
    # FliC coefficient (-20000, complexation_reactions_modified.tsv) makes
    # Gillespie SSA's combinatorial propensity calculation blow up
    # numerically -- confirmed directly via macOS `sample` on a hung ParCa
    # run (astronomically large values, ~1e19-1e35, computed inside
    # stochastic_arrow/arrowhead's propensity code) and reproduced twice: an
    # earlier fix that only filtered sim_data.py's get_complexation_config
    # (the runtime cache) was insufficient because step_05_fit_condition.py
    # reads this class's reaction set directly too, hitting the identical
    # blowup during ParCa's OWN fitting. Excluding it HERE is the actual
    # root-cause fix -- every downstream consumer of
    # sim_data.process.complexation is automatically protected. The real,
    # incremental version of flagellum completion is handled instead by
    # ecoli-flagella-filament-nucleation + ecoli-flagella-filament-elongation
    # (ordinary Step-level array arithmetic, no combinatorics) -- see
    # flagella_filament_elongation.py for the full writeup.
    # UPDATE (2026-08-06, same investigation): CPLX0-7452_RXN's exclusion
    # alone was NOT sufficient -- step_05_fit_condition.py's
    # calculateBulkDistributions runs the WHOLE complexation network through
    # the Gillespie engine with an enormous time step (2**31 s) starting from
    # raw expression-derived counts, to reach steady state. "failed
    # simulation: total propensity is NaN" reproduced within ~1 min even
    # with CPLX0-7452_RXN excluded, AND STILL reproduced with CPLX0-7450_RXN
    # also excluded. Root-caused by directly comparing coefficient
    # magnitudes network-wide: the pre-existing, unmodified LUMAZINESYN-
    # CPLX_RXN already has a coefficient of 60 and has never caused this, so
    # a single large coefficient isn't inherently fatal -- but
    # FLAGELLAR-MOTOR-COMPLEX_RXN has FIVE simultaneous double-digit
    # coefficients (FlgH:26, MotA:55, MotB:22, FlgG:24, FliF:34). Propensity
    # for a multi-reactant reaction is a PRODUCT of per-reactant
    # combinatorial terms -- several large-but-individually-survivable
    # coefficients multiplied together overflow even when none does alone.
    # All three flagella-assembly reactions are excluded here and replaced
    # by ordinary Step-level array arithmetic instead: see
    # flagella_motor_switch_assembly.py, flagella_motor_complex_assembly.py,
    # and flagella_filament_elongation.py.
    #
    # CPLX0-7451_RXN (flagella-cascade investigation, 2026-08-11): excluded
    # for a DIFFERENT reason than the three above -- its own stoichiometry
    # (max coefficient FlhA=9) was never numerically dangerous for SSA, and
    # it ran here safely for the whole session up to this point. It's
    # excluded now because complexation_reactions_modified.tsv's HIERARCHY
    # FIX made it consume CPLX0-7450 -- a molecule that only ever exists
    # transiently inside flagella_motor_switch_assembly.py's single
    # deterministic tick, produced and (previously) available to nothing
    # else. Leaving CPLX0-7451_RXN in ordinary SSA meant a
    # deterministically-produced molecule had to be consumed by a
    # stochastic-propensity mechanism with its own independent timing --
    # confirmed by direct testing to stall ongoing motor-complex
    # replenishment (pool drained 6->3 over 2400s instead of oscillating
    # 4-6 as before the hierarchy fix). Moved to
    # flagella_export_apparatus_assembly.py, matching the same deterministic
    # Step pattern as the other three, so the whole
    # C-ring -> export-apparatus -> motor-complex chain runs through one
    # consistent mechanism.
    RUNTIME_EXCLUDED_REACTIONS = {
        "CPLX0-7452_RXN", "CPLX0-7450_RXN", "FLAGELLAR-MOTOR-COMPLEX_RXN",
        "CPLX0-7451_RXN",
    }

    def __init__(self, raw_data, sim_data):
        # Build the abstractions needed for complexation
        molecules = []  # List of all molecules involved in complexation
        subunits = []  # List of all molecules that participate as subunits
        complexes = []  # List of all molecules that participate as complexes
        stoichMatrixI = []  # Molecule indices
        stoichMatrixJ = []  # Reaction indices
        stoichMatrixV = []  # Stoichiometric coefficients
        stoichMatrixMass = []  # Molecular masses of molecules in stoichMatrixI

        self.ids_reactions = []
        self.reaction_stoichiometry_unknown = []
        reaction_index = 0
        miscrnas_with_singleton_tus = sim_data.getter.get_miscrnas_with_singleton_tus()

        # Build stoichiometric matrix from given complexation reactions
        for reaction in raw_data.complexation_reactions:
            if reaction["id"] in self.RUNTIME_EXCLUDED_REACTIONS:
                continue
            self.ids_reactions.append(reaction["id"])
            stoichiometry_unknown = False

            for mol_id, coeff in reaction["stoichiometry"].items():
                # Replace miscRNA subunit IDs with TU IDs
                if mol_id in miscrnas_with_singleton_tus:
                    mol_id = sim_data.getter.get_singleton_tu_id(mol_id)

                mol_id_with_compartment = "{}[{}]".format(
                    mol_id, sim_data.getter.get_compartment(mol_id)[0]
                )

                if mol_id_with_compartment not in molecules:
                    molecules.append(mol_id_with_compartment)
                    molecule_index = len(molecules) - 1
                else:
                    molecule_index = molecules.index(mol_id_with_compartment)

                # Flag reactions whose stoichioemtric coefficients are given
                # as null and replace with -1
                if coeff is None:
                    stoichiometry_unknown = True
                    coeff = -1

                assert (coeff % 1) == 0

                stoichMatrixI.append(molecule_index)
                stoichMatrixJ.append(reaction_index)
                stoichMatrixV.append(coeff)

                # Classify molecule into subunit or complex depending on sign
                # of the stoichiometric coefficient - Note that a molecule can
                # be both a subunit and a complex
                if coeff < 0:
                    subunits.append(mol_id_with_compartment)
                else:
                    complexes.append(mol_id_with_compartment)

                # Find molecular mass of the molecule and add to mass matrix
                molecularMass = sim_data.getter.get_mass(
                    mol_id_with_compartment
                ).asNumber(units.g / units.mol)
                stoichMatrixMass.append(molecularMass)

            self.reaction_stoichiometry_unknown.append(stoichiometry_unknown)
            reaction_index += 1

        self.rates = np.full(
            (reaction_index,),
            sim_data.constants.complexation_rate.asNumber(1 / units.s),
        )

        self._stoich_matrix_I = np.array(stoichMatrixI)
        self._stoich_matrix_J = np.array(stoichMatrixJ)
        self._stoich_matrix_V = np.array(stoichMatrixV)
        self._stoich_matrix_mass = np.array(stoichMatrixMass)

        self.molecule_names = molecules
        self.ids_complexes = [
            self.molecule_names[i]
            for i in np.where(np.any(self.stoich_matrix() > 0, axis=1))[0]
        ]

        # Remove duplicate names in subunits and complexes
        self.subunit_names = set(subunits)
        self.complex_names = set(complexes)

        # Create sparse matrix for monomer to complex stoichiometry
        i, j, v, shape = self._buildStoichMatrixMonomers()
        self._stoichMatrixMonomersI = i
        self._stoichMatrixMonomersJ = j
        self._stoichMatrixMonomersV = v
        self._stoichMatrixMonomersShape = shape

        # Mass balance matrix
        # All reaction mass balances should balance out to numerical zero.
        #
        # FIX (2026-08-06, flagella-cascade investigation): a fixed absolute
        # tolerance breaks down once a reaction has large stoichiometric
        # coefficients. CPLX0-7452_RXN's corrected, real stoichiometry
        # (complexation_reactions_modified.tsv) now consumes ~20,000 copies of
        # FliC per flagellum -- the real, literature-cited filament subunit
        # count (a previous "-1" placeholder was off by four to five orders of
        # magnitude). That makes the mass terms in this reaction's balance
        # column ~1e9 in magnitude, at which point double-precision summation
        # alone produces an absolute residual (~1.9e-8) that is a RELATIVE
        # error of only ~1.9e-17 -- at the floor of what a double can even
        # represent (machine epsilon ~2.2e-16) -- yet it tripped the old fixed
        # 1e-8 absolute threshold. (The comment this replaced already shows
        # this exact failure mode recurring once before: "had to bump this up
        # to 1e-8 because of flagella supercomplex".) Rather than bump the
        # same brittle absolute threshold a third time (it will just fail
        # again the next time an even larger complex is added), the tolerance
        # is now scaled to the magnitude of each reaction's own terms: a small
        # absolute floor (atol) for ordinary small reactions, plus a tiny
        # relative allowance (rtol) proportional to that reaction's largest
        # mass term. rtol=1e-13 is ~1000x looser than the ~1e-17 relative
        # noise actually observed (comfortable safety margin for floating-
        # point roundoff) but still ~13 orders of magnitude tighter than any
        # real stoichiometry bug would produce (a genuinely wrong/missing
        # reactant leaves an imbalance comparable in magnitude to that
        # reactant's own mass contribution, i.e. a relative error of order 1,
        # not 1e-13) -- so this remains a strict correctness check, not a
        # loosened one.
        balanceMatrix = self.stoich_matrix() * self.mass_matrix()
        massBalanceArray = np.sum(balanceMatrix, axis=0)
        atol = 1e-8
        rtol = 1e-13
        max_term_magnitude = np.max(np.absolute(balanceMatrix), axis=0)
        tolerance = atol + rtol * max_term_magnitude
        assert np.all(np.absolute(massBalanceArray) < tolerance)

        stoichMatrix = self.stoich_matrix().astype(np.int64, order="F")
        if mccBuildMatrices is None:
            raise RuntimeError(
                "Failed to import Cython module "
                "``v2ecoli.processes.parca.wholecell.utils.mc_complexation``. "
                "Run ``bash scripts/parca_cython_build.sh`` (or ``make clean "
                "compile`` in the vEcoli tree) to build the extension."
            ) from _MCC_IMPORT_ERROR
        self.prebuilt_matrices = mccBuildMatrices(stoichMatrix)

        # Add boolean array to mark reactions with unknown stoichiometries
        self.reaction_stoichiometry_unknown = np.array(
            self.reaction_stoichiometry_unknown
        )

    def stoich_matrix(self):
        """
        Builds a stoichiometric matrix based on each given complexation
        reaction. One reaction corresponds to one column in the stoichiometric
        matrix.

        The result is cached on the instance because ``get_monomers`` calls
        this method once per molecule, making repeated reconstruction very
        expensive for large bulk_molecules lists (~16 k entries in E. coli).
        """
        if not hasattr(self, "_stoich_matrix_built"):
            shape = (self._stoich_matrix_I.max() + 1, self._stoich_matrix_J.max() + 1)
            out = np.zeros(shape, np.float64)
            out[self._stoich_matrix_I, self._stoich_matrix_J] = self._stoich_matrix_V
            self._stoich_matrix_built = out
        return self._stoich_matrix_built

    def mass_matrix(self):
        """
        Builds a matrix with the same shape as the stoichiometric matrix, but
        with molecular masses as elements instead of stoichiometric constants
        """
        shape = (self._stoich_matrix_I.max() + 1, self._stoich_matrix_J.max() + 1)
        out = np.zeros(shape, np.float64)
        out[self._stoich_matrix_I, self._stoich_matrix_J] = self._stoich_matrix_mass
        return out

    def stoich_matrix_monomers(self):
        """
        Returns the dense stoichiometric matrix for monomers from each complex
        """
        out = np.zeros(self._stoichMatrixMonomersShape, np.float64)
        out[self._stoichMatrixMonomersI, self._stoichMatrixMonomersJ] = (
            self._stoichMatrixMonomersV
        )
        return out

    # TODO: redesign this so it doesn't need to create a stoich matrix
    def get_monomers(self, cplxId):
        """
        Returns subunits for a complex (or any ID passed). If the ID passed is
        already a monomer returns the monomer ID again with a stoichiometric
        coefficient of one.
        """
        info = self._moleculeRecursiveSearch(
            cplxId, self.stoich_matrix(), self.molecule_names
        )
        subunits = []
        subunit_stoich = []
        for subunit, stoich in sorted(info.items()):
            subunits.append(subunit)
            subunit_stoich.append(stoich)
        return {
            "subunitIds": np.array(subunits),
            "subunitStoich": np.array(subunit_stoich),
        }

    def _buildStoichMatrixMonomers(self):
        """
        Builds a stoichiometric matrix where each column is a reaction that
        forms a complex directly from its constituent monomers. Since some
        reactions from the raw data are complexation reactions of complexes,
        this is different from the stoichiometric matrix generated by
        stoichMatrix().
        """
        stoichMatrixMonomersI = []
        stoichMatrixMonomersJ = []
        stoichMatrixMonomersV = []

        for colIdx, id_complex in enumerate(self.ids_complexes):
            D = self.get_monomers(id_complex)

            rowIdx = self.molecule_names.index(id_complex)
            stoichMatrixMonomersI.append(rowIdx)
            stoichMatrixMonomersJ.append(colIdx)
            stoichMatrixMonomersV.append(1.0)

            for subunitId, subunitStoich in zip(D["subunitIds"], D["subunitStoich"]):
                rowIdx = self.molecule_names.index(subunitId)
                stoichMatrixMonomersI.append(rowIdx)
                stoichMatrixMonomersJ.append(colIdx)
                stoichMatrixMonomersV.append(-1.0 * subunitStoich)

        stoichMatrixMonomersI = np.array(stoichMatrixMonomersI)
        stoichMatrixMonomersJ = np.array(stoichMatrixMonomersJ)
        stoichMatrixMonomersV = np.array(stoichMatrixMonomersV)

        shape = (stoichMatrixMonomersI.max() + 1, stoichMatrixMonomersJ.max() + 1)

        return (
            stoichMatrixMonomersI,
            stoichMatrixMonomersJ,
            stoichMatrixMonomersV,
            shape,
        )

    def _findRow(self, product, speciesList):
        try:
            row = speciesList.index(product)
        except ValueError:
            row = -1  # Flag if not found so not a complex
        return row

    def _findColumn(self, stoichMatrixRow):
        for i in range(0, len(stoichMatrixRow)):
            if int(stoichMatrixRow[i]) == 1:
                return i
        return -1  # Flag for monomer

    def _moleculeRecursiveSearch(self, product, stoichMatrix, speciesList):
        row = self._findRow(product, speciesList)
        if row == -1:
            return {product: 1.0}

        col = self._findColumn(stoichMatrix[row, :])
        if col == -1:
            return {product: 1.0}

        total = {}
        for i in range(0, len(speciesList)):
            if i == row:
                continue
            val = stoichMatrix[i][col]
            sp = speciesList[i]

            if val != 0:
                x = self._moleculeRecursiveSearch(sp, stoichMatrix, speciesList)
                for j in x:
                    if j in total:
                        total[j] += x[j] * (np.absolute(val))
                    else:
                        total[j] = x[j] * (np.absolute(val))
        return total
