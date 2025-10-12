import numpy as np
import numpy.typing as npt
import pytest

from src.nasap_fit_py.ode_creation import OdeRhs, create_ode_rhs
from src.nasap_fit_py.ode_creation.reaction_class import Reaction


class OdeRhsEquivalenceChecker:
    """Utility class for checking the equivalence of two OdeRhs instances."""
    def __init__(self, ode_rhs1: OdeRhs, ode_rhs2: OdeRhs):
        """
        Parameters
        ----------
        ode_rhs1 : OdeRhs
            The first OdeRhs instance.
        ode_rhs2 : OdeRhs
            The second OdeRhs instance.
        """
        self.ode_rhs1 = ode_rhs1
        self.ode_rhs2 = ode_rhs2

    def check(
            self, t: float, y: npt.NDArray, log_k_of_rxn_kinds: npt.NDArray):
        """Check if the two OdeRhs instances give the same result.

        Parameters
        ----------
        t : float
            Time.
        y : npt.NDArray
            Concentrations of assemblies.
        log_k_of_rxn_kinds : npt.NDArray
            log(k) of each reaction kind.
        """
        np.testing.assert_allclose(
            self.ode_rhs1(t, y, log_k_of_rxn_kinds),
            self.ode_rhs2(t, y, log_k_of_rxn_kinds)
            )


def test_basic():
    # A -> B  (k)
    assemblies = [0, 1]
    reactions = [
        Reaction(
            reactants=[0], products=[1], reaction_kind=0, 
            duplicate_count=1)
        ]
    reaction_kinds = [0]
    
    def expected_ode_rhs(
            t: float, y: npt.NDArray, log_k_of_rxn_kinds: npt.NDArray):
        log_k, = log_k_of_rxn_kinds
        k = 10**log_k
        return np.array([-k * y[0], k * y[0]])

    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    # parameters: t, y, log_k_of_rxn_kinds
    checker.check(0, np.array([1, 0]), np.array([0]))
    checker.check(0, np.array([0.5, 0.5]), np.array([0]))
    checker.check(0, np.array([1, 0]), np.array([2]))


def test_reversible():
    # A -> B  (k1)
    # B -> A  (k2)
    assemblies = [0, 1]
    reactions = [
        Reaction(
            reactants=[0], products=[1], reaction_kind=0, 
            duplicate_count=1),
        Reaction(
            reactants=[1], products=[0], reaction_kind=1, 
            duplicate_count=1)
        ]
    reaction_kinds = [0, 1]
    
    def expected_ode_rhs(t, y, log_k_of_rxn_kinds):
        k1 = 10**log_k_of_rxn_kinds[0]
        k2 = 10**log_k_of_rxn_kinds[1]
        return np.array([-k1 * y[0] + k2 * y[1], k1 * y[0] - k2 * y[1]])
    
    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0]), np.array([0, 0]))
    checker.check(0, np.array([0.5, 0.5]), np.array([0, 0]))
    checker.check(0, np.array([1, 0]), np.array([2, 1]))
    

def test_chain():
    # A -> B  (k1)
    # B -> C  (k2)
    assemblies = [0, 1, 2]
    reactions = [
        Reaction(
            reactants=[0], products=[1], reaction_kind=0, 
            duplicate_count=1),
        Reaction(
            reactants=[1], products=[2], reaction_kind=1, 
            duplicate_count=1)
        ]
    
    def expected_ode_rhs(t, y, log_k_of_rxn_kinds):
        k1 = 10**log_k_of_rxn_kinds[0]
        k2 = 10**log_k_of_rxn_kinds[1]
        return np.array([-k1 * y[0], k1 * y[0] - k2 * y[1], k2 * y[1]])
    
    ode_rhs = create_ode_rhs(assemblies, [0, 1], reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0, 0]), np.array([0, 0]))
    checker.check(0, np.array([0.5, 0.5, 0]), np.array([0, 0]))
    checker.check(0, np.array([1, 0, 0]), np.array([2, 1]))


def test_competition():
    # A -> B  (k1)
    # A -> C  (k2)
    assemblies = [0, 1, 2]
    reactions = [
        Reaction(
            reactants=[0], products=[1], reaction_kind=0, 
            duplicate_count=1),
        Reaction(
            reactants=[0], products=[2], reaction_kind=1, 
            duplicate_count=1)
        ]
    reaction_kinds = [0, 1]

    # d[A]/dt = -k1 * [A] - k2 * [A]
    # d[B]/dt = k1 * [A]
    # d[C]/dt = k2 * [A]

    def expected_ode_rhs(t, y, rxn_kind_to_log_k):
        k1 = 10**rxn_kind_to_log_k[0]
        k2 = 10**rxn_kind_to_log_k[1]
        return np.array(
            [-k1 * y[0] - k2 * y[0], k1 * y[0], k2 * y[0]])
    
    ode_rhs = create_ode_rhs(assemblies, [0, 1], reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0, 0]), np.array([0, 0]))
    checker.check(0, np.array([0.5, 0.5, 0.5]), np.array([0, 0]))
    checker.check(0, np.array([1, 0, 0]), np.array([2, 1]))


def test_duplicate_count():
    # A -> B  (k)
    # A has two possible pairs of reaction sites, 
    # thus the duplicate count is 2.
    assemblies = [0, 1]
    reactions = [
        Reaction(
            reactants=[0], products=[1], reaction_kind=0,
            duplicate_count=2)
        ]
    reaction_kinds = [0]

    # d[A]/dt = -2 * k * [A]  (because of the duplicate count)
    # d[B]/dt = 2 * k * [A]  (because of the duplicate count)

    def expected_ode_rhs(t, y, rxn_kind_to_log_k):
        k = 10**rxn_kind_to_log_k[0]
        return np.array([-2 * k * y[0], 2 * k * y[0]])
    
    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0]), np.array([0]))
    checker.check(0, np.array([0.5, 0.5]), np.array([0]))
    checker.check(0, np.array([1, 0]), np.array([2]))
        

def test_hetero_2_to_1():
    # A + B -> C  (k1)
    assemblies = [0, 1, 2]
    reactions = [
        Reaction(
            reactants=[0, 1], products=[2], reaction_kind=0, 
            duplicate_count=1)
        ]
    reaction_kinds = [0]
    
    # d[A]/dt = -k1 * [A] * [B]
    # d[B]/dt = -k1 * [A] * [B]
    # d[C]/dt = k1 * [A] * [B]

    def expected_ode_rhs(t, y, rxn_kind_to_log_k):
        k1 = 10**rxn_kind_to_log_k[0]
        return np.array(
            [-k1 * y[0] * y[1], -k1 * y[0] * y[1], k1 * y[0] * y[1]])
    
    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 1, 0]), np.array([0]))
    checker.check(0, np.array([0.5, 0.5, 0]), np.array([0]))
    checker.check(0, np.array([1, 1, 0]), np.array([2]))
               

def test_homo_2_to_1():
    # A + A -> B  (k1)
    assemblies = [0, 1]
    reactions = [
        Reaction(
            reactants=[0, 0], products=[1], reaction_kind=0, 
            duplicate_count=2)
        ]
    reaction_kinds = [0]

    # d[A]/dt = -2 * 2 * k1 * [A]^2
    # The first "2" is from the duplicate count. The rationale for the 
    # duplicate count is that when two reactants are the same, the 
    # chance of the reaction occurrence is doubled. In this case, 
    # one equation, A + A -> B, actually represents two reactions,
    # A1 + A2 -> B and A2 + A1 -> B. The two reactions are different
    # in the sense of where the ligand exchange occurs: on metal site in
    # A1 or A2.
    # The second "2" is from the stoichiometry; each time the reaction 
    # occurs, two A assemblies are consumed.
    
    # d[B]/dt = 2 * k1 * [A]^2
    # This "2" is from the duplicate count.

    def expected_ode_rhs(t, y, rxn_kind_to_log_k):
        k1 = 10**rxn_kind_to_log_k[0]
        return np.array(
            [-2 * 2 * k1 * y[0]**2, 2 * k1 * y[0]**2])

    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0]), np.array([0]))
    checker.check(0, np.array([0.5, 0.5]), np.array([0]))
    checker.check(0, np.array([1, 0]), np.array([2]))


def test_1_to_homo_2():
    # A -> B + B  (k1)
    assemblies = [0, 1]
    reactions = [
        Reaction(
            reactants=[0], products=[1, 1], reaction_kind=0, 
            duplicate_count=1)
        ]
    reaction_kinds = [0]

    # d[A]/dt = -k1 * [A]
    # d[B]/dt = 2 * k1 * [A]

    def expected_ode_rhs(t, y, rxn_kind_to_log_k):
        k1 = 10**rxn_kind_to_log_k[0]
        return np.array(
            [-k1 * y[0], 2 * k1 * y[0]])

    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0]), np.array([0]))
    checker.check(0, np.array([0.5, 0.5]), np.array([0]))
    checker.check(0, np.array([1, 0]), np.array([2]))


def test_homo_2_to_homo_2():
    # A + A -> B + B  (k1)
    assemblies = [0, 1]
    reactions = [
        Reaction(
            reactants=[0, 0], products=[1, 1], reaction_kind=0, 
            duplicate_count=1)
        ]
    reaction_kinds = [0]

    # d[A]/dt = -2 * k1 * [A]^2
    # d[B]/dt = 2 * k1 * [A]^2

    def expected_ode_rhs(t, y, rxn_kind_to_log_k):
        k1 = 10**rxn_kind_to_log_k[0]
        return np.array(
            [-2 * k1 * y[0]**2, 2 * k1 * y[0]**2])

    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0]), np.array([0]))
    checker.check(0, np.array([0.5, 0.5]), np.array([0]))
    checker.check(0, np.array([1, 0]), np.array([2]))


def test_homo_2_to_1_reversible():
    # A + A -> B  (k1)
    # B -> A + A  (k2)
    assemblies = [0, 1]
    reactions = [
        Reaction(
            reactants=[0, 0], products=[1], reaction_kind=0, 
            duplicate_count=1),
        Reaction(
            reactants=[1], products=[0, 0], reaction_kind=1, 
            duplicate_count=1)
        ]
    reaction_kinds = [0, 1]

    # d[A]/dt = -2 * k1 * [A]^2 + 2 * k2 * [B]
    # d[B]/dt = k1 * [A]^2 - k2 * [B]

    def expected_ode_rhs(t, y, rxn_kind_to_log_k):
        k1 = 10**rxn_kind_to_log_k[0]
        k2 = 10**rxn_kind_to_log_k[1]
        return np.array(
            [-2 * k1 * y[0]**2 + 2 * k2 * y[1], 
             k1 * y[0]**2 - k2 * y[1]])
    
    ode_rhs = create_ode_rhs(assemblies, [0, 1], reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0]), np.array([0, 0]))
    checker.check(0, np.array([0.5, 0.5]), np.array([0, 0]))
    checker.check(0, np.array([0, 1]), np.array([0, 0]))
    checker.check(0, np.array([1, 0]), np.array([2, 1]))


def test_str_assem_ids():
    # A -> B  (k)
    assemblies = ['A', 'B']
    reactions = [
        Reaction(
            reactants=['A'], products=['B'], reaction_kind='k', 
            duplicate_count=1)
        ]
    reaction_kinds = ['k']
    
    def expected_ode_rhs(t, y, log_k_of_rxn_kinds):
        log_k, = log_k_of_rxn_kinds
        k = 10**log_k
        return np.array([-k * y[0], k * y[0]])

    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    # parameters: t, y, log_k_of_rxn_kinds


def test_duplicate_reactions():
    # A -> B  (k)
    # A -> B  (k)  exactly the same as the previous reaction
    assemblies = ['A', 'B']
    reactions = [
        Reaction(
            reactants=['A'], products=['B'], reaction_kind='AtoB', 
            duplicate_count=1),
        Reaction(
            reactants=['A'], products=['B'], reaction_kind='AtoB',
            duplicate_count=1)
        ]
    reaction_kinds = ['AtoB']

    def expected_ode_rhs(t, y, log_k_of_rxn_kinds):
        log_k, = log_k_of_rxn_kinds
        k = 10**log_k
        k_tot = 2 * k
        return np.array([-k_tot * y[0], k_tot * y[0]])
    
    ode_rhs = create_ode_rhs(assemblies, reaction_kinds, reactions)

    checker = OdeRhsEquivalenceChecker(ode_rhs, expected_ode_rhs)
    checker.check(0, np.array([1, 0]), np.array([0]))
    checker.check(0, np.array([0.5, 0.5]), np.array([0]))
    checker.check(0, np.array([1, 0]), np.array([2]))


if __name__ == '__main__':
    pytest.main(['-v', __file__])
