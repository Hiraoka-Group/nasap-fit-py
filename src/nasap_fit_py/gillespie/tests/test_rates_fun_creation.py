import numpy as np

from nasap_fit_py.models.reaction import Reaction
from src.nasap_fit_py.gillespie.rates_fun_creation import create_rates_fun


def test_unimolecular_reaction():
    """Test creating conc_rates_fun for a unimolecular reversible reaction: A <-> B."""
    resolved_reactions = [
        Reaction(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            rate_constant_f=0.5,
            rate_constant_b=0.1,
        )
    ]
    species_ids = ["A", "B"]
    
    rates_fun = create_rates_fun(resolved_reactions, species_ids)
    
    particle_counts = np.array([2, 3])
    rates = rates_fun(particle_counts)
    
    # Expected: [forward_rate, backward_rate]
    # forward: k_f * [A] = 0.5 * 2 = 1.0
    # backward: k_b * [B] = 0.1 * 3 = 0.3
    expected_rates = np.array([1.0, 0.3])
    np.testing.assert_allclose(rates, expected_rates)
    

def test_bimolecular_reaction():
    """Test creating conc_rates_fun for a bimolecular reversible reaction: A + B <-> C + D."""
    resolved_reactions = [
        Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            rate_constant_f=0.5,
            rate_constant_b=0.1,
        )
    ]
    species_ids = ["A", "B", "C", "D"]
    
    rates_fun = create_rates_fun(resolved_reactions, species_ids)
    
    particle_counts = np.array([2, 3, 1, 4])
    rates = rates_fun(particle_counts)
    
    # Expected: [forward_rate, backward_rate]
    # forward: k_f * [A] * [B] = 0.5 * 2 * 3 = 3.0
    # backward: k_b * [C] * [D] = 0.1 * 1 * 4 = 0.4
    expected_rates = np.array([3.0, 0.4])
    np.testing.assert_allclose(rates, expected_rates)


def test_multiple_reactions():
    """Test creating rates_fun for multiple reversible reactions."""
    resolved_reactions = [
        Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            rate_constant_f=0.5,
            rate_constant_b=0.1,
        ),
        Reaction(
            reactant1="B",
            reactant2="C",
            product1="D",
            product2="E",
            rate_constant_f=1.0,
            rate_constant_b=0.2,
        ),
    ]
    species_ids = ["A", "B", "C", "D", "E"]
    
    rates_fun = create_rates_fun(resolved_reactions, species_ids)
    
    particle_counts = np.array([2, 3, 1, 4, 5])
    rates = rates_fun(particle_counts)
    
    # Expected: 4 elements (2 reactions × 2 directions each)
    # Reaction 0 forward: 0.5 * 2 * 3 = 3.0
    # Reaction 0 backward: 0.1 * 1 * 4 = 0.4
    # Reaction 1 forward: 1.0 * 3 * 1 = 3.0
    # Reaction 1 backward: 0.2 * 4 * 5 = 4.0
    expected_rates = np.array([3.0, 0.4, 3.0, 4.0])
    np.testing.assert_allclose(rates, expected_rates)
    np.testing.assert_allclose(rates, expected_rates)
