import numpy as np
import numpy.typing as npt
import pytest

from nasap_fit_py.simulation.rate_constant_resolution import (
    RateConstant, Reaction, ResolvedReaction, create_conc_rates_fun,
    resolve_rate_constants)


def test_single_reaction_basic():
    # A + B <-> C + D, with forward rate constant 0.5 and backward rate constant 0.1
    reactions = {
        "r1": Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    }
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == {
        "r1": ResolvedReaction(
        reactant1="A",
        reactant2="B",
        product1="C",
        product2="D",
        rate_constant_f=0.5,
        rate_constant_b=0.1,
    )}


def test_single_reaction_with_none_values():
    # A <-> B, with forward rate constant 1.0 and backward rate constant 0.5
    reactions = {
        "r1": Reaction(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            reaction_type="unimolecular",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    }
    rate_constants = {
        "unimolecular": RateConstant(forward=1.0, backward=0.5),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == {
        "r1": ResolvedReaction(
        reactant1="A",
        reactant2=None,
        product1="B",
        product2=None,
        rate_constant_f=1.0,
        rate_constant_b=0.5,
    )}


def test_duplicate_count_affects_rate_constants():
    # A + B <-> C + D, with forward rate constant 0.5 and backward rate constant 0.1, and duplicate counts of 3 and 2
    reactions = {
        "r1": Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=3,
            duplicate_count_b=2,
        ),
    }
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == {
        "r1": ResolvedReaction(
        reactant1="A",
        reactant2="B",
        product1="C",
        product2="D",
        rate_constant_f=1.5,  # 0.5 * 3
        rate_constant_b=0.2,  # 0.1 * 2
    )}


def test_multiple_reactions_different_types():
    # A + B <-> C + D (type1), with forward rate constant 0.5 and backward rate constant 0.1
    # B + C <-> D + E (type2), with forward rate constant 1.0 and backward rate constant 0.2
    reactions = {
        "r1": Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
        "r2": Reaction(
            reactant1="B",
            reactant2="C",
            product1="D",
            product2="E",
            reaction_type="type2",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    }
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
        "type2": RateConstant(forward=1.0, backward=0.2),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == {
        "r1": ResolvedReaction(
        reactant1="A",
        reactant2="B",
        product1="C",
        product2="D",
        rate_constant_f=0.5,
        rate_constant_b=0.1,
    ), 
        "r2": ResolvedReaction(
        reactant1="B",
        reactant2="C",
        product1="D",
        product2="E",
        rate_constant_f=1.0,
        rate_constant_b=0.2,
    )}


def test_undefined_reaction_type_raises_value_error():
    # Test that an undefined reaction type raises a ValueError with helpful message
    reactions = {
        "r1": Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="undefined_type",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    }
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
        "type2": RateConstant(forward=1.0, backward=0.2),
    }

    with pytest.raises(ValueError) as exc_info:
        resolve_rate_constants(reactions, rate_constants)

    assert str(exc_info.value) == (
        "Reaction type 'undefined_type' is not defined in rate_constants. "
        "This is the corresponding reaction. "
        "Reaction[r1]: A + B -> C + D. "
    )


def test_undefined_reaction_type_unimolecular_empty_rate_constants():
    # Test that error message correctly shows unimolecular reactions
    reactions = {
        "r1": Reaction(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            reaction_type="unknown_uni",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    }
    rate_constants: dict[str, RateConstant] = {}

    with pytest.raises(ValueError) as exc_info:
        resolve_rate_constants(reactions, rate_constants)

    assert str(exc_info.value) == (
        "Reaction type 'unknown_uni' is not defined in rate_constants. "
        "This is the corresponding reaction. "
        "Reaction[r1]: A -> B. "
    )


def test_undefined_reaction_type_in_second_reaction():
    # Test that error message correctly identifies which reaction in a list has the undefined type
    reactions = {
        "r1": Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
        "r2": Reaction(
            reactant1="E",
            reactant2="F",
            product1="G",
            product2="H",
            reaction_type="undefined_type",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    }
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
    }

    with pytest.raises(ValueError) as exc_info:
        resolve_rate_constants(reactions, rate_constants)

    assert str(exc_info.value) == (
        "Reaction type 'undefined_type' is not defined in rate_constants. "
        "This is the corresponding reaction. "
        "Reaction[r2]: E + F -> G + H. "
    )


def test_unimolecular_reaction():
    """Test creating conc_rates_fun for a unimolecular reversible reaction: A <-> B."""
    resolved_reactions = {
        "r1": ResolvedReaction(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            rate_constant_f=0.5,
            rate_constant_b=0.1,
        )
    }
    species_ids = ["A", "B"]
    
    conc_rates_fun = create_conc_rates_fun(resolved_reactions, species_ids)
    
    # Test with specific concentrations: [A]=2.0, [B]=3.0
    concentrations = np.array([2.0, 3.0])
    rates = conc_rates_fun(concentrations)
    
    # Expected: [forward_rate, backward_rate]
    # forward: k_f * [A] = 0.5 * 2.0 = 1.0
    # backward: k_b * [B] = 0.1 * 3.0 = 0.3
    expected_rates = np.array([1.0, 0.3])
    np.testing.assert_allclose(rates, expected_rates)
    

def test_bimolecular_reaction():
    """Test creating conc_rates_fun for a bimolecular reversible reaction: A + B <-> C + D."""
    resolved_reactions = {
        "r1": ResolvedReaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            rate_constant_f=0.5,
            rate_constant_b=0.1,
        )
    }
    species_ids = ["A", "B", "C", "D"]
    
    conc_rates_fun = create_conc_rates_fun(resolved_reactions, species_ids)
    
    # Test with specific concentrations: [A]=2.0, [B]=3.0, [C]=1.0, [D]=4.0
    concentrations = np.array([2.0, 3.0, 1.0, 4.0])
    rates = conc_rates_fun(concentrations)
    
    # Expected: [forward_rate, backward_rate]
    # forward: k_f * [A] * [B] = 0.5 * 2.0 * 3.0 = 3.0
    # backward: k_b * [C] * [D] = 0.1 * 1.0 * 4.0 = 0.4
    expected_rates = np.array([3.0, 0.4])
    np.testing.assert_allclose(rates, expected_rates)


def test_multiple_reactions():
    """Test creating conc_rates_fun for multiple reversible reactions."""
    resolved_reactions = {
        "r1": ResolvedReaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            rate_constant_f=0.5,
            rate_constant_b=0.1,
        ),
        "r2": ResolvedReaction(
            reactant1="B",
            reactant2="C",
            product1="D",
            product2="E",
            rate_constant_f=1.0,
            rate_constant_b=0.2,
         ),
    }
    species_ids = ["A", "B", "C", "D", "E"]
    
    conc_rates_fun = create_conc_rates_fun(resolved_reactions, species_ids)
    
    # Test with specific concentrations
    concentrations = np.array([2.0, 3.0, 1.0, 4.0, 5.0])
    rates = conc_rates_fun(concentrations)
    
    # Expected: 4 elements (2 reactions × 2 directions each)
    # Reaction 0 forward: 0.5 * 2.0 * 3.0 = 3.0
    # Reaction 0 backward: 0.1 * 1.0 * 4.0 = 0.4
    # Reaction 1 forward: 1.0 * 3.0 * 1.0 = 3.0
    # Reaction 1 backward: 0.2 * 4.0 * 5.0 = 4.0
    expected_rates = np.array([3.0, 0.4, 3.0, 4.0])
    np.testing.assert_allclose(rates, expected_rates)
