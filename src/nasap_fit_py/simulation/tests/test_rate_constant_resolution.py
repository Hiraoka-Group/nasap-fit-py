import pytest

from nasap_fit_py.simulation.rate_constant_resolution import (
    RateConstant, Reaction, ResolvedReaction, resolve_rate_constants)


def test_resolve_single_reaction():
    """Test resolving rate constants for a single reaction."""
    reactions = [
        Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        )
    ]
    
    rate_constants = {
        "type1": RateConstant(forward=1.5, backward=0.5)
    }
    
    result = resolve_rate_constants(reactions, rate_constants)
    
    expected = [
        ResolvedReaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
            rate_constant_f=1.5,
            rate_constant_b=0.5,
        )
    ]
    
    assert result == expected


def test_resolve_multiple_reactions():
    """Test resolving rate constants for multiple reactions."""
    reactions = [
        Reaction(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
        Reaction(
            reactant1="B",
            reactant2="C",
            product1="D",
            product2=None,
            reaction_type="type2",
            duplicate_count_f=2,
            duplicate_count_b=1,
        ),
    ]
    
    rate_constants = {
        "type1": RateConstant(forward=2.0, backward=1.0),
        "type2": RateConstant(forward=3.0, backward=0.5),
    }
    
    result = resolve_rate_constants(reactions, rate_constants)
    
    expected = [
        ResolvedReaction(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
            rate_constant_f=2.0,
            rate_constant_b=1.0,
        ),
        ResolvedReaction(
            reactant1="B",
            reactant2="C",
            product1="D",
            product2=None,
            reaction_type="type2",
            duplicate_count_f=2,
            duplicate_count_b=1,
            rate_constant_f=3.0,
            rate_constant_b=0.5,
        ),
    ]
    
    assert result == expected


def test_resolve_empty_reactions():
    """Test resolving an empty list of reactions."""
    reactions: list[Reaction] = []
    rate_constants: dict[str, RateConstant] = {}
    
    result = resolve_rate_constants(reactions, rate_constants)
    
    assert result == []
