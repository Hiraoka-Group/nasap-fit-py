import pytest

from nasap_fit_py.simulation.rate_constant_resolution import (
    RateConstant, Reaction, ResolvedReaction, resolve_rate_constants)


def test_single_reaction_basic(self):
    # A + B <-> C + D, with forward rate constant 0.5 and backward rate constant 0.1
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
        "type1": RateConstant(forward=0.5, backward=0.1),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == [ResolvedReaction(
        reactant1="A",
        reactant2="B",
        product1="C",
        product2="D",
        rate_constant_f=0.5,
        rate_constant_b=0.1,
    )]


def test_single_reaction_with_none_values(self):
    # A <-> B, with forward rate constant 1.0 and backward rate constant 0.5
    reactions = [
        Reaction(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            reaction_type="unimolecular",
            duplicate_count_f=1,
            duplicate_count_b=1,
        )
    ]
    rate_constants = {
        "unimolecular": RateConstant(forward=1.0, backward=0.5),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == [ResolvedReaction(
        reactant1="A",
        reactant2=None,
        product1="B",
        product2=None,
        rate_constant_f=1.0,
        rate_constant_b=0.5,
    )]

def test_duplicate_count_affects_rate_constants(self):
    # A + B <-> C + D, with forward rate constant 0.5 and backward rate constant 0.1, and duplicate counts of 3 and 2
    reactions = [
        Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=3,
            duplicate_count_b=2,
        )
    ]
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == [ResolvedReaction(
        reactant1="A",
        reactant2="B",
        product1="C",
        product2="D",
        rate_constant_f=1.5,  # 0.5 * 3
        rate_constant_b=0.2,  # 0.1 * 2
    )]

def test_multiple_reactions_different_types(self):
    # A + B <-> C + D (type1), with forward rate constant 0.5 and backward rate constant 0.1
    # B + C <-> D + E (type2), with forward rate constant 1.0 and backward rate constant 0.2
    reactions = [
        Reaction(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
        Reaction(
            reactant1="B",
            reactant2="C",
            product1="D",
            product2="E",
            reaction_type="type2",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    ]
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
        "type2": RateConstant(forward=1.0, backward=0.2),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == [ResolvedReaction(
        reactant1="A",
        reactant2="B",
        product1="C",
        product2="D",
        rate_constant_f=0.5,
        rate_constant_b=0.1,
    ), ResolvedReaction(
        reactant1="B",
        reactant2="C",
        product1="D",
        product2="E",
        rate_constant_f=1.0,
        rate_constant_b=0.2,
    )]
