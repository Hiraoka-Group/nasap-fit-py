import pytest

from nasap_fit_py.models.rate_constant import RateConstant
from nasap_fit_py.models.rate_constant_resolution import resolve_rate_constants
from nasap_fit_py.models.reaction import Reaction
from nasap_fit_py.models.reaction_with_type import ReactionWithType


def test_single_reaction_basic():
    # A + B <-> C + D, with forward rate constant 0.5 and backward rate constant 0.1
    reactions = [
        ReactionWithType(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    ]
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == [
        Reaction(
        reactant1="A",
        reactant2="B",
        product1="C",
        product2="D",
        rate_constant_f=0.5,
        rate_constant_b=0.1,
    )]


def test_single_reaction_with_none_values():
    # A <-> B, with forward rate constant 1.0 and backward rate constant 0.5
    reactions = [
        ReactionWithType(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            reaction_type="unimolecular",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    ]
    rate_constants = {
        "unimolecular": RateConstant(forward=1.0, backward=0.5),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == [
        Reaction(
        reactant1="A",
        reactant2=None,
        product1="B",
        product2=None,
        rate_constant_f=1.0,
        rate_constant_b=0.5,
    )]


def test_duplicate_count_affects_rate_constants():
    # A + B <-> C + D, with forward rate constant 0.5 and backward rate constant 0.1, and duplicate counts of 3 and 2
    reactions = [
        ReactionWithType(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=3,
            duplicate_count_b=2,
        ),
    ]
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
    }

    result = resolve_rate_constants(reactions, rate_constants)

    assert result == [
        Reaction(
        reactant1="A",
        reactant2="B",
        product1="C",
        product2="D",
        rate_constant_f=1.5,  # 0.5 * 3
        rate_constant_b=0.2,  # 0.1 * 2
    )]


def test_multiple_reactions_different_types():
    # A + B <-> C + D (type1), with forward rate constant 0.5 and backward rate constant 0.1
    # B + C <-> D + E (type2), with forward rate constant 1.0 and backward rate constant 0.2
    reactions = [
        ReactionWithType(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
        ReactionWithType(
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

    assert result == [
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
    )]


def test_undefined_reaction_type_raises_value_error():
    # Test that an undefined reaction type raises a ValueError with helpful message
    reactions = [
        ReactionWithType(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="undefined_type",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    ]
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
        "type2": RateConstant(forward=1.0, backward=0.2),
    }

    with pytest.raises(ValueError) as exc_info:
        resolve_rate_constants(reactions, rate_constants)

    assert str(exc_info.value) == (
        "Reaction type 'undefined_type' is not defined in rate_constants. "
        "This is the corresponding reaction: A + B -> C + D. "
    )


def test_undefined_reaction_type_unimolecular_empty_rate_constants():
    # Test that error message correctly shows unimolecular reactions
    reactions = [
        ReactionWithType(
            reactant1="A",
            reactant2=None,
            product1="B",
            product2=None,
            reaction_type="unknown_uni",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    ]
    rate_constants: dict[str, RateConstant] = {}

    with pytest.raises(ValueError) as exc_info:
        resolve_rate_constants(reactions, rate_constants)

    assert str(exc_info.value) == (
        "Reaction type 'unknown_uni' is not defined in rate_constants. "
        "This is the corresponding reaction: A -> B. "
    )


def test_undefined_reaction_type_in_second_reaction():
    # Test that error message correctly identifies which reaction in a list has the undefined type
    reactions = [
        ReactionWithType(
            reactant1="A",
            reactant2="B",
            product1="C",
            product2="D",
            reaction_type="type1",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
        ReactionWithType(
            reactant1="E",
            reactant2="F",
            product1="G",
            product2="H",
            reaction_type="undefined_type",
            duplicate_count_f=1,
            duplicate_count_b=1,
        ),
    ]
    rate_constants = {
        "type1": RateConstant(forward=0.5, backward=0.1),
    }

    with pytest.raises(ValueError) as exc_info:
        resolve_rate_constants(reactions, rate_constants)

    assert str(exc_info.value) == (
        "Reaction type 'undefined_type' is not defined in rate_constants. "
        "This is the corresponding reaction: E + F -> G + H. "
    )
    