import numpy as np
import numpy.typing as npt
import pytest

from nasap_fit_py.simulation.rate_constant_resolution import (
    RateConstant, Reaction, ResolvedReaction, create_conc_rates_fun,
    resolve_rate_constants)


class TestResolveRateConstants:
    """Tests for the resolve_rate_constants function."""

    def test_single_reaction_basic(self):
        # A + B <-> C + D, with forward rate constant 0.5 and backward rate constant 0.1
        """Test resolving a single reaction with basic parameters."""
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
        """Test resolving a reaction with None for unimolecular reactions."""
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
        """Test that duplicate counts multiply the rate constants."""
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
        """Test resolving multiple reactions with different types."""
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


class TestCreateConcRatesFun:
    """Tests for the create_conc_rates_fun function."""

    def test_bimolecular_reaction(self):
        """Test creating conc_rates_fun for a bimolecular reversible reaction: A + B <-> C + D."""
        resolved_reactions = [
            ResolvedReaction(
                reactant1="A",
                reactant2="B",
                product1="C",
                product2="D",
                rate_constant_f=0.5,
                rate_constant_b=0.1,
            )
        ]
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


    def test_multiple_reactions(self):
        """Test creating conc_rates_fun for multiple reversible reactions."""
        resolved_reactions = [
            ResolvedReaction(
                reactant1="A",
                reactant2="B",
                product1="C",
                product2="D",
                rate_constant_f=0.5,
                rate_constant_b=0.1,
            ),
            ResolvedReaction(
                reactant1="B",
                reactant2="C",
                product1="D",
                product2="E",
                rate_constant_f=1.0,
                rate_constant_b=0.2,
            ),
        ]
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
