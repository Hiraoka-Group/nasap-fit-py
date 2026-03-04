from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Reaction:
    reactant1: str
    reactant2: str | None
    product1: str
    product2: str | None
    reaction_type: str
    duplicate_count_f: int
    duplicate_count_b: int


@dataclass(frozen=True)
class RateConstant:
    forward: float
    backward: float


@dataclass(frozen=True)
class ResolvedReaction:
    reactant1: str
    reactant2: str | None
    product1: str
    product2: str | None
    rate_constant_f: float
    rate_constant_b: float


def resolve_rate_constants(
    reactions: Sequence[Reaction],
    rtype_to_rate_constant: Mapping[str, RateConstant],
) -> Sequence[ResolvedReaction]:
    """Apply rate constants to reactions, accounting for duplicate pathways.
    
    Parameters
    ----------
    reactions : Sequence[Reaction]
        Sequence of reactions to resolve.
    rtype_to_rate_constant : Mapping[str, RateConstant]
        Mapping from reaction type to rate constants.
    
    Returns
    -------
    Sequence[ResolvedReaction]
        Sequence of reactions with rate constants applied and duplicates multiplied in.
    
    Raises
    ------
    ValueError
        If a reaction type is not found in rtype_to_rate_constant.
    """
    resolved_reactions = []
    for r in reactions:
        if r.reaction_type not in rtype_to_rate_constant:
            reaction_desc = f"{r.reactant1}"
            if r.reactant2:
                reaction_desc += f" + {r.reactant2}"
            reaction_desc += " -> "
            reaction_desc += f"{r.product1}"
            if r.product2:
                reaction_desc += f" + {r.product2}"
            raise ValueError(
                f"Reaction type '{r.reaction_type}' is not defined in rate_constants. "
                f"This is the corresponding reaction: {reaction_desc}. "
            )
        resolved_reactions.append(ResolvedReaction(
                reactant1=r.reactant1,
                reactant2=r.reactant2,
                product1=r.product1,
                product2=r.product2,
                rate_constant_f=rtype_to_rate_constant[r.reaction_type].forward * r.duplicate_count_f,
                rate_constant_b=rtype_to_rate_constant[r.reaction_type].backward * r.duplicate_count_b,
        ))
    return resolved_reactions


def create_conc_rates_fun(
    resolved_reactions: Sequence[ResolvedReaction],
    species_ids: Sequence[str],
) -> Callable[[npt.NDArray], npt.NDArray]:
    """Create a function that calculates reaction rates from species concentrations.
    
    Returns a closure that computes forward and backward rates for all reactions
    based on current concentrations. For n reactions, the function produces a 2n-element
    vector where even indices contain forward rates and odd indices contain backward rates.
    
    Parameters
    ----------
    resolved_reactions : Sequence[ResolvedReaction]
        Sequence of resolved reactions with rate constants.
    species_ids : Sequence[str]
        Species IDs in order corresponding to concentration array indices.
    
    Returns
    -------
    Callable[[npt.NDArray], npt.NDArray]
        Callable that takes a concentration array and returns reaction rates.
        Output is a float64 array of shape (2*n,) where n is the number of reactions.
    """
    # Create mapping from species ID to index
    species_to_index = {species_id: i for i, species_id in enumerate(species_ids)}
    
    def conc_rates_fun(concentrations: npt.NDArray) -> npt.NDArray:
        """Calculate reaction rates from species concentrations.
        
        Parameters
        ----------
        concentrations : npt.NDArray
            Concentration array with species in order corresponding to species_ids.
        
        Returns
        -------
        npt.NDArray
            Float64 array of shape (2*n,) where n is the number of reactions.
            Even indices contain forward rates, odd indices contain backward rates.
        """
        rates = np.empty(2 * len(resolved_reactions))
        
        for i, reaction in enumerate(resolved_reactions):
            # Forward reaction (reactants -> products)
            rate_f = reaction.rate_constant_f * concentrations[species_to_index[reaction.reactant1]]
            if reaction.reactant2 is not None:
                rate_f *= concentrations[species_to_index[reaction.reactant2]]
            rates[2 * i] = rate_f
            
            # Backward reaction (products -> reactants)
            rate_b = reaction.rate_constant_b * concentrations[species_to_index[reaction.product1]]
            if reaction.product2 is not None:
                rate_b *= concentrations[species_to_index[reaction.product2]]
            rates[2 * i + 1] = rate_b
        
        return rates
    
    return conc_rates_fun
    