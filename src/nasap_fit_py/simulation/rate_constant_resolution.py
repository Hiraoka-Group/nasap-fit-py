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
    reactions: Mapping[str, Reaction],
    rtype_to_rate_constant: Mapping[str, RateConstant],
) -> Mapping[str, ResolvedReaction]:

    resolved_reactions = {}
    for rid, r in reactions.items():
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
                "This is the corresponding reaction. "
                f"Reaction[{rid}]: {reaction_desc}. "
            )
        resolved_reactions[rid] = ResolvedReaction(
                reactant1=r.reactant1,
                reactant2=r.reactant2,
                product1=r.product1,
                product2=r.product2,
                rate_constant_f=rtype_to_rate_constant[r.reaction_type].forward*r.duplicate_count_f,
                rate_constant_b=rtype_to_rate_constant[r.reaction_type].backward*r.duplicate_count_b,
        )
    return resolved_reactions


def create_conc_rates_fun(
    resolved_reactions: Mapping[str, ResolvedReaction],
    species_ids: Sequence[str],
) -> Callable[[npt.NDArray], npt.NDArray]:
    """Create a function that calculates reaction rates from concentrations.
    
    For n reactions (each reversible), returns a function that produces a 2n-element
    vector containing forward and backward rates for each reaction.
    
    Args:
        resolved_reactions: dict of rid and resolved reactions with rate constants.
        species_ids: List of species IDs corresponding to concentration array indices.
    
    Returns:
        A function that takes a concentration array and returns a reaction rate vector,
        where rates[2*i] is the forward rate and rates[2*i+1] is the backward rate
        for reaction i.
    """
    # Create mapping from species ID to index
    species_to_index = {species_id: i for i, species_id in enumerate(species_ids)}
    
    def conc_rates_fun(concentrations: npt.NDArray) -> npt.NDArray:
        """Calculate reaction rates from concentrations.
        
        Args:
            concentrations: Array of species concentrations [mol/L].
        
        Returns:
            Array of reaction rates [mol·L^-1·min^-1] with 2n elements for n reactions.
        """
        rates = np.empty(2 * len(resolved_reactions))
        
        for i, (_, reaction) in enumerate(resolved_reactions.items()):
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
    