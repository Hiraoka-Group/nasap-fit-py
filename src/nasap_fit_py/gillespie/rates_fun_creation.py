from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from ..models import ResolvedReaction


def create_rates_fun(
    resolved_reactions: Sequence[ResolvedReaction],
    species_ids: Sequence[str],
) -> Callable[[npt.NDArray[np.int_]], npt.NDArray[np.float64]]:
    """Create a function that calculates reaction rates from species particle counts.
    
    Returns a closure that computes forward and backward rates for all reactions
    based on current particle counts. For n reactions, the function produces a 2n-element
    vector where even indices contain forward rates and odd indices contain backward rates.
    
    Parameters
    ----------
    resolved_reactions : Sequence[ResolvedReaction]
        Sequence of resolved reactions with rate constants.
    species_ids : Sequence[str]
        Species IDs in order corresponding to particle count array indices.
    
    Returns
    -------
    Callable[[npt.NDArray[np.int_]], npt.NDArray[np.float64]]
        Callable that takes an integer particle count array and returns reaction rates.
        Output is a float64 array of shape (2*n,) where n is the number of reactions.
    """
    # Create mapping from species ID to index
    species_to_index = {species_id: i for i, species_id in enumerate(species_ids)}
    
    def rates_fun(
        particle_counts: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        """Calculate reaction rates from species particle counts.
        
        Parameters
        ----------
        particle_counts : npt.NDArray[np.int_]
            Integer particle count array with species in order corresponding to species_ids.
        
        Returns
        -------
        npt.NDArray[np.float64]
            Float64 array of shape (2*n,) where n is the number of reactions.
            Even indices contain forward rates, odd indices contain backward rates.
        """
        rates = np.empty(2 * len(resolved_reactions), dtype=np.float64)
        
        for i, reaction in enumerate(resolved_reactions):
            # Forward reaction (reactants -> products)
            rate_f = reaction.rate_constant_f * particle_counts[species_to_index[reaction.reactant1]]
            if reaction.reactant2 is not None:
                rate_f *= particle_counts[species_to_index[reaction.reactant2]]
            rates[2 * i] = rate_f
            
            # Backward reaction (products -> reactants)
            rate_b = reaction.rate_constant_b * particle_counts[species_to_index[reaction.product1]]
            if reaction.product2 is not None:
                rate_b *= particle_counts[species_to_index[reaction.product2]]
            rates[2 * i + 1] = rate_b
        
        return rates
    
    return rates_fun
    