from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from src.nasap_fit_py.models import Reaction


def create_rates_fun(
    reactions: Sequence[Reaction],
    species_ids: Sequence[str],
) -> Callable[[npt.NDArray[np.int_]], npt.NDArray[np.float64]]:
    """Create a function that calculates reaction rates from species particle counts.
    
    Returns a closure that computes forward and backward rates for all reactions
    based on current particle counts. For n reactions, the function produces a 2n-element
    vector where even indices contain forward rates and odd indices contain backward rates.
    
    Parameters
    ----------
    reactions : Sequence[Reaction]
        Sequence of reactions with rate constants.
    species_ids : Sequence[str]
        Species IDs in order corresponding to particle count array indices.
    
    Returns
    -------
    Callable[[npt.NDArray[np.int_]], npt.NDArray[np.float64]]
        Callable that takes an integer particle count array and returns reaction rates.
        Output is a float64 array of shape (2*n,) where n is the number of reactions.
    """
    # Precompute all indices/constants once to minimize per-step Python overhead.
    species_to_index = {species_id: i for i, species_id in enumerate(species_ids)}
    num_reactions = len(reactions)

    reactant1_indices = np.empty(num_reactions, dtype=np.intp)
    reactant2_indices = np.empty(num_reactions, dtype=np.intp)
    reactant2_exists = np.zeros(num_reactions, dtype=np.bool_)

    product1_indices = np.empty(num_reactions, dtype=np.intp)
    product2_indices = np.empty(num_reactions, dtype=np.intp)
    product2_exists = np.zeros(num_reactions, dtype=np.bool_)

    rate_constant_f = np.empty(num_reactions, dtype=np.float64)
    rate_constant_b = np.empty(num_reactions, dtype=np.float64)

    for i, reaction in enumerate(reactions):
        reactant1_indices[i] = species_to_index[reaction.reactant1]
        product1_indices[i] = species_to_index[reaction.product1]
        rate_constant_f[i] = reaction.rate_constant_f
        rate_constant_b[i] = reaction.rate_constant_b

        if reaction.reactant2 is not None:
            reactant2_indices[i] = species_to_index[reaction.reactant2]
            reactant2_exists[i] = True
        else:
            reactant2_indices[i] = 0

        if reaction.product2 is not None:
            product2_indices[i] = species_to_index[reaction.product2]
            product2_exists[i] = True
        else:
            product2_indices[i] = 0

    reactant2_positions = np.flatnonzero(reactant2_exists)
    product2_positions = np.flatnonzero(product2_exists)
    
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
        rates = np.empty(2 * num_reactions, dtype=np.float64)
        forward_rates = rates[0::2]
        backward_rates = rates[1::2]

        np.multiply(
            rate_constant_f,
            particle_counts[reactant1_indices],
            out=forward_rates,
            casting="unsafe",
        )
        if reactant2_positions.size > 0:
            forward_rates[reactant2_positions] *= particle_counts[
                reactant2_indices[reactant2_positions]
            ]

        np.multiply(
            rate_constant_b,
            particle_counts[product1_indices],
            out=backward_rates,
            casting="unsafe",
        )
        if product2_positions.size > 0:
            backward_rates[product2_positions] *= particle_counts[
                product2_indices[product2_positions]
            ]
        
        return rates
    
    return rates_fun
    