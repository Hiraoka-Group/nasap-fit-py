from collections.abc import Iterable, Sequence
from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from .lib import calc_consumed_count, calc_produced_count
from .reaction_class import Reaction

_T_co = TypeVar('_T_co', covariant=True)  # type of assembly
_S_co = TypeVar('_S_co', covariant=True)  # type of reaction kind


class OdeRhs(Protocol):
    def __call__(
            self, t: float, y: npt.NDArray, 
            log_k_of_rxn_kinds: npt.NDArray,
            ) -> npt.NDArray:
        ...


def create_ode_rhs(
        assemblies: Sequence[_T_co], 
        reaction_kinds: Sequence[_S_co],
        reactions: Iterable[Reaction[_T_co, _S_co]],
        ) -> OdeRhs:
    """Create a function that calculates the right-hand side of the ODE.

    Parameters
    ----------
    assemblies : Sequence[_T_co]
        Assembly IDs. Can be any hashable type. The order of the IDs
        will be used to determine the order of `y` parameter in the
        returned function.
    reaction_kinds : Sequence[_S_co]
        Reaction kind IDs. Can be any hashable type. The order of the IDs
        will be used to determine the order of `log_k_of_rxn_kinds`
        parameter in the returned function.
    reactions : Iterable[Reaction[_T_co, _S_co]]
        Reactions to be included in the ODE.
    """
    assemblies = list(assemblies)
    reaction_kinds = list(reaction_kinds)
    reactions = list(reactions)
    
    assem_to_index = {assem: i for i, assem in enumerate(assemblies)}
    rxn_kind_to_index = {kind: i for i, kind in enumerate(reaction_kinds)}

    # n: number of assemblies
    # m: number of reactions
    # k: number of reaction kinds
    num_of_assems = len(assemblies)
    num_of_rxns = len(reactions)
    num_of_rxn_kinds = len(reaction_kinds)
    
    # shape: (m, k), dtype: bool
    rxn_to_kind = np.full((num_of_rxns, num_of_rxn_kinds), False)
    for i, reaction in enumerate(reactions):
        rxn_kind_index = rxn_kind_to_index[reaction.reaction_kind]
        rxn_to_kind[i, rxn_kind_index] = True

    # shape: (m,)
    coefficients = np.full(num_of_rxns, np.nan)
    for i, reaction in enumerate(reactions):
        coefficients[i] = reaction.duplicate_count
    assert not np.isnan(coefficients).any()
    
    # shape: (m, n)
    consumed = calc_consumed_count(assemblies, reactions)  # shape: (m, n)
    produced = calc_produced_count(assemblies, reactions)  # shape: (m, n)
    change = produced - consumed  # shape: (m, n)
    
    # Note: `ode_rhs` should be as efficient as possible
    # to be used in `solve_ivp` for large-scale simulations
    def ode_rhs(
            t: float, y: npt.NDArray, log_k_of_rxn_kinds: npt.NDArray,
            ) -> npt.NDArray:  # shape: (n,)
        # t: time
        # y: concentrations of assemblies, shape: (n,)
        # log_k_of_rxn_kinds: log(k) of each reaction kind, shape: (k,)

        k_of_rxn_kinds = 10.0**log_k_of_rxn_kinds  # shape: (k,)
        ks = rxn_to_kind @ k_of_rxn_kinds  # shape: (m,)

        # First, determine the rate of the reaction occurrences.
        # Let's call it "event rate".
        # One value for each reaction. Thus, shape: (m,)
        # event_rate = coefficient * k * product(reactant**consumed_count)
        event_rates = coefficients * ks * np.prod(y**consumed, axis=1)

        # Then, calculate the change in the concentration of each assembly.
        # This is exactly the right-hand side of the ODE.
        # One value for each assembly. Thus, shape: (n,)
        # rhs = sum(event_rate * change)
        rhs = np.sum(event_rates[:, None] * change, axis=0)

        return rhs
    
    return ode_rhs
