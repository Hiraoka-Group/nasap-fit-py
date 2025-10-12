from collections.abc import Hashable, Sequence
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from src.nasap_fit_py.ode_creation.reaction_class import Reaction

_T = TypeVar('_T', bound=Hashable)

# n: number of assemblies
# m: number of reactions
# k: number of reaction kinds

def calc_particle_change(
        assemblies: Sequence[_T], reactions: Sequence[Reaction[_T, Any]]
        ) -> npt.NDArray:  # shape (m, n)
    assemblies = list(assemblies)
    reactions = list(reactions)
    consumed_count = calc_consumed_count(assemblies, reactions)
    produced_count = calc_produced_count(assemblies, reactions)
    return produced_count - consumed_count


def calc_consumed_count(
        assemblies: Sequence[_T], reactions: Sequence[Reaction[_T, Any]]
        ) -> npt.NDArray:  # shape (m, n)
    assemblies = list(assemblies)
    reactions = list(reactions)
    assem_to_index = {assem: i for i, assem in enumerate(assemblies)}

    consumed_count = np.zeros((len(reactions), len(assemblies)), dtype=int)

    for i, reaction in enumerate(reactions):
        for assem in reaction.reactants:
            assem_index = assem_to_index[assem]
            consumed_count[i, assem_index] += 1

    return consumed_count


def calc_produced_count(
        assemblies: Sequence[_T], reactions: Sequence[Reaction[_T, Any]]
        ) -> npt.NDArray:  # shape (m, n)
    assemblies = list(assemblies)
    reactions = list(reactions)
    assem_to_index = {assem: i for i, assem in enumerate(assemblies)}

    produced_count = np.zeros((len(reactions), len(assemblies)), dtype=int)

    for i, reaction in enumerate(reactions):
        for assem in reaction.products:
            assem_index = assem_to_index[assem]
            produced_count[i, assem_index] += 1

    return produced_count
