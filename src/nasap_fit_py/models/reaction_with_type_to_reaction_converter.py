from __future__ import annotations

from collections.abc import Mapping, Sequence

from .rate_constant import RateConstant
from .reaction import Reaction
from .reaction_with_type import ReactionWithType


def convert_reaction_with_type_to_reaction(
    reactions_with_type: Sequence[ReactionWithType],
    rtype_to_rate_constant: Mapping[str, RateConstant],
) -> Sequence[Reaction]:
    """Apply rate constants to reactions, accounting for duplicate pathways.
    
    Parameters
    ----------
    reactions_with_type : Sequence[ReactionWithType]
        Sequence of reactions with types to resolve.
    rtype_to_rate_constant : Mapping[str, RateConstant]
        Mapping from reaction type to rate constants.
    
    Returns
    -------
    Sequence[Reaction]
        Sequence of reactions with rate constants applied and duplicates multiplied in.
    
    Raises
    ------
    ValueError
        If a reaction type is not found in rtype_to_rate_constant.
    """
    reactions: list[Reaction] = []
    for rwt in reactions_with_type:
        if rwt.reaction_type not in rtype_to_rate_constant:
            reaction_desc = f"{rwt.reactant1}"
            if rwt.reactant2:
                reaction_desc += f" + {rwt.reactant2}"
            reaction_desc += " -> "
            reaction_desc += f"{rwt.product1}"
            if rwt.product2:
                reaction_desc += f" + {rwt.product2}"
            raise ValueError(
                f"Reaction type '{rwt.reaction_type}' is not defined in rate_constants. "
                f"This is the corresponding reaction: {reaction_desc}. "
            )
        reactions.append(
            Reaction(
                reactant1=rwt.reactant1,
                reactant2=rwt.reactant2,
                product1=rwt.product1,
                product2=rwt.product2,
                rate_constant_f=rtype_to_rate_constant[rwt.reaction_type].forward * rwt.duplicate_count_f,
                rate_constant_b=rtype_to_rate_constant[rwt.reaction_type].backward * rwt.duplicate_count_b,
            )
        )
    return reactions
