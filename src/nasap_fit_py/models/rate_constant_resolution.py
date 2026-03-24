from __future__ import annotations

from collections.abc import Mapping, Sequence

from . import RateConstant, Reaction, ReactionWithType


def resolve_rate_constants(
    reactions: Sequence[ReactionWithType],
    rtype_to_rate_constant: Mapping[str, RateConstant],
) -> Sequence[Reaction]:
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
        resolved_reactions.append(Reaction(
                reactant1=r.reactant1,
                reactant2=r.reactant2,
                product1=r.product1,
                product2=r.product2,
                rate_constant_f=rtype_to_rate_constant[r.reaction_type].forward * r.duplicate_count_f,
                rate_constant_b=rtype_to_rate_constant[r.reaction_type].backward * r.duplicate_count_b,
        ))
    return resolved_reactions
