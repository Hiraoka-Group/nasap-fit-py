from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


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
