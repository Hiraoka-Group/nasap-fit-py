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
    reactions: Sequence[Reaction],
    reaction_type_to_rate_constant: Mapping[str, RateConstant],
) -> Sequence[ResolvedReaction]:

    return [
        ResolvedReaction(
            reactant1=r.reactant1,
            reactant2=r.reactant2,
            product1=r.product1,
            product2=r.product2,
            rate_constant_f=reaction_type_to_rate_constant[r.reaction_type].forward*r.duplicate_count_f,
            rate_constant_b=reaction_type_to_rate_constant[r.reaction_type].backward*r.duplicate_count_b,
        )
        for r in reactions
    ]
