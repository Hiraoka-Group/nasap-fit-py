from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence, Mapping


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
    reaction_type: str
    duplicate_count_f: int
    duplicate_count_b: int
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
            reaction_type=r.reaction_type,
            duplicate_count_f=r.duplicate_count_f,
            duplicate_count_b=r.duplicate_count_b,
            rate_constant_f=reaction_type_to_rate_constant[r.reaction_type].forward,
            rate_constant_b=reaction_type_to_rate_constant[r.reaction_type].backward,
        )
        for r in reactions
    ]
