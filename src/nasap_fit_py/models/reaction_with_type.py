from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReactionWithType:
    reactant1: str
    reactant2: str | None
    product1: str
    product2: str | None
    reaction_type: str
    duplicate_count_f: int
    duplicate_count_b: int
