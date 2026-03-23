from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedReaction:
    reactant1: str
    reactant2: str | None
    product1: str
    product2: str | None
    rate_constant_f: float
    rate_constant_b: float
