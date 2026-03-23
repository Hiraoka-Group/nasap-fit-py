from dataclasses import dataclass


@dataclass(frozen=True)
class RateConstant:
    forward: float
    backward: float
