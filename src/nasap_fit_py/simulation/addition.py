from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

_T = TypeVar('_T', bound=Hashable)
_S = TypeVar('_S', bound=Hashable)


@dataclass
class Addition:
    time: float
    solute_change: npt.NDArray
    volume_change: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Addition):
            return NotImplemented
        return (
            self.time == other.time
            and np.array_equal(self.solute_change, other.solute_change)
            and self.volume_change == other.volume_change)


@dataclass
class AliasAddition(Generic[_S]):
    time: float
    solute_change: Mapping[_S, float]
    volume_change: float

    def to_addition(
            self, alias_to_id: Mapping[_S, _T], ids: Sequence[_T],
            *,
            default_solute_change: float | np.float64 = 0.0,
            ) -> Addition:
        ids = list(ids)
        if len(ids) != len(set(ids)):
            raise ValueError('ids should be unique')
        for alias in self.solute_change:
            if alias not in alias_to_id:
                raise ValueError(f'{alias} not in alias_to_id')
        for alias, id_ in alias_to_id.items():
            if id_ not in ids:
                raise ValueError(f'{id_} not in ids')
        solute_change_array = np.full(len(ids), default_solute_change)
        id_to_index = {id_: i for i, id_ in enumerate(ids)}
        for alias, change in self.solute_change.items():
            solute_change_array[id_to_index[alias_to_id[alias]]] = change
        return Addition(
            time=self.time, solute_change=solute_change_array,
            volume_change=self.volume_change)
