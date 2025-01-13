from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

_T = TypeVar('_T')


def convert_id_value_mapping_to_array(
        ids: Sequence[_T],
        id_to_value: Mapping[_T, float],
        *,
        default: float | np.float64 = np.nan,
        ) -> npt.NDArray:
    id_to_index = {id_: i for i, id_ in enumerate(ids)}
    array = np.full(len(ids), float(default))
    for id_, value in id_to_value.items():
        array[id_to_index[id_]] = value
    return array
