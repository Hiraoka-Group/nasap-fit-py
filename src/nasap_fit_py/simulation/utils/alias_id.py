"""This module provides functions for working with alias IDs."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .id_value_mapping_to_array import convert_id_value_mapping_to_array

_T = TypeVar('_T', bound=Hashable)
_S = TypeVar('_S', bound=Hashable)
_U = TypeVar('_U', int, float)


def convert_alias_mapping_to_array(
        ids: Sequence[_T],
        alias_to_id: Mapping[_S, _T],
        alias_to_value: Mapping[_S, float],
        *,
        default: float | np.float64 = np.nan,
        ) -> npt.NDArray:
    _validate_alias_assem_ids(alias_to_id, ids)
    id_to_value = {
        alias_to_id[alias]: value
        for alias, value in alias_to_value.items()}
    return convert_id_value_mapping_to_array(ids, id_to_value, default=default)


def get_extracted_y_by_alias(
        y: npt.NDArray,
        alias_assem_ids_to_extract: Iterable[_S],
        assem_ids: Sequence[_T],
        alias_to_assem_id: Mapping[_S, _T],
        ) -> npt.NDArray:
    id_to_index = {id_: i for i, id_ in enumerate(assem_ids)}
    y_extracted = np.empty((y.shape[0], len(list(alias_assem_ids_to_extract))))
    for i, alias in enumerate(alias_assem_ids_to_extract):
        assem_id = alias_to_assem_id[alias]
        y_extracted[:, i] = y[:, id_to_index[assem_id]]
    return y_extracted


def _validate_alias_assem_ids(
        alias_to_assem_id: Mapping[_S, _T],
        assem_ids: Iterable[_T],
        ) -> None:
    assem_ids = set(assem_ids)
    for alias, assem_id in alias_to_assem_id.items():
        if assem_id not in assem_ids:
            raise ValueError(
                f'Invalid assem_id "{assem_id}" for alias "{alias}"')
        if alias in assem_ids:
            raise ValueError(f'Alias "{alias}" is already an assem_id')
