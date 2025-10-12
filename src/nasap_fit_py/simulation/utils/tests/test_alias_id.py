import numpy as np
import pytest

from src.nasap_fit_py.simulation.utils import (convert_alias_mapping_to_array,
                                               get_extracted_y_by_alias)


def test_convert_alias_mapping_to_array():
    ids = ['A', 'B', 'C']
    alias_to_id = {'a': 'A', 'b': 'B'}  # C has no alias
    alias_to_value = {'a': 1.0, 'b': 2.0}
    expected = np.array([1.0, 2.0, np.nan])
    result = convert_alias_mapping_to_array(
        ids, alias_to_id, alias_to_value)
    np.testing.assert_array_equal(result, expected)


def test_convert_alias_mapping_to_array_with_default():
    ids = ['A', 'B', 'C']
    alias_to_id = {'a': 'A', 'b': 'B'}  # C has no alias
    alias_to_value = {'a': 1.0, 'b': 2.0}
    expected = np.array([1.0, 2.0, 0.0])
    result = convert_alias_mapping_to_array(
        ids, alias_to_id, alias_to_value, default=0.0)
    np.testing.assert_array_equal(result, expected)


def test_get_extracted_y_by_alias():
    # A -> B + 2C
    y = np.array([[1, 0, 0], [0.5, 1, 2]])
    alias_assem_ids_to_extract = ['a', 'b']
    assem_ids = ['A', 'B', 'C']
    alias_to_assem_id = {'a': 'A', 'b': 'B'}
    expected = np.array([[1, 0], [0.5, 1]])
    result = get_extracted_y_by_alias(
        y, alias_assem_ids_to_extract, assem_ids, alias_to_assem_id)
    np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
