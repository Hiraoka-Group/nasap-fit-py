import numpy as np
import pytest

from nasap.simulation.utils import convert_id_value_mapping_to_array


def test():
    ids = ['a', 'b', 'c']
    id_to_value = {'a': 1, 'b': 2}
    expected = [1, 2, np.nan]
    
    array = convert_id_value_mapping_to_array(ids, id_to_value)
    
    assert isinstance(array, np.ndarray)
    np.testing.assert_allclose(array, expected)


def test_default():
    ids = ['a', 'b', 'c']
    id_to_value = {'a': 1, 'b': 2}
    expected = [1, 2, 0]
    
    array = convert_id_value_mapping_to_array(ids, id_to_value, default=0)
    
    assert isinstance(array, np.ndarray)
    np.testing.assert_allclose(array, expected)


def test_int_default():
    # Should be converted to np.float64
    ids = ['a', 'b', 'c']
    id_to_value = {'a': 1, 'b': 2}
    expected = [1, 2, 0]
    
    array = convert_id_value_mapping_to_array(ids, id_to_value, default=0)

    assert array.dtype == np.float64
    assert isinstance(array, np.ndarray)
    np.testing.assert_allclose(array, expected)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
