import numpy as np
import pytest

from src.nasap_fit_py.simulation import Addition, AliasAddition


def test_alias_addition_to_addition():
    ids = ['A', 'B', 'C']
    alias_to_id = {'a': 'A', 'b': 'B'}  # No alias for 'C'
    alias_addition = AliasAddition(
        time=1, solute_change={'a': 1, 'b': 2}, volume_change=3)
    
    expected = Addition(
        time=1, solute_change=np.array([1, 2, 0]), volume_change=3)
    
    addition = alias_addition.to_addition(alias_to_id, ids)

    assert addition == expected


if __name__ == '__main__':
    pytest.main(['-v', __file__])
