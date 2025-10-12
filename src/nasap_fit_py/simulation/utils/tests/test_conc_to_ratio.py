import numpy as np
import pytest

from src.nasap_fit_py.simulation.utils import concentration_to_existence_ratio


def test_constant_100pct():
    concentrations = np.array([
        # A, B
        [1, 0],    # t = 0
        [0.5, 1],  # t = 1
        [0, 2],    # t = 2
        ])
    conc_of_100pct = np.array([1, 2])
    expected = np.array([
        # A, B
        [100, 0],    # t = 0
        [50, 50],    # t = 1
        [0, 100],    # t = 2
        ])
    
    ratio = concentration_to_existence_ratio(concentrations, conc_of_100pct)
    
    assert np.allclose(ratio, expected)


def test_variable_100pct():
    # Situation where the definition of 100% changes over time,
    # for example, due to the addition of a substance.
    concentrations = np.array([
        # A, B
        [1, 0],    # t = 0
        [0.5, 1],  # t = 1
        [0, 2],    # t = 2
        ])
    # Consuder the addition of 1 M of A at t = 1.2 min.
    # Before the addition, the 100% concentration was 1 M of A or 2 M of B.
    # After the addition, the 100% concentration is 2 M of A or 4 M of B.
    conc_of_100pct = np.array([
        [1, 2],    # t = 0
        [2, 4],    # t = 1
        [2, 4],    # t = 2
        ])
    expected = np.array([
        # A, B
        [100, 0],    # t = 0
        [25, 25],    # t = 1
        [0, 50],     # t = 2
        ])
    
    ratio = concentration_to_existence_ratio(concentrations, conc_of_100pct)

    assert np.allclose(ratio, expected)
    

if __name__ == '__main__':
    pytest.main(['-v', __file__])
