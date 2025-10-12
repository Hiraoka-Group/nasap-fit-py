import numpy as np
import pytest

from src.nasap_fit_py.fitting import make_objective_func_from_ode_rhs
from src.nasap_fit_py.fitting.sample_data import get_a_to_b_sample


def test():
    sample = get_a_to_b_sample()
    
    objective_func = make_objective_func_from_ode_rhs(
        sample.ode_rhs, sample.t, sample.y, sample.t[0], sample.y0)
    
    np.testing.assert_allclose(
        objective_func(*sample.params), 0.0, atol=1e-3)


def test_ydata_with_nan():
    sample = get_a_to_b_sample()
    ydata = sample.y.copy()

    # Pre-test
    ydata += 0.1
    # RSS should be 0.1^2 * ydata.size
    objective_func = make_objective_func_from_ode_rhs(
        sample.ode_rhs, sample.t, ydata, sample.t[0], sample.y0)
    expected = 0.1**2 * ydata.size
    np.testing.assert_allclose(
        objective_func(*sample.params), expected)
    
    # Test
    ydata[0][0] = np.nan
    objective_func = make_objective_func_from_ode_rhs(
        sample.ode_rhs, sample.t, ydata, sample.t[0], sample.y0)
    expected -= 0.1**2  # Subtract the contribution of the first element
    np.testing.assert_allclose(
        objective_func(*sample.params), expected)


if __name__ == '__main__':
    pytest.main(['-v', __file__])