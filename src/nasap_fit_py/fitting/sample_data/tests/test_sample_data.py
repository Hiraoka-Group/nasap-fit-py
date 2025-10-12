from typing import NamedTuple

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.nasap_fit_py.fitting.sample_data import SampleData


def test_init() -> None:
    class Params(NamedTuple):
        k: float
    
    ode_rhs = lambda t, y, k: np.array([k * y[0], -k * y[0]])
    t = [0.0, 1.0, 2.0]
    y0 = [0.0, 0.0]
    params = Params(k=1.0)
    
    sample = SampleData(ode_rhs, t, y0, params)

    assert sample.ode_rhs is ode_rhs
    np.testing.assert_array_equal(sample.t, t)
    np.testing.assert_array_equal(sample.y0, y0)
    assert sample.params == params


def test_immutability() -> None:
    class Params(NamedTuple):
        k: float

    ode_rhs = lambda t, y, k: np.array([k * y[0], -k * y[0]])
    t = [0.0, 1.0, 2.0]
    y0 = [0.0, 0.0]
    params = Params(k=1.0)

    sample = SampleData(ode_rhs, t, y0, params)

    with pytest.raises(AttributeError):
        sample.ode_rhs = None  # type: ignore

    with pytest.raises(AttributeError):
        sample.t = None  # type: ignore

    with pytest.raises(AttributeError):
        sample.y0 = None  # type: ignore

    with pytest.raises(AttributeError):
        sample.params = None  # type: ignore


def test_recursive_immutability() -> None:
    class Params(NamedTuple):
        k: float

    ode_rhs = lambda t, y, k: np.array([k * y[0], -k * y[0]])
    t = [0.0, 1.0, 2.0]
    y0 = [0.0, 0.0]
    params = Params(k=1.0)

    sample = SampleData(ode_rhs, t, y0, params)

    with pytest.raises(ValueError):
        sample.t[0] = 0.0  # type: ignore

    with pytest.raises(ValueError):
        sample.y0[0] = 0.0  # type: ignore

    with pytest.raises(ValueError):
        sample.y[0, 0] = 0.0  # type: ignore

    with pytest.raises(AttributeError):
        sample.params.k = 0.0  # type: ignore


def test_simulating_func() -> None:
    # A -> B
    ode_rhs = lambda t, y, k: np.array([-k * y[0], k * y[0]])
    t = np.logspace(-3, 1, 10)
    y0 = np.array([1.0, 0.0])

    class Params(NamedTuple):
        k: float
    params = Params(1.0)

    sample = SampleData(ode_rhs, t, y0, params)

    # Expected values
    def ode_rhs_with_fixed_parameters(t, y):
        return ode_rhs(t, y, params.k)
    
    sol = solve_ivp(
        ode_rhs_with_fixed_parameters, 
        (t[0], t[-1]), y0, dense_output=True)
    expected = sol.sol(t).T
    
    y = sample.simulating_func(t, y0, params.k)
    
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-6)


def test_y() -> None:
    # A -> B
    ode_rhs = lambda t, y, k: np.array([-k * y[0], k * y[0]])
    t = np.logspace(-3, 1, 10)
    y0 = np.array([1.0, 0.0])

    class Params(NamedTuple):
        k: float
    params = Params(1.0)

    sample = SampleData(ode_rhs, t, y0, params)

    # `simulating_func` is tested in `test_simulating_func`.
    expected = sample.simulating_func(t, y0, params.k)

    y = sample.y
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-6)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
