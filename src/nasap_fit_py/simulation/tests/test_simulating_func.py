
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.integrate import solve_ivp

from src.nasap_fit_py.simulation import make_simulating_func_from_ode_rhs


def test_one_reaction():
    # A -> B
    def ode_rhs(t: float, y: npt.NDArray, k: float) -> npt.NDArray:
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    t = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1

    sol = solve_ivp(ode_rhs, (t[0], t[-1]), y0, args=(k,), t_eval=t)
    expected = sol.y.T

    y = simulating_func(t, y0, k)

    assert y.shape == (len(t), len(y0))
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-6)


def test_two_reactions():
    # A -> B -> C
    def ode_rhs(t: float, y: npt.NDArray, k1: float, k2: float) -> npt.NDArray:
        return np.array([-k1 * y[0], k1 * y[0] - k2 * y[1], k2 * y[1]])

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    t = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0, 0])
    k1 = 1
    k2 = 1

    sol = solve_ivp(ode_rhs, (t[0], t[-1]), y0, args=(k1, k2), t_eval=t)
    expected = sol.y.T

    y = simulating_func(t, y0, k1, k2)

    assert y.shape == (len(t), len(y0))
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-6)


def test_reversible_reaction():
    # A <-> B
    def ode_rhs(t: float, y: npt.NDArray, k1: float, k2: float) -> npt.NDArray:
        return np.array([-k1 * y[0] + k2 * y[1], k1 * y[0] - k2 * y[1]])
    
    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    t = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k1 = 1
    k2 = 1

    sol = solve_ivp(ode_rhs, (t[0], t[-1]), y0, args=(k1, k2), t_eval=t)
    expected = sol.y.T

    y = simulating_func(t, y0, k1, k2)

    assert y.shape == (len(t), len(y0))
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-6)


@st.composite
def t_y0_log_k_mat(draw) -> tuple:
    n = draw(st.integers(min_value=3, max_value=10))  # number of time points
    m = draw(st.integers(min_value=1, max_value=3))  # number of species
    t = draw(arrays(
        dtype=float, shape=(n,), 
        elements=st.floats(min_value=0, max_value=10**5), 
        unique=True))
    y0 = draw(arrays(
        dtype=float, shape=(m,), 
        elements=st.floats(min_value=0, max_value=10)))
    lob_k_mat = draw(arrays(
        dtype=float, shape=(m, m), 
        elements=st.floats(min_value=-3, max_value=3)))
    return np.sort(t), y0, lob_k_mat


@given(t_y0_log_k_mat())
def test_simulating_func(t_y0_log_k_mat: tuple) -> None:
    t, y0, log_k_mat = t_y0_log_k_mat
    k_mat = 10 ** log_k_mat
    ode_rhs = lambda t, y, k_mat: k_mat @ y

    def ode_rhs_with_fixed_parameters(t, y):
        return ode_rhs(t, y, k_mat)
    
    sol = solve_ivp(
        ode_rhs_with_fixed_parameters, 
        (t[0], t[-1]), y0, t_eval=t)
    expected = sol.y.T

    simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)

    y = simulating_func(t, y0, k_mat)

    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-6)


def test_custom_method():
    def ode_rhs(t: float, y: npt.NDArray, k: float) -> npt.NDArray:
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(
        ode_rhs, method='RK23')

    t = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1

    sol = solve_ivp(
        ode_rhs, (t[0], t[-1]), y0, args=(k,), t_eval=t, method='RK23')
    expected = sol.y.T

    y = simulating_func(t, y0, k)

    assert y.shape == (len(t), len(y0))
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-6)


def test_custom_rtol_atol():
    def ode_rhs(t: float, y: npt.NDArray, k: float) -> npt.NDArray:
        return np.array([-k * y[0], k * y[0]])

    simulating_func = make_simulating_func_from_ode_rhs(
        ode_rhs, rtol=1e-6, atol=1e-9)

    t = np.logspace(-3, 1, 12)
    y0 = np.array([1, 0])
    k = 1

    sol = solve_ivp(
        ode_rhs, (t[0], t[-1]), y0, args=(k,), t_eval=t, 
        rtol=1e-6, atol=1e-9)
    expected = sol.y.T

    y = simulating_func(t, y0, k)

    assert y.shape == (len(t), len(y0))
    np.testing.assert_allclose(y, expected, rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    pytest.main(['-v', __file__])
