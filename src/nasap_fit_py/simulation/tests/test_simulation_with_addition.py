import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import solve_ivp

from src.nasap_fit_py.simulation import SimulationResult, simulate_solute_with_addition
from src.nasap_fit_py.simulation.addition import Addition


# A -> B
def ode_rhs(
        t: float, y: npt.NDArray, log_k_array: npt.NDArray
        ) -> npt.NDArray:
    k_array = 10.0 ** log_k_array
    k = k_array[0]
    a, b = y
    da_dt = -k * a
    db_dt = k * a
    return np.array([da_dt, db_dt])

@pytest.fixture
def t() -> npt.NDArray:
    return np.array([5, 10, 15, 20, 25, 30, 60, 120, 180, 300])

@pytest.fixture
def t0() -> float:
    return 0

@pytest.fixture
def solute0() -> npt.NDArray:
    return np.array([1, 0]) * 1e-6  # 1 µmol, 0 µmol

@pytest.fixture
def vol0() -> float:
    return 500 * 1e-6  # 500 µL

@pytest.fixture
def log_k_array() -> npt.NDArray:
    return np.array([-2])


def test_without_addition(t, t0, solute0, vol0, log_k_array):
    conc0 = solute0 / vol0
    sol = solve_ivp(
        ode_rhs, [t0, t[-1]], conc0, t_eval=t, args=(log_k_array,),
        method='LSODA')
    expected_conc = sol.y.T
    expected_solute = expected_conc * vol0

    result = simulate_solute_with_addition(
        ode_rhs, t, t0, solute0, vol0, 
        ode_rhs_args=(log_k_array,), additions=None,
        method='LSODA')
    
    assert isinstance(result, SimulationResult)
    np.testing.assert_allclose(result.t, t)
    np.testing.assert_allclose(
        result.solute, expected_solute, atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(result.vol, vol0)


def test_addition(solute0, vol0, log_k_array):
    t = np.array([5, 10, 15, 20, 25, 30, 60, 120, 180, 300])
    addition = [Addition(90, np.array([1, 0]) * 1e-6, 10 * 1e-6)]
    first_t = np.array([5, 10, 15, 20, 25, 30, 60])
    second_t = np.array([120, 180, 300])
    
    first_solute0 = solute0
    first_vol0 = vol0
    first_conc0 = first_solute0 / first_vol0
    first_sol = solve_ivp(
        ode_rhs, [0, first_t[-1]], first_conc0, t_eval=first_t, 
        args=(log_k_array,), method='LSODA')
    first_conc = first_sol.y.T

    sol_for_conc_at_90 = solve_ivp(
        ode_rhs, [0, 90], first_conc0, t_eval=np.array([90]),
        args=(log_k_array,), method='LSODA')
    conc_just_before_addition = sol_for_conc_at_90.y.T[0]
    solute_just_before_addition = conc_just_before_addition * first_vol0

    solute_just_after_addition = (
        solute_just_before_addition + np.array([1, 0]) * 1e-6)
    vol_just_after_addition = first_vol0 + 10 * 1e-6
    assert np.isclose(vol_just_after_addition, 510 * 1e-6)

    conc_just_after_addition = (
        solute_just_after_addition / vol_just_after_addition)
    second_sol = solve_ivp(
        ode_rhs, [90, second_t[-1]], conc_just_after_addition, 
        t_eval=second_t, args=(log_k_array,), method='LSODA')
    second_conc = second_sol.y.T

    expected_conc = np.vstack([first_conc, second_conc])
    expected_vol = np.array(
        [500] * len(first_t) + [510] * len(second_t)) * 1e-6
    expected_solute = expected_conc * expected_vol[:, None]

    result = simulate_solute_with_addition(
        ode_rhs, t, 0, solute0, 500 * 1e-6, ode_rhs_args=(log_k_array,), 
        additions=addition, method='LSODA')

    assert isinstance(result, SimulationResult)
    np.testing.assert_allclose(result.t, t)
    np.testing.assert_allclose(
        result.solute, expected_solute, atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(result.vol, expected_vol)
    

def test_addition_at_t0(solute0, vol0, log_k_array):
    t0 = 0
    t = np.array([5, 10])
    addition = [Addition(0, np.array([1, 0]) * 1e-3, 0)]
    with pytest.raises(ValueError):
        simulate_solute_with_addition(
            ode_rhs, t, t0, solute0, vol0, ode_rhs_args=(log_k_array,), 
            additions=addition, method='LSODA')


def test_addition_at_time_in_t(t0, solute0, vol0, log_k_array):
    t = np.array([5, 10])
    addition = [Addition(10, np.array([1, 0]) * 1e-3, 0)]
    with pytest.raises(ValueError):
        simulate_solute_with_addition(
            ode_rhs, t, t0, solute0, vol0, ode_rhs_args=(log_k_array,), 
            additions=addition, method='LSODA')


if __name__ == '__main__':
    pytest.main(['-vv', __file__])
