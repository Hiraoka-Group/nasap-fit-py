from collections.abc import Callable
from typing import Concatenate, Protocol

import numpy as np
import numpy.typing as npt
from lmfit import Parameters
from scipy.integrate import solve_ivp


class LmfitObjectiveFunc(Protocol):
    def __call__(self, params: Parameters) -> float:
        ...


def make_objective_func_for_lmfit_minimizer(
        ode_rhs: Callable[Concatenate[float, npt.NDArray, ...], npt.NDArray],
        tdata: npt.NDArray, ydata: npt.NDArray, 
        t0: float, y0: npt.NDArray,
        pass_params_as_array: bool = True,
        *,
        method: str = 'RK45', rtol: float = 1e-3, atol: float = 1e-6
        ) -> LmfitObjectiveFunc:
    """Make an objective function for the `lmfit.Minimizer` class from 
    a simulating function.

    It can be used as the `userfcn` argument of the `lmfit.Minimizer` 
    class.

    Parameters
    ----------
    ode_rhs: Callable
        The ODE right-hand side function. It should have one of the
        following signatures:

        If `pass_params_as_array` is False:
            ``ode_rhs(t: float, y: npt.NDArray, p1: float, p2: float, \
                ..., pk: float) -> npt.NDArray``

        If `pass_params_as_array` is True:
            ``ode_rhs(t: float, y: npt.NDArray, \
                params_array: npt.NDArray) -> npt.NDArray``

        where `t` is the time (float), `y` is the dependent variable
        (1-D array, shape (n,)), `p1`, `p2`, ..., `pk` are the
        parameters of the ODE right-hand side function, and `params_array`
        is an array of the parameters. The names of `t` and `y` can be
        different.
    tdata : npt.NDArray, shape (n,)
        Time points of the data.
    ydata : npt.NDArray, shape (n, m)
        Data to be compared with the simulation.
    t0 : float
        Initial time.
    y0 : npt.NDArray, shape (m,)
        Initial conditions, i.e., the dependent variables at `t0`.
    pass_params_as_array : bool, optional
        If True, the parameters are passed to the ODE right-hand side
        function as an array. If False, the parameters are unpacked
        and passed individually. Default is True.
    method : str, optional
        The method to use for the `solve_ivp` function, which is used
        to simulate the system. Default is 'RK45'.
    rtol : float, optional
        The relative tolerance for the `solve_ivp` function. Default
        is 1e-3.
    atol : float, optional
        The absolute tolerance for the `solve_ivp` function. Default
        is 1e-6.

    Returns
    -------
    Callable
        The objective function for the `lmfit.Minimizer` class. It has
        the signature:

            ``objective_func(params: Parameters) -> float``

        where `params` is the parameters of the minimization problem,
        and the return value is the sum of the squared residuals between
        the data and the simulation.
    """
    def solve_ode(params):
        try:
            if pass_params_as_array:
                param_array = np.array(list(params.values()))
                sol = solve_ivp(
                    ode_rhs, [t0, tdata[-1]], y0, t_eval=tdata,
                    args=(param_array,), 
                    method=method, rtol=rtol, atol=atol)
            else:
                sol = solve_ivp(
                    ode_rhs, [t0, tdata[-1]], y0, t_eval=tdata,
                    args=tuple(params.values()), 
                    method=method, rtol=rtol, atol=atol)
            return sol
        except Exception as e:
            raise RuntimeError(f"ODE solver failed: {e}")

    def objective_func(params: Parameters) -> float:
        sol = solve_ode(params)
        ymodel = sol.y.T
        resid = ymodel - ydata
        return np.sum(resid**2)
    
    return objective_func
