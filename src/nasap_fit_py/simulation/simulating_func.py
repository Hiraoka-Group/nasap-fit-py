"""
Module for making simulating functions from ODE right-hand side functions.
"""

from collections.abc import Callable
from typing import Concatenate, Generic, ParamSpec, Protocol

import numpy.typing as npt
from scipy.integrate import solve_ivp

_P = ParamSpec('_P')


class SimulatingFunc(Generic[_P], Protocol):
    def __call__(
            self, t: npt.NDArray, y0: npt.NDArray,
            *args: _P.args, **kwargs: _P.kwargs
            ) -> npt.NDArray:
        ...


def make_simulating_func_from_ode_rhs(
        ode_rhs: Callable[Concatenate[float, npt.NDArray, _P], npt.NDArray],
        method: str = 'RK45', rtol: float = 1e-3, atol: float = 1e-6
        ) -> SimulatingFunc[_P]:
    """Make a simulating function from an ODE right-hand side function.

    Resulting simulating function can be called with time points, 
    initial values of the dependent variables, and the parameters of 
    the ODE right-hand side function. For example, it can be called
    as ``simulating_func(t, y0, k1, k2, k3, k4)`` for a 4-parameter 
    ODE right-hand side function ``ode_rhs(t, y, k1, k2, k3, k4)``.

    Parameters
    ----------
    ode_rhs: Callable
        The ODE right-hand side function. It should have the signature
        
            ``ode_rhs(t, y, *args, **kwargs) -> dydt``

        where ``t`` is the time (float), ``y`` is the dependent
        variable (1-D array, shape (m,)), ``args`` and ``kwargs``
        are the parameters of the ODE right-hand side function, and
        ``dydt`` is the derivative of the dependent variable (1-D
        array, shape (m,)). The names of ``t`` and ``y`` can be
        different.
    method: str, optional
        The method to use for the `solve_ivp` function. Default is
        'RK45'.
    rtol: float, optional
        The relative tolerance for the `solve_ivp` function. Default
        is 1e-3.
    atol: float, optional
        The absolute tolerance for the `solve_ivp` function. Default
        is 1e-6.

    Returns
    -------
    Callable
        The simulating function for the ODE right-hand side.
        It has the signature
        
            ``simulating_func(t, y0, *args, **kwargs) -> y``
    
        where ``t`` is the time points (1-D array, shape (n,)),
        ``y0`` is the initial values of the dependent variables
        (1-D array, shape (m,)), ``args`` and ``kwargs`` are the
        parameters of the ODE right-hand side function, and ``y``
        is the dependent variables at the time points (2-D array,
        shape (n, m)).
    
    Notes
    -----
    The simulating function uses `scipy.integrate.solve_ivp` to
    solve the ODEs with the default method `RK45`. The ``solve_ivp``
    function is called without specifying the `t_eval` argument for
    efficiency, and the result is interpolated using the dense
    output.
    Default values of `rtol` and `atol` are used for the `solve_ivp`
    function. The default values are `1e-3` and `1e-6`, respectively.
    """
    def simulating_func(
            t: npt.NDArray, y0: npt.NDArray,
            *args: _P.args, **kwargs: _P.kwargs
            ) -> npt.NDArray:
        def ode_rhs_with_fixed_parameters(
                t: float, y: npt.NDArray) -> npt.NDArray:
            return ode_rhs(t, y, *args, **kwargs)
        
        sol = solve_ivp(
            ode_rhs_with_fixed_parameters, (t[0], t[-1]), y0,
            t_eval=t, method=method, rtol=rtol, atol=atol)

        return sol.y.T

    return simulating_func
