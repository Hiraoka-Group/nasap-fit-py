from collections.abc import Callable
from typing import Concatenate, Generic, NamedTuple, ParamSpec, TypeVar

import numpy as np
import numpy.typing as npt

from src.nasap_fit_py.simulation import SimulatingFunc, make_simulating_func_from_ode_rhs

_P = ParamSpec('_P')
_S = TypeVar('_S', bound=NamedTuple)


class SampleData(Generic[_P, _S]):
    """Immutable class for sample data."""
    def __init__(
            self, ode_rhs: Callable[
                Concatenate[float, npt.NDArray, _P], npt.NDArray], 
            t: npt.ArrayLike, y0: npt.ArrayLike,
            params: _S
            ) -> None:
        self._ode_rhs = ode_rhs
        self._t = np.array(t)
        self._y0 = np.array(y0)
        self._params = params

        self._simulating_func = make_simulating_func_from_ode_rhs(ode_rhs)
        self._ydata = self.simulating_func(self.t, self.y0, *params)
        assert np.array_equal(self._ydata[0], self._y0)

        # Make np.ndarray read-only
        self._t.flags.writeable = False
        self._y0.flags.writeable = False
        self._ydata.flags.writeable = False

    @property
    def t(self) -> npt.NDArray:
        """Time points of the data. (Read-only)"""
        return self._t
    
    @property
    def y(self) -> npt.NDArray:
        """Data to be compared with the simulation. (Read-only)"""
        return self._ydata

    @property
    def y0(self) -> npt.NDArray:
        """Initial conditions. (Read-only)"""
        return self._y0
    
    @property
    def ode_rhs(self) -> Callable[
            Concatenate[float, npt.NDArray, _P], npt.NDArray]:
        """ODE right-hand side function."""
        return self._ode_rhs
    
    @property
    def simulating_func(self) -> SimulatingFunc[_P]:
        """Function that simulates the system.
        
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
        THe simulating function uses the `scipy.integrate.solve_ivp`
        to simulate the system. For the tolerance, it uses the
        default values of `scipy.integrate.solve_ivp`, which are
        ``rtol=1e-3`` and ``atol=1e-6``.
        """
        return self._simulating_func
    
    @property
    def params(self) -> _S:
        """NamedTuple of parameters. (Read-only)"""
        return self._params
