import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Concatenate

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

from src.nasap_fit_py.simulation.addition import Addition


@dataclass
class SimulationResult:
    t: npt.NDArray
    solute: npt.NDArray
    vol: npt.NDArray

    @property
    def conc(self) -> npt.NDArray:
        """Return the concentration."""
        return self.solute / self.vol[:, None]


def simulate_solute_with_addition(
        ode_rhs: Callable[
            Concatenate[float, npt.NDArray, ...], npt.NDArray],
        t: npt.NDArray,
        t0: float, solute0: npt.NDArray, vol0: float,
        additions: Iterable[Addition] | None = None,
        ode_rhs_args: Iterable | None = None,
        *,
        method: str = 'RK45', rtol: float = 1e-3, atol: float = 1e-6,
        ) -> SimulationResult:
    """Simulate the system with solute and volume additions."""
    # Initializations and validations.
    if t.ndim != 1:
        raise ValueError('t should be 1D')
    if solute0.ndim != 1:
        raise ValueError('solute0 should be 1D')
    
    if not np.all(np.diff(t) > 0):
        raise ValueError('t should be monotonically increasing')
    
    if additions is None:
        additions = []
    else:
        additions = list(additions)
    for addition in additions:
        _validate_addition(t, t0, solute0, addition)
    if len(additions) != len(set(addition.time for addition in additions)):
        raise ValueError('Addition times should be unique')
    assert isinstance(additions, list)
    
    if ode_rhs_args is None:
        ode_rhs_args = ()
    ode_rhs_args = tuple(ode_rhs_args)
    
    # Concentration follows the ODE, not the amount of solutes.
    def solve_ivp_for_conc(
            t0: float, conc0: npt.NDArray, t_eval: npt.NDArray
            ) -> npt.NDArray:
        sol = solve_ivp(
            ode_rhs, [t0, t_eval[-1]], conc0, t_eval=t_eval, 
            args=ode_rhs_args, method=method, rtol=rtol, atol=atol)
        return sol.y.T

    if len(additions) == 0:
        conc0 = solute0 / vol0
        conc = solve_ivp_for_conc(t0, conc0, t)
        solute = conc * vol0
        vol = np.full_like(t, vol0, dtype=float)  # Constant volume
        assert solute.shape == (len(t), solute0.size)
        return SimulationResult(t=t, solute=solute, vol=vol)

    # Treat y0 as an addition at t0.
    additions.append(
        Addition(time=t0, solute_change=solute0, volume_change=vol0))
    
    # solute_just_before_cur_add contains solute just before the next 
    # addition. Since solute0 will be added at t0, 
    # solute_just_before_cur_add should be initialized with zeros.
    solute_just_before_cur_add = np.zeros_like(solute0)
    # Same goes for the volume.
    vol_just_before_cur_add = 0.0

    # Sort the addition by time.
    additions = sorted(additions, key=lambda x: x.time)

    solute = np.empty((0, solute0.size))
    vol = np.empty(0, dtype=float)

    for i, addition in enumerate(additions):
        # Perform the addition.
        solute_just_after_cur_add = (
            solute_just_before_cur_add + addition.solute_change)
        vol_just_after_cur_add = (
            vol_just_before_cur_add + addition.volume_change)
        conc_just_after_cur_add = (
            solute_just_after_cur_add / vol_just_after_cur_add)
        
        # If the last iteration.
        if i == len(additions) - 1:
            cur_t_eval = t[t > addition.time]
            cur_conc = solve_ivp_for_conc(
                addition.time, conc_just_after_cur_add, cur_t_eval)
            cur_solute = cur_conc * vol_just_after_cur_add
            solute = np.concatenate([solute, cur_solute])
            cur_vol = np.full_like(
                cur_t_eval, vol_just_after_cur_add, dtype=float)
            vol = np.concatenate([vol, cur_vol])
            break
        
        next_t_add = additions[i + 1].time
        cur_t_eval = t[(t >= addition.time) & (t < next_t_add)]

        # y at next_t_add should be evaluated,
        # which will be y_just_before_cur_add for the next addition.
        cur_t_eval = np.concatenate([cur_t_eval, [next_t_add]])
        
        cur_conc = solve_ivp_for_conc(
            addition.time, conc_just_after_cur_add, cur_t_eval)
        cur_solute = cur_conc * vol_just_after_cur_add

        # Update for next iter.
        solute_just_before_cur_add = cur_solute[-1]
        vol_just_before_cur_add = vol_just_after_cur_add

        # Remove y at next_t_add.
        cur_solute = cur_solute[:-1]

        solute = np.concatenate([solute, cur_solute])

        cur_vol = np.full_like(
            cur_t_eval[:-1], vol_just_after_cur_add, dtype=float)
        vol = np.concatenate([vol, cur_vol])
    
    return SimulationResult(t=t, solute=solute, vol=vol)


def _validate_addition(
        t: npt.NDArray, t0: float, y0: npt.NDArray,
        addition: Addition,
        ):
    if addition.time < t0 or addition.time > t[-1]:
        raise ValueError(
            f'time {addition.time} should be within t0 and t[-1], '
            f'i.e., {t0} < time < {t[-1]}')
    if addition.time == t0:
        raise ValueError(
            f'time {addition.time} should not be equal to t0.'
            f'Use a time point slightly greater than t0, '
            f'or simply add the solute to y0.')
    if addition.time in t:
        raise ValueError(
            f'time {addition.time} should not be in t.'
            f'Use a time point slightly greater than or less than '
            f'the addition times.')
    
    if addition.solute_change.shape != y0.shape:
        raise ValueError(
            f'solute_change shape {addition.solute_change.shape} '
            f'does not match y0 shape {y0.shape}')
    if not np.isfinite(addition.volume_change):
        raise ValueError(f'volume_change is not finite')
    if addition.volume_change < 0:
        warnings.warn('volume_change is negative')
