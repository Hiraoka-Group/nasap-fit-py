"""
Sample data for A -> B reaction.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .sample_data import SampleData


class AToBParams(NamedTuple):
    log_k: float


def get_a_to_b_sample(
        *,
        t: npt.ArrayLike = np.logspace(-3, 1, 10),
        y0: npt.ArrayLike = np.array([1, 0]),
        log_k: float = 0.0
        ):  # Intentionally not providing return type,
    # because when return type, `SampleData`, is provided,
    # somehow mypy does not infer the generic type of `SampleData`.
    """Get sample data for ``A -> B`` reaction.

    Reactions:
    - ``A -> B``

    ODEs:
    - ``d[A]/dt = -k * [A]``
    - ``d[B]/dt = k * [A]``

    Initial conditions:
    - ``[A] = 1.0``
    - ``[B] = 0.0``
    
    Kinetic constants:
    - ``k = 1.0``  i.e. ``log_k = 0.0``

    Time points: ``np.logspace(-3, 1, 10)``
    
    Returns
    -------
    SampleData
        Sample data for A -> B reaction.
        Returned object has the following read-only attributes:
        - tdata (npt.NDArray, shape (10,)): Time points.
            ``np.logspace(-3, 1, 10)``
        - ydata (npt.NDArray, shape (10, 2)): 
            Simulated concentrations of A and B.
        - simulating_func (AToBSimFunc): 
            Function that simulates the system.
        - params (MappingProxyType[str, float]):
            Read-only dictionary of parameters.
        - y0 (npt.NDArray, shape (2,)): Initial conditions.
    """
    t = np.array(t)
    y0 = np.array(y0)
    
    def ode_rhs(t: float, y: npt.NDArray, log_k: float) -> npt.NDArray:
        k = 10**log_k
        return np.array([-k * y[0], k * y[0]])
    
    return SampleData(ode_rhs, t, y0, AToBParams(log_k))
