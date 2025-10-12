from dataclasses import dataclass
from typing import Any, Protocol

import numpy.typing as npt
from lmfit import Parameters


@dataclass
class IterationRecord:
    params: dict[str, float]
    iter: int
    resid: npt.NDArray


class MinimizerIterCallback(Protocol):
    def __call__(
            self, params: Parameters, iter: int, resid: Any,
            *args, **kwargs) -> None:
        ...


# TODO: Add examples
def make_iter_cb_for_lmfit_minimizer() -> tuple[
        MinimizerIterCallback, list[IterationRecord]]:
    """Make an iteration callback function for the `lmfit.Minimizer` class.
    
    Returns
    -------
    iter_cb : MinimizerIterCallback
        The iteration callback function that records the parameters and
        residuals at each iteration.
    records : list[IterationRecord]
        The list of records that contain the parameters and residuals at
        each iteration. The records are empty at the beginning of the
        minimization, and they are filled by the iteration callback
        function.

        ``IterationRecord`` is a dataclass with the following attributes:

        - ``params``: A dictionary that contains the parameter names and
            values at the iteration.
        - ``iter``: The iteration number. The first iteration is 0.
        - ``resid``: The residuals at the iteration. It is a return value
            of the ``userfcn`` function of the `lmfit.Minimizer` class.
            It can be a 1-D array, float, or any other type that is
            returned by the ``userfcn`` function.
    """
    records = []
    def record_params(params, iter, resid, *args, **kwargs):
        records.append(IterationRecord(
            params=params.valuesdict(),
            iter=iter + 1,  # Minimizer.nfev starts from -1
            resid=resid.copy()
            ))
    return record_params, records
