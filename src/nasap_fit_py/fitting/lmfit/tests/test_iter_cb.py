import numpy as np
import numpy.typing as npt
import pytest
from lmfit import Minimizer, Parameters

from src.nasap_fit_py.fitting.lmfit import (IterationRecord,
                                            make_iter_cb_for_lmfit_minimizer,
                                            make_objective_func_for_lmfit_minimizer)
from src.nasap_fit_py.fitting.sample_data import get_a_to_b_sample


def test_use_for_lmfit_minimizer():
    # A -> B
    sample = get_a_to_b_sample()

    objective_func = make_objective_func_for_lmfit_minimizer(
        sample.ode_rhs, sample.t, sample.y, 
        sample.t[0], sample.y0,
        pass_params_as_array=False)
    # `objective_func` returns a float

    params = Parameters()
    params.add('log_k', value=1.0)  # Initial guess

    iter_cb, records = make_iter_cb_for_lmfit_minimizer()
    minimizer = Minimizer(objective_func, params, iter_cb=iter_cb)

    result = minimizer.minimize()

    np.testing.assert_allclose(
        result.params['log_k'].value, sample.params.log_k, atol=1e-3)
    assert len(records) > 0
    assert isinstance(records[0], IterationRecord)
    assert records[0].params.keys() == {'log_k'}
    assert records[0].iter == 0
    # The type of `resid` should be the same as the return type of 
    # `objective_func`
    assert isinstance(records[0].resid, float)


def test_case_where_objective_func_returns_array():
    # A -> B
    sample = get_a_to_b_sample()

    def objective_func(params: Parameters) -> npt.NDArray:
        log_k = params['log_k']
        ymodel = sample.simulating_func(sample.t, sample.y0, log_k)
        return ymodel - sample.y  # This is an array

    params = Parameters()
    params.add('log_k', value=1.0)  # Initial guess

    iter_cb, records = make_iter_cb_for_lmfit_minimizer()
    minimizer = Minimizer(objective_func, params, iter_cb=iter_cb)

    result = minimizer.minimize()

    np.testing.assert_allclose(
        result.params['log_k'].value, sample.params.log_k, atol=1e-3)
    assert len(records) > 0
    assert isinstance(records[0], IterationRecord)
    assert records[0].params.keys() == {'log_k'}
    assert records[0].iter == 0
    assert isinstance(records[0].resid, np.ndarray)  # array


if __name__ == '__main__':
    pytest.main(['-v', __file__])
