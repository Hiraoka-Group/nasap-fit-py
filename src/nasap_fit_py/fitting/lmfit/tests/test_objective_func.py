import numpy as np
import pytest
from lmfit import Minimizer, Parameters

from src.nasap_fit_py.fitting.lmfit import make_objective_func_for_lmfit_minimizer
from src.nasap_fit_py.fitting.sample_data import get_a_to_b_sample

# TODO: Add more tests


def test_use_for_lmfit_minimizer():
    # A -> B
    sample = get_a_to_b_sample()

    objective_func = make_objective_func_for_lmfit_minimizer(
        sample.ode_rhs, 
        sample.t, sample.y,
        sample.t[0], sample.y0,
        pass_params_as_array=False)

    params = Parameters()
    params.add('log_k', value=0.0)  # Initial guess

    minimizer = Minimizer(objective_func, params)

    result = minimizer.minimize()

    np.testing.assert_allclose(
        result.params['log_k'].value, sample.params.log_k, atol=1e-3)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
