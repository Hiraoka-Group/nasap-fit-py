import numpy as np
import pytest

from src.nasap_fit_py.fitting.sample_data import AToBParams, get_a_to_b_sample


def test_default_values():
    sample = get_a_to_b_sample()  # use default values
    np.testing.assert_allclose(sample.t, np.logspace(-3, 1, 10))
    assert callable(sample.simulating_func)
    assert sample.params == AToBParams(log_k=0.0)
    sim_result = sample.simulating_func(
        sample.t, np.array([1, 0]), sample.params.log_k)
    
    np.testing.assert_allclose(sim_result, sample.y, rtol=1e-3, atol=1e-6)


def test_custom_values():
    t = np.logspace(-2, 1, 20)
    y0 = np.array([0.5, 0.5])
    log_k = 1.0
    sample = get_a_to_b_sample(t=t, y0=y0, log_k=log_k)  # use custom values
    np.testing.assert_allclose(sample.t, t)
    np.testing.assert_allclose(sample.y[0], y0)
    assert sample.params.log_k == log_k

    sim_result = sample.simulating_func(
        sample.t, sample.y[0], sample.params.log_k)
    
    np.testing.assert_allclose(sim_result, sample.y, rtol=1e-3, atol=1e-6)


if __name__ == '__main__':
    pytest.main(['-v', __file__])
