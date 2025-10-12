import numpy as np
import numpy.typing as npt
import pytest
from scipy.constants import Avogadro

from src.nasap_fit_py.simulation import Gillespie
from src.nasap_fit_py.simulation.gillespie import GillespieResult, Status


def test_init():
    # A <-> B
    init_particle_counts = np.array([100, 200])
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([0.1, 0.2])
    particle_changes = [np.array([-1, 1]), np.array([1, -1])]
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        t_max=10.0)

    np.testing.assert_allclose(
        gillespie.particle_counts_seq[0], init_particle_counts)
    assert gillespie.rates_fun == rates_fun
    np.testing.assert_allclose(
        gillespie.particle_changes, particle_changes)
    assert gillespie.t_max == 10.0
    assert gillespie.max_iter == 1_000_000
    assert gillespie.t_seq[0] == 0.0


def test_solve():
    # A <-> B
    init_particle_counts = np.array([100, 200])
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([0.1, 0.2])
    particle_changes = [np.array([-1, 1]), np.array([1, -1])]
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        t_max=10.0)

    result = gillespie.solve()

    assert isinstance(result, GillespieResult)
    assert result.status in {
        Status.REACHED_T_MAX, Status.REACHED_MAX_ITER, 
        Status.TOTAL_RATE_ZERO}
    assert result.t_seq[0] == 0.0
    np.testing.assert_allclose(
        result.particle_counts_seq[0], init_particle_counts)


def test_step_count():
    # A <-> B
    init_particle_counts = np.array([100, 200])
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([0.1, 0.2])
    particle_changes = [np.array([-1, 1]), np.array([1, -1])]
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        max_iter=1)
    
    result = gillespie.solve()

    assert len(result.t_seq) == 2


def test_status_of_max_iter_reached():
    # A <-> B
    init_particle_counts = np.array([100, 200])
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([0.1, 0.2])
    particle_changes = [np.array([-1, 1]), np.array([1, -1])]
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        max_iter=1)
    
    result = gillespie.solve()
    
    assert result.status == Status.REACHED_MAX_ITER


def test_status_of_total_rate_zero():
    # A -> B (only one direction)

    # Few particles, so the total rate becomes zero
    # before the max iteration is reached.
    init_particle_counts = np.array([10, 0])
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([0.1 * x[0]])
    particle_changes = [np.array([-1, 1])]
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        max_iter=100)
    
    # A becomes zero only in 10 steps,
    # which makes the total rate zero.
    
    result = gillespie.solve()

    assert result.status == Status.TOTAL_RATE_ZERO


def test_status_of_t_max_reached():
    # A <-> B
    init_particle_counts = np.array([100, 200])
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([0.1 * x[0], 0.2 * x[1]])
    particle_changes = [np.array([-1, 1]), np.array([1, -1])]
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        t_max=0.1)
    
    # Since the reaction is reversible,
    # the number of A and B will never become zero.
    # Therefore, the simulation will stop when t_max is reached.
    
    result = gillespie.solve()

    assert result.status == Status.REACHED_T_MAX


def test_concentrations():
    # A <-> B
    init_particle_counts = np.array([100, 200])
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([0.1, 0.2])
    particle_changes = [np.array([-1, 1]), np.array([1, -1])]
    volume = 1.0
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        volume=volume, t_max=10.0)

    concentrations = gillespie.concentrations

    assert concentrations.shape == (2,)
    assert np.all(concentrations >= 0)


def test_gillespie_init_based_on_concentrations():
    # A <-> B
    init_concentrations = np.array([1e-4, 2e-4])
    def conc_rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([0.1, 0.2])
    particle_changes = [np.array([-1, 1]), np.array([1, -1])]
    volume = 1.0
    gillespie = Gillespie.init_based_on_concentrations(
        init_concentrations, conc_rates_fun, particle_changes, volume, 
        t_max=10.0)

    np.testing.assert_allclose(
        gillespie.particle_counts_seq[0], 
        init_concentrations * Avogadro * volume)
    assert gillespie.volume == volume
    assert gillespie.t_max == 10.0
    assert gillespie.max_iter == 1_000_000


def test_with_example_reaction():
    # Reaction:
    # A -> B + 2C
    # dA/dt = -k * A
    # dB/dt = k * A
    # dC/dt = 2 * k * A
    # A0 > 0, B0 = C0 = 0
    # Analytical solution:
    # A = A0 * exp(-k * t)
    # B = A0 * (1 - exp(-k * t))
    # C = 2 * A0 * (1 - exp(-k * t))
    init_particle_counts = np.array([10000, 0, 0])
    k = 0.1
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([k * x[0]])
    particle_changes = [np.array([-1, 1, 2])]
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        max_iter=10000)
    
    result = gillespie.solve()

    expected_A = 10000 * np.exp(-k * result.t_seq)
    expected_B = 10000 * (1 - np.exp(-k * result.t_seq))
    expected_C = 2 * 10000 * (1 - np.exp(-k * result.t_seq))

    atol = 500  # 5%

    np.testing.assert_allclose(
        result.particle_counts_seq[:, 0], expected_A, atol=atol)
    np.testing.assert_allclose(
        result.particle_counts_seq[:, 1], expected_B, atol=atol)
    np.testing.assert_allclose(
        result.particle_counts_seq[:, 2], expected_C, atol=atol)
    

@pytest.mark.visual
def test_with_example_reaction_by_plotting():
    import matplotlib.pyplot as plt

    # Reaction:
    # A -> B + 2C
    # dA/dt = -k * A
    # dB/dt = k * A
    # dC/dt = 2 * k * A
    # A0 > 0, B0 = C0 = 0
    # Analytical solution:
    # A = A0 * exp(-k * t)
    # B = A0 * (1 - exp(-k * t))
    # C = 2 * A0 * (1 - exp(-k * t))
    init_particle_counts = np.array([10000, 0, 0])
    k = 0.1
    def rates_fun(x: npt.NDArray[np.int_]) -> npt.NDArray:
        return np.array([k * x[0]])
    particle_changes = [np.array([-1, 1, 2])]
    gillespie = Gillespie(
        init_particle_counts, rates_fun, particle_changes, 
        max_iter=10000)
    
    result = gillespie.solve()

    expected_A = 10000 * np.exp(-k * result.t_seq)
    expected_B = 10000 * (1 - np.exp(-k * result.t_seq))
    expected_C = 2 * 10000 * (1 - np.exp(-k * result.t_seq))

    atol = 500  # 5%

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.plot(result.t_seq, result.particle_counts_seq[:, 0], label='A (sim)', color=colors[0])
    plt.plot(result.t_seq, result.particle_counts_seq[:, 1], label='B (sim)', color=colors[1])
    plt.plot(result.t_seq, result.particle_counts_seq[:, 2], label='C (sim)', color=colors[2])
    plt.plot(result.t_seq, expected_A, label='A (ana)', linestyle='--', color=colors[0])
    plt.plot(result.t_seq, expected_B, label='B (ana)', linestyle='--', color=colors[1])
    plt.plot(result.t_seq, expected_C, label='C (ana)', linestyle='--', color=colors[2])
    plt.plot(result.t_seq, expected_A - atol, 'k--', alpha=0.5, color='lightgray')
    plt.plot(result.t_seq, expected_A + atol, 'k--', alpha=0.5, color='lightgray')
    plt.plot(result.t_seq, expected_B - atol, 'k--', alpha=0.5, color='lightgray')
    plt.plot(result.t_seq, expected_B + atol, 'k--', alpha=0.5, color='lightgray')
    plt.plot(result.t_seq, expected_C - atol, 'k--', alpha=0.5, color='lightgray')
    plt.plot(result.t_seq, expected_C + atol, 'k--', alpha=0.5, color='lightgray')
    plt.title('Test for the Gillespie algorithm: A → B + 2C')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Particle count')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    pytest.main(['-vv', __file__])
    pytest.main(['-vv', __file__ + '::test_with_example_reaction_by_plotting', '-m', 'visual'])
