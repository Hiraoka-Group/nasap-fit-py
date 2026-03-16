import numpy as np
import pytest
from scipy.constants import Avogadro

from src.nasap_fit_py.simulation.gillespie import Gillespie, GillespieResult
from src.nasap_fit_py.simulation.gillespie_core import Status
from src.nasap_fit_py.simulation.rate_constant_resolution import \
    ResolvedReaction


def test_init():
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=1.0, rate_constant_b=0.5),
        ResolvedReaction('B', None, 'C', None, rate_constant_f=1.0, rate_constant_b=0.5),
    ]
    species_ids = ['A', 'B', 'C']
    init_concentrations = {'A': 2.0, 'B': 1.0, 'C': 0.5}
    volume = 1e-15

    gillespie = Gillespie(
        reactions,
        species_ids,
        init_concentrations,
        volume,
        t_max=10.0,
        )
    np.testing.assert_allclose(
        gillespie.particle_counts_seq[0], [2.0*volume*Avogadro, 1.0*volume*Avogadro, 0.5*volume*Avogadro])


def test_solve():
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=1.0, rate_constant_b=0.5),
    ]
    species_ids = ['A', 'B']
    init_concentrations = {'A': 2.0, 'B': 1.0}
    volume = 1e-15

    gillespie = Gillespie(
        reactions,
        species_ids,
        init_concentrations,
        volume,
        max_iter=1000,
    )
    result = gillespie.solve()
    assert isinstance(result, GillespieResult)
    assert len(result.concentrations_seq) == len(result.t_seq)
    assert result.concentrations_seq.shape[1] == len(species_ids)
    assert (result.concentrations_seq >= 0.0).all()


def test_internal_particle_counts_are_used_in_simulation():
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=1.0, rate_constant_b=0.0),
    ]
    species_ids = ['A', 'B']
    volume = 1e-15
    init_concentrations = {
        'A': 3 / (volume * Avogadro),
        'B': 0.0,
    }

    gillespie = Gillespie(
        reactions,
        species_ids,
        init_concentrations,
        volume,
        max_iter=2,
    )

    result = gillespie.solve()

    assert result.status == Status.REACHED_MAX_ITER
    assert np.issubdtype(gillespie.particle_counts_seq.dtype, np.integer)
    np.testing.assert_array_equal(
        gillespie.particle_counts_seq[0],
        np.array([3, 0], dtype=np.int_),
    )
    np.testing.assert_array_equal(
        gillespie.particle_counts_seq[1],
        np.array([2, 1], dtype=np.int_),
    )
    np.testing.assert_array_equal(
        gillespie.particle_counts_seq[2],
        np.array([1, 2], dtype=np.int_),
    )
    np.testing.assert_array_equal(
        np.diff(gillespie.particle_counts_seq, axis=0),
        np.array([[-1, 1], [-1, 1]], dtype=np.int_),
    )


def test_init_no_volume():
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=1.0, rate_constant_b=0.5),
    ]
    species_ids = ['A', 'B']
    init_concentrations = {'A': 2.0, 'B': 1.0}
    volume = None

    with pytest.raises(ValueError) as exc_info:
        Gillespie(
            reactions,
            species_ids,
            init_concentrations,
            volume,
            t_max=10.0,
        )

    assert str(exc_info.value) == 'volume must be specified.'


def test_init_volume_non_positive():
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=1.0, rate_constant_b=0.5),
    ]
    species_ids = ['A', 'B']
    init_concentrations = {'A': 2.0, 'B': 1.0}

    with pytest.raises(ValueError) as exc_info:
        Gillespie(
            reactions,
            species_ids,
            init_concentrations,
            volume=-1,
            t_max=10.0,
        )

    assert str(exc_info.value) == 'volume must be positive.'


def test_init_concentration_negative():
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=1.0, rate_constant_b=0.5),
    ]
    species_ids = ['A', 'B']
    init_concentrations = {'A': -1.0, 'B': 1.0}
    volume = 1e-15

    with pytest.raises(ValueError) as exc_info:
        Gillespie(
            reactions,
            species_ids,
            init_concentrations,
            volume,
            t_max=10.0,
        )

    assert str(exc_info.value) == (
        'Initial concentration for A must be non-negative.'
    )
