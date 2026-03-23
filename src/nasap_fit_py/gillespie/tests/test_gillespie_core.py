import numpy as np
import pytest

from src.nasap_fit_py.gillespie.gillespie_core import (GillespieCore,
                                                       GillespieCoreResult,
                                                       Status)
from src.nasap_fit_py.models import ResolvedReaction


def test_init_unimolecular():
    # A <-> B
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}
    gillespie_core = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        t_max=10.0
    )

    np.testing.assert_allclose(
        gillespie_core.particle_counts_seq[0], [100, 200])
    np.testing.assert_allclose(
        gillespie_core.particle_changes, [[-1, 1], [1, -1]])
    assert len(gillespie_core.reaction_counts) == 2
    np.testing.assert_array_equal(
        gillespie_core.reaction_counts, [0, 0])
    assert gillespie_core.t_max == 10.0
    assert gillespie_core.max_iter == 1_000_000
    assert gillespie_core.t_seq[0] == 0.0


def test_init_bimolecular():
    reactions = [
        ResolvedReaction('A', 'B', 'C', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B', 'C']
    init_particle_counts = {'A': 300, 'B': 200, 'C': 50}
    tmax = 10.0

    gillespie_core = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        t_max=tmax
    )

    np.testing.assert_allclose(
        gillespie_core.particle_counts_seq[0], [300, 200, 50])
    np.testing.assert_allclose(
        gillespie_core.particle_changes, [[-1, -1, 1], [1, 1, -1]])
    assert len(gillespie_core.reaction_counts) == 2


def test_init_duplicate_reactions():
    # r1: A <-> B
    # r2: B <-> C
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
        ResolvedReaction('B', None, 'C', None, rate_constant_f=0.2, rate_constant_b=0.1),
    ]
    species_ids = ['A', 'B', 'C']
    init_particle_counts = {'A': 100, 'B': 200, 'C': 50}

    gillespie_core = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        t_max=10.0
    )

    np.testing.assert_allclose(
        gillespie_core.particle_counts_seq[0], [100, 200, 50])
    # particle_changes:[r1_f, r1_b, r2_f, r2_b]
    np.testing.assert_allclose(
        gillespie_core.particle_changes, [[-1, 1, 0], [1, -1, 0], [0, -1, 1], [0, 1, -1]])
    assert len(gillespie_core.reaction_counts) == 4


def test_init_raises_when_reaction_species_not_in_species_ids():
    reactions = [
        ResolvedReaction('A', None, 'C', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}

    with pytest.raises(ValueError) as exc_info:
        GillespieCore(
            reactions,
            species_ids,
            init_particle_counts,
            t_max=10.0,
        )

    assert str(exc_info.value) == (
        'reactions contains species that are not in species_ids: C'
    )


def test_solve():
    # A <-> B
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}
    gillespie_core = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        t_max=10.0,
        max_iter=10
    )

    result = gillespie_core.solve()

    assert isinstance(result, GillespieCoreResult)
    assert result.status in {
        Status.REACHED_T_MAX, Status.REACHED_MAX_ITER,
        Status.TOTAL_RATE_ZERO}
    assert result.t_seq[0] == 0.0
    np.testing.assert_allclose(
        result.particle_counts_seq[0], [100, 200])
    assert len(result.reaction_counts) == 2
    assert result.reaction_counts[0] >= 0
    assert result.reaction_counts[1] >= 0


def test_solve_reaction_counts_sum_equals_time_steps():
    # A <-> B
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.8, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 100}

    gillespie_core = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        max_iter=100
    )

    result = gillespie_core.solve()

    assert result.status == Status.REACHED_MAX_ITER
    assert np.sum(result.reaction_counts) == len(result.t_seq) - 1


def test_solve_reaction_rate_affects_count_distribution():
    # A <-> B with significantly different rates
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=1.8, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 300, 'B': 50}
    gillespie_core = GillespieCore(
        reactions, species_ids, init_particle_counts, t_max=10.0,
        max_iter=100, seed=42)

    result = gillespie_core.solve()

    assert result.reaction_counts[0] >= result.reaction_counts[1]


def test_status_of_max_iter_reached():
    # A <-> B
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}
    gillespie_core = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        t_max=10.0,
        max_iter=1
    )

    result = gillespie_core.solve()

    assert result.status == Status.REACHED_MAX_ITER


def test_status_of_total_rate_zero():
    # A -> B (only one direction)
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.0),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 10, 'B': 0}
    gillespie_core = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        t_max=100.0,
        max_iter=100
    )

    result = gillespie_core.solve()

    assert result.status == Status.TOTAL_RATE_ZERO


def test_status_of_t_max_reached():
    # A <-> B
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}
    gillespie_core = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        t_max=0.1,
        max_iter=1000
    )

    result = gillespie_core.solve()

    assert result.status == Status.REACHED_T_MAX


def test_perform_reaction_increments_and_accumulates():
    # A <-> B
    reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 5, 'B': 5}

    gillespie_core = GillespieCore(
        reactions, species_ids, init_particle_counts,
        t_max=10.0, max_iter=10)

    np.testing.assert_array_equal(gillespie_core.reaction_counts, [0, 0])

    gillespie_core.perform_reaction(0)
    np.testing.assert_array_equal(gillespie_core.reaction_counts, [1, 0])

    gillespie_core.perform_reaction(1)
    np.testing.assert_array_equal(gillespie_core.reaction_counts, [1, 1])

    gillespie_core.perform_reaction(0)
    gillespie_core.perform_reaction(0)
    np.testing.assert_array_equal(gillespie_core.reaction_counts, [3, 1])


def test_with_example_reaction():
    # Reaction:
    # dA/dt = -k * A
    # dB/dt = k * A
    # dC/dt = k * A
    reactions = [
        ResolvedReaction('A', None, 'B', 'C', rate_constant_f=0.1, rate_constant_b=0.0),
    ]
    species_ids = ['A', 'B', 'C']
    init_particle_counts = {'A': 10000}
    gillespie = GillespieCore(
        reactions,
        species_ids,
        init_particle_counts,
        max_iter=1000,
        seed=42,
    )
    result = gillespie.solve()

    expected_a = 10000 * np.exp(-0.1 * result.t_seq)
    expected_b = 10000 * (1 - np.exp(-0.1 * result.t_seq))
    expected_c = 10000 * (1 - np.exp(-0.1 * result.t_seq))

    atol = 500  # 5%

    np.testing.assert_allclose(
        result.particle_counts_seq[:, 0], expected_a, atol=atol)
    np.testing.assert_allclose(
        result.particle_counts_seq[:, 1], expected_b, atol=atol)
    np.testing.assert_allclose(
        result.particle_counts_seq[:, 2], expected_c, atol=atol)
