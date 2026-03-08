import numpy as np
import pytest

from src.nasap_fit_py.simulation.gillespie import (Gillespie, GillespieResult,
                                                   Status)
from src.nasap_fit_py.simulation.rate_constant_resolution import \
    ResolvedReaction


def test_init_unimolecular():
    # A <-> B
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}
    gillespie = Gillespie(
        resolved_reactions,
        species_ids,
        init_particle_counts,
        volume=1.0,
        t_max=10.0
    )

    np.testing.assert_allclose(
        gillespie.particle_counts_seq[0], [100, 200])
    np.testing.assert_allclose(
        gillespie.particle_changes, [[-1, 1], [1, -1]])
    assert len(gillespie.reaction_counts) == 2
    np.testing.assert_array_equal(
        gillespie.reaction_counts, [0, 0])
    assert gillespie.volume == 1.0
    assert gillespie.t_max == 10.0
    assert gillespie.max_iter == 1_000_000
    assert gillespie.t_seq[0] == 0.0


def test_init_bimolecular():
    resolved_reactions = [
        ResolvedReaction('A', 'B', 'C', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B', 'C']  
    init_particle_counts = {'A': 300, 'B': 200, 'C': 50}
    tmax = 10.0

    gillespie = Gillespie(
        resolved_reactions,
        species_ids,
        init_particle_counts,
        t_max=tmax
    )

    np.testing.assert_allclose(
        gillespie.particle_counts_seq[0], [300, 200, 50])
    np.testing.assert_allclose(
        gillespie.particle_changes, [[-1, -1, 1], [1, 1, -1]])
    assert len(gillespie.reaction_counts) == 2
    

def test_init_duplicate_reactions():
    # r1: A <-> B
    # r2: B <-> C
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
        ResolvedReaction('B', None, 'C', None, rate_constant_f=0.2, rate_constant_b=0.1),
    ]
    species_ids = ['A', 'B', 'C']
    init_particle_counts = {'A': 100, 'B': 200, 'C': 50}

    gillespie = Gillespie(
        resolved_reactions,
        species_ids,
        init_particle_counts,
        t_max=10.0
    )

    np.testing.assert_allclose(
        gillespie.particle_counts_seq[0], [100, 200, 50])
    # particle_changes:[r1_f, r1_b, r2_f, r2_b]
    np.testing.assert_allclose(
        gillespie.particle_changes, [[-1, 1, 0], [1, -1, 0], [0, -1, 1], [0, 1, -1]])
    assert len(gillespie.reaction_counts) == 4


def test_init_raises_when_reaction_species_not_in_species_ids():
    resolved_reactions = [
        ResolvedReaction('A', None, 'C', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}

    with pytest.raises(ValueError, match='not in species_ids: C'):
        Gillespie(
            resolved_reactions,
            species_ids,
            init_particle_counts,
            t_max=10.0,
        )

def test_solve():
    # A <-> B
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}
    gillespie = Gillespie(
        resolved_reactions,
        species_ids,
        init_particle_counts,
        t_max=10.0,
        max_iter=10
    )

    result = gillespie.solve()

    assert isinstance(result, GillespieResult)
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
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.8, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 100}
    
    gillespie = Gillespie(
        resolved_reactions,
        species_ids,
        init_particle_counts,
        max_iter=100
    )
    
    result = gillespie.solve()

    assert result.status == Status.REACHED_MAX_ITER
    assert np.sum(result.reaction_counts) == len(result.t_seq) - 1


def test_solve_reaction_rate_affects_count_distribution():
    # A <-> B with significantly different rates
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=1.8, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 300, 'B': 50}
    gillespie = Gillespie(
        resolved_reactions, species_ids, init_particle_counts, t_max=10.0,
        max_iter=100, seed=42) 
    
    result = gillespie.solve()

    assert result.reaction_counts[0] >= result.reaction_counts[1]
        # Higher-rate reaction should occur more frequently
    

def test_status_of_max_iter_reached():
    # A <-> B
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}
    gillespie = Gillespie(
        resolved_reactions,
        species_ids,
        init_particle_counts,
        t_max=10.0,
        max_iter=1
    )

    result = gillespie.solve()
    
    assert result.status == Status.REACHED_MAX_ITER


def test_status_of_total_rate_zero():
    # A -> B (only one direction)

    # Few particles, so the total rate becomes zero
    # before the max iteration is reached.
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.0),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 10, 'B': 0}
    gillespie = Gillespie(
        resolved_reactions,
        species_ids,
        init_particle_counts,
        t_max=100.0,
        max_iter=100
    )
    
    # A becomes zero only in 10 steps,
    # which makes the total rate zero.
    
    result = gillespie.solve()

    assert result.status == Status.TOTAL_RATE_ZERO


def test_status_of_t_max_reached():
    # A <-> B
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 100, 'B': 200}
    gillespie = Gillespie(
        resolved_reactions,
        species_ids,
        init_particle_counts,
        t_max=0.1,
        max_iter=1000
    )
    
    # Since the reaction is reversible,
    # the number of A and B will never become zero.
    # Therefore, the simulation will stop when t_max is reached.
    
    result = gillespie.solve()

    assert result.status == Status.REACHED_T_MAX


def test_perform_reaction_increments_and_accumulates():
    # A <-> B
    resolved_reactions = [
        ResolvedReaction('A', None, 'B', None, rate_constant_f=0.1, rate_constant_b=0.2),
    ]
    species_ids = ['A', 'B']
    init_particle_counts = {'A': 5, 'B': 5}

    gillespie = Gillespie(
        resolved_reactions, species_ids, init_particle_counts,
        t_max=10.0, max_iter=10)

    # Initial state: both counters should be 0
    np.testing.assert_array_equal(gillespie.reaction_counts, [0, 0])

    # Perform forward reaction: only its counter should increase
    gillespie.perform_reaction(0)
    np.testing.assert_array_equal(gillespie.reaction_counts, [1, 0])

    # Perform reverse reaction: only its counter should increase
    gillespie.perform_reaction(1)
    np.testing.assert_array_equal(gillespie.reaction_counts, [1, 1])

    # Perform reaction 0 twice more
    gillespie.perform_reaction(0)
    gillespie.perform_reaction(0)
    np.testing.assert_array_equal(gillespie.reaction_counts, [3, 1])
