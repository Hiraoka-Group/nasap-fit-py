import numpy as np
import pytest

from src.nasap_fit_py.ode_creation import (calc_consumed_count, calc_particle_change,
                                           calc_produced_count)
from src.nasap_fit_py.ode_creation.reaction_class import Reaction


def test_1_to_1():
    # A -> B  (k)
    assems = ['A', 'B']
    reaction = Reaction(
        reactants=['A'], products=['B'], reaction_kind=0, 
        duplicate_count=1)
    np.testing.assert_array_equal(
        calc_consumed_count(assems, [reaction]), [[1, 0]])
    np.testing.assert_array_equal(
        calc_produced_count(assems, [reaction]), [[0, 1]])
    np.testing.assert_array_equal(
        calc_particle_change(assems, [reaction]), [[-1, 1]])


def test_2_to_1():
    # A + A -> B  (k)
    assems = ['A', 'B']
    reaction = Reaction(
        reactants=['A', 'A'], products=['B'], reaction_kind=0, 
        duplicate_count=2)
    np.testing.assert_array_equal(
        calc_consumed_count(assems, [reaction]), [[2, 0]])
    np.testing.assert_array_equal(
        calc_produced_count(assems, [reaction]), [[0, 1]])
    np.testing.assert_array_equal(
        calc_particle_change(assems, [reaction]), [[-2, 1]])


def test_1_to_2():
    # A -> B + B  (k)
    assems = ['A', 'B']
    reaction = Reaction(
        reactants=['A'], products=['B', 'B'], reaction_kind=0,
        duplicate_count=1)
    np.testing.assert_array_equal(
        calc_consumed_count(assems, [reaction]), [[1, 0]])
    np.testing.assert_array_equal(
        calc_produced_count(assems, [reaction]), [[0, 2]])
    np.testing.assert_array_equal(
        calc_particle_change(assems, [reaction]), [[-1, 2]])


def test_2_to_2():
    # A + A -> B + B  (k)
    assems = ['A', 'B']
    reaction = Reaction(
        reactants=['A', 'A'], products=['B', 'B'], reaction_kind=0,
        duplicate_count=2)
    np.testing.assert_array_equal(
        calc_consumed_count(assems, [reaction]), [[2, 0]])
    np.testing.assert_array_equal(
        calc_produced_count(assems, [reaction]), [[0, 2]])
    np.testing.assert_array_equal(
        calc_particle_change(assems, [reaction]), [[-2, 2]])
    

def test_reversible():
    # A -> 2B  (k1)
    # 2B -> A  (k2)
    assems = ['A', 'B']
    reactions = [
        Reaction(
            reactants=['A'], products=['B', 'B'], reaction_kind=0,
            duplicate_count=1),
        Reaction(
            reactants=['B', 'B'], products=['A'], reaction_kind=1,
            duplicate_count=2)
        ]
    np.testing.assert_array_equal(
        calc_consumed_count(assems, reactions), [[1, 0], [0, 2]])
    np.testing.assert_array_equal(
        calc_produced_count(assems, reactions), [[0, 2], [1, 0]])
    np.testing.assert_array_equal(
        calc_particle_change(assems, reactions), [[-1, 2], [1, -2]])


if __name__ == '__main__':
    pytest.main(['-v', __file__])
