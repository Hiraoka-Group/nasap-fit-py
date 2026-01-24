from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt
from scipy.constants import Avogadro


class Status(Enum):
    NOT_STARTED = auto()
    RUNNING = auto()
    REACHED_T_MAX = auto()
    REACHED_MAX_ITER = auto()
    TOTAL_RATE_ZERO = auto()


@dataclass
class GillespieResult:
    t_seq: npt.NDArray
    particle_counts_seq: npt.NDArray[np.int_]
    status: Status


class AbortGillespieError(Exception):
    def __init__(self, status: Status) -> None:
        self.status = status


class Gillespie:
    """Class for simulating stochastic chemical kinetics 
    using the Gillespie algorithm.
    """
    def __init__(
            self,
            init_particle_counts: npt.NDArray[np.int_],
            rates_fun: Callable[
                [npt.NDArray[np.int_]], npt.NDArray],
            particle_changes: Sequence[npt.NDArray[np.int_]],
            *,
            volume: float | None = None,
            t_max: float | None = None,
            max_iter: int | None = 1_000_000,
            seed: int | None = None,
            t_init: float = 0.,
            ) -> None:
        if t_max is None and max_iter is None:
            raise ValueError('Either t_max or max_iter must be specified.')
        
        self.rates_fun = rates_fun
        self.particle_changes = np.array(particle_changes)
        if self.particle_changes.shape != (
                len(self.rates_fun(init_particle_counts)),
                len(init_particle_counts)):
            raise ValueError(
                'Invalid shape of particle_changes. '
                'The shape must be (reaction_count, species_count).')
        
        self.volume = volume
        self.t_max = t_max
        self.max_iter = max_iter

        self.rng = np.random.default_rng(seed)

        self.t_seq = [t_init]
        self.particle_counts_seq = [init_particle_counts.copy()]

    @property
    def rates(self) -> npt.NDArray:
        # Each rate represents the average number of reaction occurrences 
        # per minute in the entire volume.

        # [min^-1]
        cur_particle_counts = self.particle_counts_seq[-1]
        return np.array(self.rates_fun(cur_particle_counts))
    
    @property
    def total_rate(self) -> float:
        return sum(self.rates)
    
    def solve(self) -> GillespieResult:
        while True:
            try:
                self._step()
            except AbortGillespieError as e:
                return GillespieResult(
                    np.array(self.t_seq),
                    np.array(self.particle_counts_seq),
                    e.status)
    
    def _step(self) -> None:
        cur_t = self.t_seq[-1]

        if (self.max_iter is not None 
                and len(self.t_seq) - 1 >= self.max_iter):
            raise AbortGillespieError(Status.REACHED_MAX_ITER)

        if self.total_rate == 0:
            raise AbortGillespieError(Status.TOTAL_RATE_ZERO)
        reaction_index = self.determine_reaction()
        time_step = self.determine_time_step()

        if self.t_max is not None and cur_t + time_step > self.t_max:
            raise AbortGillespieError(Status.REACHED_T_MAX)
        
        self.perform_reaction(reaction_index)
        self.t_seq.append(cur_t + time_step)

    def determine_reaction(self) -> int:
        probabilities = self.rates / self.total_rate
        return self.rng.choice(len(self.rates), p=probabilities)

    def determine_time_step(self) -> float:
        return self.rng.exponential(1.0 / self.total_rate)

    def perform_reaction(self, reaction_index: int) -> None:
        cur_particle_counts = self.particle_counts_seq[-1]
        new_particle_counts = (
            cur_particle_counts + self.particle_changes[reaction_index])
        
        # Replace negative particle counts with 0
        new_particle_counts[new_particle_counts < 0] = 0

        self.particle_counts_seq.append(new_particle_counts)

    # ====================
    # Concentration-based
    
    @property
    def concentrations(self) -> npt.NDArray:
        if self.volume is None:
            raise ValueError(
                'Volume must be specified to calculate concentrations. '
                'Consider using calc_concentrations instead.')
        return self.calc_concentrations(self.volume)
    
    @property
    def concentration_rates(self) -> npt.NDArray:
        # Each rate represents the average concentration change per minute.
        if self.volume is None:
            raise ValueError(
                'Volume must be specified to calculate concentration rates. '
                'Consider using calc_concentration_rates instead.')
        return self.calc_concentration_rates(self.volume)

    def calc_concentrations(
            self, volume: float) -> npt.NDArray:
        # mol/L
        cur_particle_counts = self.particle_counts_seq[-1]
        return cur_particle_counts / Avogadro / volume
    
    def calc_concentration_rates(
            self, volume: float) -> npt.NDArray:
        # [mol·L^-1·min^-1] = [min^-1] * [mol/L]
        return self.rates * self.calc_concentrations(volume)

    @classmethod
    def init_based_on_concentrations(
            cls,
            init_concentrations: npt.NDArray,
            conc_rates_fun: Callable[
                [npt.NDArray], npt.NDArray],
            particle_changes: Sequence[npt.NDArray[np.int_]],
            volume: float,
            *,
            t_max: float | None = None,
            seed: int | None = None,
            max_iter: int | None = 1_000_000
            ) -> Gillespie:
        init_particle_counts = init_concentrations * Avogadro * volume

        def particle_rates_fun(
                particle_counts: npt.NDArray[np.int_]
                ) -> npt.NDArray:
            concentrations = particle_counts / Avogadro / volume
            return conc_rates_fun(concentrations)
        
        return cls(
            init_particle_counts, particle_rates_fun,
            particle_changes, volume=volume, 
            t_max=t_max, seed=seed, max_iter=max_iter)
