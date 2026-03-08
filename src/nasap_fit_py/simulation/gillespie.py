from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from .rate_constant_resolution import ResolvedReaction, create_conc_rates_fun


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
    reaction_counts: npt.NDArray[np.int_]
    status: Status 

class AbortGillespieError(Exception):
    def __init__(self, status: Status) -> None:
        self.status = status

class Gillespie:
    def __init__(
            self,
            resolved_reactions: Sequence[ResolvedReaction],
            species_ids: Sequence[str],
            init_particle_counts: Mapping[str, int],
            volume: float | None = None,
            t_max: float | None = None,
            max_iter: int | None = 1_000_000,
            seed: int | None = None,
            ) -> None:
        if t_max is None and max_iter is None:
            raise ValueError('Either t_max or max_iter must be specified.')
        
        self.resolved_reactions = resolved_reactions
        self.species_ids = species_ids

        self.rates_fun = create_conc_rates_fun(resolved_reactions, species_ids)
        
        # Create particle changes for each reaction (forward and backward)
        self.particle_changes = []
        species_to_index = {sp_id: i for i, sp_id in enumerate(species_ids)}
        
        for r in resolved_reactions:
            # Forward reaction: reactants -> products
            forward_change = np.zeros(len(species_ids), dtype=np.int_)
            
            # Decrease reactants
            forward_change[species_to_index[r.reactant1]] -= 1
            if r.reactant2 is not None:
                forward_change[species_to_index[r.reactant2]] -= 1
            
            # Increase products
            forward_change[species_to_index[r.product1]] += 1
            if r.product2 is not None:
                forward_change[species_to_index[r.product2]] += 1
            
            # Backward reaction: products -> reactants
            backward_change = -forward_change
            
            self.particle_changes.extend([forward_change, backward_change])
        
        self.volume = volume
        self.t_max = t_max
        self.max_iter = max_iter

        self.rng = np.random.default_rng(seed)

        self.t_seq = [0.0]

        init_counts_array = np.array(
            [init_particle_counts.get(sp_id, 0) for sp_id in species_ids],
            dtype=np.int_
        )
        self.particle_counts_seq = [init_counts_array]

        self.reaction_counts = np.zeros(len(self.rates_fun(init_counts_array)), dtype=np.int_)

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
                    self.reaction_counts.copy(),
                    e.status,
                )
    
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

        self.reaction_counts[reaction_index] += 1
