from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from .rate_constant_resolution import ResolvedReaction, create_rates_fun


class Status(Enum):
    NOT_STARTED = auto()
    RUNNING = auto()
    REACHED_T_MAX = auto()
    REACHED_MAX_ITER = auto()
    TOTAL_RATE_ZERO = auto()


@dataclass
class GillespieCoreResult:
    t_seq: npt.NDArray[np.float64]
    particle_counts_seq: npt.NDArray[np.int_]
    reaction_counts: npt.NDArray[np.int_]
    status: Status


class AbortGillespieCoreError(Exception):
    def __init__(self, status: Status) -> None:
        self.status = status


class GillespieCore:
    """Simulate a reaction network with the GillespieCore algorithm.

    Parameters
    ----------
    reactions : Sequence[ResolvedReaction]
        Reactions with rate constants. Each reaction contributes two event channels: 
        the forward direction and the backward direction.
    species_ids : Sequence[str]
        Species IDs defining the order of particle-count vectors.
    init_particle_counts : Mapping[str, int]
        Initial particle count for some species. Species not listed here will be 
        initialized with 0 particles.
        Units are particle counts, not molar amounts and not concentration.
    t_max : float | None, optional
        Simulation will terminate when the next reaction time step would exceed this value.
        Units are arbitrary but must be consistent with the rate constants.
        t_min, the initial time, is fixed at 0.0.
    max_iter : int | None, optional
        Simulation will terminate when this number of events has been executed.
    seed : int | None, optional
        Seed for the random number generator used for reaction and waiting-time
        sampling.

    Raises
    ------
    ValueError
        If both t_max and max_iter are None, or if reactions reference species IDs 
        not present in species_ids.
    """
    def __init__(
            self,
            reactions: Sequence[ResolvedReaction],
            species_ids: Sequence[str],
            init_particle_counts: Mapping[str, int],
            *,
            t_max: float | None = None,
            max_iter: int | None = 1_000_000,
            seed: int | None = None,
            ) -> None:
        if t_max is None and max_iter is None:
            raise ValueError('Either t_max or max_iter must be specified.')
        self._validate_reaction_species_ids(reactions, species_ids)

        self.reactions = reactions
        self.species_ids = species_ids

        self.rates_fun = create_rates_fun(reactions, species_ids)
        
        self.particle_changes = self._create_particle_changes(reactions, species_ids)
        
        self.t_max = t_max
        self.max_iter = max_iter

        self.rng = np.random.default_rng(seed)

        self.t_seq = np.array([0.0], dtype=np.float64)

        init_counts_array = np.array(
            [init_particle_counts.get(sp_id, 0) for sp_id in species_ids],
            dtype=np.int_
        )
        self.particle_counts_seq = init_counts_array.reshape(1, -1)

        self.reaction_counts = np.zeros(2*len(reactions), dtype=np.int_)

    @staticmethod
    def _create_particle_changes(
        reactions: Sequence[ResolvedReaction],
        species_ids: Sequence[str],
    ) -> npt.NDArray[np.int_]:
        """Create particle-count deltas for each forward and backward reaction.

        Each returned array has the same length and ordering as species_ids.
        A negative value means particles are consumed, while a positive value
        means particles are produced. Given a system of n reversible reactions, 
        the returned sequence has length 2n, where the forward change 
        for each reaction is immediately followed by the corresponding backward change.

        Parameters
        ----------
        reactions : Sequence[ResolvedReaction]
            Reactions to convert into particle-count deltas.
        species_ids : Sequence[str]
            Species IDs defining the order of the returned arrays.

        Returns
        -------
        npt.NDArray[np.int_]
            Two-dimensional integer array with shape
            (2 * len(reactions), len(species_ids)). For each reaction,
            the forward change row is followed by the corresponding
            backward change row.
        """
        particle_changes = []
        species_to_index = {sp_id: i for i, sp_id in enumerate(species_ids)}
        
        for r in reactions:
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
            
            particle_changes.extend([forward_change, backward_change])
        
        return np.array(particle_changes, dtype=np.int_)

    @staticmethod
    def _validate_reaction_species_ids(
        reactions: Sequence[ResolvedReaction],
        species_ids: Sequence[str],
    ) -> None:
        """Validate that all species used in reactions are listed in species_ids.

        Raises
        ------
        ValueError
            If any reactant or product species ID found in reactions is missing from species_ids.
        """
        species_id_set = set(species_ids)
        missing_species_ids: set[str] = set()

        for reaction in reactions:
            reaction_species_ids = (
                reaction.reactant1,
                reaction.reactant2,
                reaction.product1,
                reaction.product2,
            )
            for species_id in reaction_species_ids:
                if species_id is not None and species_id not in species_id_set:
                    missing_species_ids.add(species_id)

        if missing_species_ids:
            missing_list = ', '.join(sorted(missing_species_ids))
            raise ValueError(
                f'reactions contains species that are not in species_ids: {missing_list}'
            )

    @property
    def rates(self) -> npt.NDArray[np.float64]:
        """Return reaction rates computed from the current particle counts.

        Rates are re-evaluated on every access using the latest state,
        self.particle_counts_seq[-1].

        Returns
        -------
        npt.NDArray[np.float64]
            One-dimensional array of nonnegative rates for all event channels
            at the current simulation state.
            For each reaction, the forward rate appears first and the backward rate 
            appears immediately after it. If you use, for example, minute as the time unit, 
            the rates should be in [min^-1].
        """
        cur_particle_counts = self.particle_counts_seq[-1]
        return self.rates_fun(cur_particle_counts)
 
    @property
    def total_rate(self) -> float:
        """Return the sum of all current reaction rates.

        The total rate is the intensity parameter of the exponential waiting
        time distribution used by the Gillespie algorithm.
        """
        return sum(self.rates)
    
    def solve(self) -> GillespieCoreResult:
        """Run the simulation until a termination condition is reached.

        The method repeatedly advances the system by one stochastic event and
        stops only when _step signals a terminal status.

        Returns
        -------
        GillespieCoreResult
            Recorded trajectories, reaction counts, and the terminal status.
        """
        while True:
            try:
                self._step()
            except AbortGillespieCoreError as e:
                return GillespieCoreResult(
                    self.t_seq.copy(),
                    self.particle_counts_seq.copy(),
                    self.reaction_counts.copy(),
                    e.status,
                )
    
    def _step(self) -> None:
        """Advance the simulation by one reaction event.

        The method checks stopping conditions in this order:
        1. maximum number of executed events,
        2. zero total rate,
        3. proposed next event would exceed t_max.

        If none of them apply, it samples the next reaction, samples the waiting
        time, applies the particle-count change, and appends the new time.

        Raises
        ------
        AbortGillespieCoreError
            If a termination condition is reached before executing the next event.
        """
        cur_t = self.t_seq[-1]

        if (self.max_iter is not None 
                and len(self.t_seq) - 1 >= self.max_iter):
            raise AbortGillespieCoreError(Status.REACHED_MAX_ITER)

        rates = self.rates
        total_rate = self.total_rate
        
        if total_rate == 0:
            raise AbortGillespieCoreError(Status.TOTAL_RATE_ZERO)
        reaction_index = self.determine_reaction(rates, total_rate)
        time_step = self.determine_time_step(total_rate)

        if self.t_max is not None and cur_t + time_step > self.t_max:
            raise AbortGillespieCoreError(Status.REACHED_T_MAX)
        
        self.perform_reaction(reaction_index)
        self.t_seq = np.append(self.t_seq, cur_t + time_step)

    def determine_reaction(
        self,
        rates: npt.NDArray[np.float64],
        total_rate: float,
    ) -> int:
        """Sample the index of the next reaction from the current rates.

        Parameters
        ----------
        rates : npt.NDArray[np.float64]
            Rates of all reactions.
        total_rate : float
            Sum of rates.

        Returns
        -------
        int
            Index of the selected reaction.
        """
        probabilities = rates / total_rate
        return self.rng.choice(len(rates), p=probabilities)

    def determine_time_step(self, total_rate: float) -> float:
        """Sample the time until the next reaction event.

        Parameters
        ----------
        total_rate : float
            Current total reaction rate.

        Returns
        -------
        float
            Positive waiting time sampled from an exponential distribution with
            mean 1 / total_rate.
        """
        return self.rng.exponential(1.0 / total_rate)

    def perform_reaction(self, reaction_index: int) -> None:
        """Apply a sampled reaction and record the updated state.

        The method updates the particle counts according to the sampled reaction 
        and reaction counts will be updated accordingly. 
        If the sampled particle change would produce a negative count for some
        species, the resulting count is clipped to 0.

        Parameters
        ----------
        reaction_index : int
            Index of the forward or backward reaction to apply.
        """
        cur_particle_counts = self.particle_counts_seq[-1]
        new_particle_counts = (
            cur_particle_counts + self.particle_changes[reaction_index])
        
        # Replace negative particle counts with 0
        new_particle_counts[new_particle_counts < 0] = 0

        self.particle_counts_seq = np.vstack(
            (self.particle_counts_seq, new_particle_counts))

        self.reaction_counts[reaction_index] += 1
