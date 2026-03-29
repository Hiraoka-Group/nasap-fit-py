from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from src.nasap_fit_py.models import Reaction

from .rates_fun_creation import create_rates_fun


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
    reaction_counts: npt.NDArray[np.int_] # shape (num_reactions, 2) with forward and backward counts
    status: Status


class AbortGillespieCoreError(Exception):
    def __init__(self, status: Status) -> None:
        self.status = status


class GillespieCore:
    """Simulate a reaction network with the GillespieCore algorithm.

    Parameters
    ----------
    reactions : Sequence[Reaction]
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
            reactions: Sequence[Reaction],
            species_ids: Sequence[str],
            init_particle_counts: Mapping[str, int],
            *,
            t_max: float | None = None,
            max_iter: int | None = 1_000_000,
            seed: int | None = None,
            ) -> None:
        self._validate_init_args(reactions, species_ids, t_max, max_iter)

        self.reactions = reactions
        self.species_ids = species_ids

        self.rates_fun = create_rates_fun(reactions, species_ids)

        self.particle_changes = self._create_particle_changes(reactions, species_ids)

        self.t_max = t_max
        self.max_iter = max_iter

        self.rng = np.random.default_rng(seed)

        init_counts_array = self._create_init_counts_array(init_particle_counts, species_ids)
        initial_capacity = self._resolve_initial_capacity(max_iter)
        self._initialize_buffers(initial_capacity, len(species_ids), init_counts_array)

        self.t_seq = self._t_buffer[:self._t_len]
        self.particle_counts_seq = self._particle_counts_buffer[:self._particle_counts_len]

        self.reaction_counts = np.zeros((len(reactions), 2), dtype=np.int_)

    @staticmethod
    def _validate_init_args(
        reactions: Sequence[Reaction],
        species_ids: Sequence[str],
        t_max: float | None,
        max_iter: int | None,
    ) -> None:
        if t_max is None and max_iter is None:
            raise ValueError('Either t_max or max_iter must be specified.')
        GillespieCore._validate_reaction_species_ids(reactions, species_ids)

    @staticmethod
    def _create_init_counts_array(
        init_particle_counts: Mapping[str, int],
        species_ids: Sequence[str],
    ) -> npt.NDArray[np.int_]:
        return np.array(
            [init_particle_counts.get(sp_id, 0) for sp_id in species_ids],
            dtype=np.int_,
        )

    @staticmethod
    def _resolve_initial_capacity(max_iter: int | None) -> int:
        initial_capacity = max_iter + 1 if max_iter is not None else 1024
        if initial_capacity < 1:
            return 1
        return initial_capacity

    def _initialize_buffers(
        self,
        initial_capacity: int,
        num_species: int,
        init_counts_array: npt.NDArray[np.int_],
    ) -> None:
        self._t_buffer = np.empty(initial_capacity, dtype=np.float64)
        self._t_buffer[0] = 0.0
        self._t_len = 1

        self._particle_counts_buffer = np.empty(
            (initial_capacity, num_species),
            dtype=np.int_,
        )
        self._particle_counts_buffer[0] = init_counts_array
        self._particle_counts_len = 1

    def _ensure_t_capacity(self, required_len: int) -> None:
        if required_len <= self._t_buffer.shape[0]:
            return

        new_capacity = max(required_len, self._t_buffer.shape[0] * 2)
        new_t_buffer = np.empty(new_capacity, dtype=np.float64)
        new_t_buffer[:self._t_len] = self._t_buffer[:self._t_len]
        self._t_buffer = new_t_buffer

    def _ensure_particle_counts_capacity(self, required_len: int) -> None:
        if required_len <= self._particle_counts_buffer.shape[0]:
            return

        new_capacity = max(required_len, self._particle_counts_buffer.shape[0] * 2)
        new_particle_counts_buffer = np.empty(
            (new_capacity, self._particle_counts_buffer.shape[1]),
            dtype=np.int_,
        )
        new_particle_counts_buffer[:self._particle_counts_len] = (
            self._particle_counts_buffer[:self._particle_counts_len]
        )
        self._particle_counts_buffer = new_particle_counts_buffer

    def _append_t(self, t: float) -> None:
        next_len = self._t_len + 1
        self._ensure_t_capacity(next_len)
        self._t_buffer[self._t_len] = t
        self._t_len = next_len
        self.t_seq = self._t_buffer[:self._t_len]

    def _append_particle_counts(self, particle_counts: npt.NDArray[np.int_]) -> None:
        next_len = self._particle_counts_len + 1
        self._ensure_particle_counts_capacity(next_len)
        self._particle_counts_buffer[self._particle_counts_len] = particle_counts
        self._particle_counts_len = next_len
        self.particle_counts_seq = self._particle_counts_buffer[:self._particle_counts_len]

    @staticmethod
    def _create_particle_changes(
        reactions: Sequence[Reaction],
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
        reactions : Sequence[Reaction]
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
        reactions: Sequence[Reaction],
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

        for r in reactions:
            reaction_species_ids = (
                r.reactant1,
                r.reactant2,
                r.product1,
                r.product2,
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
    def workable_rates(self) -> npt.NDArray[np.float64]:
        """Return rates that are executable at the current particle state.

        This applies non-negativity constraints to the raw rates by setting to
        zero any reaction channel that would produce negative particle counts.

        Returns
        -------
        npt.NDArray[np.float64]
            One-dimensional rate array for all event channels at the current
            simulation state, after masking out non-workable channels.
        """
<<<<<<< HEAD
        cur_particle_counts = self._particle_counts_buffer[self._particle_counts_len - 1]
        return self.rates_fun(cur_particle_counts)
=======
        cur_particle_counts = self.particle_counts_seq[-1]
        rates = self.rates_fun(cur_particle_counts)
        return self._create_workable_rates(rates, cur_particle_counts)
>>>>>>> negative-particle-counts-reaction-does-not-occur

    @property
    def workable_total_rate(self) -> float:
        """Return the sum of currently workable reaction rates."""
        return float(np.sum(self.workable_rates))

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
        cur_t = self._t_buffer[self._t_len - 1]

        if (self.max_iter is not None
            and self._t_len - 1 >= self.max_iter):
            raise AbortGillespieCoreError(Status.REACHED_MAX_ITER)

        workable_rates = self.workable_rates
        workable_total_rate = self.workable_total_rate

        if workable_total_rate == 0:
            raise AbortGillespieCoreError(Status.TOTAL_RATE_ZERO)
        reaction_index = self.determine_reaction(workable_rates, workable_total_rate)
        time_step = self.determine_time_step(workable_total_rate)

        if self.t_max is not None and cur_t + time_step > self.t_max:
            raise AbortGillespieCoreError(Status.REACHED_T_MAX)

        self.perform_reaction(reaction_index)
        self._append_t(cur_t + time_step)

    def _create_workable_rates(
        self,
        rates: npt.NDArray[np.float64],
        cur_particle_counts: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        """Zero-out rates for reactions that would make particle counts negative."""
        next_counts = cur_particle_counts + self.particle_changes
        workable_mask = np.all(next_counts >= 0, axis=1)

        workable_rates = rates.copy()
        workable_rates[~workable_mask] = 0.0
        return workable_rates

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

        Parameters
        ----------
        reaction_index : int
            Index of the forward or backward reaction to apply.

        Raises
        ------
        ValueError
            If applying the reaction would produce negative particle counts.
        """
        cur_particle_counts = self._particle_counts_buffer[self._particle_counts_len - 1]
        new_particle_counts = (
            cur_particle_counts + self.particle_changes[reaction_index])
        if np.any(new_particle_counts < 0):
            raise ValueError(
                f'Reaction index {reaction_index} would produce negative particle counts.'
            )

        self._append_particle_counts(new_particle_counts)

        reaction_row = reaction_index // 2
        direction_col = reaction_index % 2  # 0: forward, 1: backward
        self.reaction_counts[reaction_row, direction_col] += 1
