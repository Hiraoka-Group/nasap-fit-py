from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.constants import Avogadro

from .gillespie_core import GillespieCore, GillespieCoreResult, Status
from .rate_constant_resolution import ResolvedReaction


@dataclass
class GillespieResult:
    t_seq: npt.NDArray[np.float64]
    concentrations_seq: npt.NDArray[np.float64]
    reaction_counts: npt.NDArray[np.int_]
    status: Status


class Gillespie:
    """Concentration-based wrapper of GillespieCore.

    Initial conditions are provided as concentrations [mol/L], and the
    simulation is executed internally in particle counts.
    """

    def __init__(
            self,
            reactions: Sequence[ResolvedReaction],
            species_ids: Sequence[str],
            init_concentrations: Mapping[str, float],
            volume: float,
            *,
            t_max: float | None = None,
            max_iter: int | None = 1_000_000,
            seed: int | None = None,
            ) -> None:
        
        if volume is None:
            raise ValueError('volume must be specified.')
        if volume <= 0:
            raise ValueError('volume must be positive.')

        for species_id, concentration in init_concentrations.items():
            if concentration < 0:
                raise ValueError(
                    f'Initial concentration for {species_id} must be non-negative.'
                )

        self.volume = volume

        init_particle_counts = {
            species_id: int(np.rint(concentration * volume * Avogadro))
            for species_id, concentration in init_concentrations.items()
        }

        self._core = GillespieCore(
            reactions,
            species_ids,
            init_particle_counts,
            t_max=t_max,
            max_iter=max_iter,
            seed=seed,
        )

    @property
    def t_seq(self) -> npt.NDArray[np.float64]:
        return self._core.t_seq

    @property
    def particle_counts_seq(self) -> npt.NDArray[np.int_]:
        return self._core.particle_counts_seq

    @property
    def reaction_counts(self) -> npt.NDArray[np.int_]:
        return self._core.reaction_counts

    @property
    def rates(self) -> npt.NDArray[np.float64]:
        return self._core.rates

    @property
    def total_rate(self) -> float:
        return self._core.total_rate

    def solve(self) -> GillespieResult:
        """Run the simulation and return concentration trajectories."""
        core_result: GillespieCoreResult = self._core.solve()
        
        concentrations_seq = (
            core_result.particle_counts_seq.astype(np.float64) / (self.volume * Avogadro)
            )

        return GillespieResult(
            core_result.t_seq,
            concentrations_seq,
            core_result.reaction_counts,
            core_result.status,
        )
        