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


class Gillespie(GillespieCore):
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
        
        if volume <= 0:
            raise ValueError('volume must be positive.')

        for species_id, concentration in init_concentrations.items():
            if concentration < 0:
                raise ValueError(
                    f'Initial concentration for {species_id} must be non-negative.'
                )

        self.volume = volume

        init_particle_counts = {
            species_id: int(concentration * volume * Avogadro)
            for species_id, concentration in init_concentrations.items()
        }

        super().__init__(
            reactions,
            species_ids,
            init_particle_counts,
            t_max=t_max,
            max_iter=max_iter,
            seed=seed,
        )

    @property
    def concentrations_seq(self) -> npt.NDArray[np.float64]:
        """Return concentration trajectories [mol/L]."""
        return self.particle_counts_seq.astype(np.float64) / (self.volume * Avogadro)

    def solve(self) -> GillespieResult:  # type: ignore[override]
        """Run the simulation and return concentration trajectories."""
        core_result: GillespieCoreResult = super().solve()
        concentrations_seq = self.concentrations_seq

        return GillespieResult(
            core_result.t_seq,
            concentrations_seq,
            core_result.reaction_counts,
            core_result.status,
        )
        