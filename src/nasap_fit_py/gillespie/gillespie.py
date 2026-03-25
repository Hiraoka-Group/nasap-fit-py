from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.constants import Avogadro

from src.nasap_fit_py.models import Reaction

from .gillespie_core import GillespieCore, GillespieCoreResult, Status


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
            reactions: Sequence[Reaction],
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

        init_particle_counts = self._build_init_particle_counts(
            init_concentrations,
            volume,
        )

        self._core = GillespieCore(
            reactions,
            species_ids,
            init_particle_counts,
            t_max=t_max,
            max_iter=max_iter,
            seed=seed,
        )

    @staticmethod
    def _build_init_particle_counts(
            init_concentrations: Mapping[str, float],
            volume: float,
            ) -> dict[str, int]:
        max_int = np.iinfo(np.int_).max
        init_particle_counts: dict[str, int] = {}

        for species_id, concentration in init_concentrations.items():
            particle_count_float = np.rint(concentration * volume * Avogadro)
            if not np.isfinite(particle_count_float):
                raise ValueError(
                    f"Initial particle count for {species_id} is not finite "
                    f"(concentration={concentration}, volume={volume})."
                )
            if particle_count_float < 0:
                # This should not happen because concentrations are validated above,
                # but we guard against numerical artifacts.
                raise ValueError(
                    f"Initial particle count for {species_id} is negative after conversion "
                    f"(count={particle_count_float}, concentration={concentration}, "
                    f"volume={volume})."
                )
            if particle_count_float > max_int:
                raise ValueError(
                    f"Initial particle count for {species_id}={particle_count_float:.3e} "
                    f"exceeds the maximum supported integer {max_int} for NumPy int_ "
                    f"(concentration={concentration}, volume={volume})."
                )
            init_particle_counts[species_id] = int(particle_count_float)

        return init_particle_counts

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
