import numpy as np
import numpy.typing as npt


def concentration_to_existence_ratio(
        concentrations: npt.NDArray,
        conc_of_100pct: npt.NDArray,
        ) -> npt.NDArray:
    """Convert concentrations to existence ratios.

    Parameters
    ----------
    concentrations : npt.NDArray
        Concentrations of species over time. The shape must be 
        (n_time_points, n_species).
    conc_of_100pct : npt.NDArray
        Concentrations of 100%. If 1D, the 100% concentration is the same for
        all time points. If 2D, the 100% concentration can change over time.
        If 1D, the size must be the same as the number of species.
        If 2D, the shape must be the same as concentrations, i.e., 
        (n_time_points, n_species).

    Returns
    -------
    npt.NDArray
        Existence ratios. The shape is the same as concentrations,
        (n_time_points, n_species).
    """
    if conc_of_100pct.ndim == 2:
        if concentrations.shape != conc_of_100pct.shape:
            raise ValueError(
                'concentrations and conc_of_100pct must have the same shape')
        return concentrations / conc_of_100pct * 100

    if conc_of_100pct.ndim == 1:
        if conc_of_100pct.size != concentrations.shape[1]:
            raise ValueError(
                'conc_of_100pct must have the same size as concentrations')
        conc_of_100pct = conc_of_100pct[np.newaxis, :]  # 1D -> 2D
        conc_of_100pct = np.tile(conc_of_100pct, (concentrations.shape[0], 1))
        return concentrations / conc_of_100pct * 100

    raise ValueError('conc_of_100pct must be 1D or 2D')
