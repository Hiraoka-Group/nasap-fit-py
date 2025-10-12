import os
from pathlib import Path
import pandas as pd

from dataclasses import dataclass
import yaml


@dataclass
class OutputPaths:
    results: Path
    sim: Path
    details: Path


def mock_fit(
        data_path: os.PathLike | Path,
        reactions_path: os.PathLike | Path,
        config_path: os.PathLike | Path,
        output_dir_path: os.PathLike | Path,
        *,
        overwrite: bool = False,
        ) -> OutputPaths:
    """Mock function for fitting"""
    data_path = Path(data_path)
    reactions_path = Path(reactions_path)
    config_path = Path(config_path)
    output_dir_path = Path(output_dir_path)

    if not data_path.exists():
        raise FileNotFoundError(f'Data path {data_path} does not exist.')
    data = pd.read_csv(data_path)

    if not reactions_path.exists():
        raise FileNotFoundError(f'Reactions path {reactions_path} does not exist.')
    reactions = pd.read_csv(reactions_path)

    if not config_path.exists():
        raise FileNotFoundError(f'Config path {config_path} does not exist.')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_paths = OutputPaths(
        results=output_dir_path / 'results.csv',
        sim=output_dir_path / 'sim.csv',
        details=output_dir_path / 'details.txt',
    )

    dummy_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    dummy_dict = {'dummy': [1, 2, 3]}

    if output_paths.results.exists() and not overwrite:
        raise FileExistsError(f'Results file {output_paths.results} already exists.')
    dummy_df.to_csv(output_paths.results, index=False)

    if output_paths.sim.exists() and not overwrite:
        raise FileExistsError(f'Sim file {output_paths.sim} already exists.')
    dummy_df.to_csv(output_paths.sim, index=False)

    if output_paths.details.exists() and not overwrite:
        raise FileExistsError(f'Details file {output_paths.details} already exists.')
    with open(output_paths.details, 'w') as f:
        yaml.dump(dummy_dict, f)

    return output_paths
