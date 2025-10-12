from nasap_fit_py.mocks import mock_fit
from pathlib import Path
import pandas as pd
import yaml


def test_mock_fit(tmp_path: Path):
    data_path = tmp_path / 'data.csv'
    reactions_path = tmp_path / 'reactions.csv'
    config_path = tmp_path / 'config.yaml'
    output_dir_path = tmp_path / 'output'

    # Prepare mock input files
    pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}).to_csv(data_path, index=False)
    pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]}).to_csv(reactions_path, index=False)
    with open(config_path, 'w') as f:
        yaml.dump({'param1': 'value1', 'param2': 2}, f)

    # Run mock_fit
    output_paths = mock_fit(
        data_path=data_path,
        reactions_path=reactions_path,
        config_path=config_path,
        output_dir_path=output_dir_path,
        overwrite=True
    )

    # Assertions
    assert output_paths.results.exists()
    assert output_paths.sim.exists()
    assert output_paths.details.exists()

    pd.read_csv(output_paths.results)
    pd.read_csv(output_paths.sim)
    with open(output_paths.details, 'r') as f:
        f.read()
