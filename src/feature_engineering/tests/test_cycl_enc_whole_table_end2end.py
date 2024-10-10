import os

import numpy as np
import pandas as pd

from .. import cyclical_encoding_whole_table as cewt


def test_e2e_cyclical_encoding(tmpdir):
    # Path to save mock output
    output_dir = tmpdir.mkdir("data_ready")

    # Create mock data similar to the CSV
    mock_df = pd.DataFrame(
        {
            "date": [
                "2008-01-18",
                "2008-01-19",
                "2008-01-20",
                "2008-01-21",
                "2008-01-22",
                "2008-01-23",
                "2008-01-24",
                "2008-01-25",
                "2008-01-26",
                "2008-01-27",
                "2008-01-28",
            ],
            "total_orders": [237.0, 171.0, 191.0, 101.0, 154.0, 157.0, 124.0, 174.0, 128.0, 182.0, 64.0],
            "neerslag_10e_mm": [44, 83, 52, 8, 95, 2, 0, 8, 0, 0, 0],
        }
    )

    # Run the extract_cyclical_features function from cewt
    x, y = cewt.extract_cyclical_features(mock_df)

    # Save outputs
    x_path = os.path.join(output_dir, "x_features.csv")
    y_path = os.path.join(output_dir, "y_totalorders.csv")
    np.savetxt(x_path, x, delimiter=",", fmt="%s")
    np.savetxt(y_path, y, delimiter=",", fmt="%s")

    # Check that files were saved
    assert os.path.exists(x_path), "x_features.csv was not saved!"
    assert os.path.exists(y_path), "y_totalorders.csv was not saved!"

    # Load the saved files and check contents
    x_loaded = np.loadtxt(x_path, delimiter=",")
    y_loaded = np.loadtxt(y_path, delimiter=",")

    # Basic sanity checks
    assert x_loaded.shape == x.shape, "Saved x_features does not match expected shape!"
    assert np.allclose(x_loaded, x), "Saved x_features do not match expected values!"

    assert y_loaded.shape == y.shape, "Saved y_totalorders does not match expected shape!"
    assert np.allclose(y_loaded, y), "Saved y_totalorders do not match expected values!"
