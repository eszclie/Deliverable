import numpy as np
import pandas as pd

from .. import cyclical_encoding_whole_table as cewt


def test_cyclical_encode():
    # Mock data
    data = pd.DataFrame({"day_of_week": [0, 1, 2, 3, 4, 5, 6]})
    max_val = 7

    # Expected outputs (approximate)
    expected_sin = np.sin(2 * np.pi * np.array([0, 1, 2, 3, 4, 5, 6]) / max_val)
    expected_cos = np.cos(2 * np.pi * np.array([0, 1, 2, 3, 4, 5, 6]) / max_val)

    # Run cyclical encoding using the cyclical_encode function from cewt
    result = cewt.cyclical_encode(data.copy(), "day_of_week", max_val)

    # Check if the sine and cosine values are correct
    assert np.allclose(result["day_of_week_sin"], expected_sin, atol=1e-6)
    assert np.allclose(result["day_of_week_cos"], expected_cos, atol=1e-6)
