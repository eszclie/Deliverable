import numpy as np
import pandas as pd

from .. import cyclical_encoding_new_inputs as ceni


def test_cyclical_encode():
    # Create a sample dataframe
    data = pd.DataFrame({"day_of_week": [0, 1, 6]})
    max_val = 7

    # Perform cyclical encoding using ceni.cyclical_encode
    encoded_data = ceni.cyclical_encode(data, "day_of_week", max_val)

    # Check that the sine and cosine columns are added
    assert "day_of_week_sin" in encoded_data.columns
    assert "day_of_week_cos" in encoded_data.columns

    # Validate the encoding
    expected_sin = np.sin(2 * np.pi * data["day_of_week"] / max_val)
    expected_cos = np.cos(2 * np.pi * data["day_of_week"] / max_val)
    np.testing.assert_array_almost_equal(encoded_data["day_of_week_sin"], expected_sin)
    np.testing.assert_array_almost_equal(encoded_data["day_of_week_cos"], expected_cos)
