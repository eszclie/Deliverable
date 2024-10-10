import numpy as np

from .. import cyclical_encoding_new_inputs as ceni


def test_extract_cyclical_features():
    input_data = [["2024-09-27", 45], ["2024-09-28", 55]]

    # Extract cyclical features using ceni.extract_cyclical_features
    encoded_features = ceni.extract_cyclical_features(input_data)

    # Check that the output is a numpy array
    assert isinstance(encoded_features, np.ndarray)

    # Validate shape: 2 rows, 4 encoded features (2 for day_of_week, 2 for month) + 1 for neerslag
    assert encoded_features.shape == (2, 5)
