from io import StringIO

import pandas as pd

from .. import cyclical_encoding_whole_table as cewt


def test_extract_cyclical_features():
    # Mock CSV data as a string
    mock_data = """date,neerslag_10e_mm,total_orders
    2024-09-24,10,150
    2024-09-25,5,200
    2024-09-26,0,180
    """
    # Use StringIO to simulate a file
    mock_csv = StringIO(mock_data)
    mock_df = pd.read_csv(mock_csv)

    # Run the extract_cyclical_features function from cewt
    x, y = cewt.extract_cyclical_features(mock_df)

    # Expected output size checks
    assert x.shape == (3, 5), "The feature array should have 5 columns (2 cyclical pairs and 1 neerslag)."
    assert y.shape == (3,), "The target array should have 3 rows."
