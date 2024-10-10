import pandas as pd

from .. import train_model


def test_load_data():
    # Act
    x, y = train_model.load_data()

    # Assert
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert not x.empty
    assert not y.empty
    assert len(x.columns) == 5  # 22 if using one hot encoding instead of cyclical vars

    # Check data types
    assert x.values.dtype in [int, float], "All features in x should be numeric."
    assert y.values.dtype in [int, float], "Target variable y should be numeric."

    # Check for missing values
    assert x.isnull().sum().sum() == 0, "DataFrame x should not contain missing values."
    assert y.isnull().sum().sum() == 0, "Series y should not contain missing values."

    # Check that x and y have the same number of rows
    assert len(x) == len(y), "x and y should have the same number of samples."

    # Check that `neerslag_10e_mm` column (1st column) is non-negative and continuous
    assert (x.iloc[:, 0] >= 0).all(), "The `neerslag_10e_mm` column contains negative values."

    # Ensure the target variable y is non-negative
    assert (y >= 0).all().all(), "The target variable y contains negative values."

    # Conditional checks based on the number of features in x
    if len(x.columns) == 22:
        # One-hot encoding check
        assert (
            x.iloc[:, 1:8].sum(axis=1) == 1
        ).all(), "One-hot encoding check for first group (day of week) failed."
        assert (
            x.iloc[:, 8:20].sum(axis=1) == 1
        ).all(), "One-hot encoding check for second group (month) failed."
        assert (
            x.iloc[:, 20:22].sum(axis=1) == 1
        ).all(), "One-hot encoding check for last group (weekend_day) failed."

    elif len(x.columns) == 5:
        # Cyclical encoding check (assuming columns 2-5 contain sine/cosine encoded values for day of the week and month)
        assert (x.iloc[:, 1:5] >= -1).all().all() and (
            x.iloc[:, 1:5] <= 1
        ).all().all(), "Cyclical encoded features should be between -1 and 1."

        # Check that sine and cosine pairs have proper magnitudes
        assert (
            (x.iloc[:, 1] ** 2 + x.iloc[:, 2] ** 2 - 1).abs() < 0.01
        ).all(), "Sine/cosine pair check for day of the week failed."
        assert (
            (x.iloc[:, 3] ** 2 + x.iloc[:, 4] ** 2 - 1).abs() < 0.01
        ).all(), "Sine/cosine pair check for month failed."

    else:
        raise ValueError("Unexpected number of features in x: expected 5 or 22.")
