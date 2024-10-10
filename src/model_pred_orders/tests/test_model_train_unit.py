import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .. import train_model


def test_train_model():
    """Test the train_model function to ensure it returns a fitted XGBRegressor model."""
    # Define the number of samples (rows) and features (columns)
    num_samples = 100  # Adjust as needed

    # Generate one-hot encoded data for the first 7 features
    x_group1 = np.zeros((num_samples, 7))
    group1_indices = np.random.randint(0, 7, size=num_samples)
    x_group1[np.arange(num_samples), group1_indices] = 1

    # Generate one-hot encoded data for the next 12 features
    x_group2 = np.zeros((num_samples, 12))
    group2_indices = np.random.randint(0, 12, size=num_samples)
    x_group2[np.arange(num_samples), group2_indices] = 1

    # Generate one-hot encoded data for the last 2 features
    x_group3 = np.zeros((num_samples, 2))
    group3_indices = np.random.randint(0, 2, size=num_samples)
    x_group3[np.arange(num_samples), group3_indices] = 1

    # Generate mock continuous data for the 22nd neerslag feature (ranging from 0 to 1500)
    x_continuous = np.random.uniform(0, 1500, size=(num_samples, 1))

    # Combine all groups and continuous data into a single DataFrame
    x = pd.DataFrame(
        np.hstack([x_group1, x_group2, x_group3, x_continuous]),
        columns=[f"feature{i}" for i in range(1, 23)],  # 22 features in total
    )

    # Generate mock target total orders variable (y) with matching number of rows as a Series
    y = pd.Series(np.random.randint(1000, 2000, size=num_samples), name="target")

    # Train the model
    model = train_model.train_model(x, y)

    # Assert that the model is an XGBRegressor instance and fitted
    assert isinstance(model, XGBRegressor)

    # Check that the model has been fitted and has feature_importances_
    assert hasattr(model, "feature_importances_")

    # Assert that the feature importances match the number of features
    assert model.feature_importances_ is not None
    assert model.feature_importances_.shape[0] == x.shape[1]
