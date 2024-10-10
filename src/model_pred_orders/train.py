# %%
import os
import shutil

import mlflow
import pandas as pd
from xgboost import XGBRegressor

MODEL_DIR = "src/model/XGBRegressor"


# %%
def load():
    """Loads the xx and y data"""
    x = pd.read_csv("data/data_ready/x_features.csv", header=None)
    y = pd.read_csv("data/data_ready/y_totalorders.csv", header=None)

    x = x.drop(x.columns[0], axis=1)

    return x, y


# %%
def train(x, y):
    """Train a XQBoost model on the features x and target y."""

    x = x.values
    y = y.values

    model = XGBRegressor(
        colsample_bytree=1.0,
        learning_rate=0.3,
        max_depth=3,
        min_child_weight=1,
        n_estimators=400,
        subsample=0.8,
        alpha=1.00,
        gamma=0.5,
    )

    model.fit(x, y)

    return model


# %%
if __name__ == "__main__":
    x_features, y_orders = load()

    # Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Remove the existing model directory if it exists to allow overwriting
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
    print(f"Existing model directory '{MODEL_DIR}' has been removed to allow overwriting.")

    model = train(x_features, y_orders)

# Save the model directly to the model/ directory without MLflow tracking
mlflow.sklearn.save_model(model, path=MODEL_DIR)

print(f"Model trained and saved to {MODEL_DIR}.")
