# %%
import os

import numpy as np
import pandas as pd


# %%
def cyclical_encode(data, col, max_val):
    """Takes a dataframe, columname and maximum value in the cycle. It creates the cyclical features for the feature in the specified column."""
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    return data


# %%
def extract_features(df):
    """Takes a pandas dataframe with the order and neerslag data. Extracts and encodes features and the predicted variable. Returns pandas dataframe with the features and a pandas dataframe with the total orders (y)"""

    df = df.iloc[4:].copy()

    df["date"] = pd.to_datetime(df["date"])

    # Add day of the week
    df["day_of_week"] = df["date"].dt.weekday

    # Add month
    df["month"] = df["date"].dt.month

    # perform cyclical encoding
    df = cyclical_encode(df, "day_of_week", 7)
    df = cyclical_encode(df, "month", 12)

    # Create lag features
    df["lag_1"] = df["total_orders"].shift(1)
    df["lag_2"] = df["total_orders"].shift(2)
    df["lag_3"] = df["total_orders"].shift(3)
    df["lag_4"] = df["total_orders"].shift(4)
    df.dropna(inplace=True)

    # Weekly EMA
    df["weekly_ema"] = df["total_orders"].ewm(span=7, adjust=False).mean()

    # Monthly EMA
    df["monthly_ema"] = df["total_orders"].ewm(span=30, adjust=False).mean()

    # Rainfall binary
    df["rainfall_binary"] = (df["neerslag_10e_mm"] > 0).astype(int)

    # Drop redundant columns
    df.drop(["day_of_week", "month"], axis=1, inplace=True)
    df["date"] = df["date"].dt.date

    # split data in x and y
    x = df[
        [
            "date",
            "neerslag_10e_mm",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "weekly_ema",
            "monthly_ema",
            "rainfall_binary",
        ]
    ]
    y = df["total_orders"]

    return (x, y)


# %%
if __name__ == "__main__":
    df = pd.read_csv("data/orders_neerslag.csv")
    x, y = extract_features(df)
    # Ensure the data_ready subfolder exists
    os.makedirs("data/data_ready", exist_ok=True)
    y.to_csv("data/data_ready/y_totalorders.csv", header=False, index=False)
    x.to_csv("data/data_ready/x_features.csv", header=False, index=False)
