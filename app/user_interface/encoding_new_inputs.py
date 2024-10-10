# %%

import json
import sys

import numpy as np
import pandas as pd


# %%
def cyclical_encode(data, col, max_val):
    """Takes a dataframe, columname and maximum value in the cycle. It craetes the cyclical features for the feature in the specified column."""
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    return data


# %%
def extract_features(input_list):
    """Takes an input list with the order and neerslag data. Extracts and encodes features. Returns a numpy array with the features."""

    # Create a DataFrame from the list and convert to datetime
    df_x = pd.DataFrame(input_list, columns=["date", "neerslag"])
    df_x["date"] = pd.to_datetime(df_x["date"])

    # Add day of the week
    df_x["day_of_week"] = df_x["date"].dt.weekday

    # Add month
    df_x["month"] = df_x["date"].dt.month

    # perform cyclical encoding
    df_x = cyclical_encode(df_x, "day_of_week", 7)
    df_x = cyclical_encode(df_x, "month", 12)

    # Rainfall binary
    df_x["rainfall_binary"] = (df_x["neerslag"] > 0).astype(int)

    # Drop redundant columns
    df_x.drop(["date", "day_of_week", "month"], axis=1, inplace=True)

    x = df_x.to_numpy()

    return x


# %%
# run in command line as: python src/feature_engineering/encoding_new_inputs.py '[["2024-09-27", 45], ["2024-09-28", 55]]'

if __name__ == "__main__":
    test_inputs = json.loads(sys.argv[1])
    x_encoded = extract_features(test_inputs)
