import os
import sys

import mlflow
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), "src"))

import get_average as rainfall_funcs
import encoding_new_inputs as enc

# Fixed paths
X_PATH = "data/data_ready/x_features.csv"
Y_PATH = "data/data_ready/y_totalorders.csv"
MODEL_DIR = "src/model/XGBRegressor"


# Helper functions
def update_lag_features(df, index, lag_days):
    """Updates lag features for the specified row, calculating lag values (previous 'total_orders')"""

    # Only update lag features for the row at the specified index
    if index > 0:  # Avoid out-of-bounds errors
        df.at[index, "lag_1"] = df.at[index - 1, "total_orders"]
        for lag in range(2, lag_days + 1):
            if index - lag >= 0:
                df.at[index, f"lag_{lag}"] = df.at[index - lag, "total_orders"]


def update_ema_features(df, index, weekly_span, monthly_span):
    """Updates exponentially weighted moving average (EMA) features for the specified row, using the total orders to calculate both weekly and monthly EMA."""

    # Only update EMA features for the row at the specified index
    if index > 0:  # Ensure we don't calculate EMA for the first row
        df.at[index, "weekly_ema"] = df["total_orders"].ewm(span=weekly_span, adjust=False).mean().iloc[index]
        df.at[index, "monthly_ema"] = (
            df["total_orders"].ewm(span=monthly_span, adjust=False).mean().iloc[index]
        )


def load_data_model(xpath, ypath, modelpath):
    "Takes three paths, one for the x data, one for y, and one for the model. Loads the dataset and the model."
    x = pd.read_csv(
        xpath,
        header=None,
        index_col=False,
        names=[
            "date",
            "rainfall",
            "cyc1",
            "cyc2",
            "cyc3",
            "cyc4",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "weekly_ema",
            "monthly_ema",
            "rainfall_binary",
        ],
    )
    y = pd.read_csv(ypath, header=None, names=["total_orders"])

    xy = pd.concat([x, y], axis=1)

    xy.sort_values("date", inplace=True)

    model = mlflow.sklearn.load_model(modelpath)

    return xy, model


def make_prediction_table(input_dates, dataset):
    """Takes input dates and the existing dataset. Appends the to be predicted input rows to the existing dataset."""

    # extract the maximum date in the input
    latest_entry = max(input_dates, key=lambda x: x[0])

    # create a data range from the last known datapoint (2024-08-15) up and untill the maximum date in the inputs
    date_range = pd.date_range(start="2024-08-16", end=latest_entry[0])

    # Create the dataframe with 'date' and 'rainfall' columns
    df = pd.DataFrame(
        {
            "date": date_range,
            "rainfall": 0,  # Set rainfall to 0 for every date
        }
    )

    # Apply the get_rainfall function to each date to calculate expected rainfall
    df["rainfall"] = df["date"].apply(lambda x: rainfall_funcs.get_rainfall(x.date()))

    # Insert the given rainfall for those dates where a rainfall was given
    # First ensure that date is of type datetime
    df["date"] = pd.to_datetime(df["date"])

    # Loop through the nested list and update rainfall values
    for i in input_dates:
        input_date = pd.to_datetime(i[0])  # Convert date to datetime
        input_rainfall = i[1]

        if input_rainfall is not None:
            df.loc[df["date"] == input_date, "rainfall"] = input_rainfall

    # Extract the (non order related) features for the new predictions
    # make nested list from dataframe
    df_list = df.values.tolist()

    # create the features for the observations
    features = enc.extract_features(df_list)
    df_features = pd.DataFrame(features)
    df_features.drop(
        df_features.columns[0], axis=1, inplace=True
    )  # Drop rainfall_binary as this feature is already present in df
    df_features.columns = ["cyc1", "cyc2", "cyc3", "cyc4", "rainfall_binary"]

    # concat the inputs with the features
    df_2024 = pd.concat([df, df_features], axis=1)

    # concat the features to the existing data
    df_all = pd.concat([dataset, df_2024], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date

    return df_all


def make_predictions(prediction_table, model, input_dates):
    """Takes the prediction table with the dates to be predicted and a model. It returns the predicted total orders for the input dates as a single array"""

    # Define parameters for lag and EMA features
    lag_days = 4  # Number of lag features (lag_1 to lag_4)
    weekly_ema_span = 7  # Weekly EMA span
    monthly_ema_span = 30  # Monthly EMA span

    # Define the feature columns the model expects
    feature_columns = [
        "rainfall",
        "cyc1",
        "cyc2",
        "cyc3",
        "cyc4",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "weekly_ema",
        "monthly_ema",
        "rainfall_binary",
    ]

    # Find the index of the last valid row with lag and EMA features
    last_valid_index = prediction_table[prediction_table["lag_1"].notna()].index[-1]

    # Loop through each row for predictions
    for index in range(last_valid_index + 1, len(prediction_table)):
        # Only predict for rows where total_orders is NaN
        if pd.isna(prediction_table.at[index, "total_orders"]):
            # For the first row with NaN, use the previous valid row to calculate lag and EMA
            if index == last_valid_index + 1:
                update_lag_features(prediction_table, index, lag_days)
                update_ema_features(prediction_table, index, weekly_ema_span, monthly_ema_span)

            # Get the features for the current row
            features = prediction_table.loc[index, feature_columns].values.reshape(1, -1)

            # Predict the total_orders for the current row
            predicted_order = model.predict(features)[0]

            # Update the DataFrame with the predicted value
            prediction_table.at[index, "total_orders"] = predicted_order

            # Update lag and EMA features for the next row based on the predicted value
            if index + 1 < len(prediction_table):  # Ensure we don't go out of bounds
                update_lag_features(prediction_table, index + 1, lag_days)
                update_ema_features(prediction_table, index + 1, weekly_ema_span, monthly_ema_span)

    # Get the predicted total_orders for the input dates
    # Convert the list into a list of dates
    input_list = [item[0] for item in input_dates]

    # Ensure that both the 'date' column and the input dates are in the same format (datetime)
    prediction_table["date"] = pd.to_datetime(prediction_table["date"])
    input_list = pd.to_datetime(input_list)

    # Filter the dataframe by the dates in input_list
    filtered_prediction_table = prediction_table[prediction_table["date"].isin(input_list)]

    # Extract the total_orders column as a NumPy array
    predicted_total_orders = filtered_prediction_table["total_orders"].values

    return predicted_total_orders


if __name__ == "__main__":
    # Load the dataset and model
    existing_dataset, model = load_data_model(X_PATH, Y_PATH, MODEL_DIR)

    # Make the prediction table given the inputs
    test_inputs = [["2024-09-27", 45], ["2024-09-28", None]]
    prediction_table = make_prediction_table(test_inputs, existing_dataset)

    # Make the predictions
    predicted_orders = make_predictions(prediction_table, model, test_inputs)
    print(predicted_orders)

    # command to run: python src/model_pred_orders/predict_orders.py '[["2024-09-27", 45], ["2024-09-28", 55]]'
