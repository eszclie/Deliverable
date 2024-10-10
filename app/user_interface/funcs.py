import json
import os

import pandas as pd
import requests
import streamlit as st
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# from encoding_new_inputs import extract_features
import predict_orders as po


@st.cache_resource
def bind_socket():
    # Load configuration from environment variables
    load_dotenv()
    
    df_feature_table = pd.read_csv("data/feature_table.csv")
    prediction_table = po.make_prediction_table([["2024-12-31", None]], df_feature_table)
    scoring_uri = os.environ["score"]
    key = os.environ["key"]
    headers = {"Authorization": ("Bearer " + key)}

    return scoring_uri, headers, prediction_table


# def get_response(in_data, h, score):
#     # Display the selected dates and predicted rainfall
#     print("Input: ", in_data)
#     input_encoded = extract_features(in_data)
#     print("Encoded: ", input_encoded)
#     input_dict = {"input_data": input_encoded.tolist()}
#     print("Input_dict: ", input_dict)

#     # Send the POST request
#     response = requests.post(score, json=input_dict, headers=h).text
#     out = json.loads(response)
#     return out


def set_page_confic():
    st.set_page_config(
        page_title="Deliverable App",
        page_icon="app/images/Deliverable_logo.png",
    )


def make_predict(input_dates, h, score, prediction_table):
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
                po.update_lag_features(prediction_table, index, lag_days)
                po.update_ema_features(prediction_table, index, weekly_ema_span, monthly_ema_span)

            # Get the features for the current row
            features = prediction_table.loc[index, feature_columns].values.reshape(1, -1)

            # Predict the total_orders for the current row
            input_dict = {"input_data": features.tolist()}
            predicted_order = json.loads(requests.post(score, json=input_dict, headers=h).text)

            # Update the DataFrame with the predicted value
            prediction_table.at[index, "total_orders"] = predicted_order

            # Update lag and EMA features for the next row based on the predicted value
            if index + 1 < len(prediction_table):  # Ensure we don't go out of bounds
                po.update_lag_features(prediction_table, index + 1, lag_days)
                po.update_ema_features(prediction_table, index + 1, weekly_ema_span, monthly_ema_span)

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
