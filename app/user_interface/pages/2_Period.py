from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

import get_average as avg
from funcs import bind_socket, make_predict, set_page_confic

# Set the name and icon of the page
set_page_confic()
# Set endpoint credentials
scoring_uri, headers, pred_table = bind_socket()

st.title("Deliverable - Predict the number of orders")
st.write("This app provides a user interface for a prediction model for the orders of Deliverable.")

st.markdown("""
### Prediction for the coming period
""")

# Define the start and end of the slider range
start_date = datetime.now().date()
# end_date = start_date + timedelta(days=90)

option = st.sidebar.selectbox(
    "Specify period", ("One Week", "Two Weeks", "One Month", "Two Months", "More Than Two Months")
)
if option == "Two Weeks":
    end_date = start_date + timedelta(days=14)
elif option == "One Month":
    end_date = start_date + timedelta(days=30)
elif option == "Two Months":
    end_date = start_date + timedelta(days=60)
elif option == "More Than Two Months":
    end_date = datetime(2024, 12, 31).date()
else:  # One week
    end_date = start_date + timedelta(days=7)

# col1, col2, col3 = st.columns([0.1, 7, 0.2])
# Add a date range slider in Streamlit
date_range = st.sidebar.slider(
    "Select a date range",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
    format="YYYY-MM-DD",
)
st.sidebar.markdown("""---""")
# if st.button("Make prediction for a given period"):
selected_dates = [date_range[0] + timedelta(days=i) for i in range((date_range[1] - date_range[0]).days + 1)]

# Create input data with dates and predicted rainfall
if st.sidebar.checkbox("I want to pass my own rainfall", value=False):
    rainfall = st.sidebar.number_input("Give the amount of rainfall in mm", min_value=0.0)
    input_data = [[str(date), rainfall * 10] for date in selected_dates]

else:
    input_data = [[str(date), avg.get_rainfall(date=date)] for date in selected_dates]

# Option to display as a graph or a table
option = st.selectbox("Output choices", ("Graph", "Table"))

out = make_predict(input_data, headers, scoring_uri, pred_table)
output_list = list(map(int, out))

if option == "Table":
    # Create a table
    df = pd.DataFrame(columns=["Date", "Weekday", "Predicted number of orders"])
    df["Date"] = selected_dates
    df["Weekday"] = [dt.strftime("%A") for dt in selected_dates]
    df["Predicted number of orders"] = output_list
    st.dataframe(
        df,
        column_config={
            "Date": st.column_config.Column(width="medium"),
            "Weekday": st.column_config.Column(width="medium"),
            "Predicted number of orders": st.column_config.Column(width="medium"),
        },
    )
else:  # option == "Graph"
    # Create a figure
    fig = px.line(
        x=selected_dates, y=output_list, title="Prediction of orders for a given period", markers=True
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Orders")
    # fig.add_annotation(x=0, y=0, text="~")
    st.plotly_chart(fig)
