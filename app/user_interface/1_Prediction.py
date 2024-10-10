import streamlit as st

import get_average as avg
from funcs import bind_socket, make_predict, set_page_confic

# Set the name and icon of the page
set_page_confic()
# Set endpoint credentials
scoring_uri, headers, pred_table = bind_socket()
# Parameters for the rainfall slider

st.title("Deliverable - Predict the number of orders")
st.write("This app provides a user interface for a prediction model for the orders of Deliverable.")

date = st.date_input("Choose a date to predict orders for")

st.markdown("""---""")
min1 = 0.0
max1 = 2.0
max2 = 10.0
max3 = 50.0


def add_styling():
    st.html("""
        <style>
            /* convert radio to list of buttons */
            div[role="radiogroup"] {
                flex-direction:row;
            }
            input[type="radio"] + div {
                background: #63ADD2 !important;
                color: #FFF;
                border-radius: 38px !important;
                padding: 8px 18px !important;
            }
            input[type="radio"][tabindex="0"] + div {
                background: #E6FF4D !important;
                color: #17455C !important;
            }
            input[type="radio"][tabindex="0"] + div p {
                color: #17455C !important;
            }
            div[role="radiogroup"] label > div:first-child {
                display: none !important;
            }
            div[role="radiogroup"] label {
                margin-right: 0px !important;
            }
            div[role="radiogroup"] {
                gap: 12px;
            }
        </style>
    """)


if st.checkbox("I want to pass my own rainfall", value=False):
    add_styling()
    col1, col2 = st.columns([0.2, 10])
    if col2.checkbox("Specify a number using an input box", value=False, key="two"):
        rainfall = col2.number_input("Give the amount of rainfall in mm", min_value=0.0)
    else:
        col2.rain_option = col2.radio(
            "Specify the amount of rain?",
            options=["No rain", "Small amount of rain", "Medium amount of rain", "Heavy amount of rain"],
            captions=[
                "   🌤️",
                "   🌧️",
                "   🌧️🌧️",
                "  🌧️🌧️🌧️",
            ],
            horizontal=True,
        )
        if col2.rain_option == "Small amount of rain":
            col2.markdown(" \n")
            rainfall = col2.slider(
                "Choose the amount of rainfall in mm",
                min_value=min1,
                max_value=max1,
            )
        elif col2.rain_option == "Medium amount of rain":
            col2.markdown(" \n")
            rainfall = col2.slider(
                "Choose the amount of rainfall in mm",
                min_value=max1,
                max_value=max2,
            )
        elif col2.rain_option == "Heavy amount of rain":
            col2.markdown(" \n")
            rainfall = col2.slider(
                "Choose the amount of rainfall in mm",
                min_value=max2,
                max_value=max3,
            )
        else:
            rainfall = 0.0
    st.markdown("""---""")
    st.write(
        "For more information about rainfall check out this [link](https://www.buienradar.nl/weer/rotterdam/nl/2747891/14daagse)."
    )
    st.markdown(" \n")
    user_input = [[date, rainfall * 10]]
else:
    avg_rainfall = avg.get_rainfall(date=date)
    user_input = [[date, avg_rainfall]]

# new_input = [["2024-09-27", 45], ["2024-09-28", 55]]

if st.button("Make prediction"):
    st.markdown(" \n")
    try:
        output_list = make_predict(user_input, headers, scoring_uri, pred_table)
        st.write(f"The predicted amount of orders is **{int(output_list[0])}**")
    except Exception:
        st.write(f"**Sorry, we are not yet able to give a prediction for {date}**")

# Command to run streamlit: python -m streamlit run 'app/user_interface/1_Prediction.py'
