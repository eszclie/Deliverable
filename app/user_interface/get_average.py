"""
This script contains the function that, when given a date return the rainfall in 10th mm. If its a date that is already in the DB
it returns that rainfall, but if its a date in the future, it returns the average rainfall for that particular month, from 2008-01-18 onwards
Date format is YYYY-MM-DD
"""

import datetime
import functools
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOSTNAME = os.environ["DB_HOSTNAME"]
DB_NAME = os.environ["DB_NAME"]

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOSTNAME}:5432/{DB_NAME}")

neerslag = pd.read_sql_table(
    table_name="neerslagdata",
    con=engine,
    schema="customer_analytics",
)

# Convert the column to datetime object
neerslag["date"] = pd.to_datetime(neerslag["date"])
neerslag["month"] = neerslag["date"].dt.month
averages = neerslag.groupby("month")["neerslag_10e_mm"].mean()


@functools.cache
def get_rainfall(date: datetime.date):
    # Check if the date is already in the data
    result = neerslag[neerslag["date"] == date]

    if not result.empty:
        # Date is already in the database, so return the value from the DB
        return result["neerslag_10e_mm"].values[0]
    else:
        # Date is not yet in the database, so make a prediction, based on the average for that mont
        return averages[date.month]


if __name__ == "__main__":
    input_date = datetime.datetime.now()
    neerslag = get_rainfall(input_date)
    print(neerslag)
