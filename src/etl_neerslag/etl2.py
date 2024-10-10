import os

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect

load_dotenv()

DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOSTNAME = os.environ["DB_HOSTNAME"]
DB_NAME = os.environ["DB_NAME"]


# Extract df


def extract(engine, start=None, end=None, stns=None, table_name="neerslagdata"):
    """ "
    This function extracts the data from the API, using the parameters passed
    """

    # If you dont pass stations, it will take Bergschenhoek (453), Barendrecht (461) and Poortugaal (467) as default
    if stns is None:
        stns = ["453:461:467"]

    # If you do not provide a start date, it will automatically determine what it should be from the database
    # If no table exists yet, it will take January 18, 2008 as the start date, because that’s when the orders begin
    # If there is already something in the database, it will take the maximum date from it.

    if start is None:
        inspector = inspect(engine)

        if not inspector.has_table(table_name, schema="customer_analytics"):
            start = "20080118"
        else:
            maxdate = pd.read_sql_query(
                f"""select MAX(date) from customer_analytics.{table_name} """, con=engine
            )
            strdate = str(maxdate.iat[0, 0])
            start = strdate.replace("-", "")

    # Request the data as a JSON from the API and turn it into a dataframe
    data = {"start": start, "end": end, "stns": stns, "fmt": "json"}

    # URL for rainfall data
    url = "https://www.daggegevens.knmi.nl/klimatologie/monv/reeksen"

    # Define the headers that the response should ignore
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(url, data=data, headers=headers)

    df = pd.DataFrame(response.json())

    # The piece of code below ensures that the row with the start date is not added, as it is already in the database.

    if "strdate" in locals():
        df = df[~df["date"].str.startswith(strdate)]

    return df


def transform(df: pd.DataFrame):
    """
    This function transforms the dataframe into the desired format to be used in the model.
    """
    if not df.empty:
        # # Remove the time from the datetime
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Remove snowcode
        if "SX" in df.columns:
            df = df.drop("SX", axis=1)

        # This dict will be used to make the column names

        station_dict = {
            453: "Bergschenhoek",
            461: "Barendrecht",
            467: "Poortugaal",
        }

        # This dict will store the location:rainfall , with the keys the station code, and as value a list with the rainfalls
        rainfall_dict = {}

        # The keys will be the station codes, which are the ones present in the df
        present_stations = df["station_code"].unique()

        # Make the lists with the rainfalldata per station code, which will be the cols in the new df
        for key in present_stations:
            rainfall_dict[key] = df[df["station_code"] == key]["RD"].to_list()

        # Initialize a new df, and make the columns: Date, rainfall per station, and the average
        new_df = pd.DataFrame()
        new_df["date"] = df[df["station_code"] == 453]["date"]

        # The columns per station
        for key in rainfall_dict.keys():
            new_df[f"neerslag_{station_dict[key]}"] = rainfall_dict[key]

        # The actual column to be used will be the average (.mean), but it is called "neerslag_10e_mm"
        # to remain consistent with the rest of the codebase, which uses this as the input column
        new_df["neerslag_10e_mm"] = new_df[
            [f"neerslag_{station_dict[key]}" for key in rainfall_dict.keys()]
        ].mean(axis=1)
        df = new_df
    return df


def load(df: pd.DataFrame, engine, table_name="neerslagdata"):
    """
    Load the data to the database
    """
    if not df.empty:
        df.to_sql(table_name, con=engine, schema="customer_analytics", if_exists="append", index=False)


if __name__ == "__main__":
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOSTNAME}:5432/{DB_NAME}")

    df = extract(engine)
    transformed_df = transform(df)
    load(transformed_df, engine)
