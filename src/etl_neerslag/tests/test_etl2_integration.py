import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine

from ..etl2 import extract, load

load_dotenv()

DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOSTNAME = os.environ["DB_HOSTNAME"]
DB_NAME = os.environ["DB_NAME"]


def test_extract():
    # Arrange
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOSTNAME}:5432/{DB_NAME}")

    # Act
    df = extract(engine)

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 4
    assert df.columns[0] == "station_code"
    assert df.columns[1] == "date"


def test_load():
    # Arrange
    table_name = "test_neerslag_data"

    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOSTNAME}:5432/{DB_NAME}")

    clean_df = pd.DataFrame(
        {
            "station_code": [453],
            "date": ["2008-01-18"],
            "neerslag_10e_mm": [35],
        }
    )
    # Act
    load(clean_df, engine, table_name)

    # Assert

    result = pd.read_sql_table(
        table_name,
        con=engine,
        schema="customer_analytics",
    )

    pd.testing.assert_frame_equal(result, clean_df)

    # Cleanup
    metadata = MetaData()
    schema_name = "customer_analytics"
    table_to_drop = Table(table_name, metadata, schema=schema_name)

    with engine.begin() as conn:  # Transactional context to ensure safe execution
        table_to_drop.drop(conn, checkfirst=True)
