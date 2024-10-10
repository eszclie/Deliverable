import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine

from .. import etl2

load_dotenv()

DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOSTNAME = os.environ["DB_HOSTNAME"]
DB_NAME = os.environ["DB_NAME"]


def test_end2end_neerslag():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOSTNAME}:5432/{DB_NAME}")

    table_name = "test_e2e_neerslag"

    df = etl2.extract(engine, table_name=table_name)
    transdf = etl2.transform(df)
    etl2.load(transdf, engine, table_name=table_name)

    # Assert
    result = pd.read_sql_table(
        table_name,
        con=engine,
        schema="customer_analytics",
    )

    assert not result.empty
    assert list(result.columns) == [
        "date",
        "neerslag_Bergschenhoek",
        "neerslag_Barendrecht",
        "neerslag_Poortugaal",
        "neerslag_10e_mm",
    ]

    # Cleanup
    metadata = MetaData()
    schema_name = "customer_analytics"
    table_to_drop = Table(table_name, metadata, schema=schema_name)

    with engine.begin() as conn:  # Transactional context to ensure safe execution
        table_to_drop.drop(conn, checkfirst=True)
