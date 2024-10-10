import pandas as pd

from ..etl2 import transform


def test_transform_neerslag():
    input_dict = {
        "station_code": [453, 461, 467],
        "date": ["2008-01-18T00:00:00.000Z", "2008-01-18T00:00:00.000Z", "2008-01-18T00:00:00.000Z"],
        "RD": [35, 35, 35],
        "SX": [45, 46, 46],
    }

    df = pd.DataFrame(input_dict)
    transdf = transform(df)

    # Check of je de juiste columns hebt
    assert len(transdf.columns) == 5
    assert list(transdf.columns) == [
        "date",
        "neerslag_Bergschenhoek",
        "neerslag_Barendrecht",
        "neerslag_Poortugaal",
        "neerslag_10e_mm",
    ]

    # Check of de station code en neerslag in integers zijn
    assert transdf["neerslag_Bergschenhoek"].dtype == "int64"
    assert transdf["neerslag_Barendrecht"].dtype == "int64"
    assert transdf["neerslag_Poortugaal"].dtype == "int64"
    assert transdf["neerslag_10e_mm"].dtype == "float64"
