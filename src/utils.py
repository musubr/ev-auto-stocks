import pandas as pd
import numpy as np
from typing import Iterable


def write_df_to_local_directory(path: str, df: pd.DataFrame) -> None:
    df.to_parquet(path)


def get_10y_treasury_yield_data(index_df: pd.DataFrame, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={" value": "yield"}).set_index("date")
    df = df.reindex(index_df.set_index("date").index)

    return df.fillna(method="ffill")


def select_sample_for_backtesting(num_stocks_per_sector: dict, tickers_per_sector: dict) -> Iterable:
    sample_tickers = []

    for sector in num_stocks_per_sector.keys():
        tickers_in_sector = tickers_per_sector[sector]
        num_stocks = num_stocks_per_sector[sector]

        random_indices = np.random.randint(0, len(tickers_in_sector), num_stocks, dtype=int)

        sample_tickers += list(tickers_in_sector[random_indices])

    return sample_tickers
