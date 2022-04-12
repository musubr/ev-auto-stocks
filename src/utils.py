import pandas as pd
import numpy as np
from typing import Iterable


def write_df_to_local_directory(path: str, df: pd.DataFrame) -> None:
    df.to_parquet(path)


def calculate_beta(asset_returns: pd.Series, index_returns: pd.Series) -> pd.Series:
    cov_matrix = np.cov(asset_returns, index_returns)
    return np.round(cov_matrix[0, 1]/np.var(index_returns), 2)


def get_10y_treasury_yield_data(index_df: pd.DataFrame, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={" value": "yield"}).set_index("date")
    df = df.reindex(index_df.set_index("date").index)

    return df.fillna(method="ffill")/100


def select_sample_for_backtesting(num_stocks_per_sector: dict, tickers_per_sector: dict) -> Iterable:
    sample_tickers = []

    for sector in num_stocks_per_sector.keys():
        tickers_in_sector = tickers_per_sector[sector]
        num_stocks = num_stocks_per_sector[sector]

        random_indices = np.random.randint(0, len(tickers_in_sector), num_stocks, dtype=int)

        sample_tickers += list(tickers_in_sector[random_indices])

    return sample_tickers
