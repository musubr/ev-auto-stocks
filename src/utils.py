import pandas as pd
import numpy as np
from typing import Iterable
import gc
import multiprocessing
import time
import itertools


def write_df_to_local_directory(path: str, df: pd.DataFrame) -> None:
    df.to_parquet(path)


def calculate_beta(asset_returns: pd.Series, index_returns: pd.Series) -> pd.Series:
    mask_non_null_asset_returns = ~asset_returns.isnull()
    asset_returns = asset_returns.loc[mask_non_null_asset_returns]
    cov_matrix = np.cov(asset_returns, index_returns.loc[mask_non_null_asset_returns])

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


def compute_portfolio_return(returns_df: pd.DataFrame, ticker_to_weight: dict) -> float:
    portfolio_return = 0
    for ticker in ticker_to_weight.keys():
        w = ticker_to_weight[ticker]
        mask_non_null_asset_returns = ~returns_df[f"{ticker}_daily_returns"].isnull()
        mean_daily_return = returns_df[f"{ticker}_daily_returns"].loc[mask_non_null_asset_returns].mean()
        portfolio_return += w * mean_daily_return

    return portfolio_return


def compute_portfolio_variance(returns_df: pd.DataFrame, ticker_to_weight: dict, index_name: str) -> float:
    if any(returns_df.isnull()):
        portfolio_var = 0

        for combo in itertools.combinations_with_replacement(ticker_to_weight.keys(), 2):
            i = combo[0]
            j = combo[1]
            w_i = ticker_to_weight[i]
            w_j = ticker_to_weight[j]

            sel_i = ~returns_df[f"{i}_daily_returns"].isnull()
            sel_j = ~returns_df[f"{j}_daily_returns"].isnull()

            if sel_i.sum() <= sel_j.sum():
                cov_ij = np.cov(returns_df[f"{i}_daily_returns"].loc[sel_i], returns_df[f"{j}_daily_returns"].loc[sel_i])[0, 1]
            else:
                cov_ij = np.cov(returns_df[f"{i}_daily_returns"].loc[sel_j], returns_df[f"{j}_daily_returns"].loc[sel_j])[0, 1]

            portfolio_var += w_i * w_j * cov_ij
    else:
        weights = list(ticker_to_weight.values())
        cov_matrix = returns_df.drop(columns=[f"{index_name}_daily_returns"]).cov()
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))

    return portfolio_var


def run_parallel(func, arg_list, pool_processes):
    """
    Map function over argument list using parallel pool

    Example:

    >>> def double(x): return x * 2
    >>> arg_list = list(range(3))
    >>> output = run_parallel(double, arg_list, pool_processes=2)
    >>> assert output == [0, 2, 4]

    """
    gc.collect()
    t_start = time.time()
    with multiprocessing.get_context("fork").Pool(processes=pool_processes) as _pool:
        results = _pool.map(func, list(arg_list))

    t_end = time.time()
    print("Elapsed time: {:.2f} minutes".format((t_end - t_start) / 60))

    failed = [r is None for r in results]
    if np.any(failed):
        print("Warning: null results returned in {} functional evaluations".format(np.sum(failed)))

    return results

