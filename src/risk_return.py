import pandas as pd
import numpy as np
import multiprocessing
import time
import traceback
import os
from typing import Iterable
import yfinance as yf
import itertools
from src.utils import get_10y_treasury_yield_data, calculate_beta, run_parallel, write_df_to_local_directory, compute_portfolio_return, compute_portfolio_variance


class RiskReturn:
    def __init__(self, curr_portfolio_df, index_name: str, index_ticker: str, period: str):
        self.curr_portfolio = curr_portfolio_df
        self.ticker_list = list(curr_portfolio_df["ticker"])
        self.rf = float
        self.stock_to_beta = pd.DataFrame()
        self.df = pd.DataFrame()
        self.index_name = index_name
        self.index_ticker = index_ticker
        self.period = period

    def _calculate_expected_returns_capm(self, ticker: str) -> pd.DataFrame:
        beta = self.stock_to_beta.loc[ticker]["beta"]
        rm = self.df.set_index("date")[f"{self.index_name}_daily_returns"]
        er = beta * (rm - self.rf) + self.rf

        return er.rename(f"{ticker}_exp_daily_return")

    def _compute_avg_rf(self):
        self.rf = get_10y_treasury_yield_data(self.df, common_path + "10-year-treasury-yield.csv")["yield"] / 365

    def _compile_daily_returns(self) -> None:
        df = yf.Ticker(self.index_ticker).history(period=self.period)["Close"].reset_index().rename(columns={"Date": "date", "Close": f"{self.index_name}_close"})
        df[f"{self.index_name}_daily_returns"] = np.log(df[f"{self.index_name}_close"] / df[f"{self.index_name}_close"].shift())

        for ticker in self.ticker_list:
            t = yf.Ticker(ticker)
            ticker_df = pd.DataFrame(t.history(period=self.period)["Close"].rename(f"{ticker}_close"))
            ticker_df[f"{ticker}_daily_returns"] = np.log(ticker_df / ticker_df.shift())
            df = df.join(ticker_df, on="date")

        self.df = df

    def _add_expected_returns_capm(self) -> None:
        er_list = []

        for ticker in self.ticker_list:
            er_ticker = self._calculate_expected_returns_capm(ticker)
            er_list += [er_ticker]

        self.df = self.df.join(pd.concat(er_list, axis=1), on="date")

        self.df = self.df.drop(index=self.df.index[0], axis=0)

    def _compute_stock_to_beta(self):
        stock_to_beta = {"ticker": [], "beta": []}
        sp500_returns = self.df[f"{self.index_name}_daily_returns"]

        for ticker in self.ticker_list:
            stock_to_beta["beta"] += [calculate_beta(self.df[f"{ticker}_daily_returns"], sp500_returns)]
            stock_to_beta["ticker"] += [ticker]

        self.stock_to_beta = pd.DataFrame(stock_to_beta).set_index("ticker")

    def preprocess_data(self):
        self._compile_daily_returns()
        self._compute_stock_to_beta()
        self._compute_avg_rf()
        self._add_expected_returns_capm()

    def generate_possible_asset_weights(self, increment: float) -> dict:
        possible_weights = []
        weight_id = 0
        final_weights = {}

        for weights in itertools.combinations_with_replacement(np.arange(0, 1 + increment, increment), len(self.ticker_list)):
            if sum(weights) == 1:
                possible_weights += [weights]

        for weights in possible_weights:
            for w in itertools.permutations(weights):
                final_weights[weight_id] = w
                weight_id += 1

        return final_weights

    def iterate_risk_v_return_for_portfolios(self, final_weight_index: Iterable):
        try:
            portfolio_returns = []
            portfolio_sd = []
            portfolio_sharpe = []
            ticker_weights = []

            weights = final_weights[final_weight_index]
            ticker_to_weight = dict(zip(self.ticker_list, weights))

            p_return = compute_portfolio_return(self.df, ticker_to_weight)
            p_sd = np.sqrt(compute_portfolio_variance(self.df, ticker_to_weight))
            p_sharpe = (p_return - self.rf)/p_sd
            portfolio_returns += [p_return]
            portfolio_sd += [ann_std_multiplier*p_sd]
            portfolio_sharpe += [p_sharpe]
            ticker_weights += [weights]

            risk_return_df = pd.DataFrame({"sigma_p": np.round(portfolio_sd, 4), "er_p": np.round(portfolio_returns, 4)})
            ticker_weights_df = pd.DataFrame.from_records(ticker_weights, columns=self.ticker_list)

            return pd.concat([risk_return_df, ticker_weights_df], axis=1)

        except Exception:
            multiprocessing.get_logger().error(f"Error processing index of weights with key {final_weight_index}: {traceback.format_exc()}")


def main():
    global common_path, ann_std_multiplier, final_weights

    start_time = time.time()
    common_path = "../eda/data/"
    ann_std_multiplier = np.sqrt(252)
    POOL_PROCESSES = os.cpu_count()
    current_portfolio = pd.read_csv(common_path + "portfolio_analysis/portfolio_composition.csv")

    r = RiskReturn(current_portfolio, "sp500", "^GSPC", "1y")
    r.preprocess_data()
    final_weights = r.generate_possible_asset_weights(increment=0.1)
    risk_return_df = run_parallel(r.iterate_risk_v_return_for_portfolios, final_weights.keys(), POOL_PROCESSES)
    risk_return_df = pd.concat(risk_return_df, axis=0, ignore_index=True, sort=False)

    write_df_to_local_directory(common_path+"portfolio_analysis/compiled_df.parquet", r.df)
    write_df_to_local_directory(common_path+"portfolio_analysis/risk_return_df.parquet", risk_return_df)

    print(f"End of risk v return calculations; time taken: {(time.time() - start_time)/60} minutes")

main()
