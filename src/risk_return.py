import pandas as pd
import numpy as np
import multiprocessing
import time
import traceback
import os
from typing import Iterable
import yfinance as yf
import itertools
from scipy.optimize import minimize
from src.utils import get_10y_treasury_yield_data, calculate_beta, run_parallel, write_df_to_local_directory, compute_portfolio_return, compute_portfolio_variance


class RiskReturn:
    def __init__(self, curr_portfolio_df, index_name: str, index_ticker: str, period: str):
        self.curr_portfolio = curr_portfolio_df
        self.ticker_list = list(curr_portfolio_df["ticker"])
        self.rf = 0.0
        self.stock_to_beta = pd.DataFrame()
        self.close_df = pd.DataFrame()
        self.daily_returns_df = pd.DataFrame()
        self.index_name = index_name
        self.index_ticker = index_ticker
        self.period = period

    def _calculate_expected_returns_capm(self, ticker: str) -> pd.DataFrame:
        beta = self.stock_to_beta.loc[ticker]["beta"]
        rm = self.daily_returns_df[f"{self.index_name}_daily_returns"]
        er = beta * (rm - self.rf) + self.rf

        return er.rename(f"{ticker}_exp_daily_returns")

    def _compute_avg_rf(self):
        self.rf = (get_10y_treasury_yield_data(self.close_df, common_path + "10-year-treasury-yield.csv")["yield"] / 365).mean()

    def _compile_closing_prices(self) -> None:
        df = yf.Ticker(self.index_ticker).history(period=self.period)["Close"].reset_index().rename(
            columns={
                "Date": "date",
                "Close": f"{self.index_name}_close"
            }
        )

        for ticker in self.ticker_list:
            t = yf.Ticker(ticker)
            ticker_df = pd.DataFrame(t.history(period=self.period)["Close"].rename(f"{ticker}_close"))
            df = df.join(ticker_df, on="date")

        self.close_df = df

    def _compile_daily_returns(self) -> None:
        daily_returns = [np.log(self.close_df[f"{self.index_name}_close"] / self.close_df[f"{self.index_name}_close"].shift()).rename(f"{self.index_name}_daily_returns")]

        for ticker in self.ticker_list:
            close = self.close_df[f"{ticker}_close"]
            daily_returns += [np.log(close / close.shift()).rename(f"{ticker}_daily_returns")]

        df = pd.concat(daily_returns, axis=1)
        df = df.drop(index=df.index[0], axis=0)

        self.daily_returns_df = df

    def calculate_expected_returns_capm(self) -> pd.DataFrame:
        er_list = []

        for ticker in self.ticker_list:
            er_ticker = self._calculate_expected_returns_capm(ticker)
            er_list += [er_ticker]

        return pd.concat(er_list, axis=1)

    def _compute_stock_to_beta(self):
        stock_to_beta = {"ticker": [], "beta": []}
        sp500_returns = self.daily_returns_df[f"{self.index_name}_daily_returns"]

        for ticker in self.ticker_list:
            stock_to_beta["beta"] += [calculate_beta(self.daily_returns_df[f"{ticker}_daily_returns"], sp500_returns)]
            stock_to_beta["ticker"] += [ticker]

        self.stock_to_beta = pd.DataFrame(stock_to_beta).set_index("ticker")

    def preprocess_data(self):
        self._compile_closing_prices()
        self._compile_daily_returns()
        self._compute_stock_to_beta()
        self._compute_avg_rf()

    def generate_possible_asset_weights(self, num_portfolios: int) -> dict:
        final_weights = {}

        # for weights in itertools.combinations_with_replacement(np.arange(0.01, 1, increment), len(self.ticker_list)):
        #     if sum(weights) == 1:
        #         possible_weights += [weights]
        #
        # for weights in possible_weights:
        #     for w in itertools.permutations(weights):
        #         if w not in final_weights.values():
        #             final_weights[weight_id] = w
        #         weight_id += 1

        for n in range(num_portfolios):
            weights = np.random.rand(len(self.ticker_list))
            weights = weights/sum(weights)

            final_weights[n] = weights

        return final_weights

    def iterate_risk_v_return_for_portfolios(self, final_weight_index: Iterable):
        try:
            portfolio_returns = []
            portfolio_sd = []
            portfolio_sharpe = []
            ticker_weights = []

            weights = final_weights[final_weight_index]
            ticker_to_weight = dict(zip(self.ticker_list, weights))

            p_return = compute_portfolio_return(self.daily_returns_df, ticker_to_weight)
            p_sd = np.sqrt(compute_portfolio_variance(self.daily_returns_df, ticker_to_weight, self.index_name))
            p_sharpe = (p_return - self.rf)/p_sd
            portfolio_returns += [p_return]
            portfolio_sd += [p_sd]
            portfolio_sharpe += [p_sharpe]
            ticker_weights += [weights]

            risk_return_df = pd.DataFrame({"sigma_p": np.round(portfolio_sd, 6), "er_p": np.round(portfolio_returns, 4), "sharpe_ratio": np.round(portfolio_sharpe, 6)})
            ticker_weights_df = pd.DataFrame.from_records(ticker_weights, columns=self.ticker_list)

            return pd.concat([risk_return_df, ticker_weights_df], axis=1)

        except Exception:
            multiprocessing.get_logger().error(f"Error processing index of weights with key {final_weight_index}: {traceback.format_exc()}")

    def compute_efficient_frontier(self, min_return) -> pd.DataFrame:
        w0 = np.ones(len(self.ticker_list))/len(self.ticker_list)
        bounds = list(zip(np.zeros(len(self.ticker_list)), np.ones(len(self.ticker_list))))

        constraints = ({"type": "eq", "fun": self._check_weights_sum_to_one}, {"type": "eq", "fun": lambda w: self.get_return(w) - min_return})
        opt_sd = minimize(self.get_portfolio_sd, w0, method="SLSQP", bounds=bounds, constraints=constraints)

        return pd.DataFrame({"er_p": [min_return], "opt_sd": [opt_sd["fun"]]})

    @staticmethod
    def _check_weights_sum_to_one(weights):
        return np.sum(weights) - 1

    def get_return(self, weights):
        mean_returns = list(self.daily_returns_df.drop(columns=[f"{self.index_name}_daily_returns"]).mean())
        return np.dot(weights, mean_returns)

    def get_portfolio_sd(self, weights):
        weights = np.array(weights)
        ticker_to_weight = dict(zip(self.ticker_list, weights))
        return np.sqrt(compute_portfolio_variance(self.daily_returns_df, ticker_to_weight, self.index_name))

    def compile_optimal_portfolio_statistics(self, risk_return_df):
        optimal_portfolio = risk_return_df.iloc[risk_return_df["sharpe_ratio"].idxmax()]
        opt_portfolio_stats = optimal_portfolio.to_dict()
        opt_portfolio_stats["avg_daily_rf"] = self.rf
        opt_portfolio_stats["beta_p"] = np.dot(optimal_portfolio.loc[self.ticker_list], self.stock_to_beta.loc[self.ticker_list]["beta"])
        opt_portfolio_stats["treynor_ratio"] = (opt_portfolio_stats["er_p"] - self.rf) / (opt_portfolio_stats["beta_p"])

        for ticker in self.ticker_list:
            opt_portfolio_stats[f"weight_{ticker}"] = opt_portfolio_stats.pop(ticker)

        return pd.DataFrame(opt_portfolio_stats, index=[0]).transpose().rename(columns={0: "stat"})


def main():
    global common_path, final_weights

    start_time = time.time()
    common_path = "../eda/data/"
    POOL_PROCESSES = os.cpu_count()
    current_portfolio = pd.read_csv(common_path + "portfolio_analysis/portfolio_composition.csv")

    r = RiskReturn(current_portfolio, "sp500", "^GSPC", "1y")
    r.preprocess_data()
    expected_returns_capm_df = r.calculate_expected_returns_capm()
    final_weights = r.generate_possible_asset_weights(num_portfolios=5000)
    risk_return_df = run_parallel(r.iterate_risk_v_return_for_portfolios, final_weights.keys(), POOL_PROCESSES)
    risk_return_df = pd.concat(risk_return_df, axis=0, ignore_index=True, sort=False)
    efficient_frontier_df = run_parallel(r.compute_efficient_frontier, np.linspace(risk_return_df["er_p"].min(), risk_return_df["er_p"].max(), 150), POOL_PROCESSES)
    efficient_frontier_df = pd.concat(efficient_frontier_df, axis=0, ignore_index=True, sort=False)
    optimal_portfolio_df = r.compile_optimal_portfolio_statistics(risk_return_df)

    write_df_to_local_directory(common_path+"portfolio_analysis/daily_returns.parquet", r.daily_returns_df)
    write_df_to_local_directory(common_path+"portfolio_analysis/stock_to_beta.parquet", r.stock_to_beta)
    write_df_to_local_directory(common_path+"portfolio_analysis/capm.parquet", expected_returns_capm_df)
    write_df_to_local_directory(common_path+"portfolio_analysis/risk_return.parquet", risk_return_df)
    write_df_to_local_directory(common_path+"portfolio_analysis/optimal_portfolio.parquet", optimal_portfolio_df)
    write_df_to_local_directory(common_path+"portfolio_analysis/eff_frontier.parquet", efficient_frontier_df)

    print(f"End of risk v return calculations; time taken: {(time.time() - start_time)/60} minutes")


main()
