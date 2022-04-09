import pandas as pd
import numpy as np
import datetime as dt
from typing import Iterable, Union


class Backtesting:
    def __init__(
            self,
            initial_cash_balance: int,
            end_date: dt,
            compiled_df: pd.DataFrame,
            ten_yr_yield: pd.Series,
            stock_to_beta_df: pd.DataFrame,
            index_prices: pd.Series
    ):
        self.cash_balance = initial_cash_balance
        self.end_date = end_date
        self.bom_balance = initial_cash_balance
        self.ten_yr_yield = ten_yr_yield
        self.compiled_df = compiled_df
        self.stock_to_beta_df = stock_to_beta_df
        self.index_prices= index_prices
        self.is_sufficient_balance = True
        self.owned_stocks = []
        self.transaction_id = 1
        self.ticker_to_num_shares_monthly = dict()
        self.ticker_to_purchase_date = dict()
        self.ticker_to_purchase_price = dict()

    def get_closing_price(self, tr_date: dt, ticker: str) -> float:
        df = self.compiled_df.set_index("date")
        return df.loc[tr_date][f"closing_price_{ticker}"]

    def get_rsi(self, tr_date: dt, ticker: str) -> float:
        df = self.compiled_df.set_index("date")
        return df.loc[tr_date][f"rsi_{ticker}"]

    def order_tickers_to_buy(self, tickers_to_buy: Iterable, tr_date: dt, order_by: str = None) -> Iterable:
        if order_by is None:
            return tickers_to_buy
        else:
            ticker_to_ordered_attribute = {"ticker": [], order_by: []}
            for ticker in tickers_to_buy:
                ticker_to_ordered_attribute["ticker"] += [ticker]
                ticker_to_ordered_attribute[order_by] += [getattr(self, f"get_{order_by}")(tr_date, ticker)]

            ticker_to_ordered_attribute = pd.DataFrame(ticker_to_ordered_attribute).sort_values(by=order_by, ascending=False)

            return list(ticker_to_ordered_attribute["ticker"])

    def initialise_output_dfs(self, is_buy_df: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
        start_date = is_buy_df["date"][0]

        transactions_df = pd.DataFrame(
            {
                "transaction_id": 0,
                "date": start_date,
                "stock": None,
                "action": 0,
                "price": 0,
                "num_shares": 0,
                "cash_balance": self.cash_balance
            },
            index=[0]
        )

        capm_df = pd.DataFrame(
            {
                "sell_transaction_id": 0,
                "stock": None,
                "buy_date": None,
                "buy_price": 0,
                "sell_date": None,
                "sell_price": 0,
                "r_m": 0,
                "r_f": 0,
                "E_r": 0,
                "R_i": 0,
                "risk_adjusted_ri": 0
            },
            index=[0]
        )

        return transactions_df, capm_df

    def get_tickers_to_buy(self, is_buy: pd.Series) -> Iterable:
        tickers_to_buy = set(s[7:] for s in is_buy[is_buy == True].index)
        tickers_to_buy = tickers_to_buy.difference(self.owned_stocks)

        return tickers_to_buy

    def get_tickers_to_sell(self, is_sell: pd.Series, tr_date: dt, max_days_held: int = None) -> Iterable:
        tickers_to_sell = set(s[8:] for s in is_sell[is_sell == True].index)
        tickers_to_sell = list(tickers_to_sell.intersection(self.owned_stocks))

        if max_days_held is not None:
            for ticker in self.owned_stocks:
                if ticker not in tickers_to_sell:
                    days_held = (tr_date - self.ticker_to_purchase_date[ticker]).days
                    if days_held > max_days_held:
                        tickers_to_sell += [ticker]

        return tickers_to_sell
    
    def implement_trading_strategy(
            self, is_buy_df: pd.DataFrame, is_sell_df: pd.DataFrame, order_buy_trades_by: str, max_days_held: int = None
    ) -> Union[pd.DataFrame, pd.DataFrame]:

        transactions_df, capm_df = self.initialise_output_dfs(is_buy_df)

        for row in range(len(is_buy_df)):
            is_buy = is_buy_df.iloc[row]
            is_sell = is_sell_df.iloc[row]
            tr_date = is_buy["date"]
            is_eom = self.compiled_df.set_index("date").loc[tr_date]["is_eom"]
            investment_amount = np.round(self.bom_balance / 20, 2)

            if tr_date.date() == self.end_date:
                tickers_to_buy = []
                tickers_to_sell = self.owned_stocks.copy()
            else:
                tickers_to_buy = self.get_tickers_to_buy(is_buy)
                tickers_to_buy = self.order_tickers_to_buy(tickers_to_buy, tr_date, order_buy_trades_by)
                tickers_to_sell = self.get_tickers_to_sell(is_sell, tr_date, max_days_held)

            sell_index = 0
            while (len(tickers_to_sell) > 0) and (sell_index < len(tickers_to_sell)):
                ticker = tickers_to_sell[sell_index]
                buy_date = self.ticker_to_purchase_date[ticker]
                buy_price = self.ticker_to_purchase_price[ticker]
                days_held = (tr_date - buy_date).days
                action = -1
                price = self.get_closing_price(tr_date, ticker)
                num_shares = self.ticker_to_num_shares_monthly[ticker]
                self.cash_balance += num_shares * price
                transactions_df.loc[self.transaction_id, :] = [self.transaction_id, tr_date, ticker, action, price, num_shares, self.cash_balance]

                beta = self.stock_to_beta_df.loc[ticker]["beta"]
                rm = np.log(self.index_prices.loc[tr_date] / self.index_prices.loc[buy_date])
                rf = self.ten_yr_yield.loc[buy_date: tr_date].mean() / (365*100) * days_held
                er = rf + beta * (rm - rf)
                ri = np.log(price / buy_price)
                risk_adjusted_ri = ri - er
                capm_df.loc[self.transaction_id, :] = np.array(
                    [self.transaction_id, ticker, buy_date.date(), buy_price, tr_date.date(), price, rm, rf, er, ri, risk_adjusted_ri],
                    dtype=object)

                self.ticker_to_num_shares_monthly[ticker] = 0
                self.owned_stocks.remove(ticker)
                self.transaction_id += 1

                sell_index += 1

            if self.cash_balance > investment_amount:
                self.is_sufficient_balance = True

            buy_index = 0
            while (len(tickers_to_buy) > 0) and self.is_sufficient_balance and buy_index < len(tickers_to_buy):
                ticker = tickers_to_buy[buy_index]
                if self.cash_balance < investment_amount:
                    self.is_sufficient_balance = False
                else:
                    action = 1
                    self.cash_balance -= investment_amount
                    price = self.get_closing_price(tr_date, ticker)
                    num_shares = np.round(investment_amount / price, 2)
                    transactions_df.loc[self.transaction_id, :] = [self.transaction_id, tr_date, ticker, action, price, num_shares, self.cash_balance]
                    self.ticker_to_num_shares_monthly[ticker] = num_shares
                    self.ticker_to_purchase_date[ticker] = tr_date
                    self.ticker_to_purchase_price[ticker] = price
                    self.owned_stocks += [ticker]
                    buy_index += 1
                    self.transaction_id += 1

            if is_eom:
                self.bom_balance = self.cash_balance

                if self.bom_balance <= 0:
                    break

        return transactions_df.set_index("transaction_id"), capm_df.set_index("sell_transaction_id")
