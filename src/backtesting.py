import pandas as pd
import numpy as np
import datetime as dt
from typing import Iterable, Union


class Backtesting:
    def __init__(
            self,
            initial_cash_balance: int,
            ten_yr_yield: pd.Series,
            stock_to_beta_df: pd.DataFrame,
            index_prices: pd.Series
    ):
        self.cash_balance = initial_cash_balance
        self.bom_balance = initial_cash_balance
        self.ten_yr_yield = ten_yr_yield
        self.stock_to_beta_df = stock_to_beta_df
        self.index_prices= index_prices
        self.is_sufficient_balance = True
    
    def implement_trading_strategy(
            self, compiled_df: pd.DataFrame, is_buy_df: pd.DataFrame, is_sell_df: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.DataFrame]:

        owned_stocks = []
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

        transaction_id = 1
        ticker_to_num_shares_monthly = dict()
        ticker_to_purchase_date = dict()
        ticker_to_purchase_price = dict()

        for row in range(len(is_buy_df)):
            is_buy = is_buy_df.iloc[row]
            tr_date = is_buy["date"]
            is_eom = compiled_df.set_index("date").loc[tr_date]["is_eom"]
            investment_amount = np.round(self.bom_balance / 30, 2)

            if not is_eom:
                tickers_to_buy = set(s[7:] for s in is_buy[is_buy == True].index)
                tickers_to_buy = tickers_to_buy.difference(owned_stocks)
                tickers_to_buy = order_tickers_to_buy_by_rsi(tickers_to_buy, tr_date)

                is_sell = is_sell_df.iloc[row]
                tickers_to_sell = set(s[8:] for s in is_sell[is_sell == True].index)
                tickers_to_sell = list(tickers_to_sell.intersection(owned_stocks))
            else:
                tickers_to_buy = []
                tickers_to_sell = owned_stocks.copy()

            sell_index = 0
            while (len(tickers_to_sell) > 0) and (sell_index < len(tickers_to_sell)):
                ticker = tickers_to_sell[sell_index]
                buy_date = ticker_to_purchase_date[ticker]
                buy_price = ticker_to_purchase_price[ticker]
                days_held = (tr_date - buy_date).days
                if (days_held >= 45) or (tr_date.date() == end_date):
                    action = -1
                    price = get_closing_price(compiled_df, tr_date, ticker)
                    num_shares = ticker_to_num_shares_monthly[ticker]
                    self.cash_balance += num_shares * price
                    transactions_df.loc[transaction_id, :] = [transaction_id, tr_date, ticker, action, price, num_shares, self.cash_balance]

                    beta = stock_to_beta_df.loc[ticker]["beta"]
                    rm = np.log(sp500_prices.loc[tr_date] / self.index_prices.loc[buy_date])
                    rf = self.ten_yr_yield.loc[buy_date: tr_date].mean() / 365 * days_held
                    er = rf + beta * (rm - rf)
                    ri = np.log(price / buy_price)
                    risk_adjusted_ri = ri - er
                    capm_df.loc[transaction_id, :] = np.array(
                        [transaction_id, ticker, buy_date.date(), buy_price, tr_date.date(), price, rm, rf, er, ri, risk_adjusted_ri],
                        dtype=object)

                    ticker_to_num_shares_monthly[ticker] = 0
                    owned_stocks.remove(ticker)
                    transaction_id += 1
                sell_index += 1

            if self.cash_balance > investment_amount:
                self.is_sufficient_balance = True

            buy_index = 0
            while (len(tickers_to_buy) > 0) and (self.is_sufficient_balance) and buy_index < len(tickers_to_buy):
                ticker = tickers_to_buy[buy_index]
                if self.cash_balance < investment_amount:
                    self.is_sufficient_balance = False
                else:
                    action = 1
                    self.cash_balance -= investment_amount
                    price = get_closing_price(compiled_df, tr_date, ticker)
                    num_shares = np.round(investment_amount / price, 2)
                    transactions_df.loc[transaction_id, :] = [transaction_id, tr_date, ticker, action, price, num_shares, self.cash_balance]
                    ticker_to_num_shares_monthly[ticker] = num_shares
                    ticker_to_purchase_date[ticker] = tr_date
                    ticker_to_purchase_price[ticker] = price
                    owned_stocks += [ticker]
                    buy_index += 1
                    transaction_id += 1

            if is_eom:
                self.bom_balance = self.cash_balance

                if self.bom_balance <= 0:
                    break

        return transactions_df.set_index("transaction_id"), capm_df.set_index("sell_transaction_id")
