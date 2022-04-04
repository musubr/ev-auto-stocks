import pandas as pd


def get_closing_price(compiled_df: pd.DataFrame, tr_date: dt, ticker: str) -> float:
    df = compiled_df.set_index("date")
    return df.loc[tr_date][f"closing_price_{ticker}"]
