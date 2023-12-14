import time
import numpy as np
import pandas as pd
import datetime as dt
from python_bitvavo_api.bitvavo import Bitvavo
from pytrends.request import TrendReq

from utils.config import Config
from utils.file_handling import DataHandler


def update_data(symbol=Config().symbol) -> None:
    dh = DataHandler()
    df = dh.load_csv_data(symbol)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df.sort_values("Date", ascending=False, inplace=True)
    new = download_data(start=df["Date"].iloc[2])
    updated = pd.concat([df, new])
    updated.drop_duplicates(["Symbol", "Date"], keep="last", inplace=True)
    updated.sort_values(["Symbol", "Date"], ascending=False, inplace=True)
    dh.write_csv_data(updated, symbol)
    time.sleep(60)
    update_google_trends_data()


def download_data(
    target_symbol: str = Config().symbol,
    start=dt.datetime(2019, 3, 8),
    end=dt.datetime.now(),
    freq: str = "1d",
    limit: int = 1000,
) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq=freq)
    blocks = []
    while len(dates) > 0:
        blocks.append(dates[:limit])
        dates = dates[limit:]
    dfs = []
    for symbol in [target_symbol, "BTC"]:
        print(f"Downloading data for {symbol}")
        D, O, H, L, C, V = get_candlesticks(symbol, blocks, freq, limit)
        if len(D) == 0:
            continue
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(D).date,
                "Open": pd.Series(O, dtype=float),
                "High": pd.Series(H, dtype=float),
                "Low": pd.Series(L, dtype=float),
                "Close": pd.Series(C, dtype=float),
                "Volume": pd.Series(V, dtype=float),
            }
        )
        df.drop_duplicates("Date", keep="last", inplace=True)
        df = impute_missing_rows(df, freq)
        df["Symbol"] = symbol
        df.sort_values("Date", ascending=False, inplace=True)
        dfs.append(df)
    return pd.concat(dfs)


def get_candlesticks(symbol: str, date_blocks: list, freq: str, limit: int) -> tuple:
    D, O, H, L, C, V = [], [], [], [], [], []
    bitvavo = Bitvavo()
    for block in date_blocks:
        candles = bitvavo.candles(
            symbol + "-EUR",
            freq,
            {
                "limit": limit,
                "start": block[0].timestamp() * 1000,
                "end": block[-1].timestamp() * 1000,
            },
        )
        if len(candles) == 0:
            print(f"Found no candles for {symbol} until {block[-1]}.")
            return D, O, H, L, C, V
        for candle in candles:
            D.append(dt.datetime.fromtimestamp(candle[0] / 1000))
            O.append(candle[1])
            H.append(candle[2])
            L.append(candle[3])
            C.append(candle[4])
            V.append(candle[5])
        if bitvavo.getRemainingLimit() < 900:
            print("Stopping download because too few API Calls Remaining.")
            break
    return D, O, H, L, C, V


def impute_missing_rows(df: pd.DataFrame, freq) -> pd.DataFrame:
    expected = pd.date_range(df["Date"].min(), df["Date"].max(), freq=freq).date
    actual = pd.to_datetime(df["Date"].unique()).date
    missing = list(set(expected).difference(set(actual)))
    print(f"Filling in {len(missing)} missing rows")
    imputing_df = pd.DataFrame({"Date": missing})
    for col in df.columns:
        if col != "Date":
            imputing_df[col] = np.nan
    df = pd.concat([df, imputing_df])
    df.sort_values("Date", inplace=True)
    for col in df.columns:
        if col != "Date":
            df[col] = df[col].ffill()
    return df


def update_google_trends_data(file_name: str = "google_trends"):
    dh = DataHandler()
    prev = dh.load_csv_data(file_name)
    keywords = list(prev.columns[1:])
    symbol = Config().symbol
    if symbol not in keywords:
        raise Exception(f"File {file_name} is not suited for symbol {symbol}.")
    pytrends = TrendReq(tz=-60)
    pytrends.build_payload(keywords, timeframe="today 3-m")
    new = pytrends.interest_over_time()
    new.reset_index(names=["Date"], inplace=True)
    new = new[["Date"] + keywords].sort_values("Date", ascending=False)
    new["Date"] = new["Date"].astype(str)
    updated = pd.concat([new, prev[~prev["Date"].isin(new["Date"])]])
    if len(updated) != len(
        pd.date_range(updated["Date"].min(), updated["Date"].max(), freq="D")
    ):
        raise Exception(f"File {file_name} is missing dates.")
    dh.write_csv_data(updated, file_name)
    print("Successfully updated google trends data.")
