import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Union
from ta.trend import macd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

from utils.config import Config
from utils.data_classes import MultivariateTimeSeries
from utils.file_handling import DataHandler


def preprocess_data(
    window_size: int = None,
    forecast_horizon: int = None,
    split_type: str = "validation",
) -> MultivariateTimeSeries:
    cfg = Config()
    df = DataHandler().load_csv_data(cfg.symbol)
    df, dates = trim_and_sort_data(df, forecast_horizon or cfg.forecast_horizon)
    df.sort_values(["Symbol", "Date"], inplace=True)
    x, y = process_features(
        df,
        cfg.symbol,
        cfg.features,
        window_size or cfg.window_size,
        forecast_horizon or cfg.forecast_horizon,
        cfg.buy_threshold,
    )
    x_train, y_train, x_test, y_test = split_sets(
        x, y, cfg.train_years, cfg.eval_days, split_type
    )
    x_train, x_test = normalize(x_train, x_test)
    print("Shapes:")
    print(" x:", x_train.shape, x_test.shape)
    print(" y:", y_train.shape, y_test.shape)
    return MultivariateTimeSeries(dates, x_train, y_train, x_test, y_test)


def trim_and_sort_data(df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    n_units = df["Date"].nunique() - df["Date"].nunique() % forecast_horizon
    dates = sorted(list(df["Date"].unique()))[-n_units:]
    return df[df["Date"].isin(dates)], dates


def process_features(
    df: pd.DataFrame,
    symbol: str,
    features: list,
    window_size: int,
    forecast_horizon: int,
    buy_threshold: float,
) -> tuple[np.array]:
    x = []
    sym = df[df["Symbol"] == symbol]
    signs = compute_binary_signs(sym["Close"], buy_threshold)
    T = len(sym) - window_size - forecast_horizon
    for feature_name in features:
        match feature_name:
            case "RawClose":
                feature = sym["Close"]
            case "LogReturn":
                feature = np.log(compute_returns(sym["Close"]))
            case "Sign":
                feature = signs
            case "Volume":
                feature = sym["Volume"]
            case "RPR":  # Relative Price Range
                feature = 2 * (sym["High"] - sym["Low"]) / (sym["High"] + sym["Low"])
            case "RSI":  # Relative Strength Index
                feature = RSIIndicator(sym["Close"]).rsi()
            case "StochasticOscillator":
                feature = StochasticOscillator(
                    sym["High"], sym["Low"], sym["Close"]
                ).stoch_signal()
            case "MACD":  # Moving Average Convergence/Divergence
                feature = macd(sym["Close"])
            case "EMA5":
                feature = sym["Close"].ewm(span=5).mean()
            case "EMA20":
                feature = sym["Close"].ewm(span=20).mean()
            case "BollingerBandHigh":
                feature = BollingerBands(sym["Close"]).bollinger_hband_indicator()
            case "BollingerBandLow":
                feature = BollingerBands(sym["Close"]).bollinger_lband_indicator()
            case "RawCloseBTC":
                feature = df[df["Symbol"] == "BTC"]["Close"]
            case "LogReturnBTC":
                feature = np.log(compute_returns(df[df["Symbol"] == "BTC"]["Close"]))
            case "SignBTC":
                feature = compute_binary_signs(
                    df[df["Symbol"] == "BTC"]["Close"], buy_threshold
                )
            case "GoogleTrends":
                trends = DataHandler().load_csv_data("google_trends")
                trends = trends[trends["Date"].isin(sym["Date"].astype(str))]
                feature = np.array(trends.mean(axis=1, numeric_only=True))
                assert len(feature) == len(sym)
            case _:
                raise Exception(f"Feature {feature_name} is not implemented.")
        x.append(build_lagged_feature(feature, T, window_size))
    x = np.stack(x, 2)  # T x WindowSize x Features
    y = np.stack(
        [signs[t + window_size : t + window_size + forecast_horizon] for t in range(T)],
        0,
    )  # T x ForecastHorizon
    return x, np.int32(y)


def compute_returns(prices: pd.Series) -> np.array:
    return np.array(prices / prices.shift(1).fillna(0.0))


def compute_binary_signs(prices: pd.Series, buy_threshold: float) -> np.array:
    return np.float32(compute_returns(prices) > buy_threshold)


def build_lagged_feature(feature: pd.Series, T: int, window_size: int) -> np.array:
    return np.stack([np.array(feature)[t : t + window_size] for t in range(T)], 0)


def split_sets(
    x: np.array, y: np.array, train_years: int, eval_days: int, split_type: str
) -> None:
    if split_type == "validation":
        x_train, y_train = x[: -eval_days * 2], y[: -eval_days * 2]
        x_test, y_test = (
            x[-eval_days * 2 : -eval_days],
            y[-eval_days * 2 : -eval_days],
        )
    elif split_type == "test":
        x_train, y_train = x[:-eval_days], y[:-eval_days]
        x_test, y_test = x[-eval_days:], y[-eval_days:]
    elif split_type == "forecast":
        x_train, y_train = x, y
        x_test, y_test = np.array([]), np.array([])
    return x_train[-train_years * 365 :], y_train[-train_years * 365 :], x_test, y_test


def normalize(x_train, x_test) -> tuple:
    train, test = [], []
    for feat in range(x_train.shape[2]):
        scaler = MinMaxScaler()
        train.append(
            scaler.fit_transform(x_train[:, :, feat].reshape(-1, 1)).reshape(
                x_train[:, :, feat].shape
            )
        )
        test.append(
            scaler.transform(x_test[:, :, feat].reshape(-1, 1)).reshape(
                x_test[:, :, feat].shape
            )
        )
    return np.stack(train, 2), np.stack(test, 2)
