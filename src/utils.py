# Author: Marcelo Lima Gomes
# Email: marcelolimagomes@gmail.com
# Linkedin: https://www.linkedin.com/in/marcelolimagomes/
# Github: https://github.com/marcelolimagomes
# Date: 2023-11-20
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import glob
import os

data_dir = f"{sys.path[0]}/data"
stocks_dir = f"{data_dir}/stocks"
fiis_dir = f"{data_dir}/fiis"


def get_tickers_ibov():
    df = pd.read_csv(f"{data_dir}/tickers_ibov.csv", sep=";")
    df["ticker"] = df["ticker"] + ".SA"
    return df


def get_tickers_b3():
    df = pd.read_csv(f"{data_dir}/tickers_b3.csv", sep=";")
    df.info()
    df["ticker"] = df["ticker"] + ".SA"
    return df


def get_tickers_fiis():
    df = pd.read_json(f"{data_dir}/tickers_fiis.json")
    df["ticker"] = df["ticker"] + ".SA"
    return df


def get_data_files():
    file_list = glob.glob(os.path.join(f"{stocks_dir}/", "*.SA.csv"))
    return file_list


def download_data(ticker_list: list, start_date: str = None, type='stock'):
    print(f"Downloading data for {len(ticker_list)} tickers...")
    for ticker in ticker_list:
        filename = f"{stocks_dir}/{ticker}.csv"
        max_date = start_date
        file_exists = os.path.exists(filename)
        if file_exists:
            data = pd.read_csv(f"{stocks_dir}/{ticker}.csv", sep=";")
            if data.shape[0] > 0:
                data.info()
                print('data["open_time"].max()', data["open_time"].max())
                max_date = pd.to_datetime(data["open_time"].max(), unit='ms').strftime("%Y-%m-%d")
        else:
            data = pd.DataFrame()

        print(f"Downloading data for {ticker} ==> Max date: {max_date}")
        if max_date is None:
            downloaded_data = yf.download(ticker, auto_adjust=False, threads=20)
        else:
            downloaded_data = yf.download(ticker, start=max_date, auto_adjust=False, threads=20)
        if downloaded_data.shape[0] > 0:
            downloaded_data = downloaded_data.round(2)
            downloaded_data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}, inplace=True,)
            if file_exists:
                downloaded_data['ticker'] = ticker
                downloaded_data['open_time'] = np.int64(downloaded_data.index.values.astype(np.int64) / 1000000)
                downloaded_data['s_open_time'] = downloaded_data.index
            else:
                downloaded_data.insert(0, "ticker", ticker)
                downloaded_data.insert(0, "open_time", np.int64(downloaded_data.index.values.astype(np.int64) / 1000000))
                downloaded_data.insert(0, "s_open_time", downloaded_data.index)

            print(f"Downloaded data for {ticker} ==> Shape: {downloaded_data.shape}")
            data = pd.concat([data, downloaded_data], ignore_index=True)
            data.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
            data['type'] = type
            data.to_csv(filename, index=False, sep=";")
            print(f"Data saved to {filename}")


def calc_ema_periods(df: pd.DataFrame, periods_of_time: any, close_price="close", diff_price=True) -> pd.DataFrame:
    count_occours = df.shape[0]
    try:
        for periods in periods_of_time:
            mme_price = f'ema_{close_price}_{periods}p'
            s_diff_price = mme_price + "_diff"
            if periods > count_occours:
                print(f"calc_ema_periods: Não foi encontrado registros no período informado: {periods}")
                df[mme_price] = None
                if diff_price:
                    df[s_diff_price] = None
            else:
                df[mme_price] = (df[close_price].ewm(span=periods, min_periods=periods).mean())
                df[mme_price] = df[mme_price].astype("float32")
                if diff_price:
                    df[s_diff_price] = ((df[close_price] - df[mme_price]) / df[mme_price]) * 100
                    df[s_diff_price] = df[s_diff_price].astype("float32")
    except Exception as error:
        print(error)

    return df


def calc_RSI(df: pd.DataFrame, close_price="close") -> pd.DataFrame:
    window = 14
    try:
        df["change"] = df[close_price].diff()
        df["gain"] = df.change.mask(df.change < 0, 0.0)
        df["loss"] = -df.change.mask(df.change > 0, -0.0)
        df["avg_gain"] = rma(df.gain.to_numpy(), window)
        df["avg_loss"] = rma(df.loss.to_numpy(), window)

        df["rs"] = df.avg_gain / df.avg_loss
        df["rsi"] = 100 - (100 / (1 + df.rs))
    except Exception as error:
        print(error)
    finally:
        df.drop(columns=["change", "gain", "loss", "avg_gain", "avg_loss", "rs"], inplace=True, errors="ignore")

    return df


def rma(x, n):
    """Running moving average"""
    a = np.full_like(x, np.nan)
    a[n] = x[1: n + 1].mean()
    for i in range(n + 1, len(x)):
        a[i] = (a[i - 1] * (n - 1) + x[i]) / n
    return a
