import numpy as np
import pandas as pd

import generate_features

def preprocessing():
    filename = "MSFT.csv"
    df = source_to_df(filename)
    df = clean_df(df)
    df = new_features(df)
    # print(df.tail(6)[["date", "open", "high", "low", "close", "volume", "5d_momentum"]])
    filename = "MSFT_clean.csv"
    df_to_csv(df, filename)

def clean_df(df):
    df = missing_values(df)
    # TODO check for outlier
    return df

def missing_values(df):
    """
    Replace missing values with latest available
    """
    missing_values_count = df.isnull().sum()
    print("Original data # missing values: {}".format(missing_values_count))
    df = df.fillna(method = "ffill", axis=0).fillna("0")
    missing_values_count = df.isnull().sum()
    print("After filling na # missing values: {}".format(missing_values_count))
    return df

def new_features(df):
    df = generate_features.trend_features(df)
    df = generate_features.momentum_features(df)
    df = generate_features.volatility_features(df)
    df = generate_features.volume_features(df)
    return df

def source_to_df(filename):
    df = pd.read_csv(filename, header=0)
    df.drop(labels="Close", axis=1, inplace=True)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    return df

def df_to_csv(df, filename):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    preprocessing()