import matplotlib
import numpy as np
import pandas as pd

import generate_features

def preprocessing():
    filename = "MSFT.csv"
    df = source_to_df(filename)
    df = generate_features.return_features(df)
    df = clean_df(df)
    # print(df.head(5)[["date", "open", "high", "low", "close", "volume", "return"]])
    df = new_features(df)
    filename = "MSFT_clean.csv"
    df_to_csv(df, filename)

def clean_df(df):
    df = missing_values(df)
    df = outliers(df)
    return df

def missing_values(df):
    """
    Replace missing values with latest available
    """
    missing_values_count = df.isnull().sum()
    # print("Original data # missing values: {}".format(missing_values_count))
    df = df.fillna(method = "ffill", axis=0).fillna("0")
    missing_values_count = df.isnull().sum()
    # print("After filling na # missing values: {}".format(missing_values_count))
    return df

def outliers(df):
    df_outliers = df.loc[:,["date", "return", "close_to_open", "close_to_high", "close_to_low"]]
    df_smallest = df_outliers.sort_values(by="return", ascending=True)
    # print(df_smallest.iloc[:5])
    df_largest = df_outliers.sort_values(by="return", ascending=False)
    # print(df_largest.iloc[:5])
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