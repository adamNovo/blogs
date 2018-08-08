import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing as sklearn_preprocessing
from sklearn.decomposition import PCA

import generate_features

def preprocessing():
    filename = "MSFT.csv"
    df = yahoo_finance_source_to_df(filename)
    df = generate_features.return_features(df)
    df = generate_features.target_value(df)
    # print(df.head(5)[["date", "close", "volume", "return", "y"]])
    df = clean_df(df)
    df = new_features(df)
    X, y = remove_unused_features(df)
    X = X.values # convert to np.ndarray for sklearn
    y = y.values # convert to np.ndarray for sklearn
    X = scaling(X)
    X = dimensionality_reduction(X)
    train_test_split = 7000
    X_learn = X[:train_test_split]
    y_learn = y[:train_test_split]
    X_test = X[train_test_split:]
    y_test = y[train_test_split:]
    assert len(X_learn) + len(X_test) == len(X)
    df_to_csv(df=pd.DataFrame(X_learn), filename="MSFT_X_learn.csv")
    df_to_csv(df=pd.DataFrame(y_learn), filename="MSFT_y_learn.csv")
    df_to_csv(df=pd.DataFrame(X_test), filename="MSFT_X_test.csv")
    df_to_csv(df=pd.DataFrame(y_test), filename="MSFT_y_test.csv")

def yahoo_finance_source_to_df(filename):
    """
    Lead Yahoo Finance csv and format to DataFrame
    Args:
        filename: string
    Returns:
        pandas.DataFrame
    """
    df = pd.read_csv(filename, header=0)
    df.drop(labels="Close", axis=1, inplace=True)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    return df

def clean_df(df):
    """
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    df = missing_values(df)
    df = outliers(df)
    return df

def missing_values(df):
    """
    Replace missing values with latest available
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    missing_values_count = df.isnull().sum()
    # print("Original data # missing values: {}".format(missing_values_count))
    if sum(missing_values_count) == 0:
        return df
    else:
        print("Ffill of missing values necessary")
        df = df.fillna(method = "ffill", axis=0).fillna("0")
        missing_values_count = df.isnull().sum()
        # print("After filling na # missing values: {}".format(missing_values_count))
        assert sum(missing_values_count) == 0
        return df

def outliers(df):
    """
    Analyze outliers of dataset
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    df_outliers = df.loc[:,["date", "return", "close_to_open", "close_to_high", "close_to_low"]]
    column_to_analysts = "return"
    df_smallest = df_outliers.sort_values(by=column_to_analysts, ascending=True)
    df_largest = df_outliers.sort_values(by=column_to_analysts, ascending=False)
    # print(df_smallest.iloc[:5])
    # print(df_largest.iloc[:5])
    return df

def new_features(df):
    """
    Generate feature useful for learning
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    df = generate_features.trend_features(df)
    df = generate_features.momentum_features(df)
    df = generate_features.volatility_features(df)
    df = generate_features.volume_features(df)
    return df

def remove_unused_features(df):
    """
    - skip rows with nan values due to feature extraction
    - extract only useful columns for learning
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    X = df.loc[200:len(df)-1, ["return", "close_to_open", "close_to_high", "close_to_low",
        "macd_diff", "ma_50_200", "sar", "stochastic_oscillator",
        "cci", "rsi", "5d_volatility", "21d_volatility", "60d_volatility",
        "bollinger", "atr", "on_balance_volume", "chaikin_oscillator"]]
    y = df.loc[200:len(df)-1, ["y"]]
    return X, y

def scaling(df):
    """
    Scale all features to unit variance and 0 mean to optimize training
    Args:
        df: pandas.DataFrame
    Returns:
        numpy.ndarray
    """
    X = sklearn_preprocessing.scale(df, axis=0)
    assert sum(X.mean(axis=0)) > -0.00001 and sum(X.mean(axis=0)) < 0.00001 # zero mean
    assert sum(X.std(axis=0)) > 0.99999 and sum(X.mean(axis=0)) < 1.00001 # unit variance
    return X

def dimensionality_reduction(X):
    """
    Analyzed the usefulness of dimensionality reduction
    Args:
        numpy.ndarray
    Returns:
        numpy.ndarray
    """
    sk_model = PCA(n_components=10)
    sk_model.fit_transform(X)
    sk_model.explained = sk_model.explained_variance_ratio_.cumsum()
    # PCA not used due to a small number of original features
    return X

def df_to_csv(df, filename):
    """
    Save DataFrame to csv for later use
    Args:
        df: pandas.DataFrame
        filename: string
    Returns:
        None
    """
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    preprocessing()