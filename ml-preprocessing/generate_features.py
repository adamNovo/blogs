import pandas as pd

def return_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    df["return"] = df["close"] / df["close"].shift(1)
    df["close_to_open"] = df["close"] / df["open"]
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]
    df = df.iloc[1:] # first first row: does not have a return value
    return df

def trend_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    # TODO MACD
    # TODO MA
    # TODO Parabolic Stop and Reverse
    return df

def momentum_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    df["5d_momentum"] = df["close"] / df["close"].shift(5)
    df["21d_momentum"] = df["close"] / df["close"].shift(21)
    df["60d_momentum"] = df["close"] / df["close"].shift(60)
    # TODO Stochastic
    # TODO Commodity Channel (CCI)
    # TODO RSI
    return df

def volatility_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    df["5d_volatility"] = df["return"].rolling(5).std()
    df["21d_volatility"] = df["return"].rolling(21).std()
    df["60d_volatility"] = df["return"].rolling(60).std()
    df["bollinger"] = ((df["close"] - df["close"].rolling(21).mean()) / 
        2 * df["close"].rolling(21).std())
    # TODO Average true range
    return df

def volume_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    df["volume_rolling"] = df["volume"] / df["volume"].shift(21)
    # TODO on balance volume
    # TODO chaikin oscilator
    return df