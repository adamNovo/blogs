import pandas as pd

def price_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    df["close_to_open"] = df["close"] / df["open"]
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]
    df["return"] = df["close"] / df["close"].shift(1)
    return df

def momentum_features(df):
    # TODO
    # df['momentum'] = (df['Adj Close'] / df['Adj Close'].shift(lookback)) - 1
    pass

def volatility_features(df):
    # TODO
    # df['volatility'] = (pd.rolling_std(df['return'], lookback))
    # df['bollinger'] = ((df['close'] - 
    #     pd.rolling_mean(df['close'], lookback)) / 
    #     2 * pd.rolling_std(df['close'], lookback))
    pass

def activity_features(df):
    # TODO
    # df['volume_rolling'] = (df['volume'] / df['volume'].shift(lookback))
    pass