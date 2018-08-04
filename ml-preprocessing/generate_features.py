import matplotlib.pyplot as plt
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
    df = macd(df)
    # TODO MA
    # TODO Parabolic Stop and Reverse
    return df

def macd(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    ema_12_day = df["close"].ewm(com=(12-1)/2).mean()
    ema_26_day = df["close"].ewm(com=(26-1)/2).mean()
    df["macd_line"] = ema_12_day - ema_26_day
    df["macd_9_day"] = df["macd_line"].ewm(com=(9-1)/2).mean()
    # print(df.tail(10)[["date", "close", "macd_line", "macd_9_day"]])
    chart_macd(df)
    return df

def chart_macd(df):
    """
    Save chart to charts/macd
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "macd_line", "macd_9_day"]
    Returns:
        None
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, hspace = 0.4)
    ax1 = axes[0]
    ax1.set_title("Price")
    ax1.set_ylabel("$")
    df.tail(300)[["date", "close"]].plot(x="date", kind="line", ax=ax1)
    ax2 = axes[1]
    ax2.set_title("MACD")
    df.tail(300)[["date", "macd_line", "macd_9_day"]].plot(x="date", kind="line", ax=ax2)
    fig.savefig("charts/macd.png")

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