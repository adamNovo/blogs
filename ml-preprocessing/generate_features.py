import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    df = ma(df)
    df = parabolic_sar(df)
    return df

def macd(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    ema_12_day = df["close"].ewm(com=(12-1)/2).mean()
    ema_26_day = df["close"].ewm(com=(26-1)/2).mean()
    df["macd_line"] = ema_12_day - ema_26_day
    df["macd_9_day"] = df["macd_line"].ewm(com=(9-1)/2).mean()
    df["macd_diff"] = df["macd_line"] - df["macd_9_day"]
    # print(df.tail(10)[["date", "close", "macd_line", "macd_9_day"]])
    # chart_macd(df)
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
    df.tail(300)[["date", "macd_line", "macd_9_day"]].plot(x="date", kind="line", ax=ax2, secondary_y=False)
    # df.tail(300)[["date", "macd_diff"]].plot(x="date", kind="bar", ax=ax2, secondary_y=True)
    fig.savefig("charts/macd.png")

def ma(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    df["ma_50_day"] = df["close"].rolling(50).mean()
    df["ma_200_day"] = df["close"].rolling(200).mean()
    df["ma_50_200"] = df["ma_50_day"] - df["ma_200_day"]
    # print(df.tail(10)[["date", "close", "ma_50_200"]])
    # chart_ma(df)
    return df

def chart_ma(df):
    """
    Save chart to charts/ma
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "ma_50_200"]
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT Moving Average 50 day - 200 day", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, hspace=0.4)
    axes.set_ylabel("$")
    df.tail(1500)[["date", "close"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(1500)[["date", "ma_50_day"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(1500)[["date", "ma_200_day"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(1500)[["date", "ma_50_200"]].plot(x="date", kind="line", ax=axes, secondary_y=True)
    fig.savefig("charts/ma.png")

def parabolic_sar(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:parabolic_sar
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    df["sar"] = np.nan
    step = 5
    acc_factor = 0.02
    uptrend = False
    prior_sar = max(df.loc[1:step, "close"])
    extreme_point = min(df.loc[1:step, "close"])
    for i, row in df.iloc[step:].iterrows():
        if uptrend:
            df.at[i, "sar"] = prior_sar + acc_factor*(extreme_point - prior_sar)
            if df.at[i, "low"] < df.at[i, "sar"]:
                # reverse to downtrend
                uptrend = False
                prior_sar = max(df.loc[i-step:i, "close"])
                extreme_point = min(df.loc[i-step:i, "close"])
            else:
                # continue uptrend
                if df.at[i, "close"] > extreme_point:
                    extreme_point = df.at[i, "close"]
                    acc_factor = min(0.2, acc_factor+0.02)
        else:
            df.at[i, "sar"] = prior_sar - acc_factor*(prior_sar - extreme_point)
            if df.at[i, "high"] > df.at[i, "sar"]:
                # reverse to uptrend
                uptrend = True
                prior_sar = min(df.loc[i-step:i, "close"])
                extreme_point = max(df.loc[i-step:i, "close"])
            else:
                # continue downtrend
                if df.at[i, "close"] < extreme_point:
                    extreme_point = df.at[i, "close"]
                    acc_factor = min(0.2, acc_factor+0.02)
        prior_sar = df.at[i, "sar"]
    # chart_sar(df)
    return df

def chart_sar(df):
    """
    Save chart to charts/ma
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "sar"]
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT SAR", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, hspace = 0.4)
    axes.set_ylabel("$")
    df.tail(100)[["date", "close"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(100)[["date", "sar"]].plot(x="date", style=".", ax=axes, secondary_y=False)
    fig.savefig("charts/sar.png")

def momentum_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    df = stochastic_oscillator(df)
    df = commodity_channel_index(df)
    df = rsi(df)
    return df

def stochastic_oscillator(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochastic_oscillator_fast_slow_and_full
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    lookback = 14
    df["stochastic_oscillator"] = ((df["close"] - df["close"].rolling(lookback).min()) /
        (df["close"].rolling(lookback).max() - df["close"].rolling(lookback).min())) * 100
    # chart_stochastic_oscillator(df)
    return df

def chart_stochastic_oscillator(df):
    """
    Save chart to charts/stochastic_oscillator
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "stochastic_oscillator"]
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT Stochastic Oscillator", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, hspace = 0.4)
    df.tail(100)[["date", "close"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(100)[["date", "stochastic_oscillator"]].plot(x="date", kind="line", ax=axes, secondary_y=True)
    fig.savefig("charts/stochastic_oscillator.png")

def commodity_channel_index(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    Args:
        df: pandas.DataFrame, columns include at least ["open", "high", "low", "close"]
    Returns:
        pandas.DataFrame
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    mean_dev = abs(typical_price - typical_price.rolling(20).mean()).rolling(20).mean()
    df["cci"] = (typical_price - typical_price.rolling(20).mean()) / (0.15 * mean_dev)
    # chart_commodity_channel_index(df)
    return df

def chart_commodity_channel_index(df):
    """
    Save chart to charts/commodity_channel_index
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "stochastic_oscillator"]
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT Commodity Channel Index (CCI)", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, hspace = 0.4)
    df.tail(300)[["date", "close"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(300)[["date", "cci"]].plot(x="date", kind="line", ax=axes, secondary_y=True)
    fig.savefig("charts/commodity_channel_index.png")

def rsi(df):
    """
    Math reference: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi
    Args:
        df: pandas.DataFrame, columns include at least ["close"]
    Returns:
        pandas.DataFrame
    """
    df["dollar_pnl"] = df["close"].shift(1) - df["close"]
    avg_gains = df["dollar_pnl"].iloc[:14][df["dollar_pnl"].iloc[:14] > 0].sum() / 14
    avg_losses = abs(df["dollar_pnl"].iloc[:14][df["dollar_pnl"].iloc[:14] < 0].sum()) / 14
    for i, row in df.iloc[14:].iterrows():
        if row["dollar_pnl"] > 0:
            avg_gains = (avg_gains * 13 + row["dollar_pnl"]) / 14
        else:
            avg_losses = (avg_losses * 13 + abs(row["dollar_pnl"])) / 14
        if avg_losses == 0:
            rs = 100
        else:
            rs = avg_gains / avg_losses
        df.loc[i, "rsi"] = 100 - 100 / (1 + rs)
    print(df.tail(20)[["date", "close", "rsi"]])
    chart_rsi(df)
    return df

def chart_rsi(df):
    """
    Save chart to charts/rsi
    Args:
        df: pandas.DataFrame, columns include at least ["date", "close", "stochastic_oscillator"]
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("MSFT Relative Strength Index (RSI)", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, hspace = 0.4)
    df.tail(100)[["date", "close"]].plot(x="date", kind="line", ax=axes, secondary_y=False)
    df.tail(100)[["date", "rsi"]].plot(x="date", kind="line", ax=axes, secondary_y=True)
    fig.savefig("charts/rsi.png")

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