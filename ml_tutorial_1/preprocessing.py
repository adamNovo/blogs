import pandas as pd

import generate_features

def preprocessing():
    filename = "MSFT.csv"
    df = source_to_df(filename)
    df = generate_features.price_features(df)
    print(df.tail(5)[["date", "low", "close", "return"]])
    filename = "MSFT_clean.csv"
    df_to_csv(df, filename)

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