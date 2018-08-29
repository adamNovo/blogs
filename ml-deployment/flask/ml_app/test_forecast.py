"""
run using: 
    pytest --capture=no -vv
"""

import pandas as pd
import requests

def test_home():
    headers = {"x-api-key": "nnon43on5ion5o34n5oin53"}
    r = requests.get("http://127.0.0.1:5005",
            headers=headers)
    assert r.status_code == 200

def test_forecast():
    headers = {"x-api-key": "nnon43on5ion5o34n5oin53"}
    df = yahoo_finance_source_to_df("MSFT.csv").tail(5)
    df_json = df.to_json(orient='split')
    r = requests.post("http://127.0.0.1:5005/forecast",
            headers=headers, data=df_json)
    assert r.status_code == 200
    df = pd.read_json(r.json()["data"], orient='split')
    print(df)

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