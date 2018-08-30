"""
Run locally:
    source activate [conda_env_name]
    pip install -r ../requirements.txt
    pytest --capture=no -vv
"""

import json
import pandas as pd
import numpy as np
import requests

headers = {"x-api-key": "nnon43on5ion5o34n5oin53"} # local testing only

def test_home():
    r = requests.get("http://127.0.0.1:5005",
            headers=headers)
    assert r.status_code == 200

def test_forecast():
    df = yahoo_finance_source_to_df("MSFT.csv").tail(5)
    df_json = df.to_json(orient='split')
    r = requests.post("http://127.0.0.1:5005/forecast",
            headers=headers, data=df_json)
    assert r.status_code == 200
    performance = pd.read_json(r.json()["performance"], orient='split')
    y_pred = json.loads(r.json()["y_pred"])
    assert 1 == len(performance)
    assert len(y_pred) == len(df)
    y_pred_evaluation = np.loadtxt("evaluation_y_pred.txt")
    print(y_pred[-5:])
    print(y_pred_evaluation[-5:])

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