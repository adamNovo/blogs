import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests

from matplotlib import cm

headers = {"x-api-key": "nnon43on5ion5o34n5oin53"} # local testing only
url_local = "http://127.0.0.1:5005"
url_remote = ""
url = url_local # select which url to test

def test_home():
    """
    Test if app is running
    """
    r = requests.get(url, headers=headers) # local testing
    assert r.status_code == 200

def test_forecast_vs_evaluation():
    """
    Test if forecast by app matches results from evaluation: https://github.com/adam5ny/blogs/tree/master/ml-evaluation
    """
    y_pred_evaluation = np.loadtxt("tests/evaluation_y_pred.txt") # data from evaluation
    forecast_len = len(y_pred_evaluation)
    df = yahoo_finance_source_to_df("tests/MSFT.csv").tail(forecast_len)
    df_json = df.to_json(orient='split')
    r = requests.post("{}/forecast".format(url), headers=headers, data=df_json)
    assert r.status_code == 200
    performance = pd.read_json(r.json()["performance"], orient='split')
    y_pred = json.loads(r.json()["y_pred"])
    assert 1 == len(performance)
    assert len(y_pred) == len(df)
    y_pred_compare = y_pred[-forecast_len:]
    y_pred_evaluation_compare = y_pred_evaluation[-forecast_len:]
    for i in range(forecast_len):
        assert y_pred_compare[i] == y_pred_evaluation_compare[i]
    plot_performance(performance, "Deployment performance", "deployment_performance")

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

def plot_performance(results, title, filename):
    """
    Plot the results as subplots for each fold
    Args:
        results: pandas.DataFrame(columns=[MAE test", "MSE test", "R2 test"])
        title: str
        filename: str
    Returns:
        None
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle(title, fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.1, hspace=0.6)
    results[["MAE test"]].plot(kind="bar", 
        ax=axes[0], logy=True, colormap=cm.flag)
    axes[0].set_title("MAE")
    results[["MSE test"]].plot(kind="bar", 
        ax=axes[1], logy=True, colormap=cm.flag)
    axes[1].set_title("MSE")
    results[["R2 test"]].plot(kind="bar", 
        ax=axes[2], colormap=cm.flag)
    axes[2].set_title("R2")
    axes[2].set_ylim(0, 1)
    fig.savefig("tests/{}.png".format(filename))