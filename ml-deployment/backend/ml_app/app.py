import warnings
# Ignore warning: lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88 return f(*args, **kwds)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import json
import pandas as pd
import pickle
import os
from flask import Flask, jsonify, request, make_response
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

import generate_features
from auth import requires_auth

app = Flask(__name__)
model = None
scaler = None
yahoo_prices = None

@app.route('/')
@requires_auth
def home_get():
    return("API status 200")

@app.route('/forecast', methods=["POST"])
@requires_auth
def forecast_post():
    """
    Args:
        request.data: json pandas dataframe
            example: {
                "columns": ["date","open","high","low","close","volume"],
                "index":[1, 0],
                "data": [[1532390400000,108.57,108.82,107.260002,107.660004,26316600],
                    [1532476800000,107.959999,111.150002,107.599998,110.830002,30702100]]}
    """
    global model
    if request.data:
        df = pd.read_json(request.data, orient='split')
        X = preprocess(df)
        if model == None:
            model = pickle.load(open("dtree_model.pkl", "rb"))
        y_pred = run_model(X, model)
        resp = make_response(jsonify({"y_pred": json.dumps(y_pred.tolist())}), 200)
        return resp
    else:
        return make_response(jsonify({"message": "no data"}), 400)

def preprocess(df):
    """
    Args:
        df: pd.DataFrame(columns=["date","open","high","low","close","volume"])
    Returns:
        df: pd.DataFrame
    """
    forecast_len = len(df)
    df = combine_with_old_prices(df)
    df = generate_features.return_features(df)
    df = generate_features.target_value(df)
    df = generate_features.trend_features(df)
    df = generate_features.momentum_features(df)
    df = generate_features.volatility_features(df)
    df = generate_features.volume_features(df)
    X = remove_unused_features(df)
    X_forecast = X[len(X)-forecast_len:].values # convert to np.ndarray for sklearn
    X_forecast = scaling(X_forecast)
    return X_forecast

def combine_with_old_prices(df_new):
    """
    Combines old prices with new forecast values, leaving out duplicates
    Args:
        df: pd.DataFrame(columns=["date","open","high","low","close","volume"])
    Returns:
        df: pd.DataFrame(columns=["date","open","high","low","close","volume"])
    """
    global yahoo_prices
    if yahoo_prices is None:
        yahoo_prices = yahoo_finance_source_to_df("MSFT.csv")
    df = yahoo_prices.copy(deep=True)
    exists_idx = df_new["date"].isin(df["date"]) == False
    for i, row in df_new.loc[exists_idx].iterrows():
        df.loc[len(yahoo_prices)] = row
    return df

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
    return X

def scaling(df):
    """
    Scales X features using the scaler derived from preprocessing
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    global scaler
    if scaler == None:
        scaler = joblib.load("scaler_ml.pkl")
    X = scaler.transform(df) 
    return X

def run_model(X, model):
    """
    Calculates regression performance
    Args:
        X: numpy.ndarray
        y: numpy.ndarray
        model: sklearn.model
    Returns:
        None
    """
    X = PolynomialFeatures(degree=2).fit(X).transform(X)
    results = pd.DataFrame(columns=["MAE test", "MSE test", "R2 test"])
    y_pred = model.predict(X)
    return y_pred