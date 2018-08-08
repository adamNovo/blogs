import matplotlib
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def training():
    filename = "MSFT.csv"
    X, y = training_data_from_csv(filename_features="MSFT_X_learn.csv", 
        filename_variables="MSFT_y_learn.csv")
    linear_regression(X, y)

def training_data_from_csv(filename_features, filename_variables):
    """
    Load cvs file of features and target variables from csv
    Args:
        filename_features: string
        filename_variables: string
    Returns:
        pandas.DataFrame
    """
    X = pd.read_csv(filename_features, header=0)
    y = pd.read_csv(filename_variables, header=0)
    return X, y

def linear_regression(X, y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    # params = model.get_params()
    # coef = model.coef_
    y_pred = model.predict(X)
    y = y.values
    mae, mse, r2 = performance(y, y_pred)
    print("MAE: {}, MSE: {}, R^2 {}".format(mae, mse, r2))

def performance(y_true, y_pred):
    """
    Calculates regression performance
    Args:
        y_true: numpy.ndarray
        y_pred: numpy.ndarray
    Returns:
        mae: mean_absolute_error
        mse: mean_squared_error
        r2: r2
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

if __name__ == "__main__":
    training()