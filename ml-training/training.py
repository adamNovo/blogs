import matplotlib
import numpy as np
import pandas as pd

from sklearn import linear_model

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
    params = model.get_params()
    print(params)
    print(model.coef_)
    score = model.score(X, y)
    print(score)


if __name__ == "__main__":
    training()