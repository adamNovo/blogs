import warnings
# Ignore warning: lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88 return f(*args, **kwds)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import cm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

import linear_regression as lin_reg

def training():
    filename = "MSFT.csv"
    X, y = training_data_from_csv(filename_features="MSFT_X_learn.csv", 
        filename_variables="MSFT_y_learn.csv")
    X = X.values # convert to numpy.ndarray used by sklearn
    y = y.values # convert to numpy.ndarray used by sklearn
    tscv = TimeSeriesSplit(n_splits=3)
    tscv_idx = tscv.split(X)
    lin_reg.main(X, y, tscv_idx)

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

def plot_performance(results, filename):
    """
    Plot the results as subplots for each fold
    Args:
        results: pandas.DataFrame(columns=["fold", "MAE train", "MSE train", 
        "R2 train", "MAE test", "MSE test", "R2 test"])
        filename: str
    Returns:
        None
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.tight_layout()
    plt.suptitle("Linear Regression", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.1, hspace=0.6)
    results[["final_train_idx", "MAE train", "MAE test"]].plot(x="final_train_idx", kind="bar", 
        ax=axes[0], logy=True, colormap=cm.flag)
    axes[0].set_title("MAE")
    axes[0].set_ylabel("Error")
    axes[0].set_xlabel("Final train index of fold")
    results[["final_train_idx", "MSE train", "MSE test"]].plot(x="final_train_idx", kind="bar", 
        ax=axes[1], logy=True, colormap=cm.flag)
    axes[1].set_title("MSE")
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Final train index of fold")
    results[["final_train_idx", "R2 train", "R2 test"]].plot(x="final_train_idx", kind="bar", 
        ax=axes[2], colormap=cm.flag)
    axes[2].set_title("R2")
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Final train index of fold")
    axes[2].set_ylim(0, 1)
    fig.savefig("images/{}.png".format(filename))

if __name__ == "__main__":
    training()