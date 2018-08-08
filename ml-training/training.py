import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import cm
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

def training():
    filename = "MSFT.csv"
    X, y = training_data_from_csv(filename_features="MSFT_X_learn.csv", 
        filename_variables="MSFT_y_learn.csv")
    X = X.values # convert to numpy.ndarray used by sklearn
    y = y.values # convert to numpy.ndarray used by sklearn
    X_train_folds, y_train_folds, X_test_folds, y_test_folds = produce_cross_validation_folds(X, y)
    linear_regression(X_train_folds, y_train_folds, X_test_folds, y_test_folds)

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

def produce_cross_validation_folds(X, y):
    """
    Split data into folds such that testing fold timestamps always for training fold
    Args:
        X: numpy.ndarray
        y: numpy.ndarray
    Returns:
        X_train_folds: list of numpy.ndarray
        y_train_folds: list of numpy.ndarray
        X_test_folds: list of numpy.ndarray
        y_test_folds: list of numpy.ndarray
    """
    step = int(len(y) / 4)
    X_train_folds = [X[:step,:], X[:step*2,:], X[:step*3,:]]
    y_train_folds = [y[:step,:], y[:step*2,:], y[:step*3,:]]
    X_test_folds = [X[step:step*2,:], X[step*2:step*3,:], X[step*3:,:]]
    y_test_folds = [y[step:step*2,:], y[step*2:step*3,:], y[step*3:,:]]
    return X_train_folds, y_train_folds, X_test_folds, y_test_folds

def linear_regression(X_train_folds, y_train_folds, X_test_folds, y_test_folds):
    """
    Train linear regression model and save the plot of results to images/
    Args:
        X_train_folds: list of numpy.ndarray
        y_train_folds: list of numpy.ndarray
        X_test_folds: list of numpy.ndarray
        y_test_folds: list of numpy.ndarray
    """
    model = linear_model.LinearRegression()
    num_folds = len(y_train_folds)
    results = pd.DataFrame(columns=["fold", "MAE train", "MSE train", 
        "R2 train", "MAE test", "MSE test", "R2 test"])
    for i in range(num_folds):
        X_train = X_train_folds[i]
        y_train = y_train_folds[i]
        X_test = X_test_folds[i]
        y_test = y_test_folds[i]
        # X_train = PolynomialFeatures(degree=2).fit(X_train).transform(X_train)
        # X_test = PolynomialFeatures(degree=2).fit(X_test).transform(X_test)
        model.fit(X_train, y_train)
        # params = model.get_params()
        # coef = model.coef_
        y_pred = model.predict(X_train)
        mae_train, mse_train, r2_train = performance(y_train, y_pred)
        y_pred = model.predict(X_test)
        mae_test, mse_test, r2_test = performance(y_test, y_pred)
        results.loc[len(results)] = [i, mae_train, mse_train, r2_train,
            mae_test, mse_test, r2_test]
    plot_performance(results, "lin_reg")
    
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
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.1, hspace=0.4)
    axes[1].set_title("MAE")
    axes[0].set_ylabel("Error")
    axes[0].set_xlabel("Fold #")
    results[["fold", "MAE train", "MAE test"]].plot(x="fold", kind="bar", 
        ax=axes[0], logy=True, colormap=cm.flag)
    axes[1].set_title("MSE")
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Fold #")
    results[["fold", "MSE train", "MSE test"]].plot(x="fold", kind="bar", 
        ax=axes[1], logy=True, colormap=cm.flag)
    axes[2].set_title("R2")
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Fold #")
    axes[2].set_ylim(0, 1)
    results[["fold", "R2 train", "R2 test"]].plot(x="fold", kind="bar", 
        ax=axes[2], colormap=cm.flag)
    fig.savefig("images/{}.png".format(filename))

if __name__ == "__main__":
    training()