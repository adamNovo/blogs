import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from training import performance
from training import plot_performance

def main(X: np.ndarray, y: np.ndarray, tscv_idx) -> None:
    """
    Train linear regression model and save the plot of results to images/
    Args:
        X: features
        y: target variables
        tscv_idx: generator of indexes used for training and testing in folds
    Returns:
        None
    """
    model = linear_model.LinearRegression()
    results = pd.DataFrame(columns=["final_train_idx", "MAE train", "MSE train", 
        "R2 train", "MAE test", "MSE test", "R2 test"])
    for train_idx, test_idx in tscv_idx:
        X_train = X[train_idx,:]
        y_train = y[train_idx,:]
        X_test = X[test_idx,:]
        y_test = y[test_idx,:]
        X_train = PolynomialFeatures(degree=2).fit(X_train).transform(X_train)
        X_test = PolynomialFeatures(degree=2).fit(X_test).transform(X_test)
        model.fit(X_train, y_train)
        # params = model.get_params()
        # coef = model.coef_
        y_pred = model.predict(X_train)
        mae_train, mse_train, r2_train = performance(y_train, y_pred)
        y_pred = model.predict(X_test)
        mae_test, mse_test, r2_test = performance(y_test, y_pred)
        results.loc[len(results)] = [train_idx[-1], mae_train, mse_train, r2_train,
            mae_test, mse_test, r2_test]
    plot_performance(results, "Linear Regression", "lin_reg")