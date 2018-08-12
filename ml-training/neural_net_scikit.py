import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

from training import performance
from training import plot_performance

def main(X: np.ndarray, y: np.ndarray, tscv_idx) -> None:
    """
    Train linear Decision Tree Regressor model and save the plot of results to images/
    Args:
        X: numpy.ndarray
        y: numpy.ndarray
        tscv_idx: generator of indexes used for training and testing in folds
    Returns:
        None
    """
    X = PolynomialFeatures(degree=2).fit(X).transform(X)
    model = MLPRegressor(hidden_layer_sizes=(200, 200), solver="adam", 
        activation="relu", random_state=2)
    results = pd.DataFrame(columns=["final_train_idx", "MAE train", "MSE train", 
        "R2 train", "MAE test", "MSE test", "R2 test"])
    for train_idx, test_idx in tscv_idx:
        X_train = X[train_idx,:]
        y_train = y[train_idx,:]
        X_test = X[test_idx,:]
        y_test = y[test_idx,:]
        model.fit(X_train, y_train)
        # params = model.get_params()
        # coef = model.coef_
        y_pred = model.predict(X_train)
        mae_train, mse_train, r2_train = performance(y_train, y_pred)
        y_pred = model.predict(X_test)
        mae_test, mse_test, r2_test = performance(y_test, y_pred)
        results.loc[len(results)] = [train_idx[-1], mae_train, mse_train, r2_train,
            mae_test, mse_test, r2_test]
    plot_performance(results, "Neural Net scikit-learn", "neural_net_scikit")