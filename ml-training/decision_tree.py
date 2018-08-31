import numpy as np
import pandas as pd
import pickle
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from training import performance
from training import plot_performance

def main(X: np.ndarray, y: np.ndarray, tscv_idx) -> None:
    """
    Train linear Decision Tree Regressor model and save the plot of results to images/
    Args:
        X: numpy.ndarray
        y: numpy.ndarray
        tscv_idx: generator
    Returns:
        None
    """
    model = tree.DecisionTreeRegressor(splitter="best", max_depth=18,
        min_samples_split=5, max_leaf_nodes=50, random_state=1)
    results = pd.DataFrame(columns=["final_train_idx", "MAE train", "MSE train", 
        "R2 train", "MAE test", "MSE test", "R2 test"])
    for train_idx, test_idx in tscv_idx:
        X_train = X[train_idx,:]
        y_train = y[train_idx,:]
        X_test = X[test_idx,:]
        y_test = y[test_idx,:]
        X_train = PolynomialFeatures(degree=2).fit(X_train).transform(X_train)
        X_test = PolynomialFeatures(degree=2).fit(X_test).transform(X_test)
        model.fit(X_train, y_train) # params: model.get_params(), coef = model.coef_
        y_pred = model.predict(X_train)
        mae_train, mse_train, r2_train = performance(y_train, y_pred)
        y_pred = model.predict(X_test)
        print(y_pred[:30])
        print(y_pred[-30:])
        mae_test, mse_test, r2_test = performance(y_test, y_pred)
        results.loc[len(results)] = [train_idx[-1], mae_train, mse_train, r2_train,
            mae_test, mse_test, r2_test]
    plot_performance(results, "Decision Tree", "tree")
    print(results)
    pickle.dump(model, open("dtree_model.pkl", "wb"))

def grid_search(X: np.ndarray, y: np.ndarray, tscv_idx):
    """
    Train linear Decision Tree Regressor model and save the plot of results to images/
    Args:
        X: numpy.ndarray
        y: numpy.ndarray
        tscv_idx: generator
    Returns:
        None
    """
    X = PolynomialFeatures(degree=2).fit(X).transform(X)
    model = tree.DecisionTreeRegressor(splitter="best", max_depth=14,
        min_samples_split=5, max_leaf_nodes=None, random_state=2)
    splitter = ["best", "random"]
    max_depth = [14, 16, 18]
    min_samples_split = [5, 10, 15]
    max_leaf_nodes = [None, 50, 100, 200]
    param_grid = dict(splitter=splitter, max_depth=max_depth, 
        min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv_idx)
    grid_result = grid.fit(X, y)
    means = grid_result.cv_results_["mean_test_score"]
    params = grid_result.cv_results_["params"]
    results = pd.DataFrame(columns=["splitter", "max_depth", 
        "min_samples_split", "max_leaf_nodes", "mse"])
    for mean, param in zip(means, params):
        row = [param["splitter"], param["max_depth"], 
            param["min_samples_split"], param["max_leaf_nodes"], mean]
        results.loc[len(results)] = row
        print(row)
    print(results)
    print("Best score: {} using {}".format(grid_result.best_score_, grid_result.best_params_))