import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import cm
from numpy.random import seed
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from tensorflow import set_random_seed

from training import performance
from training import plot_performance

seed(1)
set_random_seed(2)

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
    model = build_model(input_size=len(X[0]), optimizer="RMSprop", activation="softmax", neurons=200)
    results = pd.DataFrame(columns=["final_train_idx", "MAE train", "MSE train", 
        "R2 train", "MAE test", "MSE test", "R2 test"])
    for train_idx, test_idx in tscv_idx:
        X_train = X[train_idx,:]
        y_train = y[train_idx,:]
        X_test = X[test_idx,:]
        y_test = y[test_idx,:]
        model.fit(X_train, y_train, epochs=10, batch_size=20, verbose=1)
        # params = model.get_params()
        # coef = model.coef_
        y_pred = model.predict(X_train)
        mae_train, mse_train, r2_train = performance(y_train, y_pred)
        y_pred = model.predict(X_test)
        mae_test, mse_test, r2_test = performance(y_test, y_pred)
        results.loc[len(results)] = [train_idx[-1], mae_train, mse_train, r2_train,
            mae_test, mse_test, r2_test]
    plot_performance(results, "Neural Net Keras", "neural_keras")

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
    model = KerasRegressor(build_fn=build_model, verbose=0)
    batch_size = [10, 20]
    epochs = [10, 15]
    optimizer = ["RMSprop", "Adam"]
    activation = ["softmax", "relu", "tanh"]
    neurons = [10, 50, 200]
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer,
        activation=activation, neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv_idx)
    grid_result = grid.fit(X, y, verbose=1)
    means = grid_result.cv_results_["mean_test_score"]
    params = grid_result.cv_results_["params"]
    results = pd.DataFrame(columns=["batch_size", "epochs", "optimizer", 
        "activation", "neurons", "mse"])
    for mean, param in zip(means, params):
        results.loc[len(results)] = [param["batch_size"], param["epochs"], 
            param["optimizer"], param["activation"], param["neurons"], mean]
    print(results)
    print("Best score: {} using {}".format(grid_result.best_score_, grid_result.best_params_))
    
def build_model(input_size=171, optimizer="adam", activation="relu", neurons=200):
    """
    Train linear Decision Tree Regressor model and save the plot of results to images/
    Args:
        input_size: number of neurons in the first layer
        optimizer: model compilation optimizer
        activation: layer activation function
        neurons: number of neurons in each layer
    Returns:
        Sequential model
    """
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_dim=input_size))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=optimizer, loss="mse")
    return model