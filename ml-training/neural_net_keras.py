import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import seed
from sklearn.preprocessing import PolynomialFeatures
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
        tscv_idx: list of numpy.ndarray
    Returns:
        None
    """
    X = PolynomialFeatures(degree=2).fit(X).transform(X)
    model = Sequential()
    input_size = len(X[0])
    model.add(Dense(200, activation="relu", input_dim=input_size))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")
    results = pd.DataFrame(columns=["final_train_idx", "MAE train", "MSE train", 
        "R2 train", "MAE test", "MSE test", "R2 test"])
    for train_idx, test_idx in tscv_idx:
        X_train = X[train_idx,:]
        y_train = y[train_idx,:]
        X_test = X[test_idx,:]
        y_test = y[test_idx,:]
        model.fit(X_train, y_train, epochs=25, verbose=1)
        # params = model.get_params()
        # coef = model.coef_
        y_pred = model.predict(X_train)
        mae_train, mse_train, r2_train = performance(y_train, y_pred)
        y_pred = model.predict(X_test)
        mae_test, mse_test, r2_test = performance(y_test, y_pred)
        results.loc[len(results)] = [train_idx[-1], mae_train, mse_train, r2_train,
            mae_test, mse_test, r2_test]
    plot_performance(results, "Neural Net Keras", "neural_keras")

def test_keras():
    # example of training a final regression model
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import MinMaxScaler
    # generate regression dataset
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
    scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    scalarX.fit(X)
    scalarY.fit(y.reshape(100,1))
    X = scalarX.transform(X)
    y = scalarY.transform(y.reshape(100,1))
    print(X.shape)
    print(y.shape)
    # define and fit the final model
    model = Sequential()
    model.add(Dense(4, input_dim=2, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=1000, verbose=0)