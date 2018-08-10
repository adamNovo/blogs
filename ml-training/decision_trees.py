import pandas as pd

from training import performance
from training import plot_performance

def main(X, y, tscv_idx):
    """
    Train linear Decision Tree Regressor model and save the plot of results to images/
    Args:
        X: numpy.ndarray
        y: numpy.ndarray
        tscv_idx: list of numpy.ndarray
    Returns:
        None
    """
