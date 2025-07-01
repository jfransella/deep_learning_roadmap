"""
data_loader.py

Data loading utilities for classic neural network experiments.
Part of a project to explore the history and advances in neural network AI.
"""

from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist_0_vs_1():
    """
    Load the MNIST dataset and filter to only digits 0 and 1.

    Returns
    -------
    X : np.ndarray
        Image data (features).
    y : np.ndarray
        Labels (0 or 1).
    """
    print("[DataLoader] Fetching MNIST data (digits 0 and 1)... This may take a moment.")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    is_0_or_1 = (mnist.target == '0') | (mnist.target == '1')
    X = mnist.data[is_0_or_1]
    y = mnist.target[is_0_or_1]
    X = X / 255.0  # Normalize pixel values to [0, 1]
    y = y.astype(int)  # Convert labels to integers
    print("[DataLoader] Data loaded and prepared successfully.")
    return X, y

def load_mnist_6_vs_9():
    """
    Load the MNIST dataset and filter to only digits 6 and 9.

    Returns
    -------
    X : np.ndarray
        Image data (features).
    y : np.ndarray
        Labels (0 for '6', 1 for '9').
    """
    print("[DataLoader] Fetching MNIST data for '6' vs '9' task...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    is_6_or_9 = (mnist.target == '6') | (mnist.target == '9')
    X = mnist.data[is_6_or_9]
    y = mnist.target[is_6_or_9]
    X = X / 255.0
    y = np.where(y == '6', 0, 1)
    print("[DataLoader] Data loaded and prepared successfully.")
    return X, y