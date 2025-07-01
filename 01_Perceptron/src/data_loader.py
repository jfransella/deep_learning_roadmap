"""
data_loader.py

Data loading utilities for classic neural network experiments.
Part of a project to explore the history and advances in neural network AI.
"""

from sklearn.datasets import fetch_openml
import numpy as np
from typing import Tuple

def load_mnist_binary(digit1: int, digit2: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset and filter for a binary classification task
    between two specified digits.

    Parameters
    ----------
    digit1 : int
        The first digit for the classification task. Will be mapped to label 0.
    digit2 : int
        The second digit for the classification task. Will be mapped to label 1.

    Returns
    -------
    X : np.ndarray
        Image data (features).
    y : np.ndarray
        Labels (0 for digit1, 1 for digit2).
    """
    print(f"[DataLoader] Fetching MNIST data for '{digit1}' vs '{digit2}' task... This may take a moment.")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    str_digit1, str_digit2 = str(digit1), str(digit2)
    is_binary_task = (mnist.target == str_digit1) | (mnist.target == str_digit2)

    X = mnist.data[is_binary_task]
    y = mnist.target[is_binary_task]

    X = X / 255.0  # Normalize pixel values to [0, 1]
    y = np.where(y == str_digit1, 0, 1)  # Map digit1 to 0, digit2 to 1
    print("[DataLoader] Data loaded and prepared successfully.")
    return X, y