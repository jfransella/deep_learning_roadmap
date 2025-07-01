"""
data_loader.py

Data loading utilities for the Multi-Layer Perceptron project.
Part of a project to explore the history and advances in neural network AI.
"""

from sklearn.datasets import fetch_openml, make_circles
import numpy as np
import logging
from typing import Tuple

def load_mnist_full() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the full MNIST dataset with all 10 digits.

    Returns
    -------
    X : np.ndarray
        Image data (features), normalized to [0, 1].
    y : np.ndarray
        Integer labels (0-9).
    """
    logging.info("Fetching full MNIST data... This may take a moment.")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    X = mnist.data
    y = mnist.target.astype(int)

    X = X / 255.0  # Normalize pixel values to [0, 1]

    logging.info("Full MNIST data loaded and prepared successfully.")
    return X, y

def load_circles() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates and returns the concentric circles dataset.

    This is a classic non-linearly separable problem, perfect for testing an MLP.

    Returns
    -------
    X : np.ndarray
        2D feature data.
    y : np.ndarray
        Binary labels (0 or 1).
    """
    logging.info("Generating concentric circles dataset...")
    X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=1)
    logging.info("Circles data generated successfully.")
    return X, y