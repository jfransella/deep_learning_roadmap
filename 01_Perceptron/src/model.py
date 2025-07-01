"""
model.py

Implements the classic Rosenblatt Perceptron for binary classification.
Part of a project to explore the history and advances in neural network AI.

Author: Your Name
"""

import numpy as np
import time
from typing import Optional, Tuple, List

class Perceptron:
    """
    Perceptron classifier for binary classification (Rosenblatt, 1958).

    This class implements the classic single-layer perceptron learning algorithm.

    Parameters
    ----------
    learning_rate : float, optional
        Step size for weight updates (default=0.01).
    n_iter : int, optional
        Number of training epochs (default=10).

    Attributes
    ----------
    weights : np.ndarray
        Learned weights after fitting.
    bias : float
        Learned bias after fitting.
    errors_history : list of int
        Number of misclassifications in each epoch.
    weights_history : list of tuple
        List of (weights, bias) tuples for each epoch (for visualization).
    """
    def __init__(self, learning_rate: float = 0.01, n_iter: int = 10):
        self.learning_rate: float = learning_rate
        self.n_iter: int = n_iter
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.errors_history: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Fit the perceptron model to the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training input vectors.
        y : np.ndarray of shape (n_samples,)
            Target values (0 or 1).

        Returns
        -------
        self : Perceptron
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_history = []
        self.weights_history = []

        print(f"[Perceptron] Starting training for {self.n_iter} epochs...")
        total_start_time = time.perf_counter()

        for epoch in range(self.n_iter):
            start_time = time.perf_counter()
            errors = 0
            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i)
                update = self.learning_rate * (y[idx] - prediction)
                if update != 0:
                    errors += 1
                self.weights += update * x_i
                self.bias += update
            self.errors_history.append(errors)
            self.weights_history.append((self.weights.copy(), self.bias))
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"[Perceptron] Epoch {epoch + 1}/{self.n_iter}: {errors} misclassifications ({elapsed_ms:.1f} ms)")

        total_elapsed_ms = (time.perf_counter() - total_start_time) * 1000
        avg_time_per_epoch_ms = total_elapsed_ms / self.n_iter
        print("\n[Perceptron] --- Training Summary ---")
        print(f"[Perceptron] Total epochs run: {self.n_iter}")
        print(f"[Perceptron] Total training time: {total_elapsed_ms / 1000:.2f} seconds")
        print(f"[Perceptron] Average time per epoch: {avg_time_per_epoch_ms:.1f} ms")
        print("[Perceptron] --------------------------\n")
        return self

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the net input (weighted sum plus bias).

        Parameters
        ----------
        X : np.ndarray
            Input vector(s).

        Returns
        -------
        np.ndarray
            Net input value(s).
        """
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Input vector(s).

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1).
        """
        return np.where(self._net_input(X) >= 0.0, 1, 0)