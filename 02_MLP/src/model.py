"""
model.py

Implements a Multi-Layer Perceptron (MLP) for binary classification.
Part of a project to explore the history and advances in neural network AI.
"""

import numpy as np
import time
from typing import Optional, List, Tuple

class MLP:
    """
    Multi-Layer Perceptron for binary classification.

    This class implements a feedforward neural network with a single hidden layer,
    trained using the backpropagation algorithm.

    Parameters
    ----------
    n_hidden : int, optional
        Number of neurons in the hidden layer (default=30).
    n_output : int, optional
        Number of output neurons (number of classes).
    learning_rate : float, optional
        Step size for weight updates (default=0.01).
    n_iter : int, optional
        Number of training epochs (default=50).
    shuffle : bool, optional
        Whether to shuffle the training data in every epoch (default=True).
    random_state : int, optional
        Seed for the random number generator for reproducible results.

    Attributes
    ----------
    w1, b1 : np.ndarray
        Weights and bias for the input-to-hidden layer connection.
    w2, b2 : np.ndarray
        Weights and bias for the hidden-to-output layer connection.
    cost_history : list of float
        Sum-of-squares cost function value in each epoch.
    """
    def __init__(
        self,
        n_hidden: int = 30,
        n_output: int = 10,
        learning_rate: float = 0.01,
        n_iter: int = 50,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        self.n_hidden: int = n_hidden
        self.n_output: int = n_output
        self.learning_rate: float = learning_rate
        self.n_iter: int = n_iter
        self.shuffle: bool = shuffle
        self.random_state: Optional[int] = random_state
        self.w1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.w2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.cost_history: List[float] = []
        self.weights_history: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid activation function."""
        # Clip to avoid overflow in np.exp
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Compute the softmax activation function."""
        e_z = np.exp(z - np.max(z, axis=-1, keepdims=True)) # For numeric stability
        return e_z / np.sum(e_z, axis=-1, keepdims=True)

    def _sigmoid_gradient(self, z: np.ndarray) -> np.ndarray:
        """Compute the gradient of the sigmoid function."""
        s = self._sigmoid(z)
        return s * (1.0 - s)

    def _initialize_weights(self, n_features: int, n_output: int):
        """Initialize weights with small random numbers."""
        # Weights for input -> hidden layer
        self.w1 = self.random_gen.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden)
        # Weights for hidden -> output layer
        self.w2 = self.random_gen.normal(loc=0.0, scale=0.1, size=(self.n_hidden, self.n_output))
        self.b2 = np.zeros(n_output)

    def _feedforward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the feedforward pass."""
        # Net input and activation of hidden layer
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self._sigmoid(z1)
        # Net input and activation of output layer
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self._softmax(z2)
        return z1, a1, z2, a2

    def _one_hot(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """Encode labels into a one-hot representation."""
        onehot = np.zeros((y.shape[0], n_classes))
        for i, val in enumerate(y):
            onehot[i, val] = 1.0
        return onehot

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        """
        Fit the MLP model to the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training input vectors.
        y : np.ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : MLP
            Fitted estimator.
        """
        n_samples, n_features = X.shape

        self.random_gen = np.random.RandomState(self.random_state)
        self._initialize_weights(n_features, self.n_output)
        self.weights_history = []
        self.cost_history = []

        y_onehot = self._one_hot(y, self.n_output)

        print(f"[MLP] Starting training for {self.n_iter} epochs...")
        total_start_time = time.perf_counter()

        for epoch in range(self.n_iter):
            epoch_start_time = time.perf_counter()
            print(f"\n[MLP] --- Starting Epoch {epoch + 1}/{self.n_iter} ---")
            
            if self.shuffle:
                indices = np.arange(n_samples)
                self.random_gen.shuffle(indices)
                X, y_onehot = X[indices], y_onehot[indices]
                print(f"    [Epoch {epoch + 1}] Shuffled training data.")

            epoch_cost = []
            # Set up interval for logging progress within an epoch
            log_interval = n_samples // 10
            
            for i in range(n_samples):
                x_i, y_i = X[i, :].reshape(1, -1), y_onehot[i].reshape(1, -1)

                # --- Feedforward ---
                z1, a1, z2, a2 = self._feedforward(x_i)

                # --- Backpropagation ---
                # The gradient of cross-entropy with softmax is simply (prediction - true_label)
                delta2 = a2 - y_i
                delta1 = np.dot(delta2, self.w2.T) * self._sigmoid_gradient(z1)

                # --- Weight Updates ---
                grad_w2 = np.dot(a1.T, delta2)
                grad_b2 = np.sum(delta2, axis=0)
                grad_w1 = np.dot(x_i.T, delta1)
                grad_b1 = np.sum(delta1, axis=0)

                self.w2 -= self.learning_rate * grad_w2
                self.b2 -= self.learning_rate * grad_b2
                self.w1 -= self.learning_rate * grad_w1
                self.b1 -= self.learning_rate * grad_b1

                # Cross-entropy cost
                cost = -np.sum(y_i * np.log(a2 + 1e-9)) # Add epsilon for numeric stability
                epoch_cost.append(cost)

                # --- Periodic Logging ---
                # Log progress every 10% of the way through the epoch
                if log_interval > 0 and (i + 1) % log_interval == 0 and (i + 1) < n_samples:
                    print(f"    [Epoch {epoch + 1}] Processed {i + 1}/{n_samples} samples...")

            avg_cost = np.mean(epoch_cost)
            self.cost_history.append(avg_cost)
            
            elapsed_ms = (time.perf_counter() - epoch_start_time) * 1000
            print(f"    [Epoch {epoch + 1}] Completed. Cost = {avg_cost:.6f} ({elapsed_ms:.1f} ms)")
            
            # Store weights for this epoch
            self.weights_history.append((self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()))

        total_elapsed_ms = (time.perf_counter() - total_start_time) * 1000
        avg_time_per_epoch_ms = total_elapsed_ms / self.n_iter
        print("\n[MLP] --- Training Summary ---")
        print(f"[MLP] Total epochs run: {self.n_iter}")
        print(f"[MLP] Total training time: {total_elapsed_ms / 1000:.2f} seconds")
        print(f"[MLP] Average time per epoch: {avg_time_per_epoch_ms:.1f} ms")
        print("[MLP] --------------------------\n")
        return self

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
        if self.w1 is None or self.w2 is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        _, _, _, a2 = self._feedforward(X)
        return np.argmax(a2, axis=1)