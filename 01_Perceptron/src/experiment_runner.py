"""
experiment_runner.py

Provides a generalized function to run a Perceptron experiment.
This helps to reduce code duplication across different training scripts.
"""

from typing import Callable, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from src.model import Perceptron

def run_experiment(
    data_loader_func: Callable[[], Tuple[np.ndarray, np.ndarray]],
    experiment_name: str,
    learning_rate: float,
    n_iter: int,
    test_size: float = 0.2,
    stratify: bool = True
) -> Tuple[Perceptron, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a full Perceptron experiment: loading data, training, and evaluating.

    Parameters
    ----------
    data_loader_func : Callable
        A function that returns a tuple of (X, y) data.
    experiment_name : str
        The name of the experiment for logging purposes.
    learning_rate : float
        The learning rate for the Perceptron.
    n_iter : int
        The number of training iterations (epochs).
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    stratify : bool, optional
        Whether to stratify the data split.

    Returns
    -------
    Tuple
        A tuple containing:
        - perceptron: The trained model instance.
        - X_train, y_train: The training data.
        - X_test, y_test: The testing data.
        - predictions: The model's predictions on the test data.
    """
    # 1. Load Data & Split
    print(f"[Runner] Initializing experiment: '{experiment_name}'")
    X, y = data_loader_func()
    stratify_data = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_data
    )
    print(f"[Runner] Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 2. Train Model
    print(f"\n[Runner] Training Perceptron...")
    perceptron = Perceptron(learning_rate=learning_rate, n_iter=n_iter)
    perceptron.fit(X_train, y_train)
    print("[Runner] Model training complete.")

    # 3. Evaluate Model
    print("\n[Runner] Evaluating the model...")
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"[Runner] --> Accuracy on the '{experiment_name}' test set: {accuracy * 100:.2f}%")

    return perceptron, X_train, y_train, X_test, y_test, predictions