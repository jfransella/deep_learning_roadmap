"""
experiment_runner.py

Provides a generalized function to run an MLP experiment.
This helps to reduce code duplication across different training scripts.
"""

from typing import Callable, Tuple, Dict, Any
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from src.model import MLP

def run_mlp_experiment(
    data_loader_func: Callable[[], Tuple[np.ndarray, np.ndarray]],
    experiment_name: str,
    mlp_params: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[MLP, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a full MLP experiment: loading data, training, and evaluating.

    Parameters
    ----------
    data_loader_func : Callable
        A function that returns a tuple of (X, y) data.
    experiment_name : str
        The name of the experiment for logging purposes.
    mlp_params : Dict[str, Any]
        A dictionary of parameters to initialize the MLP model.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    random_state : int, optional
        The random state for reproducibility.
    stratify : bool, optional
        Whether to stratify the data split.

    Returns
    -------
    Tuple
        A tuple containing:
        - mlp: The trained model instance.
        - X_train, y_train: The training data.
        - X_test, y_test: The testing data.
        - predictions: The model's predictions on the test data.
    """
    # 1. Load Data & Split
    logging.info(f"Initializing experiment: '{experiment_name}'")
    X, y = data_loader_func()
    stratify_data = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_data
    )
    logging.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 2. Train Model
    logging.info("Training MLP...")
    mlp = MLP(**mlp_params)
    mlp.fit(X_train, y_train)

    # 3. Evaluate Model
    logging.info("Evaluating the model...")
    predictions = mlp.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    logging.info(f"--> Accuracy on the '{experiment_name}' test set: {accuracy * 100:.2f}%")

    return mlp, X_train, y_train, X_test, y_test, predictions