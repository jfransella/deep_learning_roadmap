"""
train_circles.py

Train and evaluate the MLP on the concentric circles dataset.
This demonstrates the MLP's ability to solve non-linearly separable problems.
Part of a project to explore the history and advances in neural network AI.
"""

import os
import numpy as np
from src.data_loader import load_circles
from src.experiment_runner import run_mlp_experiment
from src.visualize import (
    plot_decision_boundary,
    plot_learning_curve,
    plot_confusion_matrix
)

if __name__ == "__main__":
    # --- 1. Setup ---
    OUTPUT_DIR = "output"
    EXPERIMENT_PREFIX = "mlp_circles"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. Define Experiment Parameters ---
    mlp_params = {
        'n_hidden': 20,        # Number of neurons in the hidden layer
        'n_output': 2,         # 2 classes for the two circles
        'learning_rate': 0.1,
        'n_iter': 200,         # Number of epochs
        'shuffle': True,
        'random_state': 42
    }

    # --- 3. Run Experiment ---
    mlp, X_train, y_train, X_test, y_test, predictions = run_mlp_experiment(
        data_loader_func=load_circles,
        experiment_name="Concentric Circles",
        mlp_params=mlp_params
    )

    # --- 4. Visualize the Results ---
    print("\n[Main] Generating visualizations for the circles experiment...")
    class_labels = ['Class 0', 'Class 1']

    plot_learning_curve(mlp, output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_learning_curve.png"))
    plot_confusion_matrix(y_test, predictions, display_labels=class_labels, output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_confusion_matrix.png"))
    plot_decision_boundary(X_train, y_train, mlp, title="MLP on Concentric Circles", output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_decision_boundary.png"))

    print("\n[Main] --- Concentric Circles Experiment Complete ---")