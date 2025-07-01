"""
train_circles.py

Train and evaluate the Perceptron on a concentric circles dataset (a non-linearly separable case).
Part of a project to explore the history and advances in neural network AI.
"""

import os
import logging
from sklearn.datasets import make_circles
from src.experiment_runner import run_experiment
from src.logger_setup import setup_logging
from src.visualize import (
    animate_scatter_reveal,
    plot_decision_boundary,
    plot_learning_curve,
    plot_confusion_matrix
)

def load_circles_data():
    """Generates and returns the concentric circles dataset."""
    return make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=1)

if __name__ == "__main__":
    # Define output directory and experiment prefix
    OUTPUT_DIR = "output"
    EXPERIMENT_PREFIX = "circles"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}.log")
    setup_logging(log_file)

    # 1. Run the experiment
    perceptron, X_train, y_train, X_test, y_test, predictions = run_experiment(
        data_loader_func=load_circles_data,
        experiment_name="Concentric Circles",
        learning_rate=0.1,
        n_iter=100,
        test_size=0.3,
        stratify=False  # Stratification not needed for this generated dataset
    )

    # 2. Visualize the results
    logging.info("Generating visualizations for the 'Circles' experiment...")

    # Input data visualization
    animate_scatter_reveal(
        X_train, y_train,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_data_reveal.gif")
    )

    # Result visualizations
    plot_learning_curve(
        perceptron,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_learning_curve.png")
    )
    plot_confusion_matrix(
        y_test, predictions, display_labels=['Class 0', 'Class 1'],
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_confusion_matrix.png")
    )
    plot_decision_boundary(
        X_train, y_train, perceptron, title="Perceptron Failing on Concentric Circles",
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_decision_boundary.png")
    )