"""
train_mnist_0_vs_1.py

Train and evaluate the Perceptron on the MNIST 0 vs 1 task.
Part of a project to explore the history and advances in neural network AI.
"""

import os
import logging
from src.data_loader import load_mnist_binary
from src.experiment_runner import run_experiment
from src.logger_setup import setup_logging
from src.visualize import (
    animate_dataset_samples,
    animate_learned_weights,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_learned_weights,
    plot_misclassified_examples
)

if __name__ == "__main__":
    # Define output directory and experiment prefix
    OUTPUT_DIR = "output"
    EXPERIMENT_PREFIX = "mnist_0_vs_1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}.log")
    setup_logging(log_file)

    # 1. Run the experiment
    perceptron, X_train, y_train, X_test, y_test, predictions = run_experiment(
        data_loader_func=lambda: load_mnist_binary(0, 1),
        experiment_name="MNIST 0 vs 1",
        learning_rate=0.01,
        n_iter=20
    )

    # 2. Visualize the results
    logging.info("Generating visualizations for the 'MNIST 0 vs 1' experiment...")

    # Input data visualization
    animate_dataset_samples(
        X_train, y_train, n_samples=100,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_data_sample.gif")
    )

    # Result visualizations
    plot_learning_curve(
        perceptron,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_learning_curve.png")
    )
    plot_confusion_matrix(
        y_test, predictions, display_labels=['Digit 0', 'Digit 1'],
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_confusion_matrix.png")
    )
    plot_learned_weights(
        perceptron,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_learned_weights.png")
    )
    plot_misclassified_examples(
        X_test, y_test, predictions, display_labels=['0', '1'],
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_misclassified.png")
    )
    animate_learned_weights(
        perceptron,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_weights_evolution.gif")
    )

    logging.info("--- MNIST 0 vs 1 Experiment Complete ---")
