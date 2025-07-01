"""
train_mnist_3_vs_5.py

Train and evaluate the Perceptron on the MNIST 3 vs 5 task.
This demonstrates the flexibility of the refactored experiment runner and data loader.
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
    EXPERIMENT_PREFIX = "mnist_3_vs_5"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}.log")
    setup_logging(log_file)

    # 1. Run the experiment
    perceptron, X_train, y_train, X_test, y_test, predictions = run_experiment(
        data_loader_func=lambda: load_mnist_binary(3, 5),
        experiment_name="MNIST 3 vs 5",
        learning_rate=0.01,
        n_iter=1000
    )

    # 2. Visualize the results
    logging.info("Generating visualizations for the 'MNIST 3 vs 5' experiment...")

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
        y_test, predictions, display_labels=['Digit 3', 'Digit 5'],
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_confusion_matrix.png")
    )
    plot_learned_weights(
        perceptron,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_learned_weights.png")
    )
    plot_misclassified_examples(
        X_test, y_test, predictions, display_labels=['3', '5'],
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_misclassified.png")
    )
    
    animate_learned_weights(
        perceptron,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_weights_evolution.gif")
    )
       
    logging.info("--- MNIST 3 vs 5 Experiment Complete ---")