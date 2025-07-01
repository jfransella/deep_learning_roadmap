"""
train_mnist_6_vs_9.py

Train and evaluate the Perceptron on the MNIST 6 vs 9 task (a difficult case).
Part of a project to explore the history and advances in neural network AI.
"""

import os
from src.data_loader import load_mnist_binary
from src.experiment_runner import run_experiment
from src.visualize import (
    animate_dataset_samples,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_learned_weights,
    plot_misclassified_examples
)

if __name__ == "__main__":
    # Define output directory and experiment prefix
    OUTPUT_DIR = "output"
    EXPERIMENT_PREFIX = "mnist_6_vs_9"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Run the experiment
    perceptron, X_train, y_train, X_test, y_test, predictions = run_experiment(
        data_loader_func=lambda: load_mnist_binary(6, 9),
        experiment_name="MNIST 6 vs 9",
        learning_rate=0.01,
        n_iter=20
    )

    # 2. Visualize the results
    print("\n[Main] Generating visualizations for the 'MNIST 6 vs 9' experiment...")

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
        y_test, predictions, display_labels=['Digit 6', 'Digit 9'],
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_confusion_matrix.png")
    )
    plot_learned_weights(
        perceptron,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_learned_weights.png")
    )
    plot_misclassified_examples(
        X_test, y_test, predictions, display_labels=['6', '9'],
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_misclassified.png")
    )

    print("\n[Main] --- MNIST 6 vs 9 Experiment Complete ---")
