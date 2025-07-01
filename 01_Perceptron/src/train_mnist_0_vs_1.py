"""
train_mnist_0_vs_1.py

Train and evaluate the Perceptron on the MNIST 0 vs 1 task.
Part of a project to explore the history and advances in neural network AI.
"""

from src.data_loader import load_mnist_0_vs_1
from src.experiment_runner import run_experiment
from src.visualize import (
    animate_dataset_samples,
    animate_learned_weights,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_learned_weights,
    plot_misclassified_examples
)

if __name__ == "__main__":
    # 1. Run the experiment
    perceptron, X_train, y_train, X_test, y_test, predictions = run_experiment(
        data_loader_func=load_mnist_0_vs_1,
        experiment_name="MNIST 0 vs 1",
        learning_rate=0.01,
        n_iter=20
    )

    # 2. Visualize the results
    print("\n[Main] Generating visualizations for the 'MNIST 0 vs 1' experiment...")

    # Input data visualization
    animate_dataset_samples(X_train, y_train, n_samples=100, output_filename="training_data_sample.gif")

    # Result visualizations
    plot_learning_curve(perceptron)
    plot_confusion_matrix(y_test, predictions, display_labels=['Digit 0', 'Digit 1'])
    plot_learned_weights(perceptron)
    plot_misclassified_examples(X_test, y_test, predictions, display_labels=['0', '1'])
    animate_learned_weights(perceptron, output_filename="perceptron_weights_evolution.gif")

    print("\n[Main] --- MNIST 0 vs 1 Experiment Complete ---")
