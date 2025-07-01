"""
train_mnist.py

Train and evaluate the Multi-Layer Perceptron on the full MNIST dataset (10 digits).
Part of a project to explore the history and advances in neural network AI.
"""

import os
import logging
import numpy as np
from scipy.ndimage import shift
from src.data_loader import load_mnist_full
from src.experiment_runner import run_mlp_experiment
from src.logger_setup import setup_logging
from src.visualize import (
    plot_learning_curve,
    plot_confusion_matrix,
    plot_misclassified_examples,
    plot_learned_weights,
    animate_dataset_samples,
    animate_learned_weights,
    visualize_shifted_test
)

def create_shifted_dataset(X: np.ndarray, shift_pixels: int = 2) -> np.ndarray:
    """
    Creates a new dataset by shifting each image by a few pixels.
    This is used to test the model's spatial invariance.
    """
    logging.info(f"Creating a new test set by shifting images by {shift_pixels} pixels...")
    X_shifted = np.zeros_like(X)
    image_size = int(np.sqrt(X.shape[1]))
    for i in range(len(X)):
        image = X[i].reshape(image_size, image_size)
        shifted_image = shift(image, [shift_pixels, shift_pixels], cval=0)
        X_shifted[i] = shifted_image.flatten()
    return X_shifted

if __name__ == "__main__":
    # --- 1. Setup ---
    OUTPUT_DIR = "output"
    EXPERIMENT_PREFIX = "mlp_mnist_full"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}.log")
    setup_logging(log_file)

    # --- 2. Define Experiment Parameters ---
    mlp_params = {
        'n_hidden': 50,        # Number of neurons in the hidden layer
        'n_output': 10,        # 10 classes for digits 0-9
        'learning_rate': 0.01,
        'n_iter': 100,          # Number of epochs
        'shuffle': True,
        'random_state': 42
    }

    # --- 3. Run Experiment ---
    mlp, X_train, y_train, X_test, y_test, predictions = run_mlp_experiment(
        data_loader_func=load_mnist_full,
        experiment_name="Full MNIST",
        mlp_params=mlp_params
    )

    # --- 4. Visualize the Results ---
    logging.info("Generating visualizations for the full MNIST experiment...")
    digit_labels = [str(i) for i in range(10)]

    # Input data visualization
    animate_dataset_samples(
        X_train, y_train, n_samples=100,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_data_sample.gif")
    )

    # Result visualizations
    plot_learning_curve(
        mlp,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_learning_curve.png")
    )
    plot_confusion_matrix(
        y_test, predictions, display_labels=digit_labels,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_confusion_matrix.png")
    )
    plot_learned_weights(
        mlp,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_learned_weights.png")
    )
    plot_misclassified_examples(
        X_test, y_test, predictions,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_misclassified.png")
    )
    animate_learned_weights(
        mlp,
        output_filename=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_weights_evolution.gif")
    )

    # --- 5. Negative Test: Evaluate on Shifted Images ---
    X_test_shifted = create_shifted_dataset(X_test, shift_pixels=4)
    logging.info("Evaluating model on the SHIFTED test set...")
    shifted_predictions = mlp.predict(X_test_shifted)
    shifted_accuracy = np.mean(shifted_predictions == y_test)
    logging.info(f"--> Accuracy on SHIFTED test set: {shifted_accuracy * 100:.2f}%")
    logging.info("This demonstrates the MLP's lack of spatial invariance, a key problem solved by CNNs like LeNet-5.")

    # Visualize why the accuracy dropped
    visualize_shifted_test(
        X_test, X_test_shifted, y_test, shifted_predictions,
        output_filename_base=os.path.join(OUTPUT_DIR, f"{EXPERIMENT_PREFIX}_shifted_test.png")
    )

    logging.info("--- Full MNIST Experiment Complete ---")