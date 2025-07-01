"""
visualize.py

Visualization utilities for perceptron training and evaluation.
Part of a project to explore the history and advances in neural network AI.
"""

import logging
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.animation as animation

def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model: 'Perceptron',
    title: str = "",
    output_filename: Optional[str] = None
) -> None:
    """
    Plot the decision boundary for a 2D dataset and the model's evolution.
    Only for 2D data (e.g., OR gate).

    Parameters
    ----------
    X : np.ndarray
        Input features (must be 2D).
    y : np.ndarray
        Target labels.
    model : 'Perceptron'
        The trained Perceptron model instance. It must have a `weights_history`.
    title : str, optional
        Title for the plot.
    output_filename : str, optional
        If provided, saves the plot to this file.
    """
    logging.info("Generating decision boundary plot...")
    fig = plt.figure()
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('viridis'), marker='o', edgecolors='k')
    for i, (weights, bias) in enumerate(model.weights_history):
        # Decision boundary: w1*x1 + w2*x2 + b = 0 => x2 = (-w1*x1 - b) / w2
        if weights[1] != 0:
            x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            x2_at_x1_min = (-weights[0] * x1_min - bias) / weights[1]
            x2_at_x1_max = (-weights[0] * x1_max - bias) / weights[1]
            linestyle = '--' if i < len(model.weights_history) - 1 else '-'
            linewidth = 1 if i < len(model.weights_history) - 1 else 2
            plt.plot([x1_min, x1_max], [x2_at_x1_min, x2_at_x1_max], linestyle, linewidth=linewidth, label=f'Epoch {i+1}')
    plt.xlabel("Input Feature 1")
    plt.ylabel("Input Feature 2")
    plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    plt.legend(loc="lower right")
    plt.grid(True)
    if output_filename:
        logging.info(f"Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    logging.info("Decision boundary plot complete.")

def plot_learned_weights(model: 'Perceptron', output_filename: Optional[str] = None) -> None:
    """
    Visualize the learned weights of the Perceptron as an image.
    The number of weights must be a perfect square (e.g., 784 -> 28x28).

    Parameters
    ----------
    model : 'Perceptron'
        The trained Perceptron model instance.
    output_filename : str, optional
        If provided, saves the plot to this file.
    """
    logging.info("Generating learned weights plot...")
    image_size = int(np.sqrt(model.weights.shape[0]))
    weights_image = model.weights.reshape(image_size, image_size)
    plt.figure()
    plt.imshow(weights_image, cmap='viridis')
    plt.title("Perceptron's Learned Weights")
    plt.colorbar(label="Weight Strength")
    if output_filename:
        logging.info(f"Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    logging.info("Learned weights plot complete.")

def plot_misclassified_examples(
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    n: int = 10,
    display_labels: Optional[List[str]] = None,
    output_filename: Optional[str] = None
) -> None:
    """
    Plot a sample of misclassified images.

    Parameters
    ----------
    X_test : array-like
        Test data features.
    y_test : array-like
        True labels for the test data.
    predictions : array-like
        Model predictions on the test data.
    n : int, optional
        Number of misclassified examples to show (default=10).
    display_labels : list of str, optional
        List of class names to display instead of integer labels.
        e.g., ['Digit 6', 'Digit 9'].
    output_filename : str, optional
        If provided, saves the plot to this file.
    """
    logging.info("Generating misclassified examples plot...")
    misclassified_mask = (predictions != y_test)
    X_misclassified = X_test[misclassified_mask]
    y_misclassified_true = y_test[misclassified_mask]
    y_misclassified_pred = predictions[misclassified_mask]
    num_to_show = min(n, len(X_misclassified))
    if num_to_show == 0:
        logging.info("No misclassified examples to show. Perfect accuracy!")
        return
    sample_indices = np.random.choice(len(X_misclassified), num_to_show, replace=False)
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 5))
    if num_to_show == 1:
        axes = [axes]
    fig.suptitle('Misclassified Examples', fontsize=16)
    image_size = int(np.sqrt(X_misclassified.shape[1]))
    for i, ax in enumerate(axes):
        idx = sample_indices[i]
        image = X_misclassified[idx].reshape(image_size, image_size)
        true_label_int = y_misclassified_true[idx]
        pred_label_int = y_misclassified_pred[idx]

        if display_labels:
            true_label_str = display_labels[true_label_int]
            pred_label_str = display_labels[pred_label_int]
        else:
            true_label_str = true_label_int
            pred_label_str = pred_label_int

        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {true_label_str}\nPred: {pred_label_str}")
        ax.axis('off')
    if output_filename:
        logging.info(f"Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    logging.info("Misclassified examples plot complete.")

def plot_learning_curve(model: 'Perceptron', output_filename: Optional[str] = None) -> None:
    """
    Plot the number of misclassifications per epoch (learning curve).

    Parameters
    ----------
    model : 'Perceptron'
        The trained Perceptron model instance. It must have an `errors_history`.
    output_filename : str, optional
        If provided, saves the plot to this file.
    """
    logging.info("Generating learning curve plot...")
    plt.figure()
    plt.plot(range(1, len(model.errors_history) + 1), model.errors_history, marker='o')
    plt.title('Perceptron Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Misclassifications on Training Data')
    plt.grid(True)
    if output_filename:
        logging.info(f"Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    logging.info("Learning curve plot complete.")

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    display_labels: Optional[List[str]] = None,
    output_filename: Optional[str] = None
) -> None:
    """
    Plot a confusion matrix using the true labels and model predictions.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Model's predicted labels.
    display_labels : list of str, optional
        Labels for the matrix axes. If None, uses unique labels from y_true/y_pred.
    output_filename : str, optional
        If provided, saves the plot to this file.
    """
    logging.info("Generating confusion matrix plot...")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=display_labels,
        cmap='Blues',
        normalize=None
    )
    plt.title('Confusion Matrix')
    if output_filename:
        logging.info(f"Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    logging.info("Confusion matrix plot complete.")

def animate_learned_weights(
    model: 'Perceptron',
    output_filename: str = "perceptron_weights.gif"
) -> None:
    """
    Create and save an animation of the learned weights over epochs.

    Parameters
    ----------
    model : Perceptron
        Trained perceptron model with weights_history.
    output_filename : str, optional
        Output GIF filename (default: 'perceptron_weights.gif').
    """
    logging.info(f"Creating learned weights animation ('{output_filename}')...")
    fig, ax = plt.subplots()
    plt.title("Evolution of Learned Weights")
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    weights = model.weights_history[0][0]
    image_size = int(np.sqrt(weights.shape[0]))
    im = ax.imshow(model.weights_history[0][0].reshape(image_size, image_size), cmap='viridis', animated=True)
    fig.colorbar(im, label="Weight Strength")
    def update(frame):
        weights_image = model.weights_history[frame][0].reshape(image_size, image_size)
        im.set_array(weights_image)
        ax.set_title(f"Evolution of Learned Weights (Epoch: {frame + 1})")
        return im,
    anim = animation.FuncAnimation(fig, update, frames=len(model.weights_history), interval=200, blit=True)
    anim.save(output_filename, writer='pillow')
    logging.info(f"Animation saved to '{output_filename}'.")
    plt.close()

def animate_dataset_samples(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 50,
    output_filename: str = "dataset_sample.gif"
) -> None:
    """
    Create and save an animation flashing random samples from a dataset.

    Parameters
    ----------
    X : array-like
        Dataset features.
    y : array-like
        Dataset labels.
    n_samples : int, optional
        Number of samples to animate (default=50).
    output_filename : str, optional
        Output GIF filename (default: 'dataset_sample.gif').
    """
    logging.info(f"Creating dataset sample animation ('{output_filename}')...")
    sample_indices = np.random.choice(len(X), n_samples, replace=False)
    fig, ax = plt.subplots()
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    image_size = int(np.sqrt(X.shape[1]))
    initial_image = X[sample_indices[0]].reshape(image_size, image_size)
    im = ax.imshow(initial_image, cmap='gray', animated=True)
    def update(frame):
        sample_index = sample_indices[frame]
        image = X[sample_index].reshape(image_size, image_size)
        label = y[sample_index]
        im.set_array(image)
        ax.set_title(f"Sample Dataset Image (Label: {label})")
        return im,
    anim = animation.FuncAnimation(fig, update, frames=n_samples, interval=300, blit=True)
    anim.save(output_filename, writer='pillow')
    logging.info(f"Animation saved to '{output_filename}'.")
    plt.close()

def animate_scatter_reveal(
    X: np.ndarray,
    y: np.ndarray,
    n_frames: int = 200,
    output_filename: str = "scatter_reveal.gif"
) -> None:
    """
    Create an animation that sequentially reveals points in a scatter plot.

    Parameters
    ----------
    X : array-like
        2D dataset features.
    y : array-like
        Dataset labels.
    n_frames : int, optional
        Number of frames in the animation (default=200).
    output_filename : str, optional
        Output GIF filename (default: 'scatter_reveal.gif').
    """
    logging.info(f"Creating scatter reveal animation ('{output_filename}')...")
    fig, ax = plt.subplots()
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Revealing the Concentric Circles Dataset")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Revealing the Dataset ({frame+1}/{n_frames} points)")
        current_X = X[:frame+1]
        current_y = y[:frame+1]
        ax.scatter(current_X[:, 0], current_X[:, 1], c=current_y, cmap='viridis', edgecolors='k')
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    anim.save(output_filename, writer='pillow')
    logging.info(f"Animation saved to '{output_filename}'.")
    plt.close()