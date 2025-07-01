"""
visualize.py

Visualization utilities for Multi-Layer Perceptron (MLP) training and evaluation.
Part of a project to explore the history and advances in neural network AI.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model: 'MLP',
    title: str = "",
    resolution: float = 0.02,
    output_filename: Optional[str] = None
) -> None:
    """
    Plot the decision boundary for a 2D dataset.

    Parameters
    ----------
    X : np.ndarray
        Input features (must be 2D).
    y : np.ndarray
        Target labels.
    model : 'MLP'
        The trained MLP model instance.
    title : str, optional
        Title for the plot.
    resolution : float, optional
        Resolution of the meshgrid for plotting the boundary.
    output_filename : str, optional
        If provided, saves the plot to this file.
    """
    print("\n[Visualize] Generating decision boundary plot...")
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker='o', label=f'Class {cl}',
                    edgecolor='black')

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc='upper left')
    plt.grid(True)

    if output_filename:
        print(f"[Visualize] Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    print("[Visualize] Decision boundary plot complete.")

def plot_learned_weights(model: 'MLP', n_cols: int = 10, output_filename: Optional[str] = None) -> None:
    """
    Visualize the learned weights of the MLP's hidden layer as images.

    Parameters
    ----------
    model : 'MLP'
        The trained MLP model instance.
    n_cols : int, optional
        Number of columns in the subplot grid.
    output_filename : str, optional
        If provided, saves the plot to this file.
    """
    print("\n[Visualize] Generating learned weights plot for hidden layer...")
    n_hidden = model.n_hidden
    n_rows = int(np.ceil(n_hidden / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    fig.suptitle("Learned Weights of Hidden Layer Neurons", fontsize=16)
    axes = axes.flatten()
    image_size = int(np.sqrt(model.w1.shape[0]))

    for i in range(n_hidden):
        weights = model.w1[:, i]
        image = weights.reshape(image_size, image_size)
        ax = axes[i]
        ax.imshow(image, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for i in range(n_hidden, len(axes)):
        axes[i].axis('off')

    if output_filename:
        print(f"[Visualize] Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    print("[Visualize] Learned weights plot complete.")

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
    """
    print("\n[Visualize] Generating misclassified examples plot...")
    misclassified_mask = (predictions != y_test)
    X_misclassified = X_test[misclassified_mask]
    y_misclassified_true = y_test[misclassified_mask]
    y_misclassified_pred = predictions[misclassified_mask]
    num_to_show = min(n, len(X_misclassified))
    if num_to_show == 0:
        print("[Visualize] No misclassified examples to show. Perfect accuracy!")
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
        true_label = y_misclassified_true[idx]
        pred_label = y_misclassified_pred[idx]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis('off')

    if output_filename:
        print(f"[Visualize] Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    print("[Visualize] Misclassified examples plot complete.")

def plot_learning_curve(model: 'MLP', output_filename: Optional[str] = None) -> None:
    """
    Plot the cost per epoch (learning curve).
    """
    print("\n[Visualize] Generating learning curve (cost) plot...")
    plt.figure()
    plt.plot(range(1, len(model.cost_history) + 1), model.cost_history, marker='o')
    plt.title('MLP Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.grid(True)
    if output_filename:
        print(f"[Visualize] Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    print("[Visualize] Learning curve plot complete.")

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    display_labels: Optional[List[str]] = None,
    output_filename: Optional[str] = None
) -> None:
    """
    Plot a confusion matrix using the true labels and model predictions.
    """
    print("\n[Visualize] Generating confusion matrix plot...")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=display_labels,
        cmap='Blues',
        normalize=None
    )
    plt.title('Confusion Matrix')
    if output_filename:
        print(f"[Visualize] Saving plot to '{output_filename}'...")
        plt.savefig(output_filename)
    plt.show()
    print("[Visualize] Confusion matrix plot complete.")

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
    X : np.ndarray
        Dataset features.
    y : np.ndarray
        Dataset labels.
    n_samples : int, optional
        Number of samples to animate (default=50).
    output_filename : str, optional
        Output GIF filename (default: 'dataset_sample.gif').
    """
    print(f"\n[Visualize] Creating dataset sample animation ('{output_filename}')...")
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
    print(f"[Visualize] Animation saved to '{output_filename}'.")
    plt.close()

def animate_learned_weights(model: 'MLP', n_cols: int = 10, output_filename: Optional[str] = None) -> None:
    """
    Create and save an animation of the learned weights of the hidden layer over epochs.

    Parameters
    ----------
    model : 'MLP'
        The trained MLP model instance with weights_history.
    n_cols : int, optional
        Number of columns in the subplot grid.
    output_filename : str, optional
        If provided, saves the animation to this file.
    """
    print(f"\n[Visualize] Creating learned weights animation ('{output_filename}')...")
    if not hasattr(model, 'weights_history') or not model.weights_history:
        print("[Visualize] 'weights_history' not found in model. Skipping animation.")
        return

    n_hidden = model.n_hidden
    n_rows = int(np.ceil(n_hidden / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten()

    try:
        image_size = int(np.sqrt(model.w1.shape[0]))
        if image_size * image_size != model.w1.shape[0]:
            raise ValueError
    except (ValueError, AttributeError):
        print("[Visualize] Weights cannot be reshaped into square images. Skipping animation.")
        plt.close(fig)
        return

    ims = []
    for i in range(n_hidden):
        ax = axes[i]
        w1_initial = model.weights_history[0][0]
        image = w1_initial[:, i].reshape(image_size, image_size)
        im = ax.imshow(image, cmap='viridis', animated=True)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)

    for i in range(n_hidden, len(axes)):
        axes[i].axis('off')

    def update(frame):
        fig.suptitle(f"Evolution of Hidden Layer Weights (Epoch: {frame + 1})", fontsize=16)
        w1_frame = model.weights_history[frame][0]
        for i in range(n_hidden):
            ims[i].set_array(w1_frame[:, i].reshape(image_size, image_size))
        return ims

    anim = animation.FuncAnimation(fig, update, frames=len(model.weights_history), interval=200, blit=False)
    if output_filename:
        anim.save(output_filename, writer='pillow')
        print(f"[Visualize] Animation saved to '{output_filename}'.")
    plt.close(fig)

def visualize_shifted_test(
    X_test_original: np.ndarray,
    X_test_shifted: np.ndarray,
    y_test: np.ndarray,
    shifted_predictions: np.ndarray,
    n_examples: int = 5,
    output_filename_base: Optional[str] = None
) -> None:
    """
    Visualizes the effect of shifting images on the MLP's performance.

    Generates two plots:
    1. A comparison of original vs. shifted images.
    2. A sample of misclassified examples from the shifted set.

    Parameters
    ----------
    X_test_original : np.ndarray
        The original, unshifted test images.
    X_test_shifted : np.ndarray
        The shifted test images.
    y_test : np.ndarray
        The true labels for the test set.
    shifted_predictions : np.ndarray
        The model's predictions on the shifted test set.
    n_examples : int, optional
        The number of examples to show in each plot.
    output_filename_base : str, optional
        The base filename for saving the output plots.
    """
    print("\n[Visualize] Generating visualization for the shifted dataset test...")
    image_size = int(np.sqrt(X_test_original.shape[1]))

    # --- Part 1: Show Misclassified Shifted Examples ---
    misclassified_mask = (shifted_predictions != y_test)
    X_misclassified = X_test_shifted[misclassified_mask]
    y_misclassified_true = y_test[misclassified_mask]
    y_misclassified_pred = shifted_predictions[misclassified_mask]

    num_to_show = min(n_examples, len(X_misclassified))
    if num_to_show > 0:
        sample_indices = np.random.choice(len(X_misclassified), num_to_show, replace=False)
        fig, axes = plt.subplots(1, num_to_show, figsize=(15, 4))
        if num_to_show == 1: axes = [axes]
        fig.suptitle('Misclassified Examples from SHIFTED Test Set', fontsize=16)

        for i, ax in enumerate(axes):
            idx = sample_indices[i]
            image = X_misclassified[idx].reshape(image_size, image_size)
            true_label = y_misclassified_true[idx]
            pred_label = y_misclassified_pred[idx]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            ax.axis('off')

        if output_filename_base:
            filename = output_filename_base.replace('.png', '_misclassified_shifted.png')
            print(f"[Visualize] Saving misclassified plot to '{filename}'...")
            plt.savefig(filename)
        plt.show()
    else:
        print("[Visualize] No misclassified examples in the shifted set to show.")

    print("[Visualize] Shifted dataset visualization complete.")
    plt.close(fig)