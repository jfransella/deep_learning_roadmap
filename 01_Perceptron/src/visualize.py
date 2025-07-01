import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.animation as animation

def plot_decision_boundary(X, y, model, title=""):
    """
    Plots the decision boundary for a 2D dataset.
    NOTE: This function is only for 2-dimensional data like the OR gate.
    """
    fig = plt.figure()
    plt.title(title)
    
    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap('viridis'), marker='o', edgecolors='k')
    
    # Plot the decision boundary for each step in history
    for i, (weights, bias) in enumerate(model.weights_history):
        # The line equation is w1*x1 + w2*x2 + b = 0
        # We can solve for x2: x2 = (-w1*x1 - b) / w2
        
        # Avoid division by zero if a weight is zero
        if weights[1] != 0:
            x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            
            x2_at_x1_min = (-weights[0] * x1_min - bias) / weights[1]
            x2_at_x1_max = (-weights[0] * x1_max - bias) / weights[1]
            
            # Use a different style for the final epoch's line
            linestyle = '--' if i < len(model.weights_history) - 1 else '-'
            linewidth = 1 if i < len(model.weights_history) - 1 else 2
            
            plt.plot([x1_min, x1_max], [x2_at_x1_min, x2_at_x1_max], linestyle, linewidth=linewidth, label=f'Epoch {i+1}')

    plt.xlabel("Input Feature 1")
    plt.ylabel("Input Feature 2")
    plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_learned_weights(model):
    """
    Visualizes the learned weights of the Perceptron as an image.
    The number of weights must be a perfect square (e.g., 784 -> 28x28).
    """
    # Reshape the weights vector into a square image
    image_size = int(np.sqrt(model.weights.shape[0]))
    weights_image = model.weights.reshape(image_size, image_size)
    
    plt.figure()
    plt.imshow(weights_image, cmap='viridis')
    plt.title("Perceptron's Learned Weights")
    plt.colorbar(label="Weight Strength")
    plt.show()

def plot_misclassified_examples(X_test, y_test, predictions, n=10):
    """
    Plots a selection of images that the model misclassified.
    
    Parameters:
    - X_test: The test data features.
    - y_test: The true labels for the test data.
    - predictions: The model's predictions on the test data.
    - n: The number of misclassified examples to show.
    """
    misclassified_mask = (predictions != y_test)
    X_misclassified = X_test[misclassified_mask]
    y_misclassified_true = y_test[misclassified_mask]
    y_misclassified_pred = predictions[misclassified_mask]
    
    # Take a sample of the misclassified images
    num_to_show = min(n, len(X_misclassified))
    if num_to_show == 0:
        print("No misclassified examples to show. Perfect accuracy!")
        return
        
    sample_indices = np.random.choice(len(X_misclassified), num_to_show, replace=False)
    
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 5))
    # Handle case where there's only one subplot
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
        
    plt.show()
    
def plot_learning_curve(model):
    """
    Plots the number of misclassifications per epoch.
    """
    plt.figure()
    plt.plot(range(1, len(model.errors_history) + 1), model.errors_history, marker='o')
    plt.title('Perceptron Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Misclassifications on Training Data')
    plt.grid(True)
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred):
    """
    Plots a confusion matrix using the true labels and model predictions.
    """
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=['Digit 0', 'Digit 1'],
        cmap='Blues',
        normalize=None # Can be 'true', 'pred', or 'all' to show percentages
    )
    plt.title('Confusion Matrix')
    plt.show()
    
def animate_learned_weights(model, output_filename="perceptron_weights.gif"):
    """
    Creates and saves an animation of the learned weights over epochs.
    """
    fig, ax = plt.subplots()
    plt.title("Evolution of Learned Weights")
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")

    image_size = int(np.sqrt(model.weights_history[0].shape[0]))
    
    # Initial image setup
    im = ax.imshow(model.weights_history[0].reshape(image_size, image_size), cmap='viridis', animated=True)
    fig.colorbar(im, label="Weight Strength")
    
    # This function will be called for each frame of the animation
    def update(frame):
        weights_image = model.weights_history[frame].reshape(image_size, image_size)
        im.set_array(weights_image)
        ax.set_title(f"Evolution of Learned Weights (Epoch: {frame + 1})")
        return im,

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(model.weights_history), interval=200, blit=True)
    
    # Save the animation as a GIF
    print(f"Saving animation to {output_filename}...")
    anim.save(output_filename, writer='pillow')
    print("Animation saved.")
    plt.close() # Close the plot window to prevent it from displaying statically
    
def animate_dataset_samples(X, y, n_samples=50, output_filename="dataset_sample.gif"):
    """
    Creates and saves an animation flashing random samples from a dataset.
    """
    # Select a random subset of the data to animate
    sample_indices = np.random.choice(len(X), n_samples, replace=False)
    
    fig, ax = plt.subplots()
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    
    image_size = int(np.sqrt(X.shape[1]))
    
    # Set up the initial image
    initial_image = X[sample_indices[0]].reshape(image_size, image_size)
    im = ax.imshow(initial_image, cmap='gray', animated=True)
    
    # This function is called for each frame
    def update(frame):
        sample_index = sample_indices[frame]
        image = X[sample_index].reshape(image_size, image_size)
        label = y[sample_index]
        
        im.set_array(image)
        ax.set_title(f"Sample Dataset Image (Label: {label})")
        return im,

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=n_samples, interval=300, blit=True)
    
    # Save the animation as a GIF
    print(f"Saving dataset sample animation to {output_filename}...")
    anim.save(output_filename, writer='pillow')
    print("Animation saved.")
    plt.close() # Close the plot window
    
def animate_scatter_reveal(X, y, n_frames=200, output_filename="scatter_reveal.gif"):
    """
    Creates an animation that sequentially reveals points in a scatter plot.
    """
    fig, ax = plt.subplots()
    
    # Set the plot limits based on the data
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Setup the initial empty plot
    ax.set_title("Revealing the Concentric Circles Dataset")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    # This function is called for each frame
    def update(frame):
        # Clear the previous frame's points to redraw
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Revealing the Dataset ({frame+1}/{n_frames} points)")
        
        # Plot all points up to the current frame
        current_X = X[:frame+1]
        current_y = y[:frame+1]
        ax.scatter(current_X[:, 0], current_X[:, 1], c=current_y, cmap='viridis', edgecolors='k')
        
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    
    # Save the animation as a GIF
    print(f"Saving scatter reveal animation to {output_filename}...")
    anim.save(output_filename, writer='pillow')
    print("Animation saved.")
    plt.close() # Close the plot window