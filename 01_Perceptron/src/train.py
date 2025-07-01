import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_mnist_0_vs_1
from src.model import Perceptron
# Import all visualization functions we've created
from src.visualize import (
    animate_dataset_samples,
    plot_learning_curve,
    plot_learned_weights,
    plot_misclassified_examples,
    plot_confusion_matrix,
    animate_learned_weights
)


# --- 1. Data Loading and Preparation ---
print("Step 1: Loading and preparing data...")
X, y = load_mnist_0_vs_1()

# Split data into training and testing sets to evaluate generalization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")


# --- 2. Input Data Visualization ---
print("\nStep 2: Visualizing the training data...")
# Create an animation of the input data to get a feel for it
animate_dataset_samples(X_train, y_train, n_samples=100, output_filename="training_data_sample.gif")


# --- 3. Model Training ---
print("\nStep 3: Training the Perceptron model...")
# Instantiate and train our Perceptron model on the training data
perceptron = Perceptron(learning_rate=0.01, n_iter=20)
perceptron.fit(X_train, y_train)
print("Model training complete.")


# --- 4. Model Evaluation ---
print("\nStep 4: Evaluating the model on the unseen test set...")
# Make predictions on the test set
predictions = perceptron.predict(X_test)
# Calculate the accuracy score
accuracy = np.mean(predictions == y_test)
print(f"--> Accuracy on the test set: {accuracy * 100:.2f}%")


# --- 5. Results Visualization ---
print("\nStep 5: Generating visualizations of the results...")
# Plot the learning curve (errors per epoch)
plot_learning_curve(perceptron)

# Plot the confusion matrix to see a detailed performance breakdown
plot_confusion_matrix(y_test, predictions)

# Show the final learned weights as an image
plot_learned_weights(perceptron)

# Show a sample of images the model got wrong
plot_misclassified_examples(X_test, y_test, predictions)

# Create an animation of the weights evolving over each epoch
animate_learned_weights(perceptron, output_filename="perceptron_weights_evolution.gif")

print("\n--- Project 01 Complete ---")