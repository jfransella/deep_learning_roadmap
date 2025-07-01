"""
train_mnist_0_vs_1.py

Train and evaluate the Perceptron on the MNIST 0 vs 1 task.
Part of a project to explore the history and advances in neural network AI.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_mnist_0_vs_1
from src.model import Perceptron
from src.visualize import (
    animate_dataset_samples,
    plot_learning_curve,
    plot_learned_weights,
    plot_misclassified_examples,
    plot_confusion_matrix,
    animate_learned_weights
)

# 1. Data Loading and Preparation
print("[Main] Step 1: Loading and preparing data...")
X, y = load_mnist_0_vs_1()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[Main] Data prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")

# 2. Input Data Visualization
print("\n[Main] Step 2: Visualizing the training data...")
animate_dataset_samples(X_train, y_train, n_samples=100, output_filename="training_data_sample.gif")

# 3. Model Training
print("\n[Main] Step 3: Training the Perceptron model...")
perceptron = Perceptron(learning_rate=0.01, n_iter=20)
perceptron.fit(X_train, y_train)
print("[Main] Model training complete.")

# 4. Model Evaluation
print("\n[Main] Step 4: Evaluating the model on the unseen test set...")
predictions = perceptron.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"[Main] --> Accuracy on the test set: {accuracy * 100:.2f}%")

# 5. Results Visualization
print("\n[Main] Step 5: Generating visualizations of the results...")
plot_learning_curve(perceptron)
plot_confusion_matrix(y_test, predictions, display_labels=['Digit 0', 'Digit 1'])
plot_learned_weights(perceptron)
plot_misclassified_examples(X_test, y_test, predictions, display_labels=['0', '1'])
animate_learned_weights(perceptron, output_filename="perceptron_weights_evolution.gif")

print("\n[Main] --- Project 01 Complete ---")
