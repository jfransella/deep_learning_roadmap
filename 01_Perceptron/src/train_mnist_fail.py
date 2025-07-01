import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_mnist_6_vs_9
from src.model import Perceptron
from src.visualize import (
    animate_dataset_samples,  # Make sure this is imported
    plot_learning_curve,
    plot_learned_weights,
    plot_misclassified_examples,
    plot_confusion_matrix
)

# --- 1. Load the difficult '6' vs '9' dataset ---
X, y = load_mnist_6_vs_9()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")


# --- 2. Input Data Visualization ---
print("\nVisualizing the '6' vs '9' training data...")
# Add the call to our animation function
animate_dataset_samples(X_train, y_train, n_samples=100, output_filename="6_vs_9_data_sample.gif")


# --- 3. Train the Perceptron ---
perceptron = Perceptron(learning_rate=0.01, n_iter=20)
perceptron.fit(X_train, y_train)
print("Model training complete.")


# --- 4. Evaluate the Model ---
print("\n--- Evaluation Results ---")
predictions = perceptron.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"--> Accuracy on the '6' vs '9' test set: {accuracy * 100:.2f}%")


# --- 5. Visualize the training results ---
print("\nGenerating visualizations...")
plot_learning_curve(perceptron)
plot_confusion_matrix(y_test, predictions)
plot_learned_weights(perceptron)
plot_misclassified_examples(X_test, y_test, predictions)