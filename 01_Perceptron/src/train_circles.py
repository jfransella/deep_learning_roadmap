import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from src.model import Perceptron
# Import our new animation function
from src.visualize import plot_decision_boundary, plot_learning_curve, plot_confusion_matrix, animate_scatter_reveal

# --- 1. Generate the Concentric Circles Dataset ---
X, y = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- 2. Input Data Visualization ---
print("\nVisualizing the 'circles' training data...")
# Call the new animation function on the training data
animate_scatter_reveal(X_train, y_train, output_filename="circles_data_reveal.gif")


# --- 3. Train the Perceptron ---
print("\nTraining the Perceptron...")
perceptron = Perceptron(learning_rate=0.1, n_iter=100)
perceptron.fit(X_train, y_train)


# --- 4. Evaluate and Visualize Results ---
print("\nEvaluating and generating result plots...")
predictions = perceptron.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Final accuracy on test set: {accuracy * 100:.2f}%")

plot_learning_curve(perceptron)
plot_confusion_matrix(y_test, predictions)
plot_decision_boundary(X_train, y_train, perceptron, title="Perceptron Failing on Concentric Circles")