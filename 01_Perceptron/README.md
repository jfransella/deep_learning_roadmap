# 01: The Perceptron

## How to Run

Each experiment is a self-contained script. To run an experiment, simply execute its Python file from the root of the `01_Perceptron` directory. All generated plots and animations will be saved to the `output/` directory.

```bash
# Run the linearly separable MNIST '0' vs '1' task
python src/train_mnist_0_vs_1.py

# Run the more difficult MNIST '6' vs '9' task
python src/train_mnist_6_vs_9.py

# Run the even more difficult MNIST '3' vs '5' experiment
python src/train_mnist_3_vs_5.py

# Run the non-linearly separable concentric circles task
python src/train_circles.py

```

## Project Structure

*   `src/model.py`: Contains the `Perceptron` class implementation.
*   `src/data_loader.py`: A generalized function to fetch and prepare binary MNIST datasets.
*   `src/visualize.py`: A comprehensive suite of functions for plotting and creating animations.
*   `src/experiment_runner.py`: The core, reusable logic for running a training and evaluation experiment.
*   `src/train_*.py`: The main executable scripts for each specific experiment.
*   `output/`: The directory where all generated visualizations are saved.

## Model Summary

This project is a Python implementation of the **Perceptron**, the original artificial neuron developed by Frank Rosenblatt in 1958. As one of the earliest algorithms for supervised learning, the Perceptron is a single-layer neural network that serves as a linear binary classifier. It learns a decision boundary to separate two classes of data. This implementation explores both its surprising capabilities on high-dimensional data and its fundamental limitations on non-linear problems.

## The Hypothesis

Our investigation was guided by three core hypotheses:
1.  The Perceptron can succeed on a real-world, high-dimensional task if the underlying data is mostly linearly separable (e.g., distinguishing handwritten MNIST '0's from '1's).
2.  It will also perform surprisingly well on more difficult, but still mostly separable, tasks (e.g., '6' vs '9').
3.  The Perceptron will fail completely on non-linearly separable problems (like the concentric circles dataset), proving its limitations and motivating the need for more complex models.

## Key Results

### Success Cases
*   **MNIST '0' vs '1'**: Achieved **over 99.8%** accuracy on the test set, demonstrating the model's ability to find a separating hyperplane in a 784-dimensional feature space.
*   **MNIST '6' vs '9'**: Surprisingly achieved **over 99.6%** accuracy, which led to a deeper understanding of high-dimensional spaces.
*   **MNIST '3' vs '5'**: Also achieved high accuracy (**over 97%**), further proving the model's effectiveness on linearly separable data.

### Failure Case (Concentric Circles)
*   **Accuracy**: Achieved only **~50%** accuracy, which is no better than random guessing.
*   **Learning Curve**: The `circles_learning_curve.png` plot showed the number of misclassifications oscillating wildly, never converging to zero. This is a classic sign of a model failing to learn a non-linear pattern.
*   **Decision Boundary**: The `circles_decision_boundary.png` plot provided clear visual proof that a single straight line cannot separate the inner circle from the outer ring.

## Key Visualizations

For each experiment, a rich set of visualizations is automatically saved to the `output/` directory, prefixed with the experiment name (e.g., `mnist_0_vs_1_...`).
*   **`*_learning_curve.png`**: Shows the model's error rate per epoch, indicating if the model is learning.
*   **`*_confusion_matrix.png`**: Details the true vs. predicted labels, giving a precise view of accuracy.
*   **`*_learned_weights.png`**: Visualizes the final weights as an image, revealing what features the model learned to recognize.
*   **`*_weights_evolution.gif`**: An animation showing the learned weights vector resolving from noise into a clear template over all epochs.

## Lessons Learned

* **The Power of Linear Models**: Even a simple, single-neuron model from the 1950s is surprisingly powerful. It can achieve near-perfect accuracy on real-world problems like MNIST if the classes are linearly separable.
* **The "Blessing of Dimensionality"**: Our most surprising takeaway came from the '6' vs '9' experiment. We learned that two classes that seem non-linear to the human eye can become linearly separable in a high-dimensional space. The Perceptron discovered a simple linear strategy that we did not anticipate.
* **Visualization is Essential for Understanding**: An accuracy score only tells part of the story. Visualizing the learning curve, the final learned weights, and the decision boundary gave us a much deeper, more intuitive understanding of *how* the model was working and *why* it was failing.
* **The Limits of Linearity are Absolute**: The concentric circles experiment provided definitive, visual proof that the Perceptron cannot solve non-linear problems, motivating the need for our next project: the Multi-Layer Perceptron.