# 01: The Perceptron

## Model Summary

This project is a Python implementation of the **Perceptron**, the original artificial neuron developed by Frank Rosenblatt in 1958. As one of the earliest algorithms for supervised learning, the Perceptron is a single-layer neural network that serves as a linear binary classifier. It learns a decision boundary to separate two classes of data. This implementation explores both its surprising capabilities on high-dimensional data and its fundamental limitations on non-linear problems.

## The Hypothesis

Our investigation was guided by three core hypotheses:
1.  The Perceptron can successfully learn simple, linearly separable problems like the `OR` logic gate.
2.  It can also succeed on a real-world, high-dimensional task if the underlying data is mostly linearly separable (e.g., distinguishing handwritten MNIST '0's from '1's).
3.  The Perceptron will fail on non-linearly separable problems (like the concentric circles dataset), proving its limitations and motivating the need for more complex models.

## Key Results

### Success Cases
* **Logic Gates**: Achieved 100% accuracy on the `OR` gate, successfully learning the decision boundary in just a few epochs.
* **MNIST '0' vs '1'**: Achieved **99.83%** accuracy on the test set, demonstrating the model's ability to find a separating hyperplane in a 784-dimensional feature space.
* **MNIST '6' vs '9'**: Surprisingly achieved **99.67%** accuracy, which led to a deeper understanding of high-dimensional spaces.

#### Key Visualizations for Success Cases:
* `training_data_sample.gif`: An animation showing the variability in the MNIST training data.
* `perceptron_weights_evolution.gif`: An animation showing the learned weights vector resolving from noise into a clear template over 20 epochs.
* **Confusion Matrix**: Showed near-perfect classification with very few false positives or false negatives.

### Failure Case (Concentric Circles)
* **Accuracy**: Achieved only **~50%** accuracy, which is no better than random guessing.
* **Learning Curve**: The visualization showed the number of misclassifications oscillating wildly, never converging to zero. This is a classic sign of a model failing to learn a non-linear pattern.
* **Decision Boundary**: The final plot provided clear visual proof that a single straight line cannot separate the inner circle from the outer ring.

## Lessons Learned

* **The Power of Linear Models**: Even a simple, single-neuron model from the 1950s is surprisingly powerful. It can achieve near-perfect accuracy on real-world problems like MNIST if the classes are linearly separable.
* **The "Blessing of Dimensionality"**: Our most surprising takeaway came from the '6' vs '9' experiment. We learned that two classes that seem non-linear to the human eye can become linearly separable in a high-dimensional space. The Perceptron discovered a simple linear strategy that we did not anticipate.
* **Visualization is Essential for Understanding**: An accuracy score only tells part of the story. Visualizing the learning curve, the final learned weights, and the decision boundary gave us a much deeper, more intuitive understanding of *how* the model was working and *why* it was failing.
* **The Limits of Linearity are Absolute**: The concentric circles experiment provided definitive, visual proof that the Perceptron cannot solve non-linear problems, motivating the need for our next project: the Multi-Layer Perceptron.