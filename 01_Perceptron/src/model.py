import numpy as np

class Perceptron:
    """
    A simple Perceptron classifier based on the original model by Rosenblatt.

    This implementation is designed for binary classification tasks. The Perceptron
    learns a linear decision boundary to separate two classes.

    Parameters
    ----------
    learning_rate : float
      Controls the step size for weight updates during training (typically between 0.0 and 1.0).
      A larger value means the model learns faster but may overshoot the optimal weights.
    n_iter : int
      The number of passes over the training dataset, also known as epochs.

    Attributes
    ----------
    weights_ : 1d-array
      The learned weights for each input feature after fitting the model.
    bias_ : float
      The learned bias unit after fitting. The bias allows the decision
      boundary to shift, which is crucial for finding the optimal separator.
    errors_history_ : list
      A list containing the number of misclassifications (errors) in each epoch.
      This is used to generate the learning curve visualization.

    """
    def __init__(self, learning_rate=0.01, n_iter=10):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.errors_history = []

    def fit(self, X, y):
        """
        Fit the model to the training data.

        This method adjusts the weights and bias of the Perceptron over
        a specified number of epochs. It iterates through the entire dataset
        multiple times, and for each sample, it makes a prediction, calculates
        the error, and updates the model's parameters to correct that error.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        The training input vectors, where n_samples is the number of samples
        and n_features is the number of features.
        y : array-like, shape = [n_samples]
        The true target values (must be 0 or 1).

        """
        # Get the number of samples and features from the input data shape.
        n_samples, n_features = X.shape

        # --- Initialization ---
        # Initialize the weights vector with zeros. There will be one weight for each feature.
        self.weights = np.zeros(n_features)
        # Initialize the bias term to zero.
        self.bias = 0.0
        # Initialize lists to store the history of learning for later visualization.
        self.errors_history = []
        self.weights_history = []

        # --- Training Loop ---
        # The main training loop, which iterates over the dataset `n_iter` times (epochs).
        for _ in range(self.n_iter):
            # Reset the error counter for the current epoch.
            errors = 0
            
            # The inner loop iterates through each individual sample in the training data.
            # 'idx' is the index, and 'x_i' is the feature vector for a single sample.
            for idx, x_i in enumerate(X):
                
                # Step 1: Make a prediction for the current sample.
                # The predict method will return either 0 or 1.
                prediction = self.predict(x_i)

                # Step 2: Calculate the update value based on the Perceptron update rule.
                # The 'update' is proportional to the error (target - prediction).
                # If the prediction is correct (e.g., target is 1, prediction is 1), the error is 0,
                # and therefore the update value is 0.
                update = self.learning_rate * (y[idx] - prediction)
                
                # An update of 0 means the prediction for this sample was correct.
                # If the update is non-zero, it means a misclassification occurred.
                if update != 0:
                    errors += 1
                
                # Step 3: Update the weights and the bias.
                # If 'update' is 0, this operation has no effect.
                # If 'update' is not 0, the weights are nudged in the direction
                # that would make the prediction more correct.
                self.weights += update * x_i
                self.bias += update
            
            # --- History Tracking ---
            # After iterating through all samples, store the results for this epoch.
            self.errors_history.append(errors)
            self.weights_history.append((self.weights.copy(), self.bias))
        
        return self

    def _net_input(self, X):
        """
        Calculate the net input (also called the activation).
        This is the weighted sum of the inputs plus the bias.
        z = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b
        """
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """
        Return the class label for a given input X after applying the unit step function.
        This serves as the activation function of the Perceptron.
        """
        # If the net input is >= 0, the Perceptron "fires" and predicts class 1.
        # Otherwise, it remains "inactive" and predicts class 0.
        return np.where(self._net_input(X) >= 0.0, 1, 0)