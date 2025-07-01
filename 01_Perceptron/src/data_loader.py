from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist_0_vs_1():
    """
    Loads the MNIST dataset and filters it to only include digits 0 and 1.
    
    Returns
    -------
    X : numpy.ndarray
        The image data (features).
    y : numpy.ndarray
        The labels (0 or 1).
    """
    # Load data from https://www.openml.org/d/554
    print("Fetching MNIST data... This may take a moment.")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Filter for digits 0 and 1
    is_0_or_1 = (mnist.target == '0') | (mnist.target == '1')
    X = mnist.data[is_0_or_1]
    y = mnist.target[is_0_or_1]
    
    # Normalize pixel values to be between 0 and 1
    # This helps with training stability.
    X = X / 255.0
    
    # Convert labels from strings ('0', '1') to integers (0, 1)
    y = y.astype(int)
    
    print("Data loaded and prepared successfully.")
    return X, y

def load_mnist_6_vs_9():
    """
    Loads the MNIST dataset and filters it to only include digits 6 and 9.
    """
    print("Fetching MNIST data for '6' vs '9' task...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Filter for digits 6 and 9
    is_6_or_9 = (mnist.target == '6') | (mnist.target == '9')
    X = mnist.data[is_6_or_9]
    y = mnist.target[is_6_or_9]
    
    # Normalize pixel values
    X = X / 255.0
    
    # Convert labels from strings to integers (e.g., class 0 for '6', class 1 for '9')
    y = np.where(y == '6', 0, 1)
    
    print("Data loaded and prepared successfully.")
    return X, y