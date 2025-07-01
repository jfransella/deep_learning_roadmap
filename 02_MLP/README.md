# 02: The Multi-Layer Perceptron (MLP)

## How to Run

Each experiment is a self-contained script. To run an experiment, simply execute its Python file from the root of the `02_MLP` directory. All generated plots and animations will be saved to the `output/` directory.

1.  **Set up the Environment**

    It is highly recommended to use a Python virtual environment to keep dependencies isolated. From the `02_MLP` directory, run:

    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (the command differs by OS)
    # On Windows:
    .\venv\Scripts\activate
    ```

2.  **Install Dependencies**

    Once your virtual environment is active, install the required packages from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run an Experiment**

    You can now run any of the training scripts:

    ```bash
    # Run the multi-class MNIST (0-9) task
    python src/train_mnist.py

    # Run the non-linearly separable concentric circles task
    python src/train_circles.py
    ```

## Project Structure

*   `src/model.py`: Contains the `MLP` class implementation from scratch using NumPy.
*   `src/data_loader.py`: Functions to fetch and prepare the full MNIST and concentric circles datasets.
*   `src/visualize.py`: A comprehensive suite of functions for plotting and creating animations.
*   `src/experiment_runner.py`: The core, reusable logic for running a training and evaluation experiment.
*   `src/train_*.py`: The main executable scripts for each specific experiment.
*   `output/`: The directory where all generated visualizations are saved.

## Model Summary

This project implements a **Multi-Layer Perceptron (MLP)** from scratch using Python and NumPy. The MLP is a foundational feedforward neural network that builds upon the single Perceptron by introducing one or more **hidden layers** between the input and output.

The key innovation of the MLP is its ability to learn **non-linear relationships** in data, a critical limitation of the single Perceptron. This is achieved through the combination of hidden layers and non-linear activation functions (like the sigmoid function used here). This project demonstrates how the MLP, trained with the **backpropagation** algorithm, can solve complex classification tasks that were previously impossible for simpler linear models.

## The Hypothesis

Our investigation was guided by three core hypotheses:
1.  The MLP can successfully solve the **non-linearly separable** concentric circles problem, a task where the single Perceptron provably failed.
2.  The MLP architecture can be extended to handle **multi-class classification**, allowing it to learn to distinguish all 10 MNIST digits simultaneously in a single training run with high accuracy.
3.  Despite its power, the MLP has a fundamental weakness for image tasks: a **lack of spatial invariance**. We hypothesize that a model trained on centered digits will fail when tested on digits that have been shifted, motivating the need for more advanced architectures like Convolutional Neural Networks (CNNs).

## Key Results

### Success Cases
*   **Concentric Circles**: Achieved **100% accuracy** on the test set. The `mlp_circles_decision_boundary.png` visualization clearly shows the model learning a non-linear, circular boundary to perfectly separate the two classes. This confirms our first hypothesis.
*   **Full MNIST (10-Class)**: Achieved **over 96% accuracy** on the test set, demonstrating the MLP's effectiveness as a powerful multi-class classifier. The `mlp_mnist_full_learned_weights.png` visualization shows that the hidden layer neurons learn to recognize various digit-like strokes and patterns.

### The Negative Test (Lack of Spatial Invariance)
*   **Shifted MNIST Test**: After training on centered digits, the model's accuracy dropped from **~96%** on the original test set to **under 50%** on a test set where the images were shifted by just a few pixels.
*   **Visualization**: The `mlp_mnist_full_shifted_test_misclassified.png` plot provides clear visual evidence of this failure, showing the model consistently misclassifying recognizable digits simply because they are not in the exact position they were during training. This confirms our third hypothesis and provides a powerful motivation for the convolutional layers in our next project, LeNet-5.

## Lessons Learned

*   **The Power of Hidden Layers**: The introduction of a single hidden layer and a non-linear activation function is all that's needed to break the "tyranny of linearity" and solve problems the Perceptron could not.
*   **Generalization to Multi-Class**: The MLP architecture, combined with a softmax output layer, provides a robust and effective framework for multi-class classification problems.
*   **The Importance of Spatial Invariance**: The shifted MNIST experiment was a critical lesson. It proved that while an MLP can learn *what* a pattern is, it doesn't learn to recognize it *wherever* it appears in an image. This limitation is the primary reason why CNNs, which are designed for spatial invariance, became the standard for computer vision tasks.

This project successfully demonstrates both the power and the limitations of the classic MLP, setting the stage perfectly for exploring the architectural innovations of LeNet-5.