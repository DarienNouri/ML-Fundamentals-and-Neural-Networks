# Fundamentals of Machine Learning and Neural Networks

**Note:** All work in this repository is authored by Darien Nouri.

This repository contains a collection of Jupyter notebooks for various fundamental topics in machine learning and neural networks.

## File Structure

```text
├── 01_Bias_Variance_Tradeoff.ipynb
├── 02_LogisticRegression_Regularization.ipynb
├── 03_Algorithmic_Performance_Scaling.ipynb
├── 04_Perceptron.ipynb
├── 05_Linear_Separability.ipynb
├── 06_Softmax_Activation_Derivation.ipynb
├── 07_NeuralNetwork_Manual_Backpropagation.ipynb
├── 08_Weight_Initialization_DeadNeurons_LeaklyReLU.ipynb
├── 09_BatchNorm_Dropout.ipynb
├── 10_LearningRate_BatchSIze_Exploration.ipynb
```

## Notebooks

- **01_Bias_Variance_Tradeoff.ipynb**
  - Explores the bias-variance tradeoff in a regression problem.
  - Includes dataset generation, polynomial estimators, bias-variance tradeoff analysis, and model selection.

- **02_LogisticRegression_Regularization.ipynb**
  - Investigates regularization techniques in logistic regression using the IRIS dataset.
  - Covers parameter significance, regularization penalties, model fitting, and coefficient analysis.

- **03_Algorithmic_Performance_Scaling.ipynb**
  - Studies algorithmic performance scaling using a large classification dataset.
  - Involves dataset summary, model training, learning curves, training time analysis, and performance comparison.

- **04_Perceptron.ipynb**
  - Implements the perceptron algorithm and explores different loss functions.
  - Includes dataset generation, training perceptron models, accuracy evaluation, and performance comparison.

- **05_Linear_Separability.ipynb**
  - Examines linear separability in a dataset with two features.
  - Includes dataset analysis, feature transformation, separating hyperplane, and importance of nonlinear transformations.

- **06_Softmax_Activation_Derivation.ipynb**
  - Derives the properties of the softmax activation function and its derivatives.
  - Uses cross-entropy loss function and proves the gradient of the loss function.

- **07_NeuralNetwork_Manual_Backpropagation.ipynb**
  - Manually implements a 3-layered neural network with scaled sigmoid activation functions.
  - Covers forward propagation, cost calculation, backpropagation, model training, and accuracy comparison.

- **08_Weight_Initialization_DeadNeurons_LeaklyReLU.ipynb**
  - Explores the effects of weight initialization, vanishing gradients, and dead neurons.
  - Analyzes the impact of ReLU and Leaky ReLU activations.

- **09_BatchNorm_Dropout.ipynb**
  - Compares the effects of batch normalization and dropout on the performance of LeNet-5 using the MNIST dataset.
  - Investigates the combination of both techniques and their impact on performance.

- **10_LearningRate_BatchSize_Exploration.ipynb**
  - Explores the cyclical learning rate policy and its effect on training a neural network.
  - Compares the effect of varying batch sizes with a fixed learning rate on model performance.
