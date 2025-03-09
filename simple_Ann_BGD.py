import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Define multiple training inputs and expected outputs
X = np.array([[7, 2], [3, 5], [6, 1], [4, 3], [8, 6]])  # 5 training samples
y = np.array([[1], [0], [1], [0], [1]])  # Expected outputs

# Initialize weights and bias
np.random.seed(0)
w_hidden = np.random.rand(2, 3)  # 2 input neurons -> 3 hidden neurons
b_hidden = np.random.rand(1, 3)
w_output = np.random.rand(3, 1)  # 3 hidden neurons -> 1 output neuron
b_output = np.random.rand(1, 1)

learning_rate = 0.1
epochs = 100  # Number of epochs

for epoch in range(epochs):
    # Forward Pass for all samples
    hidden_input = np.dot(X, w_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, w_output) + b_output
    final_output = sigmoid(final_input)

    # Compute error
    error = y - final_output
    loss = np.mean(np.abs(error))

    # Backpropagation (Batch Gradient Descent)
    output_gradient = error * sigmoid_derivative(final_output)
    w_output += learning_rate * np.dot(hidden_output.T, output_gradient) / len(X)
    b_output += learning_rate * np.mean(output_gradient, axis=0)

    hidden_error = np.dot(output_gradient, w_output.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)
    w_hidden += learning_rate * np.dot(X.T, hidden_gradient) / len(X)
    b_hidden += learning_rate * np.mean(hidden_gradient, axis=0)

    # Print loss every 100 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Print final results
print("Final Updated Weights (Hidden Layer):")
print(w_hidden)
print("Final Updated Weights (Output Layer):")
print(w_output)
print("Final Updated Bias (Hidden Layer):")
print(b_hidden)
print("Final Updated Bias (Output Layer):")
print(b_output)
