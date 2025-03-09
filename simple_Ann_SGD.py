import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Generate synthetic regression data
np.random.seed(0)
X = np.linspace(0, 10, 50).reshape(-1, 1)  # 50 data points
y = 2.5 * X + np.random.randn(50, 1) * 2  # Linear relation with noise

# Initialize weights and bias
w_hidden = np.random.rand(1, 3)  # 1 input neuron -> 3 hidden neurons
b_hidden = np.random.rand(1, 3)
w_output = np.random.rand(3, 1)  # 3 hidden neurons -> 1 output neuron
b_output = np.random.rand(1, 1)

learning_rate = 0.1
epochs = 1000  # Number of epochs
loss_history = []

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x_sample = X[i].reshape(1, -1)  # Select one sample
        y_sample = y[i].reshape(1, -1)

        # Forward Pass
        hidden_input = np.dot(x_sample, w_hidden) + b_hidden
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, w_output) + b_output
        final_output = final_input  # Using linear activation for regression

        # Compute error
        error = y_sample - final_output
        loss = np.abs(error).mean()
        total_loss += loss

        # Backpropagation (Stochastic Gradient Descent - SGD)
        output_gradient = error  # Derivative of linear activation is 1
        w_output += learning_rate * np.dot(hidden_output.T, output_gradient)
        b_output += learning_rate * output_gradient

        hidden_error = np.dot(output_gradient, w_output.T)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)
        w_hidden += learning_rate * np.dot(x_sample.T, hidden_gradient)
        b_hidden += learning_rate * hidden_gradient

    # Store loss history
    loss_history.append(total_loss / len(X))

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, label="Actual Data", color="blue")
predicted_y = np.dot(sigmoid(np.dot(X, w_hidden) + b_hidden), w_output) + b_output
plt.plot(X, predicted_y, label="Predicted Data", color="red")
plt.xlabel("Input X")
plt.ylabel("Output y")
plt.title("Regression Prediction using Neural Network with SGD")
plt.legend()

# Plot loss history
plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss_history, color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs using SGD")
plt.grid()

plt.show()

# Print final results
print("Final Updated Weights (Hidden Layer):")
print(w_hidden)
print("Final Updated Weights (Output Layer):")
print(w_output)
print("Final Updated Bias (Hidden Layer):")
print(b_hidden)
print("Final Updated Bias (Output Layer):")
print(b_output)
