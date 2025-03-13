import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Generate synthetic regression data
np.random.seed(0)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2.5 * X + np.random.randn(50, 1) * 2
print(y)

# Normalize target y to 0-1 range using min-max scaling
y_min, y_max = y.min(), y.max()
y = (y - y_min) / (y_max - y_min)

# Initialize weights and bias
w_hidden = np.random.rand(1, 3)
b_hidden = np.random.rand(1, 3)
w_output = np.random.rand(3, 1)
b_output = np.random.rand(1, 1)

learning_rate = 0.01
epochs = 1000
loss_history = []

for epoch in range(epochs):
    # Forward Pass for all samples
    hidden_input = np.dot(X, w_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, w_output) + b_output
    final_output = sigmoid(final_input)

    # Compute error
    error = y - final_output
    loss = np.mean(np.abs(error))
    loss_history.append(loss)  # Store loss for plotting

    # Backpropagation
    output_gradient = error * sigmoid_derivative(final_output)
    w_output += learning_rate * np.dot(hidden_output.T, output_gradient) / len(X)
    b_output += learning_rate * np.mean(output_gradient, axis=0)

    hidden_error = np.dot(output_gradient, w_output.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)
    w_hidden += learning_rate * np.dot(X.T, hidden_gradient) / len(X)
    b_hidden += learning_rate * np.mean(hidden_gradient, axis=0)

    # Print loss every 100 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Testing phase
X_test = X
actual_y_test = y
predicted_y_test = sigmoid(np.dot(sigmoid(np.dot(X_test, w_hidden) + b_hidden), w_output) + b_output)

# Compute error metrics for testing
error_test = actual_y_test - predicted_y_test

# Mean Absolute Percentage Error (MAPE) with epsilon to avoid inf
epsilon = 1e-8  # Small value to prevent division by zero
# MAPE_test = np.mean(np.abs(error_test / (actual_y_test + epsilon))) * 100  # Dalam persen
MAPE_test = np.mean(np.where(actual_y_test != 0, np.abs(error_test / actual_y_test), 0)) * 100
# Mean Absolute Deviation (MAD)
MAD_test = np.mean(np.abs(error_test))

# Mean Squared Error (MSE)
MSE_test = np.mean(error_test**2)

print("Testing Error Metrics:")
print(f"MAPE: {MAPE_test:.4f}%")
print(f"MAD: {MAD_test:.4f}")
print(f"MSE: {MSE_test:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test, actual_y_test, label="Actual Test Data", color="blue")
plt.plot(X_test, predicted_y_test, label="Predicted Test Data", color="red")
plt.xlabel("Input X")
plt.ylabel("Output y (Normalized)")
plt.title("Testing: Regression Prediction using Neural Network with Sigmoid Output")
plt.legend()

# Plot loss history
plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss_history, color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
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