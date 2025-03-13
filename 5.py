import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Generate synthetic regression data
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2.5 * X + np.random.randn(50, 1) * 2

# Initialize weights and bias
w_hidden = np.random.rand(1, 3)
b_hidden = np.random.rand(1, 3)
w_output = np.random.rand(3, 1)
b_output = np.random.rand(1, 1)

learning_rate = 0.01
epochs = 1000  # Number of epochs
loss_history = []  # Store loss values

for epoch in range(epochs):
    # Forward Pass for all samples
    hidden_input = np.dot(X, w_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, w_output) + b_output
    final_output = final_input

    # Compute error
    error = y - final_output
    loss = np.mean(np.abs(error))
    loss_history.append(loss)

    # Backpropagation
    output_gradient = error
    w_output += learning_rate * np.dot(hidden_output.T, output_gradient) / len(X)
    b_output += learning_rate * np.mean(output_gradient, axis=0)

    hidden_error = np.dot(output_gradient, w_output.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)
    w_hidden += learning_rate * np.dot(X.T, hidden_gradient) / len(X)
    b_hidden += learning_rate * np.mean(hidden_gradient, axis=0)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Testing phase
X_test = X
actual_y_test = y
predicted_y_test = np.dot(sigmoid(np.dot(X_test, w_hidden) + b_hidden), w_output) + b_output

# Compute error metrics for testing
error_test = actual_y_test - predicted_y_test

# Mean Absolute Percentage Error (MAPE)
MAPE_test = np.mean(np.abs(error_test / actual_y_test)) * 100  # Dalam persen

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
plt.scatter(X, y, label="Actual Data", color="blue")
predicted_y = np.dot(sigmoid(np.dot(X, w_hidden) + b_hidden), w_output) + b_output
plt.plot(X, predicted_y, label="Predicted Data", color="red")
plt.xlabel("Input X")
plt.ylabel("Output y")
plt.title("Regression Prediction using Neural Network")
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