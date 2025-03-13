import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Generate synthetic regression data
np.random.seed(0)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2.5 * X + np.random.randn(50, 1) * 2

# Initialize weights and bias
w_hidden = np.random.rand(1, 3)
b_hidden = np.random.rand(1, 3)
w_output = np.random.rand(3, 1)
b_output = np.random.rand(1, 1)

learning_rate = 0.1
epochs = 1000
mini_loop = 5
batch_size = 30
loss_history = []

for epoch in range(epochs):
    total_loss = 0
    for _ in range(mini_loop):  # Process 30 random samples in each mini-loop iteration
        indices = np.random.choice(len(X), batch_size, replace=True)  # Random selection
        X_batch = X[indices]
        y_batch = y[indices]

        # Forward Pass
        hidden_input = np.dot(X_batch, w_hidden) + b_hidden
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, w_output) + b_output
        final_output = final_input  # Using linear activation for regression

        # Compute error
        error = y_batch - final_output
        loss = np.abs(error).mean()
        total_loss += loss

        # Backpropagation
        output_gradient = error
        w_output += learning_rate * np.dot(hidden_output.T, output_gradient) / batch_size
        b_output += learning_rate * np.mean(output_gradient, axis=0)

        hidden_error = np.dot(output_gradient, w_output.T)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)
        w_hidden += learning_rate * np.dot(X_batch.T, hidden_gradient) / batch_size
        b_hidden += learning_rate * np.mean(hidden_gradient, axis=0)

    # Store loss history
    loss_history.append(total_loss / mini_loop)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / mini_loop}")

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
MSE_test = np.mean(error_test ** 2)

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
plt.ylabel("Output y")
plt.title("Testing: Regression Prediction using Neural Network")
plt.legend()

# Plot loss history
plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss_history, color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs using Mini-Batch Gradient Descent")
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
