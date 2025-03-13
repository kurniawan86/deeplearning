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
batch_size = 30
loss_history = []

for epoch in range(epochs):
    total_loss = 0
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # Forward Pass
        hidden_input = np.dot(X_batch, w_hidden) + b_hidden
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, w_output) + b_output
        final_output = final_input

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
    loss_history.append(total_loss / (len(X) / batch_size))

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / (len(X) / batch_size)}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, label="Actual Data", color="blue")
predicted_y = np.dot(sigmoid(np.dot(X, w_hidden) + b_hidden), w_output) + b_output
plt.plot(X, predicted_y, label="Predicted Data", color="red")
plt.xlabel("Input X")
plt.ylabel("Output y")
plt.title("Regression Prediction using Neural Network with Mini-Batch Gradient Descent")
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
