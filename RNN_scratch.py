import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the RNN class
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights for input to hidden and hidden to hidden connections
        self.W_xh = np.random.randn(input_size, hidden_size)  # Weights for input to hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size)  # Weights for hidden to hidden
        self.W_hy = np.random.randn(hidden_size, output_size)  # Weights for hidden to output
        
        # Biases
        self.b_h = np.zeros((1, hidden_size))  # Bias for hidden state
        self.b_y = np.zeros((1, output_size))  # Bias for output
        
        # Initialize the hidden state
        self.h = np.zeros((1, hidden_size))  # Hidden state initialized to zeros

    def forward(self, X):
        # Initialize lists to store the hidden states and outputs for each timestep
        self.hidden_states = []
        self.outputs = []
        
        # Loop over each timestep
        for t in range(X.shape[1]):
            # Compute the new hidden state (h_t = sigmoid(W_xh * x_t + W_hh * h_(t-1) + b_h))
            self.h = sigmoid(np.dot(X[:, t, :], self.W_xh) + np.dot(self.h, self.W_hh) + self.b_h)
            self.hidden_states.append(self.h)
            
            # Compute the output (y_t = W_hy * h_t + b_y)
            y = np.dot(self.h, self.W_hy) + self.b_y
            self.outputs.append(y)
        
        # Convert outputs to a numpy array for easier access
        self.outputs = np.array(self.outputs)
        return self.outputs

    def backward(self, X, y, learning_rate=0.001):
        # Initialize gradients
        d_W_xh = np.zeros_like(self.W_xh)
        d_W_hh = np.zeros_like(self.W_hh)
        d_W_hy = np.zeros_like(self.W_hy)
        d_b_h = np.zeros_like(self.b_h)
        d_b_y = np.zeros_like(self.b_y)
        
        # Compute the loss (Mean Squared Error)
        loss = np.mean((self.outputs - y) ** 2)
        
        # Compute the derivative of the loss with respect to the output
        d_output = 2 * (self.outputs - y) / y.shape[0]  # Derivative of MSE loss
        
        # Backpropagate through time (BPTT)
        d_h = np.zeros_like(self.h)
        
        for t in reversed(range(X.shape[1])):
            # Ensure the current hidden state and output are of the correct shape
            d_output_t = d_output[t]
            
            # Compute gradients for the output layer
            d_W_hy += np.dot(self.hidden_states[t].T, d_output_t)
            d_b_y += np.sum(d_output_t, axis=0, keepdims=True)
            
            # Backpropagate the error through the hidden layer
            d_h += np.dot(d_output_t, self.W_hy.T)
            d_h *= sigmoid_derivative(self.hidden_states[t])  # Gradient of the activation function
            
            # Compute gradients for the hidden-to-hidden and input-to-hidden weights
            if t > 0:  # Backpropagate to the previous timestep (not for t=0)
                d_W_hh += np.dot(self.hidden_states[t-1].T, d_h)
            d_W_xh += np.dot(X[:, t, :].T, d_h)
            d_b_h += np.sum(d_h, axis=0, keepdims=True)
        
        # Update weights and biases using gradient descent
        self.W_xh -= learning_rate * d_W_xh
        self.W_hh -= learning_rate * d_W_hh
        self.W_hy -= learning_rate * d_W_hy
        self.b_h -= learning_rate * d_b_h
        self.b_y -= learning_rate * d_b_y
        
        return loss

    def compute_accuracy(self, X, y):
        # Get the predictions
        y_pred = self.forward(X)
        
        # Convert predictions to binary values (0 or 1)
        y_pred_bin = (y_pred > 0.5).astype(int)
        
        # Compute the accuracy
        correct = np.sum(y_pred_bin == y)
        total = np.prod(y.shape)
        accuracy = correct / total
        return accuracy

# Create the RNN model
input_size = 3  # Number of features in the input (e.g., 3 features per timestep)
hidden_size = 4  # Number of units in the hidden layer
output_size = 1  # Output size (e.g., binary classification)

# Initialize the RNN
rnn = SimpleRNN(input_size, hidden_size, output_size)

# Generate some dummy data for training
X_train = np.random.randn(1000, 10, input_size)  # 1000 samples, 10 timesteps, 3 features per timestep
y_train = np.random.randint(0, 2, (1000, 10, output_size))  # Binary target for each timestep

# Lists to store the loss and accuracy at each epoch
losses = []
accuracies = []

# Training the RNN
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for i in range(X_train.shape[0]):
        X_sample = X_train[i:i+1]  # Get one sample
        y_sample = y_train[i:i+1]  # Get the target for the sample
        rnn.forward(X_sample)  # Perform forward pass
        loss = rnn.backward(X_sample, y_sample)  # Perform backward pass and update weights
        epoch_loss += loss
        
        # Calculate accuracy for this sample
        accuracy = rnn.compute_accuracy(X_sample, y_sample)
        epoch_accuracy += accuracy
    
    # Store the average loss and accuracy for this epoch
    losses.append(epoch_loss / len(X_train))
    accuracies.append(epoch_accuracy / len(X_train))
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train):.4f}, Accuracy: {epoch_accuracy/len(X_train):.4f}")

# Plot loss and accuracy
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(epochs), losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(epochs), accuracies, label='Accuracy', color='orange')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()
