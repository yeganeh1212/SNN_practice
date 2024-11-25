import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np

# Generate dummy data for the example
# Example: 1000 samples, each with 10 timesteps and 1 feature
X_train = np.random.random((1000, 10, 1))  # Shape: (samples, timesteps, features)
y_train = np.random.randint(2, size=(1000, 1))  # Binary output

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(10, 1)))  # 50 units in RNN layer
model.add(Dense(1, activation='sigmoid'))  # Binary classification output

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

