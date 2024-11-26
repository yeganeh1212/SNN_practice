import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_input = 1         # Number of input neurons
n_hidden = 10       # Number of hidden (spiking) neurons
n_output = 1        # Number of output neurons
time_steps = 200    # Number of time steps
dt = 0.1            # Time step duration
tau = 10.0          # Membrane time constant
v_reset = 0.0       # Reset voltage
v_threshold = 1.0   # Threshold voltage for spiking
learning_rate = 0.01

# Initialize weights
np.random.seed(42)  # For reproducibility
W_in = np.random.normal(0, 0.5, (n_hidden, n_input))  # Input to hidden
W_rec = np.random.normal(0, 0.5, (n_hidden, n_hidden))  # Recurrent weights
W_out = np.random.normal(0, 0.5, (n_output, n_hidden))  # Hidden to output

# Initialize state variables
v_hidden = np.zeros(n_hidden)  # Membrane potentials for hidden neurons
spike_hidden = np.zeros(n_hidden)  # Spikes from hidden neurons
trace = np.zeros(n_hidden)  # Trace for plasticity (e.g., STDP)

# Input signal: a sine wave
time = np.arange(0, time_steps * dt, dt)
input_signal = np.sin(2 * np.pi * time / (time_steps * dt))  # Sine wave input

# Storage for visualization
v_history = []
output_history = []

# Simulation loop
for t in range(time_steps):
    # Input to hidden neurons
    input_current = W_in @ input_signal[t:t + 1]

    # Recurrent input from other neurons
    recurrent_current = W_rec @ spike_hidden

    # Update membrane potential
    dv = (-v_hidden + input_current + recurrent_current) / tau
    v_hidden += dv * dt

    # Check for spikes
    spike_hidden = (v_hidden >= v_threshold).astype(float)
    v_hidden[spike_hidden == 1] = v_reset  # Reset after spike

    # Update trace (simple STDP-like rule)
    trace = 0.9 * trace + spike_hidden

    # Output computation
    output = W_out @ spike_hidden
    output_history.append(output)

    # Store for visualization
    v_history.append(v_hidden.copy())

# Visualization
plt.figure(figsize=(12, 6))

# Input signal
plt.subplot(3, 1, 1)
plt.plot(time, input_signal, label="Input Signal")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Input")

# Hidden neuron voltages
plt.subplot(3, 1, 2)
plt.imshow(np.array(v_history).T, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label="Voltage")
plt.title("Hidden Neuron Voltages")
plt.ylabel("Neuron Index")

# Output signal
plt.subplot(3, 1, 3)
plt.plot(time, output_history, label="Output Signal")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Output")

plt.tight_layout()
