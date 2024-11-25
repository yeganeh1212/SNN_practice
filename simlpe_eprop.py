import numpy as np
import matplotlib.pyplot as plt

# Constants
tau_m = 20.0  # Membrane time constant (ms)
V_th = -50.0  # Threshold voltage (mV)
V_reset = -65.0  # Reset voltage (mV)
R = 1.0  # Membrane resistance (MΩ)
tau_s = 5.0  # Synaptic time constant (ms)
eta = 0.1  # Learning rate

# Time parameters
dt = 0.1  # Time step (ms)
T = 200  # Total simulation time (ms)
time = np.arange(0, T, dt)  # Time vector

# Initialize voltages, spikes, and eligibility traces
V1 = V_reset  # Neuron 1 membrane potential
V2 = V_reset  # Neuron 2 membrane potential
spikes1 = np.zeros_like(time)
spikes2 = np.zeros_like(time)
eligibility_trace1 = 0
eligibility_trace2 = 0

# Synaptic weight (between neuron 1 and neuron 2)
W = 0.5

# Arrays to store voltages and weight updates
V1_arr = []
V2_arr = []
W_arr = []

# Simulate the network
for t in time:
    # Compute synaptic current for neuron 2 from neuron 1
    I_syn = W * spikes1[int(t/dt)]
    
    # Update the membrane potentials using the leaky integrate-and-fire model
    dV1 = (-V1 + R * I_syn) / tau_m * dt  # Update rule for neuron 1
    V1 += dV1
    if V1 >= V_th:  # If neuron 1 fires
        spikes1[int(t/dt)] = 1
        V1 = V_reset
        eligibility_trace1 = 1  # Reset eligibility trace after a spike
    else:
        eligibility_trace1 *= np.exp(-dt/tau_s)  # Update eligibility trace

    dV2 = (-V2 + R * I_syn) / tau_m * dt  # Update rule for neuron 2
    V2 += dV2
    if V2 >= V_th:  # If neuron 2 fires
        spikes2[int(t/dt)] = 1
        V2 = V_reset
        eligibility_trace2 = 1  # Reset eligibility trace after a spike
    else:
        eligibility_trace2 *= np.exp(-dt/tau_s)  # Update eligibility trace

    # Update synaptic weight using eligibility propagation (e-prop)
    W += eta * eligibility_trace1 * (spikes2[int(t/dt)] - spikes2[int(t/dt)-1])
    
    # Store values for plotting
    V1_arr.append(V1)
    V2_arr.append(V2)
    W_arr.append(W)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot membrane potentials
plt.subplot(2, 1, 1)
plt.plot(time, V1_arr, label="Neuron 1 Membrane Potential")
plt.plot(time, V2_arr, label="Neuron 2 Membrane Potential")
plt.axhline(V_th, color='r', linestyle='--', label="Threshold")
plt.title('Membrane Potentials')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend()

# Plot the synaptic weight
plt.subplot(2, 1, 2)
plt.plot(time, W_arr, label="Synaptic Weight")
plt.title('Synaptic Weight Evolution')
plt.xlabel('Time (ms)')
plt.ylabel('Weight')
plt.legend()

plt.tight_layout()
plt.show()
