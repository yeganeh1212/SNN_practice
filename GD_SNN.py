import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
tau = 20.0  # Synaptic time constant (ms)
g_max = 1.0  # Maximum synaptic conductance
t_end = 200.0  # End time of simulation (ms)

# Gate function (example: exponential decay with spike input)
def gate_function(t, spike_times, width=2.0):
    """
    Returns a gate function value at time t based on spike times.
    Spikes occur at spike_times, and the function is an exponential decay.
    """
    gate = 0.0
    for spike_time in spike_times:
        gate += g_max * np.exp(-(t - spike_time) / width) * (t >= spike_time)
    return gate

# Synaptic current dynamics equation
def synaptic_dynamics(t, I, spike_times):
    """
    Computes the rate of change of synaptic current.
    """
    g = gate_function(t, spike_times)
    dI_dt = -I / tau + g
    return dI_dt

# Simulate the spikes
spike_times = [50, 100, 150]  # Example spike times in ms

# Solve the differential equation using solve_ivp
time_points = np.linspace(0, t_end, 1000)
solution = solve_ivp(synaptic_dynamics, [0, t_end], [0.0], args=(spike_times,), t_eval=time_points)

# Plot the synaptic current over time
plt.plot(solution.t, solution.y[0], label="Synaptic Current (I)")
plt.scatter(spike_times, [g_max] * len(spike_times), color='r', label="Spikes", zorder=5)
plt.title("Synaptic Current Dynamics with Gate Function")
plt.xlabel("Time (ms)")
plt.ylabel("Synaptic Current (I)")
plt.legend()
plt.show()
