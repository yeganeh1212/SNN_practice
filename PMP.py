import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1  # Final time
lambda_ = -2 / T  # Adjoint variable
u_opt = lambda t: 1 / T  # Optimal control
x_opt = lambda t: (1 / T) * t  # Optimal state

# Time grid
t = np.linspace(0, T, 100)

# Compute state and control
x_vals = x_opt(t)
u_vals = np.array([u_opt(time) for time in t])  # Ensure u_vals has the same shape as t

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t, x_vals, label="State x(t)")
plt.title("Optimal State")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, u_vals, label="Control u(t)", color="r")
plt.title("Optimal Control")
plt.xlabel("Time")
plt.ylabel("Control Effort")
plt.legend()

plt.tight_layout()
plt.show()
