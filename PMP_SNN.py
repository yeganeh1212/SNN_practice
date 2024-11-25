import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0  # Decay constant
beta = 0.5   # Control weighting
T = 10.0     # Total time
x0 = 1.0     # Initial state
lambda_T = 0.0  # Terminal condition for adjoint

# Dynamics and adjoint equations
def dynamics(t, y):
    x, lambda_ = y
    u = -lambda_ / (2 * beta)
    dxdt = -alpha * x + u
    dlambdadt = -2 * x + alpha * lambda_
    return [dxdt, dlambdadt]

# Solve the system
t_eval = np.linspace(0, T, 100)
sol = solve_ivp(dynamics, [0, T], [x0, lambda_T], t_eval=t_eval, method='RK45')

# Extract solutions
t = sol.t
x = sol.y[0]
lambda_ = sol.y[1]
u = -lambda_ / (2 * beta)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, x, label="State x(t)")
plt.xlabel("Time")
plt.ylabel("Synaptic Current")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, u, label="Control u(t)", color="r")
plt.xlabel("Time")
plt.ylabel("Control Input")
plt.legend()

plt.tight_layout()
plt.show()
