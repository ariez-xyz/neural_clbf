"""Minimal test to check if neural_clbf works without cvxpylayers."""
import torch
import matplotlib.pyplot as plt

# Load the pretrained model
from neural_clbf.controllers import NeuralCLBFController

print("Loading checkpoint...")
log_file = "saved_models/review/inverted_pendulum_clf.ckpt"
controller = NeuralCLBFController.load_from_checkpoint(log_file)
print(f"Loaded! System: {type(controller.dynamics_model).__name__}")
print(f"State dims: {controller.dynamics_model.n_dims}")
print(f"Control dims: {controller.dynamics_model.n_controls}")
print(f"dt: {controller.dynamics_model.dt}")

# Get the dynamics model
dynamics = controller.dynamics_model

# Test: simulate a trajectory
print("\nSimulating trajectory...")
x0 = torch.tensor([[1.0, 0.5]])  # Initial state: theta=1, theta_dot=0.5
num_steps = 500

# Manual simulation loop (avoids QP solver issues)
trajectory = [x0]
x = x0
for _ in range(num_steps):
    # Get control-affine dynamics
    f, g = dynamics.control_affine_dynamics(x)

    # Simple nominal control (LQR)
    u = dynamics.u_nominal(x)

    # Euler step: x_next = x + dt * (f + g @ u)
    xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
    x = x + dynamics.dt * xdot
    trajectory.append(x)

trajectory = torch.cat(trajectory, dim=0)
print(f"Trajectory shape: {trajectory.shape}")

# Compute CLF values along trajectory
print("\nComputing CLF values...")
V_values = controller.V(trajectory).detach().numpy()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# State space trajectory
ax = axes[0]
ax.plot(trajectory[:, 0].numpy(), trajectory[:, 1].numpy(), 'b-', linewidth=1)
ax.plot(trajectory[0, 0].numpy(), trajectory[0, 1].numpy(), 'go', markersize=10, label='Start')
ax.plot(trajectory[-1, 0].numpy(), trajectory[-1, 1].numpy(), 'ro', markersize=10, label='End')
ax.set_xlabel('theta')
ax.set_ylabel('theta_dot')
ax.set_title('State Space Trajectory')
ax.legend()
ax.grid(True)
ax.set_aspect('equal')

# States over time
ax = axes[1]
t = torch.arange(num_steps + 1) * dynamics.dt
ax.plot(t.numpy(), trajectory[:, 0].numpy(), label='theta')
ax.plot(t.numpy(), trajectory[:, 1].numpy(), label='theta_dot')
ax.set_xlabel('Time (s)')
ax.set_ylabel('State')
ax.set_title('States over Time')
ax.legend()
ax.grid(True)

# CLF value over time
ax = axes[2]
ax.plot(t.numpy(), V_values)
ax.axhline(y=0, color='r', linestyle='--', label='V=0 (safe boundary)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('V(x)')
ax.set_title('CLF Value over Time')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('test_output.png', dpi=150)
print("\nSaved plot to test_output.png")
plt.show()
