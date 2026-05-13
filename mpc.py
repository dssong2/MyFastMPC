"""
Milestone 1: LQ-MPC for spacecraft rendezvous with thrust limits only.

Reproduces the 3D scenario from Bashnick & Ulrich (2023), Section IV.A,
Table 1, but WITHOUT obstacles, holding sphere, or entry cone.

The OCP at each step:

    min   sum_{t=1..N-1} (x_t - x_t*)^T Q (x_t - x_t*) + (u_t)^T R u_t
          + (x_N - x_N*)^T Qf (x_N - x_N*)
    s.t.  x_{t+1} = A x_t + B u_t      (CW dynamics, ZOH-discretized)
          |u_t|   <= u_max             (per-axis thrust limit)
          x_0     = current state

solved by cvxpy/OSQP. Apply only u_0, advance one step, repeat.
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# -----------------------------------------------------------------------------
# Physical and orbital parameters (Section IV.A, Table 1)
# -----------------------------------------------------------------------------
mu_E   = 3.986e5          # km^3/s^2, Earth gravitational parameter
R_E    = 6378.0           # km, Earth radius
alt    = 705.0            # km, target altitude
a_orb  = R_E + alt        # km, semi-major axis of target orbit
n_mm   = np.sqrt(mu_E / a_orb**3)   # rad/s, mean motion of target
m_chs  = 6500.0           # kg, chaser mass

print(f"Mean motion n = {n_mm:.6e} rad/s   (orbital period {2*np.pi/n_mm/60:.1f} min)")

# -----------------------------------------------------------------------------
# Continuous-time Clohessy-Wiltshire dynamics, Eq. (3)
#   state x = [x, y, z, x_dot, y_dot, z_dot]   (radial, in-track, cross-track)
#   input u = [u_x, u_y, u_z]                  (thrust force, N)
# -----------------------------------------------------------------------------
A_c = np.array([
    [0,           0,        0,    1,        0,    0],
    [0,           0,        0,    0,        1,    0],
    [0,           0,        0,    0,        0,    1],
    [3*n_mm**2,   0,        0,    0,    2*n_mm,   0],
    [0,           0,        0, -2*n_mm,     0,    0],
    [0,           0, -n_mm**2,    0,        0,    0],
])
B_c = np.vstack([np.zeros((3, 3)), np.eye(3) / m_chs])

# -----------------------------------------------------------------------------
# Discretize with zero-order hold (Section II.B)
# -----------------------------------------------------------------------------
Td = 15.0  # s, sample period (Table 1)
A_d, B_d, _, _, _ = cont2discrete(
    (A_c, B_c, np.eye(6), np.zeros((6, 3))), Td, method='zoh'
)

# -----------------------------------------------------------------------------
# Cost weights and Riccati terminal weight
# -----------------------------------------------------------------------------
Q  = np.diag([1, 1, 1, 100, 100, 100])
R  = np.diag([100, 100, 100])
Qf = solve_discrete_are(A_d, B_d, Q, R)

# -----------------------------------------------------------------------------
# MPC parameters
# -----------------------------------------------------------------------------
N            = 30      # horizon length (Table 1)
u_max        = 20.0    # N, per-axis thrust limit
hold_dist    = 2.3     # m, mission terminates within this radius of target
max_steps    = 200     # safety cap

# Initial conditions (Table 1)
x0 = np.array([50.0, -1000.0, 10.0, 0.0, 0.0, 0.0])
xt = np.zeros(6)


# -----------------------------------------------------------------------------
# Build the OCP once with parameters; resolve at each step by updating x_init
# (cvxpy Parameter avoids re-building the problem graph every step)
# -----------------------------------------------------------------------------
x_init_par = cp.Parameter(6)
x_target   = cp.Parameter(6)

x_var = cp.Variable((N + 1, 6))
u_var = cp.Variable((N, 3))

cost = 0
constr = [x_var[0] == x_init_par]

for t in range(N):
    cost  += cp.quad_form(u_var[t], cp.psd_wrap(R))
    constr += [
        x_var[t + 1] == A_d @ x_var[t] + B_d @ u_var[t],
        cp.abs(u_var[t]) <= u_max,
    ]

for t in range(1, N):
    cost += cp.quad_form(x_var[t] - x_target, cp.psd_wrap(Q))
cost += cp.quad_form(x_var[N] - x_target, cp.psd_wrap(Qf))

problem = cp.Problem(cp.Minimize(cost), constr)


def solve_mpc(x_now, x_des):
    x_init_par.value = x_now
    x_target.value   = x_des
    problem.solve(solver=cp.OSQP, warm_start=True)
    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"MPC infeasible: {problem.status}")
    return u_var.value[0], problem.solver_stats.solve_time


# -----------------------------------------------------------------------------
# Closed-loop simulation
# -----------------------------------------------------------------------------
x         = x0.copy()
traj      = [x.copy()]
ctrls     = []
solve_t   = []

for k in range(max_steps):
    u_apply, t_solve = solve_mpc(x, xt)
    ctrls.append(u_apply)
    solve_t.append(t_solve)
    x = A_d @ x + B_d @ u_apply
    traj.append(x.copy())
    if np.linalg.norm(x[:3]) <= hold_dist:
        print(f"Reached holding distance ({np.linalg.norm(x[:3]):.3f} m) "
              f"at step {k+1}")
        break
else:
    print(f"Did not converge within {max_steps} steps; "
          f"final dist = {np.linalg.norm(x[:3]):.2f} m")

traj   = np.array(traj)
ctrls  = np.array(ctrls)
solve_t = np.array(solve_t)

# Total impulse: integral of |u| dt approximated with sample period
total_impulse = np.sum(np.linalg.norm(ctrls, axis=1)) * Td  # N*s

print(f"Mission steps:       {len(ctrls)}  ({len(ctrls)*Td:.0f} s = "
      f"{len(ctrls)*Td/60:.1f} min)")
print(f"Total impulse:       {total_impulse/1000:.2f} kN*s")
print(f"Mean solve time:     {1000*solve_t.mean():.2f} ms / step")
print(f"Final dist to target {np.linalg.norm(traj[-1, :3]):.3f} m")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 5))

# 3D trajectory: paper plots in-track on x-axis, radial on y-axis, cross-track on z
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(traj[:, 1], traj[:, 0], traj[:, 2], 'r-', linewidth=1.6, label='chaser path')
ax1.scatter([0], [0], [0], c='black', s=120, marker='o', label='target')
ax1.scatter(traj[0, 1], traj[0, 0], traj[0, 2],
            c='red', s=80, marker='s', label='chaser start')
ax1.set_xlabel('In-track y [m]')
ax1.set_ylabel('Radial x [m]')
ax1.set_zlabel('Cross-track z [m]')
ax1.set_title('Trajectory')
ax1.legend(loc='upper left', fontsize=9)

# Thrust profile
t_axis = np.arange(len(ctrls)) * Td
ax2 = fig.add_subplot(132)
ax2.step(t_axis, ctrls[:, 0], where='post', label=r'$u_x$')
ax2.step(t_axis, ctrls[:, 1], where='post', label=r'$u_y$')
ax2.step(t_axis, ctrls[:, 2], where='post', label=r'$u_z$')
ax2.axhline( u_max, color='k', linestyle='--', alpha=0.4)
ax2.axhline(-u_max, color='k', linestyle='--', alpha=0.4,
            label=r'$\pm u_{\max}$')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Thrust [N]')
ax2.set_title('Commanded thrust')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Distance to target
ax3 = fig.add_subplot(133)
dists = np.linalg.norm(traj[:, :3], axis=1)
t_axis_s = np.arange(len(traj)) * Td
ax3.semilogy(t_axis_s, dists, 'b-')
ax3.axhline(hold_dist, color='g', linestyle='--', label=f'hold dist = {hold_dist} m')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Distance to target [m]')
ax3.set_title('Range vs time (log scale)')
ax3.legend()
ax3.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('mpc_milestone1.png', dpi=120, bbox_inches='tight')
print("Saved plot to mpc_milestone1.png")