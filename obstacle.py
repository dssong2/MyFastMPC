"""
Milestone 2: LQ-MPC with static obstacle avoidance via direct linearization.

Supports N static ellipsoidal keep-out zones (KOZs). Each non-convex constraint
    (r - r_obj)^T S (r - r_obj) >= 1                  [Eq. 6]
is replaced by a linear half-space constraint
    a^T r >= c                                         [from Eq. 8]
where the tangent hyperplane is taken at r_0 on the KOZ boundary along the ray
from r_obj to the chaser's reference position. The reference trajectory comes
from the previous MPC iteration's predicted states (bootstrapped on the first step).
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dynamics import ClohessyWiltshire


class CWObstacle(ClohessyWiltshire):
    """Extends the base LQ-MPC with N static ellipsoidal obstacles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles = []  # list of dicts: {r_obj, v_obj, semi_axes, S}
        self.obs_start_pos = []
        
    def add_obstacle(self, r_obj, v_obj, semi_axes):
        """Append a static ellipsoidal KOZ. semi_axes=(a,b,c); use (r,r,r) for a sphere."""
        r_obj     = np.asarray(r_obj,     dtype=float)
        v_obj     = np.asarray(v_obj,     dtype=float)
        semi_axes = np.asarray(semi_axes, dtype=float)
        self.obstacles.append({
            'r_obj':     r_obj,
            'v_obj':     v_obj,
            'semi_axes': semi_axes,
            'S':         np.diag(1.0 / semi_axes**2),
        })
        self.obs_start_pos.append(r_obj.copy())

    # ------------------------------------------------------------------
    # Linearization
    # ------------------------------------------------------------------

    def compute_linearization_points(self, r_ref, r_obj, S):
        """Project reference positions onto a KOZ boundary along the ray from
        the obstacle center.

            r_0 = r_obj + (r_ref - r_obj) / sqrt((r_ref-r_obj)^T S (r_ref-r_obj))

        Args:
            r_ref : (N, 3) reference positions over the horizon
            r_obj : (3,)   obstacle center
            S     : (3, 3) shape matrix  diag(1/a², 1/b², 1/c²)
        Returns:
            r_0   : (N, 3) points on the KOZ boundary
        """
        d      = r_ref - r_obj
        s_norm = np.sqrt(np.einsum('ij,jk,ik->i', d, S, d))
        s_norm = np.maximum(s_norm, 1e-9)
        return r_obj + d / s_norm[:, None]

    # ------------------------------------------------------------------
    # CVXPY problem
    # ------------------------------------------------------------------

    def build_problem(self):
        """Base dynamics + one linearized half-space constraint per obstacle per step."""
        self._x_init_par = cp.Parameter(6)
        self._x_target   = cp.Parameter(6)

        # One (a, c) parameter pair per obstacle
        self._a_pars = [cp.Parameter((self.N, 3)) for _ in self.obstacles]
        self._c_pars = [cp.Parameter(self.N)      for _ in self.obstacles]

        x_var = cp.Variable((self.N + 1, 6))
        u_var = cp.Variable((self.N, 3))
        self._x_var, self._u_var = x_var, u_var

        cost   = 0
        constr = [x_var[0] == self._x_init_par]

        for t in range(self.N):
            cost += cp.quad_form(u_var[t], cp.psd_wrap(self.R))
            constr += [
                x_var[t + 1] == self.A_d @ x_var[t] + self.B_d @ u_var[t],
                cp.abs(u_var[t]) <= self.u_max,
            ]
            for a_par, c_par in zip(self._a_pars, self._c_pars):
                constr.append(a_par[t] @ x_var[t + 1, :3] >= c_par[t])

        for t in range(1, self.N):
            cost += cp.quad_form(x_var[t] - self._x_target, cp.psd_wrap(self.Q))
        cost += cp.quad_form(x_var[self.N] - self._x_target, cp.psd_wrap(self.Qf))

        self._problem = cp.Problem(cp.Minimize(cost), constr)

    def update_obstacle_states(self):
        """Update obstacle positions based on their position and velocities."""
        for obs in self.obstacles:
            obs['r_obj'] = obs['r_obj'] + obs['v_obj'] * self.Td
    
    def update_obstacle_params(self, r_ref):
        """Compute (a, c) for each obstacle's linearized constraint.

        With p = r_0 - r_obj:
            a = 2 S p
            c = 1 + p^T S (r_0 + r_obj)            [Eq. 12]
        """
        self.update_obstacle_states()
        for obs, a_par, c_par in zip(self.obstacles, self._a_pars, self._c_pars):
            r_0 = self.compute_linearization_points(r_ref, obs['r_obj'], obs['S'])
            p   = r_0 - obs['r_obj']
            pS  = p @ obs['S']
            a_par.value = 2.0 * pS
            c_par.value = 1.0 + np.sum(pS * (r_0 + obs['r_obj']), axis=1)

    def solve_mpc(self, x_now, x_des, r_ref):
        self._x_init_par.value = x_now
        self._x_target.value   = x_des
        self.update_obstacle_params(r_ref)
        self._problem.solve(solver=cp.CLARABEL, warm_start=True)
        if self._problem.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(f"MPC infeasible: {self._problem.status}")
        return (self._u_var.value[0],
                self._x_var.value,
                self._problem.solver_stats.solve_time)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def initial_reference(self):
        """Bootstrap: linear interp from x0 toward xt, clipped to reachable fraction."""
        horizon_time   = self.N * self.Td
        accel_max      = self.u_max / self.m_chs
        distance       = np.linalg.norm(self.xt[:3] - self.x0[:3])
        if distance < 1e-6:
            return np.tile(self.xt[:3], (self.N, 1))
        bang_bang_time = 2.0 * np.sqrt(distance / accel_max)
        frac           = min(horizon_time / bang_bang_time, 1.0)
        alphas         = np.linspace(frac / self.N, frac, self.N)[:, None]
        return self.x0[:3][None, :] + alphas * (self.xt[:3] - self.x0[:3])[None, :]

    def simulate(self):
        self.discretize()
        self.build_problem()

        r_ref = self.initial_reference()
        x     = self.x0.copy()
        traj, ctrls, solve_t = [x.copy()], [], []

        for k in range(self.max_steps):
            u_apply, x_pred, t_solve = self.solve_mpc(x, self.xt, r_ref)
            ctrls.append(u_apply)
            solve_t.append(t_solve)

            x = self.A_d @ x + self.B_d @ u_apply
            traj.append(x.copy())
            r_ref = x_pred[1:self.N + 1, :3]

            if np.linalg.norm(x[:3]) <= self.hold_dist:
                print(f"Reached holding distance ({np.linalg.norm(x[:3]):.3f} m) "
                      f"at step {k+1}")
                break
        else:
            print(f"Did not converge within {self.max_steps} steps; "
                  f"final dist = {np.linalg.norm(x[:3]):.2f} m")

        self.trajs       = np.array(traj)
        self.ctrls       = np.array(ctrls)
        self.solve_times = np.array(solve_t)

        total_impulse = np.sum(np.linalg.norm(self.ctrls, axis=1)) * self.Td
        print(f"Total impulse:   {total_impulse/1000:.2f} kN*s")
        print(f"Mean solve time: {1000*self.solve_times.mean():.2f} ms / step")
        for i, obs in enumerate(self.obstacles):
            d        = self.trajs[:, :3] - obs['r_obj']
            dists    = np.linalg.norm(d, axis=1)
            s_metric = np.sqrt(np.einsum('ij,jk,ik->i', d, obs['S'], d))
            print(f"Obstacle {i}: min Euclidean dist = {dists.min():.2f} m  "
                  f"(KOZ radius {obs['semi_axes'][0]:.0f} m), "
                  f"min S-metric (>=1) = {s_metric.min():.4f}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_trajectory(self, traj, ctrls, filename="mpc_obstacle.png"):
        fig = super().plot_trajectory(traj, ctrls, filename=filename)
        ax1, ax2, ax3 = fig.axes

        colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(self.obstacles), 1)))
        u_, v_ = np.mgrid[0:2*np.pi:24j, 0:np.pi:14j]

        for i, obs in enumerate(self.obstacles):
            ox = obs['r_obj'][0] + obs['semi_axes'][0] * np.cos(u_) * np.sin(v_)
            oy = obs['r_obj'][1] + obs['semi_axes'][1] * np.sin(u_) * np.sin(v_)
            oz = obs['r_obj'][2] + obs['semi_axes'][2] * np.cos(v_)
            ax1.plot_wireframe(ox, oy, oz, color=colors[i], alpha=0.25, lw=0.6,
                               label=f'obstacle {i}')
            ax1.plot([self.obs_start_pos[i][0], obs['r_obj'][0]],
                     [self.obs_start_pos[i][1], obs['r_obj'][1]],
                     [self.obs_start_pos[i][2], obs['r_obj'][2]],
                     color=colors[i], ls='--', lw=0.8, label=f'obstacle {i} path')
        ax1.set_title('Trajectory with obstacles')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_xlim(-200, 100)
        ax1.set_ylim(-1000, 0)
        ax1.set_zlim(-100, 50)

        # Label the target-distance line already drawn by the parent, then add obstacles
        ax3.get_lines()[0].set_label('to target')

        t_ax_s = np.arange(len(traj)) * self.Td
        for i, obs in enumerate(self.obstacles):
            dists_obs = np.linalg.norm(traj[:, :3] - obs['r_obj'], axis=1)
            ax3.semilogy(t_ax_s, dists_obs, color=colors[i], ls='-',
                         label=f'to obstacle {i}')
            ax3.axhline(obs['semi_axes'][0], color=colors[i], ls=':',
                        label=f'KOZ {i} = {obs["semi_axes"][0]:.0f} m')

        ax3.set_title('Distances vs time')
        ax3.legend(fontsize=8)
        
        plt.savefig(filename, dpi=120, bbox_inches='tight')


# -----------------------------------------------------------------------------
# Run it
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Physical and orbital parameters (Section IV.A, Table 1)
    # -----------------------------------------------------------------------------
    mu_E   = 3.986e5          # km^3/s^2, Earth gravitational parameter
    R_E    = 6378.0           # km, Earth radius
    alt    = 705.0            # km, target altitude
    a_orb  = R_E + alt        # km, semi-major axis of target orbit
    n_mm   = np.sqrt(mu_E / a_orb**3)   # rad/s, mean motion of target
    m_chs  = 6500.0           # kg, chaser mass

    # -----------------------------------------------------------------------------
    # Discretize with zero-order hold (Section II.B)
    # -----------------------------------------------------------------------------
    Td = 15.0  # s, sample period (Table 1)

    # -----------------------------------------------------------------------------
    # Cost weights and Riccati terminal weight
    # -----------------------------------------------------------------------------
    Q  = np.diag([1, 1, 1, 100, 100, 100])
    R  = np.diag([100, 100, 100])

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
    # Obstacle parameters (Section IV.B, Table 2)
    # -----------------------------------------------------------------------------
    obs1_radius = 20.0   # m, obstacle radius
    obs1_pos = np.array([-30.0, -800.0, 10.0])  # m, obstacle position in LVLH frame
    obs1_vel = np.array([-0.075, -0.005, 0.015])  # m/s, obstacle velocity in LVLH frame
    obs2_radius = 50.0   # m, obstacle radius
    obs2_pos = np.array([-100.0, 0, -40.0])  # m, obstacle position in LVLH frame
    obs2_vel = np.array([0.07, -0.16, 0.01])  # m/s, obstacle velocity in LVLH frame
    # obs2_pos = np.array([0.0, -500.0, -30.0])  # m, obstacle position in LVLH frame
    
    cw = CWObstacle(mu_E, R_E, alt, m_chs, Td)
    cw.set_cost_params(Q, R)
    cw.set_mpc_params(x0, xt, N, u_max, hold_dist, max_steps)
    cw.add_obstacle(r_obj=obs1_pos, semi_axes=[obs1_radius]*3, v_obj=obs1_vel)
    cw.add_obstacle(r_obj=obs2_pos, semi_axes=[obs2_radius]*3, v_obj=obs2_vel)
    cw.simulate()
    
    cw.plot_trajectory(cw.trajs, cw.ctrls, filename="test.png")
    plt.show()