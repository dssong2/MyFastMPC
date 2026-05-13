import numpy as np
import sympy as sp
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from mpl_toolkits.mplot3d import Axes3D

class ClohessyWiltshire:
    """
    Clohessy-Wiltshire (Hill's) equations of relative motion for spacecraft rendezvous.
    See Eq. (3) in the paper.

    The MPC problem is formulated as:

        minimize   sum_{t=0}^{N-1} (x_t' Q x_t + u_t' R u_t) + x_N' Qf x_N
        s.t.  x_{t+1} = A x_t + B u_t      (CW dynamics, ZOH-discretized)
              |u_t|   <= u_max             (per-axis thrust limit)
              x_0     = current state
    """
    def __init__(self, mu_E, R_E, alt, m_chs, Td):
        # Clohessy-Wiltshire parameters
        self.mu_E = mu_E
        self.R_E = R_E
        self.alt = alt
        self.m_chs = m_chs
        
        # Discretization time step
        self.Td = Td
        
        # Discrete-time dynamics matrices (to be computed)
        self.A_d = None
        self.B_d = None
        
        # Cost parameters
        self.Q = None
        self.R = None
        self.Qf = None
        
        # MPC parameters
        self.x0 = None
        self.xt = None
        self.N = None
        self.u_max = None
        self.hold_dist = None
        self.max_steps = None
        
        # Compute mean motion n and continuous-time A_c, B_c matrices
        a_orb = R_E + alt
        self.n_mm = np.sqrt(mu_E / a_orb**3)
        
        x, y, z, x_dot, y_dot, z_dot = sp.symbols('x y z x_dot y_dot z_dot')
        self.state = sp.Matrix([x, y, z, x_dot, y_dot, z_dot])
        u_x, u_y, u_z = sp.symbols('u_x u_y u_z')
        self.input = sp.Matrix([u_x, u_y, u_z])
        
        # Outputs
        self.trajs = None
        self.ctrls = None
        self.solve_times = None
        
    def set_cost_params(self, Q, R):
        self.Q = Q
        self.R = R
        
    def set_mpc_params(self, x0, xt, N, u_max, hold_dist, max_steps):
        self.x0 = x0
        self.xt = xt
        self.N = N
        self.u_max = u_max
        self.hold_dist = hold_dist
        self.max_steps = max_steps
        
    def dynamics(self):
        n = self.n_mm
        m = self.m_chs
        
        # Continuous-time dynamics (Eq. 3)
        x_dot = sp.Matrix([
            self.state[3],  # x_dot
            self.state[4],  # y_dot
            self.state[5],  # z_dot
            3*n**2*self.state[0] + 2*n*self.state[4] + self.input[0]/m,  # x_ddot
            -2*n*self.state[3] + self.input[1]/m,  # y_ddot
            -n**2*self.state[2] + self.input[2]/m   # z_ddot
        ])
        
        return x_dot
    
    def discretize(self):
        # Get continuous-time A_c, B_c matrices from the symbolic dynamics
        x_dot = self.dynamics()
        A_c = x_dot.jacobian(self.state)
        B_c = x_dot.jacobian(self.input)
        
        # After jacobian, A_c and B_c have no symbolic variables, so we can convert to numeric arrays directly
        A_c_num = np.array(A_c).astype(np.float64)
        B_c_num = np.array(B_c).astype(np.float64)
        
        # Discretize using zero-order hold
        A_d, B_d, _, _, _ = cont2discrete((A_c_num, B_c_num, np.eye(6), np.zeros((6, 3))), self.Td, method='zoh')
        
        self.A_d = A_d
        self.B_d = B_d
        self.Qf = solve_discrete_are(A_d, B_d, self.Q, self.R)
    
    def build_problem(self):
        self._x_init_par = cp.Parameter(6)
        self._x_target   = cp.Parameter(6)

        x_var = cp.Variable((self.N + 1, 6))
        self._u_var = cp.Variable((self.N, 3))

        cost   = 0
        constr = [x_var[0] == self._x_init_par]

        for t in range(self.N):
            cost += cp.quad_form(self._u_var[t], cp.psd_wrap(self.R))
            constr += [
                x_var[t + 1] == self.A_d @ x_var[t] + self.B_d @ self._u_var[t],
                cp.abs(self._u_var[t]) <= self.u_max,
            ]

        for t in range(1, self.N):
            cost += cp.quad_form(x_var[t] - self._x_target, cp.psd_wrap(self.Q))
        cost += cp.quad_form(x_var[self.N] - self._x_target, cp.psd_wrap(self.Qf))

        self._problem = cp.Problem(cp.Minimize(cost), constr)

    def solve_mpc(self, x_now, x_des):
        self._x_init_par.value = x_now
        self._x_target.value   = x_des
        self._problem.solve(solver=cp.OSQP, warm_start=True)
        if self._problem.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(f"MPC infeasible: {self._problem.status}")
        return self._u_var.value[0], self._problem.solver_stats.solve_time
    
    def simulate(self):
        self.discretize()
        self.build_problem()
        x         = self.x0.copy()
        traj      = [x.copy()]
        ctrls     = []
        solve_t   = []

        for k in range(self.max_steps):
            u_apply, t_solve = self.solve_mpc(x, self.xt)
            ctrls.append(u_apply)
            solve_t.append(t_solve)
            x = self.A_d @ x + self.B_d @ u_apply
            traj.append(x.copy())
            if np.linalg.norm(x[:3]) <= self.hold_dist:
                print(f"Reached holding distance ({np.linalg.norm(x[:3]):.3f} m) "
                    f"at step {k+1}")
                break
        else:
            print(f"Did not converge within {self.max_steps} steps; "
                f"final dist = {np.linalg.norm(x[:3]):.2f} m")

        traj   = np.array(traj)
        ctrls  = np.array(ctrls)
        solve_t = np.array(solve_t)

        # Total impulse: integral of |u| dt approximated with sample period
        total_impulse = np.sum(np.linalg.norm(ctrls, axis=1)) * self.Td  # N*s

        print(f"Mission steps:       {len(ctrls)}  ({len(ctrls)*self.Td:.0f} s = "
            f"{len(ctrls)*self.Td/60:.1f} min)")
        print(f"Total impulse:       {total_impulse/1000:.2f} kN*s")
        print(f"Mean solve time:     {1000*solve_t.mean():.2f} ms / step")
        print(f"Final dist to target {np.linalg.norm(traj[-1, :3]):.3f} m")
        
        self.trajs = traj
        self.ctrls = ctrls
        self.solve_times = solve_t

    def plot_trajectory(self, traj, ctrls, filename='mpc_dynamics.png'):
        fig = plt.figure(figsize=(14, 5))

        # 3D trajectory: paper plots in-track on x-axis, radial on y-axis, cross-track on z
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', linewidth=1.6, label='chaser path')
        ax1.scatter([0], [0], [0], c='black', s=120, marker='o', label='target')
        ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                    c='red', s=80, marker='s', label='chaser start')
        ax1.set_xlabel('Radial x [m]')
        ax1.set_ylabel('In-track y [m]')
        ax1.set_zlabel('Cross-track z [m]')
        ax1.set_title('Trajectory')
        ax1.legend(loc='upper left', fontsize=9)

        # Thrust profile
        t_axis = np.arange(len(ctrls)) * self.Td
        ax2 = fig.add_subplot(132)
        ax2.step(t_axis, ctrls[:, 0], where='post', label=r'$u_x$')
        ax2.step(t_axis, ctrls[:, 1], where='post', label=r'$u_y$')
        ax2.step(t_axis, ctrls[:, 2], where='post', label=r'$u_z$')
        ax2.axhline( self.u_max, color='k', linestyle='--', alpha=0.4)
        ax2.axhline(-self.u_max, color='k', linestyle='--', alpha=0.4,
                    label=r'$\pm u_{\max}$')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Thrust [N]')
        ax2.set_title('Commanded thrust')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # Distance to target
        ax3 = fig.add_subplot(133)
        dists = np.linalg.norm(traj[:, :3], axis=1)
        t_axis_s = np.arange(len(traj)) * self.Td
        ax3.semilogy(t_axis_s, dists, 'b-')
        ax3.axhline(self.hold_dist, color='g', linestyle='--', label=f'hold dist = {self.hold_dist} m')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Distance to target [m]')
        ax3.set_title('Range vs time (log scale)')
        ax3.legend()
        ax3.grid(alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        
        return fig