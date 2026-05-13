"""
Milestone 3, step 1: class skeleton and static QP matrix builders for the
Infeasible Start Newton Method (ISNM) solver.

Variable layout (Section III.A of the paper):
    z = [u_0, x_1, u_1, x_2, ..., u_{N-1}, x_N]

QP in standard form:
    minimize   (1/2) z^T H z + g^T z
    subject to C z = b           (dynamics)
               P z <= h           (thrust limits + linearized obstacle)
"""

import numpy as np
import scipy.linalg

from obstacle import CWObstacle


class CWMPCWithObstacleISNM(CWObstacle):
    """LQ-MPC with obstacle avoidance, solved with a custom Infeasible Start
    Newton Method using a log-barrier interior-point formulation."""

    # Paper hyperparameters (Section III.A, Algorithm 1)
    KAPPA = 0.05          # log-barrier weight; smaller = sharper approximation
    ALPHA = 0.01          # Armijo sufficient-decrease parameter
    BETA = 0.95           # backtracking step-shrink factor
    MAX_NEWTON_ITERS = 10
    EPS = 1e-9            # convergence tolerance on residual norm

    # ------------- index helpers (z-vector layout) -------------------------
    def _idx_u(self, t):
        """Slice of z holding u_t, valid for t = 0..N-1."""
        start = t * (self.n + self.m)
        return slice(start, start + self.m)

    def _idx_x(self, t):
        """Slice of z holding x_t, valid for t = 1..N (x_0 is NOT in z)."""
        start = (t - 1) * (self.n + self.m) + self.m
        return slice(start, start + self.n)

    # ------------- the top-level builder -----------------------------------
    def build_problem(self):
        """Build the static QP matrices.

        Static (built once): H, g, dynamics rows of C, thrust rows of P/h.
        Dynamic (filled per MPC step): first n entries of b, obstacle rows
        of P/h.
        """
        self.n, self.m = 6, 3
        self.nz = self.N * (self.n + self.m)
        self.neq = self.N * self.n
        self.n_thrust = 2 * self.m * self.N
        self.n_obstacle = self.N
        self.nineq = self.n_thrust + self.n_obstacle

        self.H = self._build_H()
        self.C = self._build_C()
        self.P = np.zeros((self.nineq, self.nz))
        self.h = np.zeros(self.nineq)
        self._fill_thrust_constraints()         # writes top n_thrust rows of P, h
        self.g = self._build_g()
        self.b = np.zeros(self.neq)             # first n entries set per MPC step

    # ------------- individual matrix builders ------------------------------
    def _build_H(self):
        """Block-diagonal Hessian: [2R, 2Q, 2R, 2Q, ..., 2R, 2Qf].

        The factor of 2 converts the cvxpy-style cost (z^T M z used in
        quad_form) to the standard QP form (1/2) z^T H z.
        """
        blocks = []
        for t in range(self.N):
            blocks.append(2 * self.R)
            blocks.append(2 * (self.Q if t < self.N - 1 else self.Qf))
        return scipy.linalg.block_diag(*blocks)

    def _build_C(self):
        """Dynamics equality matrix: encodes x_{t+1} - A x_t - B u_t = 0.

        Row block t (n rows) contains:
          - column for u_t      :  -B
          - column for x_{t+1}  :  +I_n
          - column for x_t      :  -A    (only for t >= 1; x_0 not in z)
        For t = 0, x_0 = x_now goes into b (not into C).
        """
        n, m, N = self.n, self.m, self.N
        C = np.zeros((self.neq, self.nz))
        for t in range(N):
            row = slice(t * n, (t + 1) * n)
            C[row, self._idx_u(t)] = -self.B_d
            C[row, self._idx_x(t + 1)] = np.eye(n)
            if t >= 1:
                C[row, self._idx_x(t)] = -self.A_d
        return C

    def _fill_thrust_constraints(self):
        """Top 2mN rows of (P, h): per-axis thrust limits |u_t[i]| <= u_max.

        For each t, six rows: three of +I_m and three of -I_m at the u_t
        columns; corresponding h entries are all u_max.
        """
        m = self.m
        for t in range(self.N):
            row_pos = t * 2 * m
            col_u = self._idx_u(t).start
            self.P[row_pos:row_pos + m, col_u:col_u + m] = np.eye(m)
            self.P[row_pos + m:row_pos + 2 * m, col_u:col_u + m] = -np.eye(m)
            self.h[row_pos:row_pos + 2 * m] = self.u_max

    def _build_g(self):
        """Linear cost term: -2 Q x_des at each x_t (or -2 Qf x_des at x_N).

        Comes from expanding (x_t - x_des)^T Q (x_t - x_des):
            = x_t^T Q x_t - 2 x_des^T Q x_t + (constant)
        Constants don't affect the optimizer, so we drop them.
        """
        g = np.zeros(self.nz)
        for t in range(self.N):
            Q_eff = self.Q if t < self.N - 1 else self.Qf
            g[self._idx_x(t + 1)] = -2.0 * Q_eff @ self.xt
        return g

    # ------------- stubs to be implemented in step 3 -----------------------
    def solve_mpc(self, x_now, x_des, r_ref):
        raise NotImplementedError("ISNM solver coming in step 3")


# ===========================================================================
# Verification: build everything and check structure
# ===========================================================================
if __name__ == "__main__":
    sim = CWMPCWithObstacleISNM(mu_E=3.986e5, R_E=6378.0, alt=705.0,
                                m_chs=6500.0, Td=15.0)
    sim.set_cost_params(Q=np.diag([1, 1, 1, 100, 100, 100]),
                        R=np.diag([100, 100, 100]))
    sim.set_mpc_params(
        x0=np.array([50.0, -1000.0, 10.0, 0.0, 0.0, 0.0]),
        xt=np.zeros(6), N=30, u_max=20.0, hold_dist=2.3, max_steps=200,
    )
    sim.set_obstacle(r_obj=[-100.0, -500.0, 0.0], semi_axes=[50.0, 50.0, 50.0])
    sim.discretize()
    sim.build_problem()

    # ----- 1. Dimensions ----------------------------------------------------
    print("=== Dimensions ===")
    print(f"nz       = {sim.nz}   (variables;            expect 270)")
    print(f"neq      = {sim.neq}   (equality constraints; expect 180)")
    print(f"nineq    = {sim.nineq}   (inequalities;         expect 210)")
    print(f"  n_thrust   = {sim.n_thrust}   (2 m N)")
    print(f"  n_obstacle = {sim.n_obstacle}   (one per horizon step)")

    # ----- 2. Matrix shapes -------------------------------------------------
    print("\n=== Matrix shapes ===")
    for name, M in [('H', sim.H), ('C', sim.C), ('P', sim.P),
                    ('h', sim.h), ('g', sim.g), ('b', sim.b)]:
        print(f"  {name}: {M.shape}")

    # ----- 3. H structural checks -------------------------------------------
    print("\n=== H structure ===")
    print(f"  symmetric:        {np.allclose(sim.H, sim.H.T)}")
    print(f"  min eigenvalue:   {np.linalg.eigvalsh(sim.H).min():.4f}  (>0 ⇒ pos. def.)")
    print(f"  H[0:3, 0:3]  (should be 2R = 200*I):")
    print("   ", sim.H[0:3, 0:3].tolist())
    print(f"  H[3:9, 3:9]  (should be 2Q = diag(2, 2, 2, 200, 200, 200)):")
    print("   ", np.diag(sim.H[3:9, 3:9]))

    # ----- 4. C structural check: forward-sim a feasible trajectory ---------
    print("\n=== C structural check (forward-sim feasibility) ===")
    x0_test = sim.x0.copy()
    rng = np.random.default_rng(0)
    u_traj = (rng.random((sim.N, sim.m)) - 0.5) * sim.u_max     # bounded controls
    x_traj = np.zeros((sim.N + 1, sim.n))
    x_traj[0] = x0_test
    for t in range(sim.N):
        x_traj[t + 1] = sim.A_d @ x_traj[t] + sim.B_d @ u_traj[t]

    z_test = np.zeros(sim.nz)
    for t in range(sim.N):
        z_test[sim._idx_u(t)] = u_traj[t]
        z_test[sim._idx_x(t + 1)] = x_traj[t + 1]

    b_expected = np.zeros(sim.neq)
    b_expected[:sim.n] = sim.A_d @ x0_test
    eq_residual = sim.C @ z_test - b_expected
    print(f"  max |C z - b| on a dynamics-consistent z: {np.max(np.abs(eq_residual)):.2e}  (≈0 expected)")

    # ----- 5. P/h structural check ------------------------------------------
    print("\n=== Thrust constraint check ===")
    print(f"  P[:6, :3]  (should be I_3 stacked on -I_3):")
    print("   ", sim.P[:6, :3].astype(int).tolist())
    Pz_thrust = sim.P[:sim.n_thrust] @ z_test
    slack_thrust = sim.h[:sim.n_thrust] - Pz_thrust
    print(f"  min slack on thrust rows: {slack_thrust.min():.3f}  (>0 ⇒ test controls within ±u_max)")

    print("\n=== Obstacle rows are still zero (filled per MPC step) ===")
    print(f"  ||P[obstacle rows]|| = {np.linalg.norm(sim.P[sim.n_thrust:]):.4f}")
    print(f"  ||h[obstacle rows]|| = {np.linalg.norm(sim.h[sim.n_thrust:]):.4f}")

    print("\nStep 1 complete. ✓")