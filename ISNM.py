import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la

from obstacle import CWObstacle

class ISNM(CWObstacle):
    """
    Impulsive Spacecraft Navigation and Maneuvering (ISNM) problem with obstacles.
    This class extends CWObstacle to implement the specific ISNM scenario described in the paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = len(self.state)  # state dimension
        self.m = len(self.input)  # input dimension (impulsive maneuvers in x, y, z)
        self.nz = (self.n + self.m) * self.N  # total decision variable dimension
        self.nh = self.n * self.N
        self.nf = None
        
        self.z = np.zeros((self.nz, 1))  # [u^k x^(k+1) u^(k+1) ... x^(k+N-1) u^(k+N-1) x^(k+N)]^T
        self.z_t = np.zeros((self.nz, 1))  # [0 xt^(k+1) 0 ... xt^(k+N-1) 0 xt^(k+N)]^T

    def _idx_u(self, t):
        """Slice of z holding u_t, valid for t = 0..N-1."""
        start = t * (self.n + self.m)
        return slice(start, start + self.m)

    def _idx_x(self, t):
        """Slice of z holding x_t, valid for t = 1..N (x_0 is NOT in z)."""
        start = (t - 1) * (self.n + self.m) + self.m
        return slice(start, start + self.n)
    
    def _build_H(self):
        """Block-diagonal Hessian: [2R, 2Q, 2R, 2Q, ..., 2R, 2Qf].

        The factor of 2 converts the cvxpy-style cost (z^T M z used in
        quad_form) to the standard QP form (1/2) z^T H z.
        """
        blocks = []
        for t in range(self.N):
            blocks.append(self.R)
            blocks.append(self.Q if t < self.N - 1 else self.Qf)
        self.H = la.block_diag(*blocks)
        
    def _build_C(self):
        """Dynamics equality matrix: encodes x_{t+1} - A x_t - B u_t = 0.

        Row block t (n rows) contains:
          - column for u_t      :  -B
          - column for x_{t+1}  :  +I_n
          - column for x_t      :  -A    (only for t >= 1; x_0 not in z)
        For t = 0, x_0 = x_now goes into b (not into C).
        """
        n, m, N = self.n, self.m, self.N
        C = np.zeros((self.nh, self.nz))
        for t in range(N):
            row = slice(t * n, (t + 1) * n)
            C[row, self._idx_u(t)] = -self.B_d
            C[row, self._idx_x(t + 1)] = np.eye(n)
            if t >= 1:
                C[row, self._idx_x(t)] = -self.A_d
        self.C = C
        
    def _build_b(self, x_now):
        """Right-hand side of dynamics equality constraints: encodes x_{t+1} - A x_t - B u_t = 0.

        For t = 0, this is x_1 - A x_0 - B u_0 = 0, where x_0 is the current state and goes into b.
        For t >= 1, this is x_{t+1} - A x_t - B u_t = 0, where both x_t and x_{t+1} are in z, so the right-hand side is zero.
        """
        self.b = np.zeros((self.nh,))  # first n entries set per MPC step
        self.b[:self.n] = self.A_d @ x_now.reshape(-1, 1)  # x_0 goes into b for the first block of constraints
        
    def _build_P(self):
        m = self.m
        for t in range(self.N):
            row_pos = t * 2 * m
            col_u = self._idx_u(t).start
            self.P[row_pos:row_pos + m, col_u:col_u + m] = np.eye(m)
            self.P[row_pos + m:row_pos + 2 * m, col_u:col_u + m] = -np.eye(m)
            self.h[row_pos:row_pos + 2 * m] = self.u_max
    
    def build_problem(self):
        J = (self.z - self.z_t).T @ self.H @ (self.z - self.z_t)
        constr1 = self.C @ self.z == self.b
        constr2 = self.P @ self.z <= self.h
    
    def solve_mpc(self):
        return super().solve_mpc()
