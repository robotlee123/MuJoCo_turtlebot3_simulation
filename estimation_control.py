import numpy as np
import config as cfg
from drive import wrap

import control as ct
import sympy as sp

class DiffDriveOdometry:
    """Dead reckoning from wheel encoder angles."""

    def __init__(self, left0, right0, x=0.0, y=0.0, th=0.0):
        self.prev_l, self.prev_r = left0, right0
        self.x, self.y, self.th = x, y, th

    def update(self, left, right):
        dl = (left - self.prev_l) * cfg.WHEEL_RADIUS
        dr = (right - self.prev_r) * cfg.WHEEL_RADIUS
        self.prev_l, self.prev_r = left, right
        ds, dth = (dl + dr) / 2, (dr - dl) / cfg.WHEEL_BASE
        self.x += ds * np.cos(self.th + dth / 2)
        self.y += ds * np.sin(self.th + dth / 2)
        self.th = wrap(self.th + dth)
        return self.pose

    @property
    def pose(self):
        return np.array([self.x, self.y, self.th])


class ZigZagController:
    """original zig-zag: flips turn direction every second (sim time)."""

    def command(self, t):
        omega = -1.82 if int(t) % 2 == 0 else 1.82
        return 0.2, omega  # (v, omega)

class LQIController:
    def __init__(self, model, Ad, Bd):
        self.T = model.opt.timestep
        
        self.Ad = Ad
        self.Bd = Bd

        self.z = 0.0
        self.K = np.zeros((3,3))

    def reset(self):
        self.z = 0.0

    def calculate_gain(self):
        n = 2
        Ad_i = np.block([[self.Ad,    np.zeros((n, 1))],
                    [self.T, 0,   np.ones((1, 1))]])
        Bd_i = np.vstack([self.Bd, np.zeros((1, 1))])
        
        Qa = np.diag([1/0.2**2, 1/np.deg2rad(15)**2, 100.0]) 
        Ra = np.array([[1/0.1**2]])
        self.K, _, Ea = ct.dlqr(Ad_i, Bd_i, Qa, Ra)
        print("closed-loop poles :", np.abs(Ea))

    def step(self, e_y, e_theta, kappa, v):
        # 1. Command from the augmented state [e_y, e_theta, z], plus feedforward
        # 2. Clip to +/- OMEGA_MAX
        # 3. Update the integrator (with anti-windup) and return the command

        omega = v*kappa - self.K[:, :2] @ np.array([e_y,e_theta]).transpose() - self.K[:, 2:] * self.z
        self.z = self.z + self.T * e_y
        return np.clip(omega, -cfg.OMEGA_MAX, cfg.OMEGA_MAX)[0][0]

class ErrorModel:
    """Error model for the robot
        Linearised at the equilibrium e=0 domega = 0 kappa= 0.0 (straight) at a given v
        discretised with zoh at sample time Ts (model.opt.timestep)
    """
    def __init__(self, model, v):
        self.model = model
        self.v = v

        self.Ts = self.model.opt.timestep
        self.kappa = 0.0

        self.Ad = np.zeros((2,2))
        self.Bd = np.zeros((2,1))
        self.C = np.zeros((2,2))

    def calculate_error_model(self):
        # Discrete linearised path-following error model.
        # State e = [e_y, e_theta], input d_omega = omega - v*kappa. Returns (Ad, Bd).
        # week 5 part b
        # assesment 3 part a.1

        v_, e_theta, kappa_,  d_omega, e_y = sp.symbols("v etheta kappa domega ey")

        f = sp.Matrix([v_*sp.sin(e_theta),
                        v_*kappa_ + d_omega - ((v_*kappa_*sp.cos(e_theta))/(1-kappa_*e_y))])

        # 1. Continuous Jacobians A (2x2) and B (2x1) at e = 0, d_omega = 0
        A = f.jacobian(sp.Matrix([e_y, e_theta]))
        B = f.jacobian(sp.Matrix([d_omega]))

        sub = {v_: self.v, kappa_: self.kappa, e_y: 0.0, e_theta: 0.0, d_omega: 0.0 }

        A = np.array(A.subs(sub), dtype=float)
        B = np.array(B.subs(sub), dtype=float)
        
        C = np.array([[1.0, 0.0],
                      [0.0, 1.0]]) # position and heading measured
        D = np.zeros((2, 1))

        # 2. Zero-order-hold discretisation at Ts
        model_c_ss = ct.ss(A, B, C, D)
        model_d = ct.c2d(model_c_ss, self.Ts, method="zoh")

        self.Ad, self.Bd, self.C = np.array(model_d.A), np.array(model_d.B), C

    def check_ctrl_and_obsv_matrix(self):
        Cm = ct.ctrb(self.Ad, self.Bd)
        print("controllability matrix\n", Cm)
        print("rank =", np.linalg.matrix_rank(Cm), " (n = 2)")
        print("controllable:", np.linalg.matrix_rank(Cm) == self.Ad.shape[0])

        Om = ct.obsv(self.Ad, self.C)
        sv = np.linalg.svd(Om, compute_uv=False)
        print("rank O = %d (n = 2) singular values: %s" % (np.linalg.matrix_rank(Om), np.round(sv, 3)))
        print("Observable:", np.linalg.matrix_rank(Om) == self.Ad.shape[0])

class GroundTruthLogger:
    """Reads ground truth for evaluation/plots only. Never import this
    from estimator or controller code."""

    def __init__(self, data):
        self.data = data
        self.t, self.true_pose, self.est_pose = [], [], []

    def true(self):
        p = self.data.body(cfg.BASE_BODY).xpos
        R = self.data.body(cfg.BASE_BODY).xmat.reshape(3, 3)
        return np.array([p[0], p[1], np.arctan2(R[1, 0], R[0, 0])])

    def record(self, t, est_pose):
        self.t.append(t)
        self.true_pose.append(self.true())
        self.est_pose.append(np.asarray(est_pose))

    def summary(self):
        t = np.array(self.true_pose)
        e = np.array(self.est_pose)
        if len(t) == 0:
            return
        err = e - t
        err[:, 2] = wrap(err[:, 2])
        print(f"final true: {t[-1]}\nfinal est:  {e[-1]}")
        print(f"pos RMSE: {np.sqrt((err[:, :2] ** 2).sum(1).mean()):.4f} m | "
              f"heading RMSE: {np.sqrt((err[:, 2] ** 2).mean()):.4f} rad")