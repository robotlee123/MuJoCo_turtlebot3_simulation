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
        self.dl, self.dr = 0.0, 0.0

    def update(self, left, right):
        self.dl = (left - self.prev_l) * cfg.WHEEL_RADIUS
        self.dr = (right - self.prev_r) * cfg.WHEEL_RADIUS
        self.prev_l, self.prev_r = left, right
        ds, dth = (self.dl + self.dr) / 2, (self.dr - self.dl) / cfg.WHEEL_BASE
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
        return 0.0, omega  # (v, omega)

class AssortedPathGenerator:
    def path_error_pill(self, x, y, theta):
        # Nearest point on the loop. Returns (e_y, e_theta, kappa, s).
        a, R = 3.0, 2.0
        if -a <= x <= a:
            if y < 0:                                        # bottom straight, heading +x
                e_y, th_p, kappa, s = y + R, 0.0, 0.0, x + a
            else:                                            # top straight, heading -x
                e_y, th_p, kappa, s = R - y, np.pi, 0.0, 2*a + np.pi*R + (a - x)
        elif x > a:                                          # right-hand turn
            phi = np.arctan2(y, x - a)
            e_y, th_p, kappa, s = R - np.hypot(x - a, y), phi + np.pi/2, 1/R, 2*a + R*(phi + np.pi/2)
        else:                                                # left-hand turn
            phi = np.arctan2(y, x + a) % (2*np.pi)
            e_y, th_p, kappa, s = R - np.hypot(x + a, y), phi + np.pi/2, 1/R, 4*a + np.pi*R + R*(phi - np.pi/2)
        return float(e_y), float(wrap(theta - th_p)), float(kappa), float(s)
    def path_error_circle(self, x, y, theta):
        # Nearest point on a circle of radius R centered at (0, 0), traversed CCW.
        # Returns (e_y, e_theta, kappa, s).
        R = 1.0
        phi = np.arctan2(y, x) % (2 * np.pi)       # polar angle in [0, 2*pi)
        e_y = R - np.hypot(x, y)                   # + inside the circle (left of travel)
        th_p = phi + np.pi / 2                     # CCW tangent heading
        kappa = 1.0 / R
        s = R * phi                                # arc length from (R, 0), in [0, 2*pi*R)
        return float(e_y), float(wrap(theta - th_p)), float(kappa), float(s)
    def path_error_square(self, x, y, theta):
        # Nearest point on a square of half-side a centered at (0, 0), traversed CCW.
        # Starts at the bottom-right corner (a, -a). Returns (e_y, e_theta, kappa, s).
        a = 0.5
        if abs(x) >= abs(y):
            if x >= 0:                                   # right edge, heading +y
                e_y, th_p = a - x, np.pi / 2
                s = np.clip(y + a, 0.0, 2*a)
            else:                                        # left edge, heading -y
                e_y, th_p = x + a, -np.pi / 2
                s = 4*a + np.clip(a - y, 0.0, 2*a)
        else:
            if y >= 0:                                   # top edge, heading -x
                e_y, th_p = a - y, np.pi
                s = 2*a + np.clip(a - x, 0.0, 2*a)
            else:                                        # bottom edge, heading +x
                e_y, th_p = y + a, 0.0
                s = 6*a + np.clip(x + a, 0.0, 2*a)
        kappa = 0.0
        return float(e_y), float(wrap(theta - th_p)), float(kappa), float(s)

class EKF_Estimator:
    def __init__(self, model):
        self.T = model.opt.timestep

        self.Q_D = np.diag([(0.2*0.1)**2, (0.2*0.1)**2, np.deg2rad(4.5)**2]) #change this
        self.R_N = np.diag([0.3**2 , np.deg2rad(10)**2])                    #change this

    def f_uni(self, x, dl, dr):
        v = (dl + dr) / (2 * self.T)
        w = (dl + dr) / (cfg.WHEEL_BASE * self.T)
        x, y, th = x
        if abs(w) < 1e-6:                       # straight-line limit
            return np.array([x + v * self.T * np.cos(th),
                                y + v * self.T * np.sin(th),
                                wrap(th)])
        th_n = th + w * self.T
        return np.array([x + v / w * (np.sin(th_n) - np.sin(th)),
                            y + v / w * (-np.cos(th_n) + np.cos(th)),
                            wrap(th_n)])

    def F_uni(self, x, dl, dr):
        # Jacobian of f_uni with respect to the state, evaluated at (x, u). 3x3.
        v = (dl + dr) / (2 * self.T)
        w = (dl + dr) / (cfg.WHEEL_BASE * self.T)
        th = x[2]
        F = np.eye(3)
        if abs(w) < 1e-6:
            F[0, 2] = -v * self.T * np.sin(th)
            F[1, 2] =  v * self.T * np.cos(th)
        else:
            th_n = th + w * self.T
            F[0, 2] = v / w * (np.cos(th_n) - np.cos(th))
            F[1, 2] = v / w * (np.sin(th_n) - np.sin(th))
        return F
    
    def h_aruco(self, x, marker):
        '''Range and bearing from pose xh to marker mk.'''
        dx, dy = marker[0] - x[0], marker[1] -  x[1]
        return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])

    def H_aruco(self, x, marker):
        '''Jacobian H_t = dh/dx, evaluated at the predicted estimate.'''
        dx, dy = marker[0] - x[0], marker[1] - x[1]
        q = dx**2 + dy**2
        r = np.sqrt(q)
        return np.array([[-dx / r, -dy / r,  0.0],
                        [ dy / q, -dx / q, -1.0]])
    
    def predict(self, xh, P, dl, dr):
        xh_pred = self.f_uni(xh, dl, dr)              # nonlinear f propagates the state
        F = self.F_uni(xh, dl, dr)                    # Jacobian propagates the covariance
        P_pred = F @ P @ F.T + self.Q_D
        return xh_pred, P_pred
    
    def update(self, xh_pred, P_pred, y, marker):
        y_hat = self.h_aruco(xh_pred, marker)            # nonlinear h gives the expected measurement
        H = self.H_aruco(xh_pred, marker)
        S = H @ P_pred @ H.T + self.R_N          # innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)      # Kalman gain
        nu = y - y_hat
        xh = xh_pred + K @ nu
        P = (np.eye(len(xh_pred)) - K @ H) @ P_pred
        return xh, P, nu, S

class LQIController:
    def __init__(self, model, Ad, Bd):
        self.T = model.opt.timestep
        
        self.Ad = Ad
        self.Bd = Bd

        self.z = 0.0
        self.calculate_gain()

    def reset(self):
        self.z = 0.0

    def calculate_gain(self):
        n = 2
        Ad_i = np.block([[self.Ad,    np.zeros((n, 1))],
                    [self.T, 0,   np.ones((1, 1))]])
        Bd_i = np.vstack([self.Bd, np.zeros((1, 1))])

        Qa = np.diag([1/0.1**2, 1/np.deg2rad(5)**2, 100.0]) # CHANGE THIS BRO
        Ra = np.array([[1/0.20**2]])                          # AND THIS!!!!!!!!!!!!!!!
        self.K, _, Ea = ct.dlqr(Ad_i, Bd_i, Qa, Ra)
        print("closed-loop poles :", np.abs(Ea))
        print("gain: ", self.K)

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

        self.calculate_error_model()
        self.check_ctrl_and_obsv_matrix()

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
        print( np.sqrt((np.asarray(est_pose) - self.true())**2))

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
