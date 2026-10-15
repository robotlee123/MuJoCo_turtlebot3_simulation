import numpy as np
import config as cfg
from drive import wrap


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


class ScriptedController:
    """Your original zig-zag: flips turn direction every second (sim time)."""

    def command(self, t):
        omega = -1.82 if int(t) % 2 == 0 else 1.82
        return 0.2, omega  # (v, omega)


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