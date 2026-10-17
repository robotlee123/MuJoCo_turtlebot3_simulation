"""Platform-level helpers: multi-rate timer, kinematics, actuation."""
import numpy as np
import config as cfg


class RateTimer:
    """Fires at a fixed rate in sim time. Replaces the *_cnt variables."""

    def __init__(self, rate_hz):
        self.period = 1.0 / rate_hz
        self.next_t = 0.0

    def due(self, t):
        if t >= self.next_t:
            self.next_t += self.period
            return True
        return False


def wrap(a):
    # Wrap an angle, or an array of angles, to (-pi, pi).
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def quantize_rads(rads, ticks_per_rev=cfg.TICKS_PER_REV):
    """Quantize an angle (rad) to encoder tick resolution."""
    tick = 2 * np.pi / ticks_per_rev
    return np.round(rads / tick) * tick


def ddr_ik(v, omega, max_w, r=cfg.WHEEL_RADIUS, l=cfg.WHEEL_BASE):
    """
    Differential drive inverse kinematics: calculate wheels speeds from desired velocity.
    reference: https://www.roboticsbook.org/S52_diffdrive_actions.html
    If either wheel exceeds max_w, both are scaled so the turn ratio is kept."""
    wl = (2 * v - l * omega) / (2 * r)
    wr = (2 * v + l * omega) / (2 * r)
    s = max(1.0, abs(wl) / max_w, abs(wr) / max_w)
    return wl / s, wr / s


class Drive:
    """Turns (v, omega) commands into actuator controls."""

    def __init__(self, model, data):
        self.data = data
        hi = model.actuator_ctrlrange[0, 1]
        self.max_w = hi if hi > 0 else cfg.DEFAULT_MAX_WHEEL_SPEED

    def apply(self, v, omega):
        self.data.ctrl[0], self.data.ctrl[1] = ddr_ik(v, omega, self.max_w)