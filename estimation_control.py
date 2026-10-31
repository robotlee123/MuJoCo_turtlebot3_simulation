import random
random.seed(0)
import numpy as np
np.random.seed(0)
import config as cfg
from drive import wrap

import control as ct
import sympy as sp

from spatialmath import SE2
import open3d as o3d

import matplotlib.pyplot as plt

from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *

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

class PathGenerator:
    def __init__(self, map_path, inflation_radius = 8.0, visualise_replan=False):
        self.raw_map = np.load(map_path)
        self.inflation_radius = inflation_radius
        self.visualise_replan = visualise_replan
        
        self.X0, self.Y0, self.res = -4.05, -3.05, 0.05
        self.y_bound, self.x_bound = 0, 0

        self.path = [] #stores current path being tracked
        self.map_ = None
        self.path_history = [] #stores all the generated path
        self.load_map()

    def w2g(self, coords):
        """world (x, y) -> grid (col, row)"""
        return (int(round((coords[0] - self.X0) / self.res - 0.5)),
                int(round((coords[1] - self.Y0) / self.res - 0.5)))

    def g2w(self, col, row):
        """grid (col, row) -> world (x, y) at the cell center"""
        return (round(self.X0 + (col + 0.5) * self.res, 2),
                round(self.Y0 + (row + 0.5) * self.res, 2))
    
    def load_map(self):
        # Load raw map into python_motion map
        self.y_bound, self.x_bound = self.raw_map.shape
        self.map_ = Grid(bounds=[[0, self.x_bound], [0, self.y_bound]],  strict_collision=True)
        self.map_.fill_boundary_with_obstacles()
        for y, i in enumerate(self.raw_map):
            for x, j in enumerate(i):
                if j == 1:
                    self.map_[x,y] = TYPES.OBSTACLE
        self.map_.inflate_obstacles(radius=self.inflation_radius)

    def to_o3d(self, pts_2d: np.ndarray, normals_2d: np.ndarray = None) -> o3d.geometry.PointCloud:
        """o3d only accepts 3d coordinates, this add a z axis that is 0, and adds the normals if needed"""
        pts_3d = np.hstack([pts_2d, np.zeros((len(pts_2d), 1))])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_3d)
        if normals_2d is not None:
            n3 = np.hstack([normals_2d, np.zeros((len(normals_2d), 1))])
            pcd.normals = o3d.utility.Vector3dVector(n3)
        return pcd

    def estimate_normals_2d(self, pts_2d: np.ndarray, radius=0.2, max_nn=30) -> np.ndarray:
        """claude created this so that point to plane works with 2d points """
        pcd = self.to_o3d(pts_2d)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        n = np.asarray(pcd.normals)[:, :2]
        # Re-normalize after dropping z (for planar points it should already be ~0)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(norms, 1e-12)
    
    def add_local_map(self, local_map):
        self.load_map() # clear the map

        #convert new points to grid coords
        #from here the maps are still numpy arrays and in grid coords
        obstacle_new = []
        for coords in local_map:
            obstacle_new.append(self.w2g((coords[0], coords[1])))

        obstacle_map = []
        for y, i in enumerate(self.raw_map):
            for x, j in enumerate(i):
                if j == 1:
                    obstacle_map.append([x,y])

        obstacle_new = np.array(obstacle_new)
        obstacle_map = np.array(obstacle_map)

        target_normals = self.estimate_normals_2d(obstacle_map, radius=5)
        #ICP is used to align the local map to global map, because the predicted state not correct.                                                                 
        result = o3d.pipelines.registration.registration_icp(
            self.to_o3d(obstacle_new), #source,
            self.to_o3d(obstacle_map, target_normals), #target,
            25, # max_dist,
            np.eye(4),#init
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500),
            ) # tested point to point it did not work

        T = result.transformation
        # preserve x, y and theta
        yaw = np.arctan2(T[1, 0], T[0, 0])
        T_icp =  SE2(T[0, 3], T[1, 3], yaw)
        # transform new obstacles to align with map
        aligned = (T_icp * obstacle_new.T).T

        for coords in aligned:
            if int(coords[0]) < self.x_bound and int(coords[1]) < self.y_bound: # rejects coords that are out of bounds
                self.map_[int(coords[0]), int(coords[1])] = TYPES.OBSTACLE
        self.map_.inflate_obstacles(radius=self.inflation_radius)
    
    def plan(self, start:tuple[float, float], goal:tuple[float, float], local_map=None):
        # start and goal position in world coordinate
        # Convert start and goal position to grid coordinate
        start = self.w2g(start)  
        goal = self.w2g(goal)   
        if self.visualise_replan:
            strict_vis = Visualizer2D()
        # Run planner
        if local_map != None:
            self.add_local_map(np.asarray(local_map)) 
            if self.visualise_replan:           
                strict_vis.plot_grid_map(self.map_)

        planner = LazyThetaStar(map_=self.map_, start=start, goal=goal)
        path, _ = planner.plan()

        if local_map != None and self.visualise_replan:
            strict_vis.plot_path(path, style="--", color="C4")
            strict_vis.show()
            strict_vis.close()

        path_result = []
        for waypoint in path:
            #change back to world coords
            path_result.append(self.g2w(waypoint[0], waypoint[1]))
        self.path_history.append(np.asarray(path_result))
        self.path = path_result
    
    @staticmethod
    def compute_path_errors(x, path):
        """
        Claude made this path error function.
        """
        # print(path)
        x = np.asarray(x, dtype=float)
        path = np.asarray(path, dtype=float)

        R = x[:2] # x, y
        theta = x[2] # theta

        # Consecutive waypoint pairs define the segments
        A = path[:-1]
        B = path[1:]
        V = B - A

        # Ignore zero-length segments
        length_sq = np.sum(V**2, axis=1)
        valid = length_sq > 1e-12 # its valid only if its not zero

        if not np.any(valid):
            raise ValueError("Path has no valid segments")

        # Project robot onto each segment
        t = np.sum((R - A) * V, axis=1) / np.maximum(length_sq, 1e-12)
        t = np.clip(t, 0.0, 1.0)

        # Closest point on each segment
        Q = A + t[:, None] * V

        # Squared distances to closest points
        W = R - Q
        dist_sq = np.sum(W**2, axis=1)
        dist_sq[~valid] = np.inf

        # Select closest segment
        idx = np.argmin(dist_sq)

        v = V[idx]
        w = W[idx]

        # Signed cross-track error
        cte = (
            v[0] * w[1] - v[1] * w[0]
        ) / np.sqrt(length_sq[idx])

        # Desired heading along the selected segment
        theta_path = np.arctan2(v[1], v[0])

        # Heading error, wrapped to [-pi, pi]
        heading_error = wrap(theta - theta_path)

        return cte, heading_error, idx
    
class LocalMap:
    def __init__(self):
        self.map = []
        self.FOV = np.deg2rad(15)
        self.min_dist = 0.1
        self.DIST_THRESHOLD = 0.8
        self.RESET_DIST = 1.2
        self.obstacle = False

    def detect_obstacle(self, points):
        point_in_robot_frame = points + np.array([-0.049, 0, 0.1255])
        ranges = np.hypot(point_in_robot_frame[:, 0], point_in_robot_frame[:, 1])
        angles = wrap(np.arange(ranges.size) * ((2*np.pi)/ranges.size))
        valid = (np.isfinite(ranges) & (ranges >= self.min_dist) & (np.abs(angles) <= self.FOV))
        if not valid.any():
            self.obstacle = False
            return
        masked = np.where(valid, ranges, np.inf)
        i = np.argmin(masked)
        self.range = masked[i]

        self.obstacle = self.range < self.DIST_THRESHOLD

    def update_map(self, points, xh):
        self.map = []
        self.detect_obstacle(points)
        for idx, range_ in enumerate(np.hypot(points[:, 0], points[:, 1])):
            if range_ < 3.0 and not np.isnan(range_):
                point_in_robot_frame = points[idx] + np.array([-0.049, 0, 0.1255])
                
                Lidar_rotation = np.array([[-1.0,0.0, 0.0],
                                           [0.0, -1, 0.0],
                                           [0.0, 0.0, 1.0]
                                        ]) # Correct from the rotation in MJCF. In the MJCF its rotated 180 degrees because it was facing backwards.
                
                cos_t = np.cos(xh[2])
                sin_t = np.sin(xh[2])
                R = np.array([[cos_t, -sin_t, 0.0],
                              [sin_t,  cos_t, 0.0],
                              [  0.0,    0.0, 1.0]
                            ])
                t = np.array([xh[0], xh[1], 0.0])
                point_in_world_frame=((point_in_robot_frame @ Lidar_rotation) @ R.T) + t
                self.map.append(point_in_world_frame[:2])
    
class EKF_Estimator:
    def __init__(self, model, Qd, Rn):
        self.T = model.opt.timestep

        self.Q_D = Qd #The larger, the model assumes the model is bad and rely on sensors more
        self.R_N = Rn #The larger, the model assumes the sensor is bad and rely on model more

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
        y_hat = self.h_aruco(xh_pred, marker)            # what the range should be
        H = self.H_aruco(xh_pred, marker)
        S = H @ P_pred @ H.T + self.R_N          # innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)      # Kalman gain
        nu = y - y_hat
        xh = xh_pred + K @ nu
        P = (np.eye(len(xh_pred)) - K @ H) @ P_pred
        return xh, P, nu, S
    
class LQIController:
    def __init__(self, model, Ad, Bd, Qa, Ra):
        self.T = model.opt.timestep
        
        self.Ad = Ad
        self.Bd = Bd

        self.z = 0.0

        self.Qa = Qa # the larger the controller will try to quickly fix the errors
        self.Ra = Ra # the larger the slower the response (control input becomes more expensive)

        self.calculate_gain()

    def reset(self):
        self.z = 0.0

    def calculate_gain(self):
        n = 2
        Ad_i = np.block([[self.Ad,    np.zeros((n, 1))],
                    [self.T, 0,   np.ones((1, 1))]])
        Bd_i = np.vstack([self.Bd, np.zeros((1, 1))])

        self.K, _, Ea = ct.dlqr(Ad_i, Bd_i, self.Qa, self.Ra)
        print("==========LQI STATS============")
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
        print("================ERROR MODEL STATS=============")
        print("================Controllability=============")
        Cm = ct.ctrb(self.Ad, self.Bd)
        print("controllability matrix\n", Cm)
        print("rank =", np.linalg.matrix_rank(Cm), " (n = 2)")
        print("controllable:", np.linalg.matrix_rank(Cm) == self.Ad.shape[0])

        Om = ct.obsv(self.Ad, self.C)
        sv = np.linalg.svd(Om, compute_uv=False)
        print("================Observeability=============")
        print("rank O = %d (n = 2) singular values: %s" % (np.linalg.matrix_rank(Om), np.round(sv, 3)))
        print("Observable:", np.linalg.matrix_rank(Om) == self.Ad.shape[0])

class GroundTruthLogger:
    """Reads ground truth for evaluation/plots only. Never import this
    from estimator or controller code."""

    def __init__(self, data):
        self.data = data
        # sim time, x_true, xh/x_predict
        self.t, self.true_pose, self.est_pose = [], [], []
        self.P, self.U = [], []
        self.e_y_true, self.e_y_est = [], []

    def true(self):
        p = self.data.body(cfg.BASE_BODY).xpos
        R = self.data.body(cfg.BASE_BODY).xmat.reshape(3, 3)
        return np.array([p[0], p[1], np.arctan2(R[1, 0], R[0, 0])])

    def record(self, t, est_pose, P, U, e_y_est, path_gen:PathGenerator):
        self.t.append(t)
        self.true_pose.append(self.true())
        self.est_pose.append(np.asarray(est_pose))
        self.P.append(np.asarray(P))
        self.U.append(np.asarray(U))
        e_y_true, _, _, = path_gen.compute_path_errors(self.true(), path_gen.path)
        self.e_y_true.append(e_y_true)
        self.e_y_est.append(e_y_est)

    def summary(self, path_gen:PathGenerator):
        self.e_y_true = np.asarray(self.e_y_true)
        self.e_y_est = np.asarray(self.e_y_est)

        self.t, self.true_pose, self.est_pose = np.asarray(self.t), np.asarray(self.true_pose), np.asarray(self.est_pose)
        self.P, self.U = np.asarray(self.P), np.asarray(self.U)
        sig_pos = np.sqrt(self.P[:, 0, 0] + self.P[:, 1, 1])
        CHI2_3_95 = 7.815
        
        m = {"rms_ey": np.sqrt(np.mean(self.e_y_true**2)),
            "mean_ey": np.mean(self.e_y_true),
            "max_ey": np.max(np.abs(self.e_y_true))
            }
        
        fb = self.U[:, 1] #path is line segments so kappa is 0.0
        m.update({"rms_fb":np.sqrt(np.mean(fb**2))})

        nees = []
        t = np.array(self.true_pose)
        e = np.array(self.est_pose)
        if len(t) == 0:
            return
        err = e - t
        err[:, 2] = wrap(err[:, 2])

        pos_error = err[:,:2]
        pos_rmse = np.sqrt(np.mean(np.linalg.norm(pos_error)**2)) 
        for k, e in enumerate(err):
            nees_ = e.T @ np.linalg.solve(self.P[k], e) # claude help me with np.linalg.solve
            nees.append(nees_)
        anees = np.mean(np.array(nees))
        nees_inside = np.mean(np.array(nees) < CHI2_3_95) # only include nees of < CHI2_3_95, claude helped me with this
        m.update({
            "pos_rmse":pos_rmse,
            "nees":nees,
            "anees":anees,
            "nees_inside":nees_inside
            })
        
        print("Position RMSE: ", m["pos_rmse"])        
        print("Max true crosstrack error: ", m["max_ey"])
        print("######"*10)
        print("S1: RMS true crosstrack: ", m["rms_ey"])         #s1 of prev assignment
        print("S2: Mean true crosstrack error: ", m["mean_ey"]) #s2

        print("S3: RMS feedback steering: ", m["rms_fb"])       #s3
        print("S4: ANEES (ideal about 3): ", m["anees"])        #s4
        print("######"*10)
        print("NEES inside 95% bound  : ", m["nees_inside"])
        

        fig, axs = plt.subplots(2, 2, figsize=(16, 11))
        n = len(path_gen.path_history)
        colors = plt.cm.tab10(np.arange(n) % 10)   # distinct colors, cycles after 10
        for idx, path_ in enumerate(path_gen.path_history):
            axs[0, 0].plot(path_[:, 0], path_[:, 1], color=colors[idx], lw=1.2, label=f"Target path{idx + 1}")
        axs[0, 0].plot(self.true_pose[:, 0], self.true_pose[:, 1], "b-", lw=1.2, label="true path")
        axs[0, 0].plot(self.est_pose[:, 0], self.est_pose[:, 1], "r-", lw=1.5, alpha=0.7, label="EKF estimate")
        axs[0, 0].set_title("Trajectory", fontweight="bold"); axs[0, 0].legend(loc="upper center", ncol=3, fontsize=8)
        
        axs[0, 1].plot(self.t, 100 * self.e_y_true, "k-", lw=1, label="true $e_y$")
        axs[0, 1].plot(self.t, 100 * self.e_y_est, "r-", lw=0.8, alpha=0.7, label="estimated $e_y$ (what the controller sees)")
        axs[0, 1].set_ylim(-30, 30); axs[0, 1].set_xlabel("time [s]"); axs[0, 1].set_ylabel("cross-track error [cm]")
        axs[0, 1].set_title(f"Cross-track error (RMS {100*m["rms_ey"]:.1f} cm, mean {100*m["mean_ey"]:+.1f} cm)", fontweight="bold")
        axs[0, 1].legend(loc="upper right", fontsize=8); axs[0, 1].grid(True, ls="--", alpha=0.5)

        axs[1, 0].semilogy(self.t, m["nees"], "tab:purple", lw=0.8, label="NEES")
        axs[1, 0].axhline(CHI2_3_95, color="k", ls="--", label="95% $\\chi^2_3$")
        axs[1, 0].axhline(3, color="tab:green", ls=":", label="expected value, 3")
        axs[1, 0].set_xlabel("time [s]"); axs[1, 0].set_ylabel("NEES")
        axs[1, 0].set_title(f"Consistency (ANEES {m['anees']:.2f}, {100*m['nees_inside']:.0f}% inside bound)", fontweight="bold")
        axs[1, 0].legend(); axs[1, 0].grid(True, which="both", ls="--", alpha=0.4)

        axs[1, 1].plot(self.t, 100 * np.linalg.norm(err[:, :2], axis=1), "tab:blue", lw=1, label="position error")
        axs[1, 1].plot(self.t, 200 * sig_pos, "tab:blue", ls="--", lw=1, label="$2\\sigma$ (from $P$)")
        axs[1, 1].set_xlabel("time [s]"); axs[1, 1].set_ylabel("[cm]"); axs[1, 1].set_ylim(bottom=0)
        axs[1, 1].set_title(f"Position estimation error (RMSE {100*m['pos_rmse']:.1f} cm)", fontweight="bold")
        axs[1, 1].legend(); axs[1, 1].grid(True, ls="--", alpha=0.5)
        plt.savefig('performance_metrics.png' , dpi=300, bbox_inches="tight")

