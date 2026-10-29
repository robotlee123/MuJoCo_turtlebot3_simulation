"""Sensor wrappers. These are the ONLY way estimator/controller code
should obtain information about the world."""
import cv2
import mujoco
import numpy as np
from mujoco_lidar import MjLidarWrapper, scan_gen  # external lidar library

import config as cfg
from drive import quantize_rads


class Encoders:
    """Quantized wheel angles (rad)."""

    def __init__(self, data):
        self.data = data

    def read(self):
        left = quantize_rads(self.data.sensor(cfg.LEFT_ENC).data[0])
        right = quantize_rads(self.data.sensor(cfg.RIGHT_ENC).data[0])
        return left, right


class Lidar:
    def __init__(self, model, data):
        self.data = data
        self.site_id = model.site(cfg.LIDAR_SITE).id
        self.sensor = MjLidarWrapper(
            model,
            site_name=cfg.LIDAR_SITE,
            backend="taichi",
            cutoff_dist=50.0,
            args={
                "max_candidates": 64,
                "ti_init_args": {"device_memory_GB": 8},
                "bodyexclude": model.body(cfg.LIDAR_BODY).id,  # don't hit lidar housing
            },
        )
        self.theta, self.phi = scan_gen.create_lidar_single_line()
    def scan(self):
        """Returns hit points (N,3)."""
        self.sensor.trace_rays(self.data, self.theta, self.phi)
        return self.sensor.get_hit_points()

    def draw_points(self, viewer, pts, rgba=(1, 0, 0, 1), size=0.01):
        """Claude made this visualisation"""
        # Drop misses (NaN/inf or at max range)
        valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) < 49.9)
        pts = pts[valid]

        R = self.data.site_xmat[self.site_id].reshape(3, 3)
        p = self.data.site_xpos[self.site_id]
        pts_world = pts @ R.T + p

        with viewer.lock():
            scn = viewer.user_scn
            n = min(len(pts_world), scn.maxgeom)
            for i in range(n):
                mujoco.mjv_initGeom(
                    scn.geoms[i],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([size, 0, 0]),
                    np.asarray(pts_world[i], dtype=np.float64),
                    np.eye(3).flatten(),
                    np.array(rgba, dtype=np.float32),
                )
            scn.ngeom = n

class ArucoMarker:
    def __init__(self, data):
        self.aruco_position = [
            data.body("aruco_marker0").xpos,
            data.body("aruco_marker1").xpos,
            data.body("aruco_marker2").xpos,
            data.body("aruco_marker3").xpos,
            data.body("aruco_marker4").xpos,
            data.body("aruco_marker5").xpos,
            data.body("aruco_marker6").xpos,
            data.body("aruco_marker7").xpos,
            data.body("aruco_marker8").xpos,
            data.body("aruco_marker9").xpos,
            data.body("aruco_marker10").xpos,
            data.body("aruco_marker11").xpos,
            data.body("aruco_marker12").xpos,
            data.body("aruco_marker13").xpos,
            data.body("aruco_marker14").xpos,
            data.body("aruco_marker15").xpos,
            data.body("aruco_marker16").xpos,
            data.body("aruco_marker17").xpos,
            data.body("aruco_marker18").xpos,
            data.body("aruco_marker19").xpos,
            data.body("aruco_marker20").xpos,
            data.body("aruco_marker21").xpos,
            data.body("aruco_marker22").xpos,
            data.body("aruco_marker23").xpos,
            data.body("aruco_marker24").xpos,
            data.body("aruco_marker25").xpos,
            data.body("aruco_marker26").xpos,
            data.body("aruco_marker27").xpos,
            data.body("aruco_marker28").xpos,
            data.body("aruco_marker29").xpos,
            data.body("aruco_marker30").xpos,
            data.body("aruco_marker31").xpos,
            data.body("aruco_marker32").xpos,
            data.body("aruco_marker33").xpos,
            data.body("aruco_marker34").xpos,
            data.body("aruco_marker35").xpos,
            data.body("aruco_marker36").xpos,
            data.body("aruco_marker37").xpos,
            data.body("aruco_marker38").xpos,
            data.body("aruco_marker39").xpos,
            data.body("aruco_marker40").xpos,
            data.body("aruco_marker41").xpos,
            data.body("aruco_marker42").xpos,
            data.body("aruco_marker43").xpos,
            data.body("aruco_marker44").xpos,
            data.body("aruco_marker45").xpos,
            data.body("aruco_marker46").xpos,
            data.body("aruco_marker47").xpos,
            data.body("aruco_marker48").xpos,
            data.body("aruco_marker49").xpos
        ]
class Camera:
    def __init__(self, model, data, height=480, width=640):
        self.data = data

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.CAMERA_NAME)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{cfg.CAMERA_NAME}' not found in MJCF")
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_parameter = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameter)
        fovy = model.cam_fovy[self.cam_id] 
        f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        self.camera_mtx = np.array([ [f, 0, width / 2.0 ],
                                     [0, f, height / 2.0], 
                                     [0, 0,      1.0    ]], 
                                     dtype=np.float64)
        self.distortion_coeff = np.zeros((1, 5), dtype=np.float64)  # mujoco camera have no distortion
        MARKER_SIZE = 0.2 #size in meters
        self.MARKER_POINTS = np.array([
                                [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
                                [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
                                [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                                [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
                                ], dtype=np.float32)

        
    def read(self):
        """Returns an RGB image (H, W, 3, uint8)."""
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    def aruco_pose(self, frame, draw = False):
        # src: https://automaticaddison.com/how-to-perform-pose-estimation-using-an-aruco-marker/
        #      https://www.geeksforgeeks.org/computer-vision/detecting-aruco-markers-with-opencv-and-python-1/
        #      https://zbotic.in/aruco-marker-detection-pose-estimation-with-opencv-python/
        #      https://medium.com/@lalitsaraf1/aruco-marker-cv2-updated-version-67119b1500e3
        # Detect ArUco markers in the video frame
        ranges_array = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        bearing_array = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, marker_ids, _ = self.detector.detectMarkers(gray)
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)
            rvecs = []
            tvecs = []
            ranges_array = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            bearing_array = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            for c in corners:
                _, R, t = cv2.solvePnP(self.MARKER_POINTS, c, self.camera_mtx, self.distortion_coeff, False, cv2.SOLVEPNP_IPPE_SQUARE)

                rvecs.append(R)
                tvecs.append(t)

            for i in range(len(tvecs)):
                if draw:
                    # Use cv2.drawFrameAxes to draw the axis
                    cv2.drawFrameAxes(frame, self.camera_mtx, self.distortion_coeff, rvecs[i], tvecs[i], 0.03)
                # print(f"Marker ID: {marker_ids[i]}")  # Display the ID of the marker
                # print(f"Rotation Vector: {rvecs[i].flatten()}")
                # print(f"Translation Vector: {tvecs[i].flatten()}")
                
                # transform TO THE BASE JOINT and mujoco convention?
                
                x, y, z = tvecs[i].flatten()

                x, y, z = z + 0.078, -x, -y + 0.107
                
                est_range = np.linalg.norm([x, y])
                est_bearing = np.arctan2(y,x)

                ranges_array[marker_ids[i]] = est_range
                bearing_array[marker_ids[i]] = est_bearing

        return frame, ranges_array, bearing_array

    @staticmethod
    def show(rgb, title="MuJoCo Camera Feed"):
        cv2.imshow(title, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self.renderer.close()