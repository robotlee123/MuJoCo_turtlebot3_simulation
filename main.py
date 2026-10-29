import time
import mujoco
import mujoco.viewer
import numpy as np

from drive import wrap

import config as cfg
from drive import Drive, RateTimer
from sensors import Encoders, Lidar, Camera, ArucoMarker
from estimation_control import DiffDriveOdometry, GroundTruthLogger
from estimation_control import ErrorModel, LQIController, EKF_Estimator
from estimation_control import PathGenerator, LocalMap

def main():
    model = mujoco.MjModel.from_xml_path(cfg.MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # populate sensor values before first read

    # --- build components ---
    drive = Drive(model, data)
    encoders = Encoders(data)
    lidar = Lidar(model, data)
    camera = Camera(model, data)

    logger = GroundTruthLogger(data)

    path_gen = PathGenerator("room_map.npy")
    path_gen.plan((3.5,1.5),(-2.5,-2.5))

    error_model = ErrorModel(model, cfg.V)

    controller = LQIController(model, error_model.Ad, error_model.Bd)

    marker = ArucoMarker(data)

    estimator = EKF_Estimator(model)

    local_map = LocalMap()
    # Known start pose is an initial condition, not feedback.
    x0, y0, th0 = logger.true()
    odom = DiffDriveOdometry(*encoders.read(), x=x0, y=y0, th=th0)
    xh = np.array([x0, y0, th0])
    P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])
    # --- rate gates (sim time) ---
    enc_timer = RateTimer(cfg.ENCODER_RATE)
    cam_timer = RateTimer(cfg.CAMERA_RATE)
    lidar_timer = RateTimer(cfg.LIDAR_RATE)
    view_timer = RateTimer(cfg.VIEWER_RATE)

    
    robot_navigation_state = True
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time_since_last_recalculate = data.time
        while viewer.is_running() and data.time < cfg.SIM_DURATION and robot_navigation_state:
            step_start = time.time()
            # 1. command (later: LQR using the EKF estimate)
            if xh[0] > -2.9 and xh[0] <-2.1:
                if xh[1] > -2.9 and xh[1] <-2.1:
                    robot_navigation_state = False
                    
            if local_map.obstacle and data.time - time_since_last_recalculate > 5.0:
                time_since_last_recalculate = data.time
                print("OBSTACLE AHEAD RECALCULATING PATH")
                path_gen.plan((xh[0],xh[1]),(-2.5,-2.5), local_map.map)
                print("DONE, NAVIGATE AGAIN")

            e_y, e_theta, _= path_gen.compute_path_errors(xh, path_gen.path)


            v, omega = 0.26, controller.step(e_y, e_theta, 0.0, 0.26)

            drive.apply(v, omega)

            # 2. physics
            mujoco.mj_step(model, data)
            
            t = data.time

            # 3. sensors -> estimator
            if enc_timer.due(t):
                est = odom.update(*encoders.read())      
                xh, P = estimator.predict(xh, P, odom.dl, odom.dr)
                logger.record(t, xh, P, [v, omega], e_y, path_gen)

            if lidar_timer.due(t):
                points = lidar.scan()                    # later: occupancy grid
                lidar.draw_points(viewer=viewer, pts=points)
                local_map.update_map(points, xh)
                
            if cam_timer.due(t):
                frame = camera.read()                    
                frame, ranges, bearings = camera.aruco_pose(frame)
                for i, marker_ in enumerate(marker.aruco_position):
                    if np.isnan(ranges[i]):
                        continue
                    y = np.array([ranges[i], bearings[i]])
                    xh, P, nu, S = estimator.update(xh, P, y, marker_)
                    xh[2] = wrap(xh[2])
                # camera.show(frame)

            if view_timer.due(t):
                viewer.sync()

            # 4. real-time pacing
            sleep = model.opt.timestep - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

    camera.close()
    logger.summary(path_gen)


if __name__ == "__main__":
    main()