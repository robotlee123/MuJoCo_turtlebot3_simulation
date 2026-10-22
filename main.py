import time
import mujoco
import mujoco.viewer
import numpy as np

import config as cfg
from drive import Drive, RateTimer
from sensors import Encoders, Lidar, Camera
from estimation_control import DiffDriveOdometry, ZigZagController, GroundTruthLogger, ErrorModel, LQIController, LinePathGenerator


def main():
    model = mujoco.MjModel.from_xml_path(cfg.MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # populate sensor values before first read

    # --- build components ---
    drive = Drive(model, data)
    encoders = Encoders(data)
    lidar = Lidar(model, data)
    camera = Camera(model, data)
    controller_temp = ZigZagController()
    logger = GroundTruthLogger(data)

    path = LinePathGenerator()

    error_model = ErrorModel(model, 0.26)

    controller = LQIController(model, error_model.Ad, error_model.Bd)

    # Known start pose is an initial condition, not feedback.
    x0, y0, th0 = logger.true()
    odom = DiffDriveOdometry(*encoders.read(), x=x0, y=y0, th=th0)

    # --- rate gates (sim time) ---
    enc_timer = RateTimer(cfg.ENCODER_RATE)
    cam_timer = RateTimer(cfg.CAMERA_RATE)
    lidar_timer = RateTimer(cfg.LIDAR_RATE)
    view_timer = RateTimer(cfg.VIEWER_RATE)

    points = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < cfg.SIM_DURATION:
            step_start = time.time()
            # 1. command (later: LQR using the EKF estimate)

            v, omega = controller_temp.command(data.time)
            # R = data.body(cfg.BASE_BODY).xmat.reshape(3, 3)

            # x = data.body(cfg.BASE_BODY).xpos[0]
            # y = data.body(cfg.BASE_BODY).xpos[1]
            # e_y, e_theta, kappa, _ = path.path_error(x,y,np.arctan2(R[1, 0], R[0, 0]))
            # v, omega = 0.26, controller.step(e_y, e_theta, kappa, 0.26)
            drive.apply(v, omega)

            # 2. physics
            mujoco.mj_step(model, data)
            t = data.time

            # 3. sensors -> estimator
            if enc_timer.due(t):
                est = odom.update(*encoders.read())      # later: ekf.predict(...)
                logger.record(t, est)

            if lidar_timer.due(t):
                points = lidar.scan()                    # later: occupancy grid
                lidar.draw_points(viewer=viewer, pts=points)

            if cam_timer.due(t):
                frame = camera.read()                    # later: ArUco -> ekf.update(...)
                frame, ranges, headings = camera.aruco_pose(frame)
                print(f"Ranges: {ranges}")
                print(f"Headings: {headings}")
                camera.show(frame)

            if view_timer.due(t):
                viewer.sync()

            # 4. real-time pacing
            sleep = model.opt.timestep - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

    camera.close()
    logger.summary()


if __name__ == "__main__":
    main()