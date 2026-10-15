import time
import mujoco
import mujoco.viewer
import numpy as np

import config as cfg
from drive import Drive, RateTimer
from sensors import Encoders, Lidar, Camera
from estimation_control import DiffDriveOdometry, ScriptedController, GroundTruthLogger


def main():
    model = mujoco.MjModel.from_xml_path(cfg.MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # populate sensor values before first read

    # --- build components ---
    drive = Drive(model, data)
    encoders = Encoders(data)
    lidar = Lidar(model, data)
    camera = Camera(model, data)
    controller = ScriptedController()
    logger = GroundTruthLogger(data)

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
            v, omega = controller.command(data.time)
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