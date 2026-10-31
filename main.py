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
"""tuned with no walls or obstacle and no bias on the omega. Then fine tuned with it."""
def main():
    model = mujoco.MjModel.from_xml_path(cfg.MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # populate sensor values before first read

    # --- build components ---
    drive = Drive(model, data)
    encoders = Encoders(data)
    lidar = Lidar(model, data)
    camera = Camera(model, data)
    marker = ArucoMarker(data)
    logger = GroundTruthLogger(data)

    #Draw Camera
    draw = True
    
    #Generate initial plan from room map
    path_gen = PathGenerator("room_map.npy", 8.0, visualise_replan=False)
    path_gen.plan((3.5,1.5),(-2.5,-2.5))

    #Linearised and discretised model
    error_model = ErrorModel(model, cfg.V)

    #LQI Settings
    """ Tuned Ra, Qa constant. 
        started with np.array([[1/0.5**2]]).
        used the room_scene_no_obstacle scene.
        tuned to minimise rms true_crosstrack 
        and mean true_crosstrack
        """
    Qa = np.diag([1/0.01**2, 1/np.deg2rad(1)**2, 180.0]) # the larger the controller will try to quickly fix the errors
    Ra = np.array([[1/0.42**2]])                          # the larger the slower the response (control input becomes more expensive)
    controller = LQIController(model, error_model.Ad, error_model.Bd, Qa, Ra)

    #EKF Settings
    """started  with (1.0*0.1) and np.deg2rad(5.0)
        then used the room_scene_no_obstacle scene 
        to get the anees as close to 3."""
    Qd = np.diag([(0.28*0.1)**2, (0.28*0.1)**2, np.deg2rad(6)**2]) #The larger, the model assumes the model is bad and rely on sensors more
    Rn = np.diag([0.3**2 , np.deg2rad(10)**2])                     #The larger, the model assumes the sensor is bad and rely on model more
    estimator = EKF_Estimator(model,Qd,Rn)

    local_map = LocalMap()

    # Known start pose is an initial condition, not feedback.
    x0, y0, th0 = logger.true()
    odom = DiffDriveOdometry(*encoders.read(), x=x0, y=y0, th=th0)

    xh = np.array([x0, y0, th0]) #initial position
    P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])

    # --- rate gates (sim time) ---
    enc_timer = RateTimer(cfg.ENCODER_RATE)
    cam_timer = RateTimer(cfg.CAMERA_RATE)
    lidar_timer = RateTimer(cfg.LIDAR_RATE)
    view_timer = RateTimer(cfg.VIEWER_RATE)

    #Goal state
    goal_reached = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        time_since_last_recalculate = data.time
        simulation_time_start = time.time()
        while viewer.is_running() and data.time < cfg.SIM_DURATION and not goal_reached:
            step_start = time.time()
            
            # 1. command (later: LQR using the EKF estimate)
            if xh[0] > -2.9 and xh[0] <-2.1:
                if xh[1] > -2.9 and xh[1] <-2.1:
                    goal_reached = True
                    
            if local_map.obstacle and data.time - time_since_last_recalculate > 5.0:
                time_since_last_recalculate = data.time
                print("OBSTACLE AHEAD RECALCULATING PATH")
                path_gen.plan((xh[0],xh[1]),(-2.5,-2.5), local_map.map)
                print("DONE!, NAVIGATE AGAIN")

            e_y, e_theta, _= path_gen.compute_path_errors(xh, path_gen.path)

            v, omega = cfg.V, controller.step(e_y, e_theta, 0.0, cfg.V)

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
                points = lidar.scan()
                lidar.draw_points(viewer=viewer, pts=points)
                local_map.update_map(points, xh)
                
            if cam_timer.due(t):
                frame = camera.read()                    
                frame, ranges, bearings = camera.aruco_pose(frame, draw)
                for i, marker_ in enumerate(marker.aruco_position):
                    if np.isnan(ranges[i]):
                        continue
                    y = np.array([ranges[i], bearings[i]])
                    xh, P, nu, S = estimator.update(xh, P, y, marker_)
                    xh[2] = wrap(xh[2])
                if draw:
                    camera.show(frame)

            if view_timer.due(t):
                viewer.sync()

            # 4. real-time pacing
            sleep = model.opt.timestep - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)
    print()
    print("Real time elapsed", time.time() - simulation_time_start)
    print()
    camera.close()
    logger.summary(path_gen)


if __name__ == "__main__":
    main()