import mujoco
import mujoco.viewer

import cv2

from mujoco_lidar import MjLidarWrapper, scan_gen #external lidar library

import time
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path(r"Turtlebot3\scene_turtlebot3_waffle_pi.xml")
data = mujoco.MjData(model)

WHEEL_RADIUS = 0.033 # radius of wheel in m
WHEEL_BASE = 0.288 # distance between wheel in m
TICKS_PER_REV = 1024 #encoder ticks per revolution

def ddr_ik(v, omega, r=WHEEL_RADIUS, l=WHEEL_BASE):
    """Differential drive inverse kinematics: calculate wheels speeds from desired velocity.
        reference: https://www.roboticsbook.org/S52_diffdrive_actions.html
    """
    omega_L = (2 * v - l * omega) / (2 * r)
    omega_R = (2 * v + l * omega) / (2 * r)
    s = max(1.0, abs(omega_L)/7.88, abs(omega_R)/7.88)
    return omega_L/s, omega_R/s #clip based on MJCF

def calculate_encoder_rads(rads, ticks_per_rev = TICKS_PER_REV):
    """it first find the number of ticks via the radians, 
        quantizes it, then find the radians using the quantized ticks
    """
    tick_size = 2 * np.pi / ticks_per_rev
    return np.round(rads / tick_size) * tick_size

#camera setup
renderer = mujoco.Renderer(model, height=480, width=640)
camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_0"
)
camera_rate = 30
camera_cnt =  0

# Create LiDAR
lidar_sim_rate = 10
lidar_sim_cnt = 0

exclude_body_id = model.body("lidar_base").id #so the ray does not intersect the LiDAR model
lidar_sensor = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="taichi", 
    cutoff_dist=50.0,
    args={
        'max_candidates': 64,
        'ti_init_args': {'device_memory_GB': 8},
        'bodyexclude': exclude_body_id
    }
)

# Generate scan pattern
rays_theta, rays_phi = scan_gen.create_lidar_single_line()

# Endcoder variables
encoder_rate = 50   # Hz
encoder_cnt = 0

left_enc = 0.0      
right_enc = 0.0

#Odometry cariables
x, y, th = 0.0, 0.0, 0.0
prev_l, prev_r = 0.0, 0.0

viewer_rate = 30
viewer_cnt = 0
with mujoco.viewer.launch_passive(model, data) as viewer: # ref: https://mujoco.readthedocs.io/en/stable/python.html & https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/examples/example_mjcf.py
    start = time.time()
    print(data.body('base').xpos)
    INTERVAL = 0
    while viewer.is_running() and data.time < 30: #timing based on simulation time
        step_start = time.time()
    
        if data.time > INTERVAL: #timing based on simulation time
            if INTERVAL % 2:
                data.ctrl[0], data.ctrl[1] = ddr_ik(0.2, 1.82)
            else:
                data.ctrl[0], data.ctrl[1] = ddr_ik(0.2, -1.82)
            INTERVAL += 1
        mujoco.mj_step(model, data)
        
        if data.time * lidar_sim_rate > lidar_sim_cnt:
            lidar_sensor.trace_rays(data, rays_theta, rays_phi)
            points = lidar_sensor.get_hit_points()
            lidar_sim_cnt += 1

        if data.time * encoder_rate > encoder_cnt:
            left_enc = calculate_encoder_rads(data.sensor('left_wheel_encoder').data[0])
            right_enc = calculate_encoder_rads(data.sensor('right_wheel_encoder').data[0])
            #integration of wheel encoder to find the x,y,th
            dl = (left_enc - prev_l) * WHEEL_RADIUS
            dr = (right_enc - prev_r) * WHEEL_RADIUS
            prev_l, prev_r = left_enc, right_enc
            ds, dth = (dl + dr) / 2, (dr - dl) / WHEEL_BASE
            x  += ds * np.cos(th + dth / 2)
            y  += ds * np.sin(th + dth / 2)
            th += dth
            print("State:", x,y,th)
            encoder_cnt += 1

        if data.time * camera_rate > camera_cnt:
            renderer.update_scene(data, camera=camera_id)
            cv2.imshow("MuJoCo Camera Feed", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            camera_cnt += 1
        
        if data.time * viewer_rate > viewer_cnt:
            viewer.sync()
            viewer_cnt += 1

        # Rudimentary time keeping, will drift relative to wall clock.
        # Makes the sim run in realtime by delaying if the step is faster 
        # than model.opt.timestep (timestep)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    cv2.destroyAllWindows()
    renderer.close()
    print(data.body('base').xpos)
    plt.scatter(points[:,0], points[:,1])
    plt.show()