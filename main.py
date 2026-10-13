import mujoco
import mujoco.viewer

from mujoco_lidar import MjLidarWrapper, scan_gen #external lidar library

import time
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path(r"Turtlebot3\scene_turtlebot3_waffle_pi.xml")
data = mujoco.MjData(model)

WHEEL_RADIUS = 0.033 # radius of wheel in m
WHEEL_BASE = 0.288 # distance between wheel in m

def ddr_ik(v, omega, r=WHEEL_RADIUS, l=WHEEL_BASE):
    """Differential drive inverse kinematics: calculate wheels speeds from desired velocity.
        reference: https://www.roboticsbook.org/S52_diffdrive_actions.html
    """
    omega_L = (2 * v - l * omega) / (2 * r)
    omega_R = (2 * v + l * omega) / (2 * r)
    return np.clip(omega_L,-7.88, 7.88), np.clip(omega_R,-7.88, 7.88) #clip based on MJCF

# Create LiDAR

lidar_sim_rate = 10
lidar_sim_cnt = 0
exclude_body_id = model.body("lidar_base").id #so the ray does not intersect the LiDAR model, ref https://github.com/hansolo12334/MuJoCo-LiDAR-cpp
lidar_sensor = MjLidarWrapper(
    model,
    site_name="lidar_site",
    backend="taichi", 
    cutoff_dist=50.0,
    args={
        'max_candidates': 64,
        'ti_init_args': {'device_memory_GB': 4},
        'bodyexclude': exclude_body_id
    }
)

# Generate scan pattern
rays_theta, rays_phi = scan_gen.create_lidar_single_line()

with mujoco.viewer.launch_passive(model, data) as viewer: # ref: https://mujoco.readthedocs.io/en/stable/python.html
    start = time.time()
    INTERVAL = 0
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        if step_start - start > INTERVAL:
            if INTERVAL % 2:
                data.ctrl[0], data.ctrl[1] = ddr_ik(0, 1.82)
            else:
                data.ctrl[0], data.ctrl[1] = ddr_ik(0, -1.82)
            INTERVAL += 1
        mujoco.mj_step(model, data)

        
        if data.time * lidar_sim_rate > lidar_sim_cnt:
            lidar_sensor.trace_rays(data, rays_theta, rays_phi)
            points = lidar_sensor.get_hit_points()
            
            lidar_sim_cnt += 1

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    plt.scatter(points[:,0], points[:,1])
    plt.show()