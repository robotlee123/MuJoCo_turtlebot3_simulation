"""All tunable constants in one place."""

MODEL_PATH = r"Turtlebot3\scene_turtlebot3_waffle_pi.xml"
SIM_DURATION = 5.0  # seconds of SIM time



# Platform
WHEEL_RADIUS = 0.033   # m
WHEEL_BASE = 0.288     # m, distance between wheels
TICKS_PER_REV = 1024
DEFAULT_MAX_WHEEL_SPEED = 7.88  # rad/s, fallback if MJCF has no ctrlrange
OMEGA_MAX = 1.82 # rad/s, from https://docs.robotis.com/docs/systems/turtlebot3/features/

# Sensor / loop rates (Hz, in sim time)
ENCODER_RATE = 50
CAMERA_RATE = 30
LIDAR_RATE = 5
VIEWER_RATE = 30

# Names in the MJCF
CAMERA_NAME = "camera_0"
LIDAR_SITE = "lidar_site"
LIDAR_BODY = "lidar_base"
LEFT_ENC = "left_wheel_encoder"
RIGHT_ENC = "right_wheel_encoder"
BASE_BODY = "base"