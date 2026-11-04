"""Fixed dimensions shared by the URDF generator, the simulator scene and the skills. Units: m, rad."""
import math

HEIGHT = 0.9196429
RADIUS, TRACK = 0.10, 0.58
BASE_HALF = (0.275, 0.32)

ARM_REST = (0.0, math.pi/8, 0.0, -5*math.pi/8, 0.0, 3*math.pi/4, math.pi/4, 0.04, 0.04)

BASE_CAMERA_EYE = (0.0, -0.32, 0.65)
BASE_CAMERA_TARGET = (0.615, 0.0, 0.025)
BASE_CAMERA_SIZE = (640, 480)
HAND_MOUNT_XYZ = (0.035, 0.0, 0.036)
HAND_MOUNT_RPY = (0.0, -1.5707, 3.1415)
HAND_CAMERA_XYZ = (0.0, 0.02, 0.0115)
HAND_CAMERA_SIZE = (320, 240)
CAMERA_FOV_Y = 1.0

TABLE_CENTER = (0.35, 0.0)
TABLE_HALF = (0.55, 0.60)
TABLE_LEGS = ((0.0, -0.5), (0.0, 0.5), (0.75, -0.5), (0.75, 0.5))
TABLE_LEG_HALF = 0.035
CUBE_HALF = 0.02
CUBES = {"red_cube": (0.0, -0.08), "blue_cube": (0.0, 0.0)}
ZONES = {"green_zone": (0.0, 0.12), "yellow_zone": (0.0, -0.16)}
ZONE_HALF = 0.065
DEFAULT_START = (-1.15, -0.30, 0.0)

MAX_LINEAR, MAX_ANGULAR = 0.25, 0.7
MAX_JOINT_DELTA = 0.1
GRIPPER_RANGE = (-0.01, 0.04)
