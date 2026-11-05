"""MuJoCo backend. Collected only with mujoco installed and the Menagerie model downloaded (see conftest.py)."""
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as E

import mujoco
import numpy as np
import pytest

from mobile_panda_sim.geometry import ARM_REST
from mobile_panda_sim.mj.model import ARM_JOINTS
from mobile_panda_sim.mj.sim import MujocoScene
from mobile_panda_sim.paths import REPO, ros_urdf

sys.path.insert(0, str(REPO / "scripts"))
import mj_contact_check  # noqa: E402

# Measured on the ManiSkill backend at reset with the default start.
MANISKILL_TCP = {"wheels": (-0.535, -0.3, 0.1768), "planar": (-0.535, -0.3, 0.1698)}
MANISKILL_JOINTS = {"wheels": ["left_wheel_joint", "right_wheel_joint"], "planar": ["base_x_joint", "base_y_joint", "base_yaw_joint"]}
MANISKILL_FOCAL = {"base_camera": 439.317, "hand_camera": 219.659}


@pytest.fixture(params=["wheels", "planar"])
def scene(request):
    result = MujocoScene("tabletop", request.param)
    result.reset(0)
    yield result
    result.close()


def yaw_of(q):
    return math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))


def urdf_transform(element, angle=0.0):
    """4x4 transform of a URDF joint: origin, then rotation about its z axis."""
    origin = element.find("origin")
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    transform = np.eye(4)
    transform[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    quat, turn, matrix = np.zeros(4), np.array([math.cos(angle/2), 0, 0, math.sin(angle/2)]), np.zeros(9)
    mujoco.mju_euler2Quat(quat, rpy, "XYZ")
    mujoco.mju_mulQuat(quat, quat.copy(), turn)
    mujoco.mju_quat2Mat(matrix, quat)
    transform[:3, :3] = matrix.reshape(3, 3)
    return transform


def test_rest_pose_matches_maniskill(scene):
    assert scene.joint_names == MANISKILL_JOINTS[scene.base] + ARM_JOINTS + ["panda_finger_joint1", "panda_finger_joint2"]
    assert np.allclose(scene.joint_positions()[-9:], ARM_REST)
    assert np.allclose(scene.tcp_pose()[0], MANISKILL_TCP[scene.base], atol=2e-4)
    assert np.allclose(np.abs(scene.tcp_pose()[1]), [0, 1, 0, 0], atol=1e-3)


def test_arm_kinematics_match_urdf():
    scene = MujocoScene("tabletop", "planar")
    scene.reset(0)
    joints = {j.get("name"): j for j in E.parse(ros_urdf("planar")).getroot().findall("joint")}
    link0 = scene.model.body("panda_link0").id
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(scene.model.actuator_ctrlrange[scene.arm_ctrl, 0], scene.model.actuator_ctrlrange[scene.arm_ctrl, 1])
        scene.data.qpos[scene.arm_q] = q
        mujoco.mj_kinematics(scene.model, scene.data)
        expected = np.eye(4)
        for i, angle in enumerate(q, start=1):
            expected = expected @ urdf_transform(joints[f"panda_joint{i}"], angle)
        for name in ("panda_joint8", "panda_hand_joint", "panda_hand_tcp_joint"):
            expected = expected @ urdf_transform(joints[name])
        base = np.eye(4)
        base[:3, :3], base[:3, 3] = scene.data.xmat[link0].reshape(3, 3), scene.data.xpos[link0]
        assert np.allclose((base @ expected)[:3, 3], scene.data.site_xpos[scene.tcp], atol=1e-4)
    scene.close()


def test_base_rests_and_drives(scene):
    for _ in range(20):
        scene.step(0, 0, None, 0.04)
    position = scene.pose(scene.chassis)[0]
    assert np.linalg.norm(position[:2] - scene.start[:2]) < 0.005
    for _ in range(40):
        scene.step(0.15, 0, None, 0.04)
    moved = np.linalg.norm(scene.pose(scene.chassis)[0][:2] - position[:2])
    assert moved == pytest.approx(0.3, abs=0.02)
    yaw = yaw_of(scene.pose(scene.chassis)[1])
    for _ in range(40):
        scene.step(0, 0.4, None, 0.04)
    assert yaw_of(scene.pose(scene.chassis)[1]) - yaw == pytest.approx(0.8, abs=0.04)


def test_arm_delta_and_gripper(scene):
    start = scene.joint_positions()[-9]
    for _ in range(10):
        scene.step(0, 0, [0.02, 0, 0, 0, 0, 0, 0], 0.04)
    for _ in range(20):
        scene.step(0, 0, None, 0.04)
    assert scene.joint_positions()[-9] - start > 0.05
    held = scene.joint_positions()[-9]
    for _ in range(20):
        scene.step(0, 0, None, 0.04)
    assert scene.joint_positions()[-9] == pytest.approx(held, abs=1e-3)
    for _ in range(30):
        scene.step(0, 0, None, 0.0)
    assert np.all(np.abs(scene.joint_positions()[-2:]) < 0.002)


def test_cameras(scene):
    images = scene.capture_rgb()
    assert images["base_camera"].shape == (480, 640, 3) and images["hand_camera"].shape == (240, 320, 3)
    assert all(a.dtype == np.uint8 and a.std() > 5 for a in images.values())
    for name, matrix in scene.camera_intrinsics().items():
        assert matrix[0][0] == pytest.approx(MANISKILL_FOCAL[name], abs=0.01)


def test_contact_physics():
    assert all(r["success"] for r in mj_contact_check.pick(0).values())
    assert mj_contact_check.drawer(0)["success"]


def test_worker_protocol():
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    process = subprocess.Popen([sys.executable, "-u", "-m", "mobile_panda_sim.ros2.worker", "--backend", "mujoco", "--scene", "tabletop"],
                               cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        seen = {}
        while "state" not in seen:
            message = json.loads(process.stdout.readline())
            seen[message["type"]] = message
        assert seen["scene"]["layout"]["robot"] == "mobile_panda_wheels"
        assert seen["ready"]["control_timestep"] == pytest.approx(0.05)
        state = seen["state"]
        assert {"base_pose", "root_pose", "tcp_pose", "joint_positions", "joint_velocities", "gripper_width", "objects", "images"} <= set(state)
        assert set(state["objects"]) == {"red_cube", "blue_cube"} and state["images"]["base_camera"]["width"] == 640
    finally:
        process.stdin.close()
        assert process.wait(timeout=20) == 0
