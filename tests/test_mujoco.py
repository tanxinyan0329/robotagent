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
from mobile_panda_sim.paths import MUJOCO_KITCHEN, REPO, ros_urdf

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


def run_worker(scene):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    process = subprocess.Popen([sys.executable, "-u", "-m", "mobile_panda_sim.ros2.worker", "--backend", "mujoco", "--scene", scene],
                               cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        seen = {}
        while "state" not in seen:
            message = json.loads(process.stdout.readline())
            seen[message["type"]] = message
        return seen
    finally:
        process.stdin.close()
        assert process.wait(timeout=20) == 0


def test_worker_protocol():
    seen = run_worker("tabletop")
    assert seen["scene"]["layout"]["robot"] == "mobile_panda_wheels"
    assert seen["ready"]["control_timestep"] == pytest.approx(0.05)
    state = seen["state"]
    assert {"base_pose", "root_pose", "tcp_pose", "joint_positions", "joint_velocities", "gripper_width", "objects", "images"} <= set(state)
    assert set(state["objects"]) == {"red_cube", "blue_cube"} and state["images"]["base_camera"]["width"] == 640


needs_kitchen = pytest.mark.skipif(not (MUJOCO_KITCHEN / "scene.json").is_file(), reason="run scripts/export_kitchen_mujoco.py")


@needs_kitchen
def test_kitchen_worker_state():
    state = run_worker("kitchen")["state"]
    assert len(state["objects"]) == 42
    assert set(state["articulations"]) == {"kitchen_counter", "fridge", "sideboard"}
    assert state["articulations"]["fridge"]["top_door_hinge"] == 0.0


@needs_kitchen
def test_kitchen_joint_limits_are_radians():
    scene = MujocoScene("kitchen", "wheels")
    assert scene.model.jnt_range[scene.model.joint("fridge/top_door_hinge").id][1] == pytest.approx(2.3)
    scene.close()


@needs_kitchen
@pytest.mark.parametrize("base", ["wheels", "planar"])
def test_kitchen_props_stay_put(base):
    scene = MujocoScene("kitchen", base)
    scene.reset(0)
    start = {name: scene.pose(body)[0] for name, body in scene.objects.items()}
    for _ in range(400):
        scene.step(0, 0, None, 0.04)
    moved = {name: float(np.linalg.norm(scene.pose(body)[0] - start[name])) for name, body in scene.objects.items()}
    assert max(moved.values()) < 0.005, moved
    scene.close()


@needs_kitchen
def test_kitchen_reset_restores_scene():
    scene = MujocoScene("kitchen", "wheels", (-3.0, -1.15, 3.1416))
    scene.reset(0)
    start = {name: scene.pose(body)[0] for name, body in scene.objects.items()}
    address = scene.articulations["fridge"]["top_door_hinge"]
    scene.data.qpos[address] = 0.5
    for _ in range(300):
        scene.step(0, 0, None, 0.04)
    scene.reset(0)
    assert scene.articulation_positions()["fridge"]["top_door_hinge"] == 0.0
    for _ in range(5):
        scene.step(0, 0, None, 0.04)
    assert max(np.linalg.norm(scene.pose(body)[0] - start[name]) for name, body in scene.objects.items()) < 0.005
    scene.close()


@needs_kitchen
def test_kitchen_contact():
    assert mj_contact_check.actuate((0.249, 1.492, 1.107), "fridge", "top_door_hinge", 0.6)["success"]


@pytest.fixture
def kitchen_skills():
    pytest.importorskip("mplib")
    from mobile_panda_sim.manipulation.skills import ManipulationSkills
    from mobile_panda_sim.mj.world import MujocoWorld
    scene = MujocoScene("kitchen", "wheels")
    world = MujocoWorld(scene)

    def park(start):
        scene.reset(0, start)
        return ManipulationSkills(world)
    yield park, world
    scene.close()


@needs_kitchen
def test_world_reads_the_scene(kitchen_skills):
    park, world = kitchen_skills
    park((-2.95, -1.15, 3.1416))
    assert world.surface(-3.40, -1.25) == pytest.approx(0.918, abs=0.01)
    assert world.object_points("sugar_box")[:, 2].max() > 1.05
    assert world.joint_info("fridge", "top_door_hinge")["kind"] == "revolute"
    assert world.joint_info("kitchen_counter", "middle_slide_top")["kind"] == "prismatic"
    pose, handle = world.handle("kitchen_counter", "middle_slide_top")
    assert handle.radius == pytest.approx(0.0088, abs=1e-3)
    assert pose.to_transformation_matrix()[:3, 2] @ [0, -1, 0] > 0.99


@needs_kitchen
def test_pick_and_place_skill(kitchen_skills):
    park, world = kitchen_skills
    skills = park((-2.95, -1.15, 3.1416))
    assert skills.pick("lemon").success
    assert skills.place((-3.38, -1.42)).success
    assert np.linalg.norm(world.object_position("lemon")[:2] - [-3.38, -1.42]) < 0.04


@needs_kitchen
def test_drawer_skill(kitchen_skills):
    park, world = kitchen_skills
    skills = park((-1.15, 1.83, 1.989))
    assert skills.open("kitchen_counter", "middle_slide_top").success
    assert world.joint_position("kitchen_counter", "middle_slide_top") > 0.2
    assert skills.close("kitchen_counter", "middle_slide_top").success
    assert abs(world.joint_position("kitchen_counter", "middle_slide_top")) < 0.03


@needs_kitchen
def test_suggested_base_pose_opens_and_closes(kitchen_skills):
    park, world = kitchen_skills
    skills = park((-2.55, -1.95, 1.5708))
    pose, coverage = skills.suggest_base_pose("kitchen_counter", "middle_slide_top")
    assert coverage > 0.99
    skills = park(pose)
    assert skills.open("kitchen_counter", "middle_slide_top").success
    assert skills.close("kitchen_counter", "middle_slide_top").success
