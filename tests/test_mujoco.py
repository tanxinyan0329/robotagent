"""Model, scene and skills. Collected only with mujoco installed and the Menagerie model downloaded (see conftest.py)."""
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
from mobile_panda_sim.mj.world import MujocoWorld
from mobile_panda_sim.paths import REPO, ros_urdf

REST_TCP = {"wheels": (-2.55, -1.335, 1.0964), "planar": (-2.55, -1.335, 1.0894)}
BASE_JOINTS = {"wheels": ["left_wheel_joint", "right_wheel_joint"], "planar": ["base_x_joint", "base_y_joint", "base_yaw_joint"]}
FOCAL = {"base_camera": 439.317, "hand_camera": 219.659}


@pytest.fixture(scope="module", params=["wheels", "planar"])
def scene(request):
    result = MujocoScene(request.param)
    yield result
    result.close()


@pytest.fixture(scope="module")
def wheels():
    result = MujocoScene("wheels")
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


def test_rest_pose(scene):
    scene.reset()
    assert scene.joint_names == BASE_JOINTS[scene.base] + ARM_JOINTS + ["panda_finger_joint1", "panda_finger_joint2"]
    assert np.allclose(scene.joint_positions()[-9:], ARM_REST)
    assert np.allclose(scene.tcp_pose()[0], REST_TCP[scene.base], atol=2e-3)


def test_arm_kinematics_match_urdf(wheels):
    wheels.reset()
    joints = {j.get("name"): j for j in E.parse(ros_urdf("wheels")).getroot().findall("joint")}
    link0 = wheels.model.body("panda_link0").id
    data = mujoco.MjData(wheels.model)
    data.qpos[:] = wheels.data.qpos
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(wheels.model.actuator_ctrlrange[wheels.arm_ctrl, 0], wheels.model.actuator_ctrlrange[wheels.arm_ctrl, 1])
        data.qpos[wheels.arm_q] = q
        mujoco.mj_kinematics(wheels.model, data)
        expected = np.eye(4)
        for i, angle in enumerate(q, start=1):
            expected = expected @ urdf_transform(joints[f"panda_joint{i}"], angle)
        for name in ("panda_joint8", "panda_hand_joint", "panda_hand_tcp_joint"):
            expected = expected @ urdf_transform(joints[name])
        base = np.eye(4)
        base[:3, :3], base[:3, 3] = data.xmat[link0].reshape(3, 3), data.xpos[link0]
        assert np.allclose((base @ expected)[:3, 3], data.site_xpos[wheels.tcp], atol=1e-4)


def test_base_rests_and_drives(scene):
    scene.reset()
    for _ in range(20):
        scene.step(0, 0, None, 0.04)
    position = scene.pose(scene.chassis)[0]
    assert np.linalg.norm(position[:2] - scene.start[:2]) < 0.005
    for _ in range(40):
        scene.step(0.15, 0, None, 0.04)
    assert np.linalg.norm(scene.pose(scene.chassis)[0][:2] - position[:2]) == pytest.approx(0.3, abs=0.02)
    yaw = yaw_of(scene.pose(scene.chassis)[1])
    for _ in range(40):
        scene.step(0, 0.4, None, 0.04)
    assert yaw_of(scene.pose(scene.chassis)[1]) - yaw == pytest.approx(0.8, abs=0.04)


def test_arm_delta_and_gripper(scene):
    scene.reset()
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
    scene.reset()
    images = scene.capture_rgb()
    assert images["base_camera"].shape == (480, 640, 3) and images["hand_camera"].shape == (240, 320, 3)
    assert all(a.dtype == np.uint8 and a.std() > 5 for a in images.values())
    for name, matrix in scene.camera_intrinsics().items():
        assert matrix[0][0] == pytest.approx(FOCAL[name], abs=0.01)


def test_joint_limits_are_radians(wheels):
    assert wheels.model.jnt_range[wheels.model.joint("fridge/top_door_hinge").id][1] == pytest.approx(2.3)


def test_props_stay_put(scene):
    scene.reset()
    start = {name: scene.pose(body)[0] for name, body in scene.objects.items()}
    for _ in range(400):
        scene.step(0, 0, None, 0.04)
    moved = {name: float(np.linalg.norm(scene.pose(body)[0] - start[name])) for name, body in scene.objects.items()}
    assert max(moved.values()) < 0.005, moved


def test_reset_restores_scene(wheels):
    wheels.reset((-3.0, -2.2, 3.1416))
    start = {name: wheels.pose(body)[0] for name, body in wheels.objects.items()}
    address = wheels.articulations["fridge"]["top_door_hinge"]
    wheels.data.qpos[address] = 0.5
    for _ in range(100):
        wheels.step(0, 0, None, 0.04)
    wheels.reset()
    assert wheels.articulation_positions()["fridge"]["top_door_hinge"] == 0.0
    for _ in range(5):
        wheels.step(0, 0, None, 0.04)
    assert max(np.linalg.norm(wheels.pose(body)[0] - start[name]) for name, body in wheels.objects.items()) < 0.005


def test_worker_state():
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    process = subprocess.Popen([sys.executable, "-u", "-m", "mobile_panda_sim.ros2.worker"], cwd=REPO, env=env,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        seen = {}
        while "state" not in seen:
            message = json.loads(process.stdout.readline())
            seen[message["type"]] = message
    finally:
        process.stdin.close()
        assert process.wait(timeout=20) == 0
    assert seen["scene"]["layout"]["robot"] == "mobile_panda_wheels"
    assert seen["ready"]["control_timestep"] == pytest.approx(0.05)
    state = seen["state"]
    assert {"base_pose", "root_pose", "tcp_pose", "joint_positions", "joint_velocities", "gripper_width", "images"} <= set(state)
    assert len(state["objects"]) == 42 and state["images"]["base_camera"]["width"] == 640
    assert set(state["articulations"]) == {"kitchen_counter", "fridge", "sideboard"}


@pytest.fixture(scope="module")
def skills_on():
    pytest.importorskip("mplib")
    from mobile_panda_sim.manipulation.skills import ManipulationSkills
    scene = MujocoScene("wheels")
    world = MujocoWorld(scene)

    def park(start):
        scene.reset(start)
        return ManipulationSkills(world)
    yield park, world
    scene.close()


def test_world_reads_the_scene(skills_on):
    park, world = skills_on
    park((-2.95, -1.15, 3.1416))
    assert world.surface(-3.40, -1.25) == pytest.approx(0.918, abs=0.01)
    assert world.object_points("sugar_box")[:, 2].max() > 1.05
    assert world.joint_info("fridge", "top_door_hinge")["kind"] == "revolute"
    assert world.joint_info("kitchen_counter", "middle_slide_top")["kind"] == "prismatic"
    pose, handle = world.handle("kitchen_counter", "middle_slide_top")
    assert handle.radius == pytest.approx(0.0088, abs=1e-3)
    assert pose.to_transformation_matrix()[:3, 2] @ [0, -1, 0] > 0.99


def test_pick_and_place_skill(skills_on):
    park, world = skills_on
    skills = park((-2.95, -1.15, 3.1416))
    assert skills.pick("lemon").success
    assert skills.place((-3.38, -1.42)).success
    assert np.linalg.norm(world.object_position("lemon")[:2] - [-3.38, -1.42]) < 0.04


def test_drawer_skill(skills_on):
    park, world = skills_on
    skills = park((-1.15, 1.83, 1.989))
    assert skills.open("kitchen_counter", "middle_slide_top").success
    assert world.joint_position("kitchen_counter", "middle_slide_top") > 0.2
    assert skills.close("kitchen_counter", "middle_slide_top").success
    assert abs(world.joint_position("kitchen_counter", "middle_slide_top")) < 0.03


def test_suggested_base_pose_opens_and_closes(skills_on):
    park, world = skills_on
    skills = park((-2.55, -1.95, 1.5708))
    pose, coverage = skills.suggest_base_pose("kitchen_counter", "middle_slide_top")
    assert coverage > 0.99
    skills = park(pose)
    assert skills.open("kitchen_counter", "middle_slide_top").success
    assert skills.close("kitchen_counter", "middle_slide_top").success


def test_policy_observation_matches_training():
    from mobile_panda_sim.mj.rl_skills import RLSkills
    from mobile_panda_sim.mj.rl_tasks import DOCK, PickMoveTask
    task = PickMoveTask(1, seed=3, threads=1)
    expected = task.observe()[0]
    data = task.datas[0]
    scene = MujocoScene("planar", DOCK)
    scene.reset()
    for name, address in task.addresses.items():
        target = scene.model.joint(name).qposadr[0]
        scene.data.qpos[target:target + 7] = data.qpos[address:address + 7]
    scene.data.qpos[scene.arm_q] = data.qpos[task.arm_q]
    mujoco.mj_forward(scene.model, scene.data)
    skills = RLSkills(scene)
    skills.select(task.target[0])
    observed = skills.observation(task.goal[0])
    assert np.allclose(observed, expected, atol=1e-3), np.abs(observed - expected).argmax()
    scene.close()
    task.close()


def test_task_resets_and_rewards():
    from mobile_panda_sim.mj.rl_tasks import ACTION_SIZE, MAX_STEPS, OBS_SIZE, PickMoveTask
    task = PickMoveTask(4, seed=0, threads=1)
    obs = task.reset()
    assert obs.shape == (4, OBS_SIZE) and np.isfinite(obs).all()
    for _ in range(MAX_STEPS):
        obs, reward, failed, ended, final, _ = task.step(np.zeros((4, ACTION_SIZE), dtype=np.float32))
        assert np.all(reward <= 1)
    assert ended.all() and (task.steps == 0).all()
    task.close()


def test_closed_fingers_on_top_do_not_count_as_a_grasp():
    from mobile_panda_sim.mj.rl_tasks import MIN_FINGER_GAP, PickMoveTask, is_grasped
    task = PickMoveTask(1, seed=0, threads=1)
    data = task.datas[0]
    data.qpos[task.finger_q] = 0.0
    assert not is_grasped(task.model, data, task.target_geoms[task.target[0]], task.finger_geoms, task.finger_q)
    assert MIN_FINGER_GAP < 0.05
    task.close()


def test_moving_a_neighbour_fails_the_episode():
    from mobile_panda_sim.mj.rl_tasks import ACTION_SIZE, PickMoveTask
    task = PickMoveTask(2, seed=0, threads=1)
    data = task.datas[0]
    address = task.addresses["cracker_box"]
    data.qpos[address] += 0.1
    mujoco.mj_forward(task.model, data)
    _, _, failed, _, _, _ = task.step(np.zeros((2, ACTION_SIZE), dtype=np.float32))
    assert failed[0] and not failed[1]
    task.close()


def test_trained_policies_run_on_the_robot():
    from mobile_panda_sim.mj.rl_skills import RLSkills
    from mobile_panda_sim.mj.rl_tasks import DOCK, TABLE_Z
    scene = MujocoScene("planar", DOCK)
    scene.reset()
    skills = RLSkills(scene)
    assert skills.pick_move("lemon", np.array([-3.40, -1.50, TABLE_Z + 0.15]))["success"]
    assert skills.place((-3.34, -1.56))["success"]
    scene.close()
