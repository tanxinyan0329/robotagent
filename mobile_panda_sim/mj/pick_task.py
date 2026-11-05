"""Pick a cube and carry it to a goal: ManiSkill's PickCube-v1 task (same observation, action, reward and success)
on the MuJoCo tabletop, with many copies stepped in parallel on the CPU."""
from concurrent.futures import ThreadPoolExecutor
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..geometry import ARM_REST, CUBE_HALF, GRIPPER_RANGE, HEIGHT, MAX_JOINT_DELTA  # noqa: E402
from .model import ARM_JOINTS, CONTROL_SUBSTEPS, FINGER_JOINTS, add_tabletop, build  # noqa: E402

OBS_SIZE, ACTION_SIZE = 42, 8
MAX_STEPS = 50
DOCK = (-0.615, 0.0, 0.0)
CUBE_RANGE, GOAL_RANGE, GOAL_HEIGHT = 0.1, 0.1, 0.3
PLACED, STATIC = 0.025, 0.2
MIN_FINGER_GAP = 0.03
RESTING_CUBE = (0.8, 0.0)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def is_grasped(model, data, cube_geoms, finger_geoms, finger_q):
    """Both fingers press on the cube and it is between them; closed fingertips on top must not count."""
    n = data.ncon
    if n == 0 or data.qpos[finger_q].sum() < MIN_FINGER_GAP:
        return False
    first, second, dist = data.contact.geom1[:n], data.contact.geom2[:n], data.contact.dist[:n]
    pressed = dist < 0
    seen = [False, False]
    for a, b in zip(first[pressed], second[pressed]):
        if a in cube_geoms or b in cube_geoms:
            other = b if a in cube_geoms else a
            for i, geoms in enumerate(finger_geoms):
                if other in geoms:
                    seen[i] = True
    return seen[0] and seen[1]


class PickCubeTask:
    def __init__(self, num_envs, seed=0, threads=None):
        self.model, _ = build(add_tabletop, -HEIGHT, "planar", DOCK)
        model = self.model
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.datas = [mujoco.MjData(model) for _ in range(num_envs)]
        threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self.pool = ThreadPoolExecutor(threads)
        self.chunks = [self.datas[i::threads] for i in range(threads) if self.datas[i::threads]]
        self.arm_q = [model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.arm_v = [model.joint(n).dofadr[0] for n in ARM_JOINTS]
        self.finger_q = [model.joint(n).qposadr[0] for n in FINGER_JOINTS]
        self.finger_v = [model.joint(n).dofadr[0] for n in FINGER_JOINTS]
        self.arm_ctrl = [model.actuator(n).id for n in ARM_JOINTS]
        self.gripper = model.actuator("gripper").id
        self.base_ctrl = [model.actuator(n).id for n in ("base_x_joint", "base_y_joint", "base_yaw_joint")]
        self.cube_q = model.joint("red_cube").qposadr[0]
        self.parked_q = model.joint("blue_cube").qposadr[0]
        self.cube = model.body("red_cube").id
        self.tcp = model.site("tcp").id
        self.cube_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == self.cube}
        self.finger_geoms = [{g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(name).id and
                              (model.geom_contype[g] or model.geom_conaffinity[g])} for name in ("panda_leftfinger", "panda_rightfinger")]
        self.goal = np.zeros((num_envs, 3))
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.grasped = np.zeros(num_envs, dtype=bool)
        for i in range(num_envs):
            self.reset_one(i)

    def reset_one(self, i):
        model, data, rng = self.model, self.datas[i], self.rng
        mujoco.mj_resetData(model, data)
        data.qpos[self.arm_q + self.finger_q] = ARM_REST
        data.qpos[self.arm_q] += rng.normal(0, 0.02, 7)
        yaw = rng.uniform(0, 2*np.pi)
        data.qpos[self.cube_q:self.cube_q + 3] = [*rng.uniform(-CUBE_RANGE, CUBE_RANGE, 2), CUBE_HALF]
        data.qpos[self.cube_q + 3:self.cube_q + 7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        data.qpos[self.parked_q:self.parked_q + 3] = [*RESTING_CUBE, CUBE_HALF]
        data.ctrl[self.arm_ctrl] = data.qpos[self.arm_q]
        data.ctrl[self.gripper] = GRIPPER_RANGE[1]
        mujoco.mj_forward(model, data)
        self.goal[i] = [*rng.uniform(-GOAL_RANGE, GOAL_RANGE, 2), rng.uniform(0, GOAL_HEIGHT) + CUBE_HALF]
        self.steps[i] = 0
        self.grasped[i] = False

    def _advance(self, datas):
        for data in datas:
            mujoco.mj_step(self.model, data, CONTROL_SUBSTEPS)

    def observe(self):
        """(observations, tcp-to-cube distance, cube-to-goal distance, arm speed)."""
        n = self.num_envs
        obs = np.zeros((n, OBS_SIZE), dtype=np.float32)
        tcp_quat = np.zeros(4)
        for i, data in enumerate(self.datas):
            self.grasped[i] = is_grasped(self.model, data, self.cube_geoms, self.finger_geoms, self.finger_q)
            cube_p, cube_q, tcp_p = data.xpos[self.cube], data.xquat[self.cube], data.site_xpos[self.tcp]
            mujoco.mju_mat2Quat(tcp_quat, data.site_xmat[self.tcp])
            obs[i, :7], obs[i, 7:9] = data.qpos[self.arm_q], data.qpos[self.finger_q]
            obs[i, 9:16], obs[i, 16:18] = data.qvel[self.arm_v], data.qvel[self.finger_v]
            obs[i, 18] = self.grasped[i]
            obs[i, 19:22], obs[i, 22:26] = tcp_p, tcp_quat
            obs[i, 26:29] = self.goal[i]
            obs[i, 29:32], obs[i, 32:36] = cube_p, cube_q
            obs[i, 36:39], obs[i, 39:42] = cube_p - tcp_p, self.goal[i] - cube_p
        return obs

    def reward(self, obs):
        reach = np.linalg.norm(obs[:, 36:39], axis=1)
        to_goal = np.linalg.norm(obs[:, 39:42], axis=1)
        speed = np.linalg.norm(obs[:, 9:16], axis=1)
        placed = to_goal <= PLACED
        success = placed & (speed < STATIC)
        reward = 1 - np.tanh(5*reach) + self.grasped + (1 - np.tanh(5*to_goal))*self.grasped + (1 - np.tanh(5*speed))*placed
        reward[success] = 5.0
        return reward / 5.0, success

    def reset(self):
        for i in range(self.num_envs):
            self.reset_one(i)
        return self.observe()

    def step(self, actions):
        """actions (N, 8) in [-1, 1]. Returns obs, reward, terminated, truncated, final_obs (for finished envs), success."""
        low, high = GRIPPER_RANGE
        for i, data in enumerate(self.datas):
            data.ctrl[self.arm_ctrl] = data.qpos[self.arm_q] + actions[i, :7]*MAX_JOINT_DELTA
            data.ctrl[self.gripper] = low + (actions[i, 7] + 1)/2*(high - low)
        list(self.pool.map(self._advance, self.chunks))
        self.steps += 1
        obs = self.observe()
        reward, success = self.reward(obs)
        truncated = (self.steps >= MAX_STEPS) & ~success
        done = success | truncated
        final_obs = obs.copy()
        if done.any():
            for i in np.flatnonzero(done):
                self.reset_one(i)
            obs = self.observe()
        return obs, reward, success, truncated, final_obs, success

    def close(self):
        self.pool.shutdown()
