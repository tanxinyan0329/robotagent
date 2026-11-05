"""The PPO pick policy trained on MuJoCo (training/mujoco/train_pick.py), on the mobile robot."""
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from ..manipulation.pose import Pose
from ..paths import CHECKPOINTS
from .pick_task import ACTION_SIZE, CUBE_RANGE, DOCK, OBS_SIZE, is_grasped

CHECKPOINT = CHECKPOINTS / "mujoco/pick_cube.pt"
MOUNT = Pose(DOCK[:2] + (0.0,))


class PickPolicy(nn.Module):
    def __init__(self, checkpoint=CHECKPOINT):
        super().__init__()
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{checkpoint} missing; train it with training/mujoco/train_pick.py")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.actor_mean = nn.Sequential(nn.Linear(OBS_SIZE, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                                        nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, ACTION_SIZE))
        self.load_state_dict({k: v for k, v in state.items() if k.startswith("actor_mean.")})
        self.eval()

    @torch.no_grad()
    def forward(self, observation):
        return self.actor_mean(torch.from_numpy(observation)).clamp(-1, 1).numpy()


class PickObserver:
    """Builds the training observation on a MujocoScene: world poses are moved so that panda_link0 sits where it
    did in training, which lets the policy run wherever the base is parked."""

    def __init__(self, scene, cube):
        self.scene, self.model, self.data = scene, scene.model, scene.data
        model = self.model
        self.cube = model.body(cube).id
        self.link0 = model.body("panda_link0").id
        self.finger_q = [model.joint(n).qposadr[0] for n in ("panda_finger_joint1", "panda_finger_joint2")]
        self.finger_v = [model.joint(n).dofadr[0] for n in ("panda_finger_joint1", "panda_finger_joint2")]
        self.arm_v = [model.joint(f"panda_joint{i}").dofadr[0] for i in range(1, 8)]
        self.cube_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == self.cube}
        self.finger_geoms = [{g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(name).id and
                              (model.geom_contype[g] or model.geom_conaffinity[g])} for name in ("panda_leftfinger", "panda_rightfinger")]

    def to_training(self):
        return (MOUNT * Pose(self.data.xpos[self.link0], self.data.xquat[self.link0]).inv()).to_transformation_matrix()

    def grasped(self):
        return is_grasped(self.model, self.data, self.cube_geoms, self.finger_geoms, self.finger_q)

    def cube_in_range(self):
        frame = self.to_training()
        p = frame[:3, :3] @ self.data.xpos[self.cube] + frame[:3, 3]
        return abs(p[0]) <= CUBE_RANGE and abs(p[1]) <= CUBE_RANGE and abs(p[2] - 0.02) <= 0.025

    def observation(self, goal_world):
        data, frame = self.data, self.to_training()
        rotation, shift = frame[:3, :3], frame[:3, 3]
        turn = np.zeros(4)
        mujoco.mju_mat2Quat(turn, rotation.reshape(-1))
        cube_p, tcp_p = rotation @ data.xpos[self.cube] + shift, rotation @ data.site_xpos[self.scene.tcp] + shift
        goal = rotation @ np.asarray(goal_world) + shift
        cube_q, tcp_q = np.zeros(4), np.zeros(4)
        mujoco.mju_mulQuat(cube_q, turn, data.xquat[self.cube])
        mujoco.mju_mat2Quat(tcp_q, (rotation @ data.site_xmat[self.scene.tcp].reshape(3, 3)).reshape(-1))
        return np.concatenate([data.qpos[self.scene.arm_q], data.qpos[self.finger_q], data.qvel[self.arm_v], data.qvel[self.finger_v],
                               [float(self.grasped())], tcp_p, tcp_q, goal, cube_p, cube_q, cube_p - tcp_p, goal - cube_p]).astype(np.float32)
