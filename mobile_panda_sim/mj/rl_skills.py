"""The two trained policies on the mobile robot: pick_move() and place()."""
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from ..geometry import GRIPPER_RANGE, MAX_JOINT_DELTA
from ..manipulation.pose import Pose
from ..paths import POLICIES
from .rl_tasks import (ACTION_SIZE, AT_GOAL, DISTURBED, DOCK, HAND_ABOVE, MAX_STEPS, NEIGHBOURS, OBS_SIZE, OPEN_GAP, STILL, TABLE_Z, TARGETS,
                       body_box, body_geoms, is_grasped)

LINK0_HEIGHT = 0.9196429
TRAINED_LINK0 = Pose((DOCK[0], DOCK[1], LINK0_HEIGHT), (np.cos(DOCK[2]/2), 0, 0, np.sin(DOCK[2]/2)))


def load_policy(path):
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    actor = nn.Sequential(nn.Linear(OBS_SIZE, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                          nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, ACTION_SIZE))
    actor.load_state_dict({k[len("actor_mean."):]: v for k, v in state.items() if k.startswith("actor_mean.")})
    return actor.eval()


class RLSkills:
    """Works from the docking pose DOCK. Observations are moved so that panda_link0 sits where it did in training."""

    def __init__(self, scene, pick_policy=POLICIES / "pick_move.pt", place_policy=POLICIES / "place.pt"):
        self.scene, self.model, self.data = scene, scene.model, scene.data
        model = self.model
        self.pick_policy, self.place_policy = load_policy(pick_policy), load_policy(place_policy)
        self.link0 = model.body("panda_link0").id
        self.finger_q = [model.joint(n).qposadr[0] for n in ("panda_finger_joint1", "panda_finger_joint2")]
        self.finger_v = [model.joint(n).dofadr[0] for n in ("panda_finger_joint1", "panda_finger_joint2")]
        self.arm_v = [model.joint(f"panda_joint{i}").dofadr[0] for i in range(1, 8)]
        self.finger_geoms = [body_geoms(model, model.body(n).id) for n in ("panda_leftfinger", "panda_rightfinger")]
        self.boxes = {}

    def select(self, name):
        if name not in TARGETS:
            raise ValueError(f"the policies were trained on {TARGETS}, not {name}")
        self.name, self.body = name, self.model.body(name).id
        self.target_geoms = body_geoms(self.model, self.body)
        if name not in self.boxes:
            data = mujoco.MjData(self.model)
            data.qpos[:] = self.data.qpos
            address = self.model.joint(name).qposadr[0]
            data.qpos[address:address + 3], data.qpos[address + 3:address + 7] = self.scene.settled[name]
            mujoco.mj_forward(self.model, data)
            self.boxes[name] = body_box(self.model, data, self.body)
        self.centre_offset, self.size = self.boxes[name]

    def grasped(self):
        return is_grasped(self.model, self.data, self.target_geoms, self.finger_geoms, self.finger_q)

    def centre(self):
        return self.data.xpos[self.body] + self.data.xmat[self.body].reshape(3, 3) @ self.centre_offset

    def to_training(self):
        return (TRAINED_LINK0 * Pose(self.data.xpos[self.link0], self.data.xquat[self.link0]).inv()).to_transformation_matrix()

    def observation(self, goal):
        """`goal` is in world coordinates."""
        data, frame = self.data, self.to_training()
        rotation, shift = frame[:3, :3], frame[:3, 3]
        turn = np.zeros(4)
        mujoco.mju_mat2Quat(turn, rotation.reshape(-1))
        centre, tcp = rotation @ self.centre() + shift, rotation @ data.site_xpos[self.scene.tcp] + shift
        goal = rotation @ np.asarray(goal) + shift
        quat, tcp_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, data.xquat[self.body])
        mujoco.mju_mat2Quat(tcp_quat, (rotation @ data.site_xmat[self.scene.tcp].reshape(3, 3)).reshape(-1))
        return np.concatenate([data.qpos[self.scene.arm_q], data.qpos[self.finger_q], data.qvel[self.arm_v], data.qvel[self.finger_v],
                               [float(self.grasped())], tcp, tcp_quat, goal, centre, quat, centre - tcp, goal - centre, self.size]).astype(np.float32)

    def act(self, policy, goal, on_step):
        low, high = GRIPPER_RANGE
        with torch.no_grad():
            action = policy(torch.from_numpy(self.observation(goal))).clamp(-1, 1).numpy()
        self.scene.step(0.0, 0.0, action[:7]*MAX_JOINT_DELTA, low + (action[7] + 1)/2*(high - low))
        if on_step:
            on_step()

    def neighbours(self):
        return {n: self.data.xpos[self.model.body(n).id].copy() for n in (*TARGETS, *NEIGHBOURS) if n != self.name}

    def moved(self, before):
        now = self.neighbours()
        return {n: round(float(np.linalg.norm(now[n] - before[n])), 3) for n in now if np.linalg.norm(now[n] - before[n]) > DISTURBED}

    def pick_move(self, name, goal, on_step=None, limit=MAX_STEPS):
        """Take `name` and hold it still with its centre at the world point `goal`."""
        self.select(name)
        before = self.neighbours()
        for step in range(limit):
            self.act(self.pick_policy, goal, on_step)
            speed = np.linalg.norm(self.data.qvel[self.arm_v])
            if np.linalg.norm(self.centre() - goal) <= AT_GOAL and speed < STILL and self.grasped():
                moved = self.moved(before)
                return dict(success=not moved, steps=step + 1, moved=moved)
        return dict(success=False, steps=limit, moved=self.moved(before), error=round(float(np.linalg.norm(self.centre() - goal)), 3))

    def place(self, xy, on_step=None, limit=MAX_STEPS):
        """Set the held object down at (x, y) on the console, let go and lift the hand."""
        goal = np.array([xy[0], xy[1], TABLE_Z + self.size[2]])
        before = self.neighbours()
        for step in range(limit):
            self.act(self.place_policy, goal, on_step)
            released = np.linalg.norm(self.centre() - goal) <= AT_GOAL and not self.grasped()
            gap = self.data.qpos[self.finger_q].sum()
            above = self.data.site_xpos[self.scene.tcp][2] - self.centre()[2]
            still = np.linalg.norm(self.data.cvel[self.body]) < 0.1
            if released and gap > OPEN_GAP and above > HAND_ABOVE and still:
                moved = self.moved(before)
                return dict(success=not moved, steps=step + 1, moved=moved)
        return dict(success=False, steps=limit, moved=self.moved(before), error=round(float(np.linalg.norm(self.centre() - goal)), 3))
