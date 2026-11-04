"""Pick skill: ManiSkill's official PickCube-v1 PPO (pd_joint_delta_pos), reused on the mobile Panda."""
from pathlib import Path

import torch
from torch import nn
from mani_skill.utils import common
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply, quaternion_raw_multiply
from mani_skill.utils.structs.pose import Pose

from ..paths import CHECKPOINTS

CHECKPOINT = CHECKPOINTS / "maniskill/PickCube-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt"
TRAINING_MOUNT = (-0.615, 0.0, 0.0)
TRAINING_RANGE = (0.08, 0.10)
GOAL_HEIGHT = 0.16


class PickPolicy(nn.Module):
    def __init__(self, checkpoint=CHECKPOINT):
        super().__init__()
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{checkpoint} missing; run python scripts/download_checkpoints.py PickCube-v1")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.actor_mean = nn.Sequential(nn.Linear(42, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                                        nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 8))
        self.load_state_dict({k: v for k, v in state.items() if k.startswith("actor_mean.")})
        self.eval()

    @torch.no_grad()
    def forward(self, obs):
        if obs.shape != (1, 42) or not torch.isfinite(obs).all():
            raise RuntimeError("invalid PickCube observation")
        return self.actor_mean(obs).clamp(-1, 1)


def _to_training_frame(env):
    """Rigid transform taking world poses to the PickCube training world (keeps raw quaternion signs)."""
    mount = Pose.create_from_pq(p=torch.tensor([TRAINING_MOUNT]))
    inverse = env.agent.robot.links_map["panda_link0"].pose.inv()
    rot = quaternion_raw_multiply(mount.q, inverse.q)
    translation = mount.p + quaternion_apply(mount.q, inverse.p)

    def transform(pose):
        return Pose.create_from_pq(p=translation + quaternion_apply(rot, pose.p),
                                   q=quaternion_raw_multiply(rot, pose.q))
    return transform


def in_training_range(env, cube):
    p = _to_training_frame(env)(cube.pose).p[0]
    return (abs(float(p[0])) <= TRAINING_RANGE[0] and abs(float(p[1])) <= TRAINING_RANGE[1]
            and abs(float(p[2]) - 0.02) <= 0.025)


def encode(env, cube, goal_world):
    """42-D PickCube-v1 'state' observation for `cube`, goal given in world coordinates (1x3 tensor)."""
    robot = env.agent.robot
    names = [j.name for j in robot.get_active_joints()]
    ids = [names.index(n) for n in env.agent.arm_joint_names + env.agent.gripper_joint_names]
    agent = dict(qpos=robot.get_qpos()[:, ids], qvel=robot.get_qvel()[:, ids])
    transform = _to_training_frame(env)
    obj = transform(cube.pose)
    tcp = transform(env.agent.tcp.pose)
    goal = transform(Pose.create_from_pq(p=goal_world)).p
    extra = dict(is_grasped=env.agent.is_grasping(cube), tcp_pose=tcp.raw_pose, goal_pos=goal,
                 obj_pose=obj.raw_pose, tcp_to_obj_pos=obj.p - tcp.p, obj_to_goal_pos=goal - obj.p)
    return common.flatten_state_dict(dict(agent=agent, extra=extra), use_torch=True).float()
