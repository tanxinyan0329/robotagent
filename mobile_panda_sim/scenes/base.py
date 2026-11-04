"""Shared parts of every mobile Panda scene: robot placement, the two RGB cameras, no task logic."""
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig

from .. import robot as _robot  # noqa: F401
from ..geometry import ARM_REST, BASE_CAMERA_SIZE, CAMERA_FOV_Y, HAND_CAMERA_SIZE

ROOT_ABOVE_FLOOR = {"mobile_panda_wheels": 0.205 + 0.002, "mobile_panda_planar": 0.0}
CAMERAS = {"base_camera": BASE_CAMERA_SIZE, "hand_camera": HAND_CAMERA_SIZE}


class MobilePandaSceneBase(BaseEnv):
    SUPPORTED_ROBOTS = ["mobile_panda_wheels", "mobile_panda_planar"]
    FLOOR_Z = 0.0
    DEFAULT_START = (0.0, 0.0, 0.0)

    def __init__(self, *args, robot_uids="mobile_panda_wheels", start=None, **kwargs):
        self.start = tuple(float(v) for v in (start if start is not None else self.DEFAULT_START))
        self.movable_objects = {}
        self.articulations = {}
        self.handles = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        links = self.agent.robot.links_map
        return [CameraConfig(name, sapien.Pose(), width, height, CAMERA_FOV_Y, 0.01, 10.0, mount=links[f"{name}_link"])
                for name, (width, height) in CAMERAS.items()]

    def _root_z(self):
        return self.FLOOR_Z + ROOT_ABOVE_FLOOR[self.robot_uids]

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(p=[self.start[0], self.start[1], self._root_z()]))

    def _initialize_robot(self):
        robot = self.agent.robot
        names = [j.name for j in robot.get_active_joints()]
        q = torch.zeros((1, len(names)), device=self.device)
        for name, value in zip(self.agent.arm_joint_names + self.agent.gripper_joint_names, ARM_REST):
            q[:, names.index(name)] = float(value)
        self.agent.reset(q)
        x, y, yaw = self.start
        robot.set_pose(sapien.Pose(p=[x, y, self._root_z()], q=[np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))
        if not self.agent.fix_root_link:
            robot.set_root_linear_velocity(torch.zeros((1, 3)))
            robot.set_root_angular_velocity(torch.zeros((1, 3)))

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info):
        return {}

    def capture_rgb(self):
        """Render all sensors once and return {camera: uint8 HxWx3 numpy array}."""
        self.scene.update_render()
        images = {}
        for name in CAMERAS:
            camera = self._sensors[name]
            camera.capture()
            rgb = camera.get_obs(rgb=True, depth=False, position=False, segmentation=False)["rgb"]
            images[name] = rgb[0, ..., :3].to(torch.uint8).cpu().numpy()
        return images

    def camera_intrinsics(self):
        return {name: self._sensors[name].get_params()["intrinsic_cv"][0].cpu().numpy().tolist() for name in CAMERAS}

    def joint(self, articulation, joint):
        """ManiSkill joint `joint` of articulation `articulation` (names as in /state 'articulations')."""
        for j in self.articulations[articulation].get_active_joints():
            if j.name == joint:
                return j
        raise KeyError(f"{articulation} has no joint {joint}")

    def handle_pose(self, articulation, joint):
        """World pose of the handle bar moved by this joint (x along the bar, z out of the panel) and the
        assets.Handle (length, radius)."""
        handle = self.handles[articulation][joint]
        link = self.joint(articulation, joint).child_link
        return link.pose.sp * handle.local, handle

    def layout(self):
        raise NotImplementedError
