"""MuJoCo physics and rendering for one scene; same control conventions as the ManiSkill backend."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ..geometry import BASE_CAMERA_SIZE, CAMERA_FOV_Y, CUBE_HALF, CUBES, DEFAULT_START, HAND_CAMERA_SIZE, RADIUS, TRACK, tabletop_layout  # noqa: E402
from .model import ARM_JOINTS, CONTROL_SUBSTEPS, FINGER_JOINTS, REST_POSITIONS, TIMESTEP, build, yaw_quat  # noqa: E402

BASE_JOINTS = {"wheels": ["left_wheel_joint", "right_wheel_joint"], "planar": ["base_x_joint", "base_y_joint", "base_yaw_joint"]}
CAMERAS = {"base_camera": BASE_CAMERA_SIZE, "hand_camera": HAND_CAMERA_SIZE}
DEFAULT_STARTS = {"tabletop": DEFAULT_START}


class MujocoScene:
    control_timestep = TIMESTEP * CONTROL_SUBSTEPS

    def __init__(self, scene="tabletop", base="wheels", start=None, viewer=False, extend=None):
        if scene not in DEFAULT_STARTS:
            raise ValueError(f"scene {scene!r} is not available on the MuJoCo backend yet; use tabletop")
        self.scene, self.base = scene, base
        self.start = tuple(float(v) for v in (start if start is not None else DEFAULT_STARTS[scene]))
        self.model, self.spec = build(scene, base, self.start, extend)
        self.data = mujoco.MjData(self.model)
        self.joint_names = BASE_JOINTS[base] + ARM_JOINTS + FINGER_JOINTS
        self.qpos_index = [self.model.joint(n).qposadr[0] for n in self.joint_names]
        self.qvel_index = [self.model.joint(n).dofadr[0] for n in self.joint_names]
        self.arm_q = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.arm_ctrl = [self.model.actuator(n).id for n in ARM_JOINTS]
        self.base_ctrl = [self.model.actuator(n).id for n in BASE_JOINTS[base]]
        self.gripper_ctrl = self.model.actuator("gripper").id
        self.chassis = self.model.body("chassis_link").id
        self.root = self.model.body("chassis_link" if base == "wheels" else "world_root").id
        self.tcp = self.model.site("tcp").id
        self.objects = {name: self.model.body(name).id for name in CUBES}
        self.renderers = {name: mujoco.Renderer(self.model, height=h, width=w) for name, (w, h) in CAMERAS.items()}
        self.viewer = None
        if viewer:
            from mujoco import viewer as passive_viewer
            self.viewer = passive_viewer.launch_passive(self.model, self.data)

    def reset(self, seed=0):
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        for name, value in REST_POSITIONS.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        if self.base == "wheels":
            address = model.joint("root").qposadr[0]
            data.qpos[address:address + 3] = model.body("chassis_link").pos
            data.qpos[address + 3:address + 7] = yaw_quat(self.start[2])
        rng = np.random.default_rng(seed)
        for name, xy in CUBES.items():
            address = model.joint(name).qposadr[0]
            yaw = rng.uniform(0, 2*np.pi) if name == "red_cube" else 0.0
            data.qpos[address:address + 3] = [*xy, CUBE_HALF]
            data.qpos[address + 3:address + 7] = yaw_quat(yaw)
        data.ctrl[self.arm_ctrl] = data.qpos[self.arm_q]
        data.ctrl[self.gripper_ctrl] = REST_POSITIONS["panda_finger_joint1"]
        mujoco.mj_forward(model, data)

    def base_targets(self, linear, angular):
        if self.base == "wheels":
            return [(linear - angular*TRACK/2) / RADIUS, (linear + angular*TRACK/2) / RADIUS]
        yaw = self.data.qpos[self.model.joint("base_yaw_joint").qposadr[0]]
        return [linear*np.cos(yaw), linear*np.sin(yaw), angular]

    def step(self, linear, angular, arm_delta, gripper):
        """One control step. arm_delta is relative to the measured joints; None holds the current target."""
        data = self.data
        if arm_delta is not None:
            data.ctrl[self.arm_ctrl] = data.qpos[self.arm_q] + np.asarray(arm_delta)
        data.ctrl[self.gripper_ctrl] = gripper
        data.ctrl[self.base_ctrl] = self.base_targets(linear, angular)
        if self.viewer is None:
            mujoco.mj_step(self.model, data, CONTROL_SUBSTEPS)
        else:
            with self.viewer.lock():
                mujoco.mj_step(self.model, data, CONTROL_SUBSTEPS)
            self.viewer.sync()

    def pose(self, body):
        return self.data.xpos[body].copy(), self.data.xquat[body].copy()

    def tcp_pose(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.tcp])
        return self.data.site_xpos[self.tcp].copy(), quat

    def joint_positions(self):
        return self.data.qpos[self.qpos_index].copy()

    def joint_velocities(self):
        return self.data.qvel[self.qvel_index].copy()

    def capture_rgb(self):
        images = {}
        for name, renderer in self.renderers.items():
            renderer.update_scene(self.data, camera=name)
            images[name] = renderer.render().copy()
        return images

    def camera_intrinsics(self):
        result = {}
        for name, (width, height) in CAMERAS.items():
            focal = height / (2*np.tan(CAMERA_FOV_Y/2))
            result[name] = [[focal, 0.0, width/2], [0.0, focal, height/2], [0.0, 0.0, 1.0]]
        return result

    def layout(self):
        return tabletop_layout(self.start, f"mobile_panda_{self.base}")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        for renderer in self.renderers.values():
            renderer.close()
