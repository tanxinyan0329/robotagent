"""ManiSkill agents for the mobile Panda (ported from agent_demo/mobile_panda/mobile_panda.py)."""
from copy import deepcopy

from mani_skill.agents.controllers import PDBaseForwardVelControllerConfig, PDJointVelControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda

from .geometry import RADIUS, TRACK
from .paths import simulator_urdf

CHASSIS_LINKS = ("chassis_link", "pedestal_link", "left_wheel", "right_wheel", "front_support", "rear_support")


class MobilePandaBase(Panda):
    keyframes = {}
    base_mode = None

    @property
    def _controller_configs(self):
        original = super()._controller_configs
        result = {}
        for name in ("pd_joint_delta_pos", "pd_joint_pos"):
            config = deepcopy(original[name])
            if self.fix_root_link:
                config["base"] = PDBaseForwardVelControllerConfig(
                    ["base_x_joint", "base_y_joint", "base_yaw_joint"],
                    lower=[-0.25, -0.7], upper=[0.25, 0.7], damping=3000,
                    force_limit=5000, normalize_action=False,
                )
            else:
                config["base"] = PDJointVelControllerConfig(
                    ["left_wheel_joint", "right_wheel_joint"],
                    lower=-10, upper=10, damping=30, force_limit=40, normalize_action=False,
                )
            result[name] = config
        return result

    def base_action(self, linear, angular):
        """Body twist (m/s, rad/s) -> this model's base controller action."""
        if self.fix_root_link:
            return [linear, angular]
        return [(linear - angular*TRACK/2) / RADIUS, (linear + angular*TRACK/2) / RADIUS]


@register_agent()
class MobilePandaPlanar(MobilePandaBase):
    uid = "mobile_panda_planar"
    urdf_path = str(simulator_urdf("planar"))
    fix_root_link = True
    base_mode = "planar"


@register_agent()
class MobilePandaWheels(MobilePandaBase):
    uid = "mobile_panda_wheels"
    urdf_path = str(simulator_urdf("wheels"))
    fix_root_link = False
    base_mode = "wheels"

    urdf_config = deepcopy(Panda.urdf_config)
    urdf_config["_materials"].update(
        tyre=dict(static_friction=1.2, dynamic_friction=1.0, restitution=0),
        support=dict(static_friction=0.015, dynamic_friction=0.01, restitution=0),
    )
    for _name in ("left_wheel", "right_wheel"):
        urdf_config["link"][_name] = dict(material="tyre")
    for _name in ("front_support", "rear_support"):
        urdf_config["link"][_name] = dict(material="support")

    def _after_init(self):
        super()._after_init()
        for name in CHASSIS_LINKS:
            self.robot.links_map[name].disable_gravity = False
