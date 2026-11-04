"""Tabletop skill-test scene (merged from agent_demo mobile_env.py + robot_stack multi_env.py)."""
import sapien
import torch

from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from ..geometry import CUBE_HALF, CUBES, DEFAULT_START, HEIGHT, TABLE_CENTER, TABLE_HALF, TABLE_LEG_HALF, TABLE_LEGS, ZONE_HALF, ZONES
from .base import MobilePandaSceneBase

ZONE_COLORS = {"green_zone": [0.06, 0.65, 0.15, 1], "yellow_zone": [0.9, 0.75, 0.04, 1]}
CUBE_COLORS = {"red_cube": [1, 0, 0, 1], "blue_cube": [0.05, 0.12, 0.95, 1]}


@register_env("MobilePandaTabletop-v1", max_episode_steps=None)
class MobilePandaTabletop(MobilePandaSceneBase):
    FLOOR_Z = -HEIGHT
    DEFAULT_START = DEFAULT_START

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-2.5, -3.4, 1.5], target=[-0.45, 0, -0.35])
        return [CameraConfig("overview", pose, 800, 608, 1.0, 0.01, 100)]

    def _load_scene(self, options):
        self.ground = build_ground(self.scene, floor_width=15, altitude=-HEIGHT)
        self.table = actors.build_box(self.scene, half_sizes=[*TABLE_HALF, 0.035], color=[0.55, 0.36, 0.20, 1],
                                      name="table", body_type="kinematic",
                                      initial_pose=sapien.Pose(p=[*TABLE_CENTER, -0.035]))
        self.table_legs = [
            actors.build_box(self.scene, half_sizes=[TABLE_LEG_HALF, TABLE_LEG_HALF, (HEIGHT-0.07)/2],
                             color=[0.25, 0.25, 0.28, 1], name=f"table_leg_{i}", body_type="kinematic",
                             initial_pose=sapien.Pose(p=[x, y, (-HEIGHT-0.07)/2]))
            for i, (x, y) in enumerate(TABLE_LEGS)]
        self.cubes = {name: actors.build_cube(self.scene, half_size=CUBE_HALF, color=CUBE_COLORS[name], name=name,
                                              initial_pose=sapien.Pose(p=[*xy, CUBE_HALF]))
                      for name, xy in CUBES.items()}
        self.movable_objects = dict(self.cubes)
        self.zones = {name: actors.build_box(self.scene, half_sizes=[ZONE_HALF, ZONE_HALF, 0.0005],
                                             color=ZONE_COLORS[name], name=name, body_type="kinematic",
                                             add_collision=False, initial_pose=sapien.Pose(p=[*xy, 0.001]))
                      for name, xy in ZONES.items()}

    def _initialize_episode(self, env_idx, options):
        self._initialize_robot()
        for name, cube in self.cubes.items():
            q = (randomization.random_quaternions(1, lock_x=True, lock_y=True) if name == "red_cube"
                 else torch.tensor([[1.0, 0, 0, 0]]))
            cube.set_pose(Pose.create_from_pq(p=torch.tensor([[*CUBES[name], CUBE_HALF]]), q=q))
            cube.set_linear_velocity(torch.zeros((1, 3)))
            cube.set_angular_velocity(torch.zeros((1, 3)))

    def layout(self):
        """Static scene description (world frame, metres)."""
        return dict(scene="tabletop", frame="map", floor_z=-HEIGHT, table_top_z=0.0,
                    table=dict(center=list(TABLE_CENTER), half_extents=list(TABLE_HALF)),
                    table_legs=[list(p) for p in TABLE_LEGS], table_leg_half=TABLE_LEG_HALF,
                    zones={name: dict(center=list(xy), half_extent=ZONE_HALF) for name, xy in ZONES.items()},
                    cube_half=CUBE_HALF, start=list(self.start), robot=self.robot_uids)
