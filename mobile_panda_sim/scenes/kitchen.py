"""Home scene: open-plan kitchen + dining + living room (about 8 m x 6 m), floor at z=0."""
import math

import numpy as np
import sapien

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env

from .assets import ArticulationMirror, HandleSpec, drawer_handles, place_articulated, place_rcad, place_ycb, set_mass
from .base import MobilePandaSceneBase

ROOM = (-4.0, 4.0, -3.0, 3.0)
WALL_HEIGHT, WALL_THICK = 2.6, 0.1
START = (-2.55, -1.95, math.pi / 2)
FACE = {"south": 0.0, "north": math.pi, "east": math.pi / 2, "west": -math.pi / 2}
FACE_ART = {"east": 0.0, "north": math.pi / 2, "west": math.pi, "south": -math.pi / 2}
PANTRY_SHELVES = (0.159, 0.505, 0.797, 1.099, 1.374, 1.664)
# Round collision bars in fridge.urdf link frames; the top one replaces the URDF's square handle box.
FRIDGE_HANDLES = {
    "top_door": HandleSpec(centre=(0.085, -0.665, 0.223), axis=(0, 0, 1), outward=(1, 0, 0), length=0.59, radius=0.02,
                           visual=False, replace_box=True, panel_gap=0.0235),
    "bottom_door": HandleSpec(centre=(0.085, -0.665, -0.17), axis=(0, 0, 1), outward=(1, 0, 0), length=0.35, radius=0.02,
                              visual=False, panel_gap=0.0235),
}
# Real masses (kg): YCB published values, real fruit, typical household items.
MIN_MASS = 0.02
MASSES = {
    "mustard_bottle": 0.603, "cleanser": 1.131, "mug_counter": 0.118, "mug_table": 0.118,
    "plate_dirty_0": 0.279, "plate_dirty_1": 0.279, "fork": 0.034, "spoon": 0.030, "cereal_bowl": 0.147,
    "tuna_can": 0.171, "meat_can": 0.370, "pudding_box": 0.187, "gelatin_box": 0.097, "pitcher": 0.178,
    "paper_cups": 0.013, "foam_brick": 0.028, "sponges": 0.006, "cracker_box": 0.411, "sugar_box": 0.514,
    "soup_can": 0.349, "coffee_can": 0.414,
    "banana": 0.15, "apple": 0.20, "orange": 0.18, "lemon": 0.10, "pear": 0.18,
    "chopping_board": 0.9, "knife_block": 1.5, "sponge_dish": 0.15, "frying_pan": 1.0, "coffee_machine": 4.0,
    "fruit_bowl": 0.8, "plate_clean_0": 0.45, "plate_clean_1": 0.45, "bowl_clean": 0.5, "vase": 0.6,
    "grocery_basket": 0.8, "remote_control": 0.12, "book": 0.5, "shoe_left": 0.4, "shoe_right": 0.4, "umbrella": 0.5,
}


@register_env("MobilePandaKitchen-v1", max_episode_steps=None)
class MobilePandaKitchen(MobilePandaSceneBase):
    FLOOR_Z = 0.0
    DEFAULT_START = START

    @property
    def _default_human_render_camera_configs(self):
        return [
            CameraConfig("overview", sapien_utils.look_at([0.4, -5.2, 9.0], [0.0, 0.2, 0.0]), 1280, 800, 0.85, 0.01, 100),
            CameraConfig("top", sapien_utils.look_at([0.0, -0.001, 11.0], [0.0, 0.0, 0.0]), 1100, 840, 0.62, 0.01, 100),
            CameraConfig("kitchen", sapien_utils.look_at([-0.4, -0.2, 2.1], [-1.6, 2.6, 0.9]), 960, 640, 1.1, 0.01, 100),
            CameraConfig("entry", sapien_utils.look_at([-1.2, -2.4, 1.9], [-3.6, -0.6, 0.8]), 960, 640, 1.1, 0.01, 100),
            CameraConfig("dining", sapien_utils.look_at([0.3, -1.0, 1.7], [1.5, 0.4, 0.75]), 960, 640, 1.0, 0.01, 100),
        ]

    def _load_lighting(self, options):
        self.scene.set_ambient_light([0.35, 0.35, 0.35])
        self.scene.add_directional_light([0.3, 0.4, -1.0], [1.4, 1.35, 1.25], shadow=True, shadow_scale=8, shadow_map_size=4096)
        for x, y in ((-1.6, 1.6), (1.4, 0.3), (2.4, -1.8), (-2.8, -1.4)):
            self.scene.add_point_light([x, y, 2.4], color=[1.6, 1.4, 1.1])

    def _room(self):
        x0, x1, y0, y1 = ROOM
        floor = self.scene.create_actor_builder()
        floor.add_box_collision(half_size=[(x1-x0)/2 + 1, (y1-y0)/2 + 1, 0.05])
        floor.add_box_visual(half_size=[(x1-x0)/2 + WALL_THICK, (y1-y0)/2 + WALL_THICK, 0.05],
                             material=sapien.render.RenderMaterial(base_color=[0.62, 0.47, 0.33, 1], roughness=0.7))
        floor.initial_pose = sapien.Pose([(x0+x1)/2, (y0+y1)/2, -0.05])
        self.floor = floor.build_static(name="floor")
        wall_color = [0.90, 0.88, 0.84, 1]
        t, h = WALL_THICK, WALL_HEIGHT
        walls = {"wall_north": ((x0+x1)/2, y1 + t/2, (x1-x0)/2 + t, t/2),
                 "wall_south": ((x0+x1)/2, y0 - t/2, (x1-x0)/2 + t, t/2),
                 "wall_west": (x0 - t/2, (y0+y1)/2, t/2, (y1-y0)/2),
                 "wall_east": (x1 + t/2, (y0+y1)/2, t/2, (y1-y0)/2)}
        self.walls = [actors.build_box(self.scene, half_sizes=[hx, hy, h/2], color=wall_color, name=name,
                                       body_type="static", initial_pose=sapien.Pose([cx, cy, h/2]))
                      for name, (cx, cy, hx, hy) in walls.items()]
        actors.build_box(self.scene, half_sizes=[0.48, 0.02, 1.02], color=[0.42, 0.28, 0.17, 1], name="entrance_door",
                         body_type="static", initial_pose=sapien.Pose([-2.9, y0 + 0.02, 1.02]))
        actors.build_box(self.scene, half_sizes=[0.42, 0.42, 0.003], color=[0.15, 0.35, 0.75, 1], name="charging_pad",
                         body_type="static", add_collision=False, initial_pose=sapien.Pose([START[0], START[1], 0.003]))

    def _load_scene(self, options):
        self._room()
        s, x0, x1, y0, y1 = self.scene, *ROOM
        self.furniture, self.props = [], []
        mirror = ArticulationMirror()

        def furniture(name, x, y, face, support=0.0, label=None):
            self.furniture.append(place_rcad(s, name, x, y, FACE[face], support, label=label))
            return self.furniture[-1]

        def surface(x, y, below=2.5):
            """Height of the first surface under (x, y), casting down from z=below."""
            hit = s.px.raycast(np.array([x, y, below], dtype=np.float32), np.array([0, 0, -1], dtype=np.float32), below + 0.1)
            heights = [h for h in (None if hit is None else float(hit.position[2]), mirror.raycast_down(x, y, below)) if h is not None]
            if not heights:
                raise RuntimeError(f"no support surface under ({x}, {y})")
            return max(heights)

        def prop(name, x, y, yaw=0.0, support=None, label=None, ycb=False, below=2.5):
            support = surface(x, y, below) if support is None else support
            placed = (place_ycb(s, name, x, y, yaw, support, label=label) if ycb
                      else place_rcad(s, name, x, y, yaw, support, dynamic=True, label=label))
            if placed.name in MASSES:
                set_mass(placed.entity, max(MASSES[placed.name], MIN_MASS))
            self.props.append(placed)
            self.movable_objects[placed.name] = placed.entity
            return placed

        counter = place_articulated(s, "kitchen_counter", -1.65, y1 - 0.51, FACE_ART["south"], scale=0.98,
                                    label="kitchen_counter", mirror=mirror, handles=drawer_handles("kitchen_counter"))
        fridge = place_articulated(s, "fridge", 0.30, y1 - 0.34, FACE_ART["south"], label="fridge", mirror=mirror,
                                   handles=FRIDGE_HANDLES)
        self.articulated = [counter, fridge]
        furniture("bin_02", -3.55, y1 - 0.32, "south", label="trash_bin")
        furniture("bin_01", 0.95, y1 - 0.25, "south", label="recycling_bin")
        prop("choppingboard_02", -2.75, y1 - 0.40, 0.0, label="chopping_board")
        prop("knifeblock", -3.05, y1 - 0.20, 0.0, label="knife_block")
        prop("011_banana", -2.70, y1 - 0.42, 0.4, label="banana", ycb=True)
        prop("sponge_dish", -1.25, y1 - 0.16, 0.0, label="sponge_dish")
        prop("pan_01", -2.05, y1 - 0.40, 0.3, label="frying_pan")
        prop("small_appliance_01", -0.48, y1 - 0.22, 0.0, label="coffee_machine")
        prop("025_mug", -0.80, y1 - 0.45, 1.2, label="mug_counter", ycb=True)
        prop("006_mustard_bottle", -0.95, y1 - 0.22, 0.0, label="mustard_bottle", ycb=True)
        prop("021_bleach_cleanser", -1.05, y1 - 0.18, 0.5, label="cleanser", ycb=True)

        furniture("table_01", 1.35, 0.25, "south", label="dining_table")
        for i, (cx, cy, face) in enumerate(((1.0, 0.25 + 0.68, "south"), (1.7, 0.25 + 0.68, "south"), (2.32, 0.25, "west"))):
            furniture("chair_04", cx, cy, face, label=f"chair_{i}")
        prop("029_plate", 1.00, 0.45, 0.0, label="plate_dirty_0", ycb=True)
        prop("029_plate", 1.70, 0.45, 0.0, label="plate_dirty_1", ycb=True)
        prop("030_fork", 1.00, 0.18, 1.57, label="fork", ycb=True)
        prop("031_spoon", 1.72, 0.18, 1.57, label="spoon", ycb=True)
        prop("025_mug", 2.00, 0.40, 0.3, label="mug_table", ycb=True)
        prop("bowl_01", 1.35, 0.05, 0.0, label="fruit_bowl")
        prop("013_apple", 1.31, 0.08, 0.0, label="apple", ycb=True)
        prop("017_orange", 1.39, 0.02, 0.0, label="orange", ycb=True)
        prop("024_bowl", 0.80, 0.02, 0.0, label="cereal_bowl", ycb=True)

        sideboard = place_articulated(s, "cabinet", x1 - 0.24, 1.55, FACE["west"], label="sideboard", mirror=mirror)
        self.articulated.append(sideboard)
        prop("plate_01", x1 - 0.24, 1.15, 0.0, label="plate_clean_0")
        prop("plate_01", x1 - 0.24, 1.15, 0.0, label="plate_clean_1")
        prop("bowl_02", x1 - 0.24, 1.55, 0.0, label="bowl_clean")
        prop("vase_01", x1 - 0.24, 2.05, 0.0, label="vase")

        furniture("wall_cabinet_02", x0 + 0.20, 0.75, "east", label="pantry_shelf")
        px_ = x0 + 0.20
        for name, y, board, label in (("007_tuna_fish_can", 0.45, 2, "tuna_can"), ("010_potted_meat_can", 0.62, 2, "meat_can"),
                                      ("008_pudding_box", 0.85, 2, "pudding_box"), ("009_gelatin_box", 1.05, 2, "gelatin_box"),
                                      ("019_pitcher_base", 0.55, 3, "pitcher"), ("065-a_cups", 0.95, 3, "paper_cups"),
                                      ("061_foam_brick", 0.60, 1, "foam_brick"), ("026_sponge", 0.95, 1, "sponges")):
            prop(name, px_, y, 0.0, label=label, ycb=True, below=PANTRY_SHELVES[board] + 0.05)
        furniture("table_02", x0 + 0.42, -1.05, "east", label="entry_console")
        prop("basket", x0 + 0.40, -0.80, 0.0, label="grocery_basket")
        prop("003_cracker_box", x0 + 0.42, -1.35, 1.57, label="cracker_box", ycb=True)
        prop("004_sugar_box", x0 + 0.55, -1.12, 1.57, label="sugar_box", ycb=True)
        prop("005_tomato_soup_can", x0 + 0.30, -1.15, 0.0, label="soup_can", ycb=True)
        prop("002_master_chef_can", x0 + 0.30, -1.45, 0.0, label="coffee_can", ycb=True)
        prop("014_lemon", x0 + 0.58, -1.55, 0.0, label="lemon", ycb=True)
        prop("016_pear", x0 + 0.58, -1.40, 0.0, label="pear", ycb=True)

        furniture("rug_01", 2.35, -1.55, "east", label="rug")
        furniture("sofa", 2.35, y0 + 0.55, "north", label="sofa")
        furniture("table_03", 2.35, -1.25, "south", label="coffee_table")
        prop("remote-control_01", 2.15, -1.25, 0.4, label="remote_control")
        prop("book_02", 2.60, -1.20, 0.2, label="book")
        furniture("indoor_plant_01", x1 - 0.32, y0 + 0.32, "north", label="plant")
        furniture("lamp_01", 0.95, y0 + 0.30, "north", label="floor_lamp")

        prop("shoe_01", -3.55, -2.55, 0.3, label="shoe_left")
        prop("shoe_02", -3.30, -2.65, -0.2, label="shoe_right")
        prop("umbrella", -3.75, -2.20, 0.0, label="umbrella")
        for p in self.articulated:
            self.articulations[p.name] = p.entity
            for j in p.entity.get_active_joints():
                j.set_drive_properties(stiffness=0, damping=2.0, force_limit=1e6)
                j.set_friction(0.3)
        self.handles = {p.name: {j.name: p.handles[j.child_link.name] for j in p.entity.get_active_joints()
                                 if j.child_link.name in (p.handles or {})} for p in self.articulated}

    def _initialize_episode(self, env_idx, options):
        for p in self.props:
            p.entity.set_pose(p.pose)
            p.entity.set_linear_velocity(np.zeros(3))
            p.entity.set_angular_velocity(np.zeros(3))
        for p in self.articulated:
            p.entity.set_qpos(p.entity.qpos * 0)
            p.entity.set_qvel(p.entity.qvel * 0)
        self._initialize_robot()

    def layout(self):
        def entry(p):
            return dict(name=p.name, kind=p.kind, footprint=[round(v, 3) for v in p.footprint],
                        height=[round(v, 3) for v in p.height])
        return dict(scene="kitchen", frame="map", floor_z=0.0, room=dict(xmin=ROOM[0], xmax=ROOM[1], ymin=ROOM[2], ymax=ROOM[3]),
                    wall_thickness=WALL_THICK, start=list(self.start), robot=self.robot_uids,
                    furniture=[entry(p) for p in self.furniture + self.articulated],
                    objects=[entry(p) for p in self.props])
