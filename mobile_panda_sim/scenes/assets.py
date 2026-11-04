"""Place ReplicaCAD furniture/props and YCB objects by footprint: (x, y, yaw, support height)."""
from dataclasses import dataclass
from functools import lru_cache
import json
import math

import numpy as np
import sapien
import transforms3d

from mani_skill import ASSET_DIR

RCAD = ASSET_DIR / "scene_datasets/replica_cad_dataset"
YCB = ASSET_DIR / "assets/mani_skill2_ycb"
Y_UP_TO_Z_UP = sapien.Pose(q=transforms3d.quaternions.axangle2quat([1, 0, 0], np.deg2rad(90)))


def yaw_q(yaw):
    return [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]


@dataclass
class Placed:
    name: str
    kind: str
    entity: object
    footprint: tuple
    height: tuple
    pose: sapien.Pose = None
    handles: dict = None


def _rotated_bounds(vertices, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    xy = vertices[:, :2] @ np.array([[c, s], [-s, c]])
    return xy.min(0), xy.max(0), vertices[:, 2].min(), vertices[:, 2].max()


@lru_cache(maxsize=None)
def rcad_vertices(path, scale=1.0):
    """Mesh vertices in the asset's Z-up frame (after the +90 deg x rotation), scaled."""
    import trimesh
    mesh = trimesh.load(str(path), force="scene").dump(concatenate=True)
    mesh.apply_transform(Y_UP_TO_Z_UP.to_transformation_matrix())
    return np.asarray(mesh.vertices) * scale


@lru_cache(maxsize=None)
def object_config(name):
    path = RCAD / "configs/objects" / f"{name}.object_config.json"
    data = json.loads(path.read_text())
    render = (path.parent / data["render_asset"]).resolve()
    collision = (path.parent / data["collision_asset"]).resolve() if "collision_asset" in data else None
    return render, collision, bool(data.get("use_bounding_box_for_collision"))


def _footprint(vertices, x, y, yaw, z):
    lo, hi, zmin, zmax = _rotated_bounds(vertices, yaw)
    return (x + lo[0], x + hi[0], y + lo[1], y + hi[1]), (z, z + zmax - zmin)


def place_rcad(scene, name, x, y, yaw=0.0, support=0.0, dynamic=False, label=None):
    """ReplicaCAD object `frl_apartment_<name>`. Static furniture gets exact (non-convex) collision;
    dynamic props use the dataset's convex decomposition."""
    full = f"frl_apartment_{name}"
    render, collision, use_box = object_config(full)
    vertices = rcad_vertices(render)
    z = support - vertices[:, 2].min()
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(render), pose=Y_UP_TO_Z_UP)
    if not dynamic:
        builder.add_nonconvex_collision_from_file(str(render), pose=Y_UP_TO_Z_UP)
    elif use_box or collision is None:
        builder.add_convex_collision_from_file(str(render), pose=Y_UP_TO_Z_UP)
    else:
        builder.add_multiple_convex_collisions_from_file(str(collision), pose=Y_UP_TO_Z_UP)
    builder.initial_pose = sapien.Pose([x, y, z], yaw_q(yaw))
    label = label or name
    actor = builder.build(name=label) if dynamic else builder.build_static(name=label)
    footprint, height = _footprint(vertices, x, y, yaw, support)
    return Placed(label, "prop" if dynamic else "furniture", actor, footprint, height, builder.initial_pose)


class ArticulationMirror:
    """A private SAPIEN scene holding copies of the articulated furniture at their final poses."""

    def __init__(self):
        self.scene = sapien.Scene()

    def add(self, name, scale, pose):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = scale
        art = loader.load(str(RCAD / "urdf" / name / f"{name}.urdf"))
        art.set_root_pose(pose)
        art.set_qpos(np.zeros(art.dof))
        self.scene.step()

    def raycast_down(self, x, y, below):
        hit = self.scene.physx_system.raycast(np.array([x, y, below], dtype=np.float32),
                                              np.array([0, 0, -1], dtype=np.float32), below + 0.1)
        return None if hit is None else float(hit.position[2])


HANDLE_MATERIAL = sapien.render.RenderMaterial(base_color=[0.75, 0.75, 0.78, 1], metallic=0.8, roughness=0.3)


@dataclass
class Handle:
    """A round graspable bar on an articulated link. `local` is its pose in the link frame (already
    scaled): local +x along the bar, local +z the outward normal of the panel it sits on."""
    link: str
    local: sapien.Pose
    length: float
    radius: float
    standoff: float


@dataclass
class HandleSpec:
    """Bar handle in URDF link coordinates (unscaled). `standoff` > 0 adds two posts from the panel to the
    bar (leaving a gap behind it for the fingers); `replace_box` removes an existing URDF box collision
    centred at `centre` (a square handle the bar replaces)."""
    centre: tuple
    axis: tuple
    outward: tuple
    length: float
    radius: float
    standoff: float = 0.0
    visual: bool = True
    replace_box: bool = False
    panel_gap: float = None


def _frame(axis, outward):
    x = np.asarray(axis, float); x /= np.linalg.norm(x)
    z = np.asarray(outward, float); z -= x * (x @ z); z /= np.linalg.norm(z)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2] = x, np.cross(z, x), z
    return m


def _add_handle(link_builder, spec, scale):
    m = _frame(spec.axis, spec.outward)
    m[:3, 3] = np.asarray(spec.centre) * scale
    bar = sapien.Pose(m)
    radius, half = spec.radius * scale, spec.length / 2 * scale - spec.radius * scale
    if spec.replace_box:
        link_builder.collision_records = [r for r in link_builder.collision_records
                                          if not (r.type == "box" and np.allclose(r.pose.p, bar.p, atol=1e-3))]
    link_builder.add_capsule_collision(bar, radius, half, density=300)
    if spec.visual:
        link_builder.add_capsule_visual(bar, radius, half, material=HANDLE_MATERIAL)
    if spec.standoff > 0:
        post_half = (0.006 * scale, 0.006 * scale, spec.standoff / 2 * scale)
        for end in (-1, 1):
            post = bar * sapien.Pose([end * (half - 0.004 * scale), 0, -spec.standoff / 2 * scale])
            link_builder.add_box_collision(post, post_half, density=300)
            if spec.visual:
                link_builder.add_box_visual(post, post_half, material=HANDLE_MATERIAL)
    gap = spec.standoff if spec.panel_gap is None else spec.panel_gap
    return Handle(link_builder.name, bar, spec.length * scale, radius, gap * scale)


def place_articulated(scene, name, x, y, yaw=0.0, support=0.0, scale=1.0, base_rotation=None, label=None, mirror=None,
                      handles=None):
    """ReplicaCAD articulated URDF (fridge, kitchen_counter, cabinet, chestOfDrawers_01, ...), fixed base."""
    urdf = RCAD / "urdf" / name / f"{name}.urdf"
    loader = scene.create_urdf_loader()
    loader.name = label or name
    loader.fix_root_link = True
    loader.disable_self_collisions = True
    loader.scale = scale
    builder = loader.parse(str(urdf))["articulation_builders"][0]
    placed_handles = {}
    for link_builder in builder.link_builders:
        if link_builder.name in (handles or {}):
            placed_handles[link_builder.name] = _add_handle(link_builder, handles[link_builder.name], scale)
    inner = Y_UP_TO_Z_UP * sapien.Pose(q=base_rotation or [1, 0, 0, 0])
    vertices = articulated_vertices(name, scale, tuple(inner.q))
    z = support - vertices[:, 2].min()
    pose = sapien.Pose([x, y, z], yaw_q(yaw)) * inner
    builder.initial_pose = pose
    articulation = builder.build(name=label or name)
    if mirror is not None:
        mirror.add(name, scale, pose)
    footprint, height = _footprint(vertices, x, y, yaw, support)
    return Placed(label or name, "articulated", articulation, footprint, height, builder.initial_pose, placed_handles)


def drawer_handles(name, standoff=0.04, radius=0.009, length=0.12):
    """Bar handles for every prismatic drawer of ReplicaCAD articulation `name`, centred on the drawer
    front. The front panel is the drawer's thin collision box furthest along the slide axis (URDF +x);
    the bar runs along the drawer's width (URDF z), `standoff` m in front of the panel."""
    import xml.etree.ElementTree as ET
    root = ET.parse(RCAD / "urdf" / name / f"{name}.urdf").getroot()
    links = {link.get("name"): link for link in root.iter("link")}
    result = {}
    for joint in root.iter("joint"):
        if joint.get("type") != "prismatic" or [float(v) for v in joint.find("axis").get("xyz").split()] != [1, 0, 0]:
            continue
        child = joint.find("child").get("link")
        boxes = []
        for col in links[child].iter("collision"):
            box = col.find("geometry/box")
            if box is not None:
                xyz = [float(v) for v in col.find("origin").get("xyz").split()]
                boxes.append((xyz, [float(v) for v in box.get("size").split()]))
        panels = [b for b in boxes if b[1][0] < 0.1]
        (fx, fy, fz), (sx, sy, sz) = max(panels, key=lambda b: b[0][0])
        result[child] = HandleSpec(centre=(fx + sx/2 + standoff, fy, fz), axis=(0, 0, 1), outward=(1, 0, 0),
                                   length=min(length, sz - 0.06), radius=radius, standoff=standoff)
    return result


@lru_cache(maxsize=None)
def articulated_vertices(name, scale, inner_q):
    """Collision vertices of a closed articulation, in a frame rotated by inner_q (no yaw/translation)."""
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    scene = sapien.Scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = scale
    art = loader.load(str(RCAD / "urdf" / name / f"{name}.urdf"))
    art.set_root_pose(sapien.Pose(q=list(inner_q)))
    points = []
    for link in art.get_links():
        for mesh in get_component_meshes(link):
            mesh.apply_transform(link.get_pose().to_transformation_matrix())
            points.append(np.asarray(mesh.vertices))
    return np.concatenate(points)


@lru_cache(maxsize=None)
def ycb_info(model_id):
    data = json.loads((YCB / "info_pick_v0.json").read_text())[model_id]
    scale = data.get("scales", [1.0])[0]
    return np.array(data["bbox"]["min"]) * scale, np.array(data["bbox"]["max"]) * scale


def place_ycb(scene, model_id, x, y, yaw=0.0, support=0.0, label=None):
    from mani_skill.utils.building import actors
    lo, hi = ycb_info(model_id)
    builder = actors.get_actor_builder(scene, id=f"ycb:{model_id}")
    builder.initial_pose = sapien.Pose([x, y, support - lo[2] + 0.002], yaw_q(yaw))
    actor = builder.build(name=label or model_id)
    corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], 0], [hi[0], lo[1], 0]])
    footprint, height = _footprint(corners, x, y, yaw, support)
    return Placed(label or model_id, "ycb", actor, footprint, height, builder.initial_pose)


def set_mass(actor, kg):
    """Set a dynamic actor's mass, scaling its inertia by the same factor (shape and centre of mass kept)."""
    for body in actor._bodies:
        ratio = kg / body.mass
        body.set_mass(kg)
        body.set_inertia(body.inertia * ratio)
