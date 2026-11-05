"""Export the ManiSkill kitchen to assets/mujoco_kitchen/ for the MuJoCo backend (run once, in the mshab env).

    python scripts/export_kitchen_mujoco.py

Reads the live scene, so poses, masses, joints, handles and textures stay identical to the ManiSkill one. MuJoCo
collides convex shapes only: static triangle meshes are split into connected components; a convex one stays a
hull, a clearly non-convex closed one (table frame, chair) becomes boxes from a voxel grid.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402
import trimesh  # noqa: E402

from mobile_panda_sim.paths import MUJOCO_KITCHEN  # noqa: E402
from mobile_panda_sim.scenes import SCENES  # noqa: E402

TEXTURE_SIZE = 512
VOXEL_PITCH = 0.02
CONVEX_RATIO = 0.8
ROBOT = "mobile_panda_wheels"


def pose_dict(pose):
    return dict(pos=np.asarray(pose.p, float).reshape(-1).tolist(), quat=np.asarray(pose.q, float).reshape(-1).tolist())


def merge_boxes(filled):
    """Cover a boolean voxel grid with axis-aligned boxes: yields (first index, last index) pairs."""
    todo = filled.copy()
    for start in np.argwhere(filled):
        i, j, k = start
        if not todo[i, j, k]:
            continue
        end = start.copy()
        for axis in range(3):
            while end[axis] + 1 < todo.shape[axis]:
                grown = end.copy()
                grown[axis] += 1
                first = start.copy()
                first[axis] = grown[axis]
                if not todo[first[0]:grown[0] + 1, first[1]:grown[1] + 1, first[2]:grown[2] + 1].all():
                    break
                end = grown
        todo[i:end[0] + 1, j:end[1] + 1, k:end[2] + 1] = False
        yield start, end


def voxel_boxes(part):
    """(centre, half size) boxes approximating a closed mesh."""
    grid = part.voxelized(VOXEL_PITCH).fill()
    origin = np.asarray(grid.translation)
    for first, last in merge_boxes(grid.matrix):
        low, high = origin + (first - 0.5)*VOXEL_PITCH, origin + (last + 0.5)*VOXEL_PITCH
        yield (low + high)/2, (high - low)/2


class Exporter:
    def __init__(self, folder):
        self.folder = folder
        self.meshes, self.textures = {}, {}

    def mesh(self, **arrays):
        key = hashlib.sha1(b"".join(np.asarray(a, np.float32).tobytes() for a in arrays.values())).hexdigest()[:16]
        self.meshes.setdefault(key, arrays)
        return key

    def texture(self, texture):
        if texture is None:
            return None
        image = np.asarray(texture.download())
        key = hashlib.sha1(image.tobytes()).hexdigest()[:16]
        if key not in self.textures:
            picture = Image.fromarray(image[..., :3].astype(np.uint8))
            if max(picture.size) > TEXTURE_SIZE:
                picture.thumbnail((TEXTURE_SIZE, TEXTURE_SIZE))
            picture.save(self.folder / "textures" / f"{key}.png")
            self.textures[key] = True
        return key

    def collisions(self, component):
        result = []
        for shape in component.collision_shapes:
            kind = type(shape).__name__.replace("PhysxCollisionShape", "")
            common = dict(pose_dict(shape.local_pose), friction=float(shape.physical_material.static_friction))
            if kind == "Box":
                result.append(dict(common, type="box", half=np.asarray(shape.half_size, float).tolist()))
            elif kind == "Capsule":
                result.append(dict(common, type="capsule", radius=float(shape.radius), half_length=float(shape.half_length)))
            elif kind == "ConvexMesh":
                vertices = np.asarray(shape.vertices, float) * np.asarray(shape.scale, float)
                result.append(dict(common, type="mesh", mesh=self.mesh(v=vertices)))
            elif kind == "TriangleMesh":
                vertices = np.asarray(shape.vertices, float) * np.asarray(shape.scale, float)
                whole = trimesh.Trimesh(vertices, np.asarray(shape.triangles), process=True)
                for part in whole.split(only_watertight=False):
                    points = np.asarray(part.vertices)
                    if np.linalg.matrix_rank(points - points.mean(0), tol=1e-4) < 3:
                        continue
                    if part.is_watertight and abs(part.volume) < CONVEX_RATIO * part.convex_hull.volume:
                        for centre, half in voxel_boxes(part):
                            box = shape.local_pose * sapien.Pose(centre)
                            result.append(dict(common, type="box", half=half.tolist(), **pose_dict(box)))
                    else:
                        result.append(dict(common, type="mesh", mesh=self.mesh(v=points)))
            else:
                raise ValueError(f"unsupported collision shape {kind}")
        return result

    def visuals(self, render):
        result = []
        for shape in render.render_shapes:
            kind = type(shape).__name__.replace("RenderShape", "")
            common = pose_dict(shape.local_pose)
            if kind == "Box":
                result.append(dict(common, type="box", half=np.asarray(shape.half_size, float).tolist(),
                                   rgba=list(shape.material.base_color)))
            elif kind == "Capsule":
                result.append(dict(common, type="capsule", radius=float(shape.radius), half_length=float(shape.half_length),
                                   rgba=list(shape.material.base_color)))
            elif kind == "TriangleMesh":
                scale = np.asarray(shape.scale, float)
                for part in shape.parts:
                    material = part.material
                    mesh = self.mesh(v=part.get_vertices()*scale, f=part.get_triangles(), n=part.get_vertex_normal(),
                                     uv=part.get_vertex_uv())
                    result.append(dict(common, type="mesh", mesh=mesh, rgba=list(material.base_color),
                                       texture=self.texture(material.base_color_texture)))
            else:
                raise ValueError(f"unsupported render shape {kind}")
        return result


def components(entity):
    physics = next((c for c in entity.components if hasattr(c, "collision_shapes")), None)
    render = next((c for c in entity.components if type(c).__name__ == "RenderBodyComponent"), None)
    return physics, render


def body_dict(exporter, entity):
    physics, render = components(entity)
    item = dict(collision=exporter.collisions(physics) if physics is not None else [],
                visual=exporter.visuals(render) if render is not None else [])
    if physics is not None and hasattr(physics, "mass"):
        item.update(mass=float(physics.mass), inertia=np.asarray(physics.inertia, float).tolist(),
                    inertial=pose_dict(physics.cmass_local_pose))
    return item


def export(env, exporter):
    world = env.unwrapped
    world_scene = world.scene
    props = {p.name: p for p in world.props}
    actors = []
    for name, actor in world_scene.actors.items():
        entity = actor._objs[0]
        item = dict(body_dict(exporter, entity), name=name, movable=name in props)
        item.update(pose_dict(props[name].pose if name in props else entity.pose))
        actors.append(item)
    articulations = []
    for name, articulation in world_scene.articulations.items():
        if name == ROBOT:
            continue
        links = []
        short = lambda full, prefix=f"scene-0-{name}_": full[len(prefix):] if full.startswith(prefix) else full
        by_name = {link._objs[0].name: link._objs[0] for link in articulation.links}
        for link in by_name.values():
            parent = link.parent.name if link.parent is not None else None
            item = dict(body_dict(exporter, link.entity), name=short(link.name), parent=parent and short(parent))
            item.update(pose_dict(link.entity_pose))
            joint = link.joint
            if joint.type in ("prismatic", "revolute", "revolute_unwrapped"):
                frame = link.entity_pose.inv() * joint.global_pose
                axis = frame.to_transformation_matrix()[:3, 0]
                item["joint"] = dict(name=short(joint.name), type="slide" if joint.type == "prismatic" else "hinge", axis=axis.tolist(),
                                     pos=np.asarray(frame.p, float).tolist(), range=np.asarray(joint.limit, float).reshape(-1).tolist(),
                                     damping=float(joint.damping), friction=float(joint.friction))
            links.append(item)
        handles = {joint: dict(link=handle.link, length=handle.length, radius=handle.radius, standoff=handle.standoff, **pose_dict(handle.local))
                   for joint, handle in world.handles[name].items()}
        articulations.append(dict(name=name, links=links, handles=handles))
    return dict(actors=actors, articulations=articulations, layout=world.layout(), start=list(world.start))


def main():
    folder = MUJOCO_KITCHEN
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "textures").mkdir(parents=True)
    env = gym.make(SCENES["kitchen"], robot_uids=ROBOT, obs_mode="none", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cpu", reward_mode="none", render_mode="rgb_array")
    env.reset(seed=0)
    exporter = Exporter(folder)
    scene = export(env, exporter)
    env.close()
    (folder / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    np.savez_compressed(folder / "meshes.npz", **{f"{key}/{field}": np.asarray(values, np.int32 if field == "f" else np.float32)
                                                  for key, arrays in exporter.meshes.items() for field, values in arrays.items()})
    print(f"{len(scene['actors'])} actors, {len(scene['articulations'])} articulations, {len(exporter.meshes)} meshes, "
          f"{len(exporter.textures)} textures -> {folder}")


if __name__ == "__main__":
    main()
