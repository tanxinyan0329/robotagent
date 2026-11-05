"""What the skills need from a simulation, implemented on the ManiSkill environment."""
import numpy as np
import torch
import trimesh

from mani_skill.utils.geometry.trimesh_utils import get_component_meshes

from ..geometry import MAX_JOINT_DELTA
from .pose import Pose

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def to_pose(pose):
    return Pose(pose.p, pose.q)


class ManiSkillWorld:
    def __init__(self, env, step_fn=None):
        self.env = env
        self.agent = env.agent
        self.robot = env.agent.robot
        self.step_fn = step_fn or env.step
        self.control_timestep = float(env.control_timestep)
        names = [j.name for j in self.robot.get_active_joints()]
        self.arm_ids = [names.index(n) for n in ARM_JOINTS]
        self.finger_ids = [names.index(n) for n in self.agent.gripper_joint_names]

    def link0_pose(self):
        return to_pose(self.robot.links_map["panda_link0"].pose.sp)

    def tcp_pose(self):
        return to_pose(self.agent.tcp.pose.sp)

    def qpos(self):
        """7 arm joints + 2 fingers."""
        q = self.robot.get_qpos()[0].cpu().numpy()
        return np.concatenate([q[self.arm_ids], q[self.finger_ids]]).astype(np.float64)

    def step(self, delta, gripper):
        action = self.agent.controller.from_action_dict(dict(
            arm=torch.tensor(delta[None] / MAX_JOINT_DELTA, dtype=torch.float32),
            gripper=torch.tensor([[gripper]], dtype=torch.float32),
            base=torch.tensor([self.agent.base_action(0.0, 0.0)], dtype=torch.float32)))
        self.step_fn(action)

    def surface_points(self, exclude, origin, radius, density, tick):
        """World points sampled from the scene's collision surfaces within `radius` of `origin`."""
        robot_bodies = {link._objs[0].entity for link in self.robot.links}
        bodies = [(actor.name, actor._objs[0]) for actor in self.env.scene.actors.values()]
        for art in self.env.scene.articulations.values():
            if art is not self.robot:
                bodies += [(link.name, link._objs[0].entity) for link in art.links]
        points = []
        for name, entity in bodies:
            tick()
            if name in exclude or entity in robot_bodies:
                continue
            for component in entity.components:
                if not hasattr(component, "get_collision_shapes"):
                    continue
                for mesh in get_component_meshes(component):
                    mesh.apply_transform(entity.pose.to_transformation_matrix())
                    lo, hi = mesh.bounds
                    if np.linalg.norm(np.clip(origin, lo, hi) - origin) > radius:
                        continue
                    count = int(min(max(mesh.area * density, 50), 200000))
                    sampled, _ = trimesh.sample.sample_surface(mesh, count)
                    points.append(sampled[np.linalg.norm(sampled - origin, axis=1) < radius])
        return np.concatenate(points) if points else np.zeros((0, 3))

    def object_points(self, name):
        meshes = self.env.movable_objects[name].get_collision_meshes(to_world_frame=True)
        return np.asarray((meshes[0] if isinstance(meshes, list) else meshes).vertices)

    def object_position(self, name):
        return self.env.movable_objects[name].pose.p[0].cpu().numpy()

    def object_positions(self):
        return {name: self.object_position(name) for name in self.env.movable_objects}

    def is_grasping(self, name):
        return bool(self.agent.is_grasping(self.env.movable_objects[name])[0])

    def link_grasped(self, articulation, joint):
        link = self.env.joint(articulation, joint).child_link
        scene = self.env.scene
        forces = [scene.get_pairwise_contact_forces(self.agent.finger1_link, link),
                  scene.get_pairwise_contact_forces(self.agent.finger2_link, link)]
        return all(float(torch.linalg.norm(f, dim=-1)[0]) > 0.5 for f in forces)

    def surface(self, x, y, exclude=None, top=2.0):
        """Height of the first surface under (x, y), ignoring the robot and object `exclude`."""
        skip = {link._objs[0].entity for link in self.robot.links}
        if exclude is not None:
            skip.add(self.env.movable_objects[exclude]._objs[0])
        z = top
        for _ in range(400):
            hit = self.env.scene.px.raycast(np.array([x, y, z], np.float32), np.array([0, 0, -1], np.float32), z + 0.1)
            if hit is None:
                return None
            if hit.component.entity not in skip:
                return float(hit.position[2])
            z = min(float(hit.position[2]) - 0.002, z - 0.01)
            if z < 0:
                return None
        return None

    def joint_info(self, articulation, joint):
        raw = self.env.joint(articulation, joint)._objs[0]
        limits = raw.get_limits()[0]
        frame = raw.get_global_pose()
        return dict(kind="prismatic" if raw.type == "prismatic" else "revolute", limits=(float(limits[0]), float(limits[1])),
                    position=self.joint_position(articulation, joint), anchor=np.asarray(frame.p, dtype=np.float64),
                    axis=frame.to_transformation_matrix()[:3, 0].astype(np.float64))

    def joint_position(self, articulation, joint):
        art = self.env.articulations[articulation]
        return float(art.qpos[0, [j.name for j in art.get_active_joints()].index(joint)])

    def moving_link(self, articulation, joint):
        """Name of the link `joint` moves, as surface_points() takes it in `exclude`."""
        return self.env.joint(articulation, joint).child_link.name

    def handle(self, articulation, joint):
        pose, handle = self.env.handle_pose(articulation, joint)
        return to_pose(pose), handle

    def panel_box(self, articulation, joint):
        """(4x4 world transform, half sizes) of the largest collision box on the link `joint` moves."""
        link = self.env.joint(articulation, joint).child_link
        boxes = [shape for shape in link._objs[0].get_collision_shapes() if hasattr(shape, "half_size")]
        box = max(boxes, key=lambda b: np.prod(b.half_size))
        return (link.pose.sp * box.get_local_pose()).to_transformation_matrix().astype(np.float64), np.asarray(box.half_size, float)

    def layout(self):
        return self.env.layout()
