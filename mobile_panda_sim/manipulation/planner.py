"""Arm motion planning for the mobile Panda: IK, collision-free RRT and straight-line (screw) motion."""
import os
from pathlib import Path
import time

import mplib
import numpy as np
import sapien
import trimesh

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.utils.geometry.trimesh_utils import get_component_meshes

PANDA_URDF = Path(PACKAGE_ASSET_DIR) / "robots/panda/panda_v2.urdf"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
CLOUD_RADIUS = 1.25
CLOUD_DENSITY = 8000
IDLE_PERIOD = 1 / 30
CLOUD_POINT_RADIUS = 0.015


def to_pose(p, q=(1, 0, 0, 0)):
    return sapien.Pose(np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64))


def _slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)


def pose_from_axes(position, approach, closing):
    """TCP pose from the gripper approach direction (TCP +z) and finger closing direction (TCP +y)."""
    z = np.asarray(approach, float); z /= np.linalg.norm(z)
    y = np.asarray(closing, float); y = y - z * (y @ z); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, position
    return sapien.Pose(m)


def planner_description():
    """Panda URDF/SRDF for mplib with the mobile base (pedestal + chassis boxes) fixed under panda_link0,
    so arm-vs-own-base collisions are self-collisions. Written to outputs/planner/."""
    from ..geometry import HEIGHT
    from ..paths import OUTPUTS
    pillar = HEIGHT - 0.27
    out = OUTPUTS / "planner"
    out.mkdir(parents=True, exist_ok=True)
    meshes = out / "franka_description"
    if not meshes.exists():
        try:
            meshes.symlink_to(PANDA_URDF.parent / "franka_description", target_is_directory=True)
        except FileExistsError:
            pass
    urdf = PANDA_URDF.read_text()
    pad = 0.01
    base = f"""
  <link name="mobile_base">
    <collision><origin xyz="0 0 {-pillar/2:.4f}"/><geometry><box size="{0.18+2*pad} {0.18+2*pad} {pillar:.4f}"/></geometry></collision>
    <collision><origin xyz="0 0 {-(pillar + 0.07):.4f}"/><geometry><box size="{0.55+2*pad} {0.46+2*pad} 0.14"/></geometry></collision>
  </link>
  <joint name="mobile_base_mount" type="fixed"><parent link="panda_link0"/><child link="mobile_base"/><origin xyz="0 0 0"/></joint>
</robot>"""
    urdf = urdf[:urdf.rindex("</robot>")] + base
    srdf = PANDA_URDF.with_suffix(".srdf").read_text()
    srdf = srdf[:srdf.rindex("</robot>")] + \
        '    <disable_collisions link1="panda_link0" link2="mobile_base" reason="Adjacent"/>\n</robot>'
    for name, text in (("mobile_panda_arm.urdf", urdf), ("mobile_panda_arm.srdf", srdf)):
        path = out / name
        if path.exists() and path.read_text() == text:
            continue
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    return out / "mobile_panda_arm.urdf"


class ArmPlanner:
    def __init__(self, env, joint_vel_limit=0.8, joint_acc_limit=1.5):
        self.env = env
        self.agent = env.agent
        self.robot = env.agent.robot
        self.dt = float(env.control_timestep)
        urdf = planner_description()
        self.planner = mplib.Planner(
            urdf=str(urdf), srdf=str(urdf.with_suffix(".srdf")), move_group="panda_hand_tcp",
            joint_vel_limits=np.full(7, joint_vel_limit), joint_acc_limits=np.full(7, joint_acc_limit))
        self.planner.set_base_pose(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64))
        names = [j.name for j in self.robot.get_active_joints()]
        self.arm_ids = [names.index(n) for n in ARM_JOINTS]
        self.finger_ids = [names.index(n) for n in self.agent.gripper_joint_names]
        self.cloud_ready = False
        self.idle = None
        self._last_idle = 0.0

    def tick(self):
        """Give the caller a chance to run (e.g. keep a viewer window responsive) while planning."""
        if self.idle is not None:
            now = time.monotonic()
            if now - self._last_idle >= IDLE_PERIOD:
                self._last_idle = now
                self.idle()

    def base_pose(self):
        """World pose of panda_link0 (the planner's origin); `virtual_base` overrides it to evaluate
        reachability from a base pose the robot is not at yet."""
        if getattr(self, "virtual_base", None) is not None:
            return self.virtual_base
        return self.robot.links_map["panda_link0"].pose.sp

    def link0_at(self, x, y, yaw):
        """panda_link0 world pose if the chassis were parked at (x, y, yaw)."""
        live = self.robot.links_map["panda_link0"].pose.sp
        return sapien.Pose([x, y, live.p[2]], [np.cos(yaw/2), 0, 0, np.sin(yaw/2)])

    def to_base(self, world_pose):
        return self.base_pose().inv() * world_pose

    def tcp_pose(self):
        return self.agent.tcp.pose.sp

    def qpos(self):
        """Planner qpos: 7 arm joints + 2 fingers (mplib order)."""
        q = self.robot.get_qpos()[0].cpu().numpy()
        return np.concatenate([q[self.arm_ids], q[self.finger_ids]]).astype(np.float64)

    @staticmethod
    def _vec(pose):
        return np.concatenate([pose.p, pose.q]).astype(np.float64)

    def update_obstacles(self, exclude=()):
        """Build the obstacle cloud (see build_cloud) and make it the active one. Returns its size."""
        self.use_cloud(self.build_cloud(exclude))
        return len(self._cloud)

    def use_cloud(self, cloud):
        """Make a cloud from build_cloud() the active obstacle set."""
        self._cloud = cloud if len(cloud) else np.array([[50.0, 50.0, 50.0]])
        self.cloud_ready = True
        self._use_cloud(True)

    def build_cloud(self, exclude=(), clear=None):
        """Sample the scene's collision geometry near the arm into the planner's point cloud."""
        base = self.base_pose()
        origin = base.p
        inv = base.inv().to_transformation_matrix()
        robot_bodies = {link._objs[0].entity for link in self.robot.links}
        points = []
        bodies = []
        for actor in self.env.scene.actors.values():
            bodies.append((actor.name, actor._objs[0]))
        for art in self.env.scene.articulations.values():
            if art is self.robot:
                continue
            for link in art.links:
                bodies.append((link.name, link._objs[0].entity))
        for name, entity in bodies:
            self.tick()
            if name in exclude or entity in robot_bodies:
                continue
            for component in entity.components:
                if not hasattr(component, "get_collision_shapes"):
                    continue
                for mesh in get_component_meshes(component):
                    mesh.apply_transform(entity.pose.to_transformation_matrix())
                    lo, hi = mesh.bounds
                    nearest = np.clip(origin, lo, hi)
                    if np.linalg.norm(nearest - origin) > CLOUD_RADIUS:
                        continue
                    n = int(min(max(mesh.area * CLOUD_DENSITY, 50), 200000))
                    pts, _ = trimesh.sample.sample_surface(mesh, n)
                    pts = pts[np.linalg.norm(pts - origin, axis=1) < CLOUD_RADIUS]
                    points.append(pts)
        cloud = np.concatenate(points) if points else np.zeros((0, 3))
        if clear is not None:
            cloud = cloud[np.linalg.norm(cloud - np.asarray(clear[0]), axis=1) > clear[1]]
        cloud = cloud @ inv[:3, :3].T + inv[:3, 3]
        return cloud.astype(np.float64)

    def attach_box(self, size, tcp_to_box):
        """Treat a held object as a box (full size, pose relative to the TCP) during planning."""
        self.planner.update_attached_box(np.asarray(size, dtype=np.float64), self._vec(tcp_to_box))
        self.attached = True

    def detach(self):
        self.attached = False

    def _use_cloud(self, enabled):
        """mplib's IK rejects solutions touching the obstacle cloud. Contact poses (fingers around a
        handle, on a table) must ignore it, so swap in a single far-away point when disabled."""
        cloud = self._cloud if enabled and self.cloud_ready else np.array([[50.0, 50.0, 50.0]])
        self.planner.update_point_cloud(cloud, radius=CLOUD_POINT_RADIUS)

    def ik(self, world_pose, seed=None, n_init=30, avoid=False):
        """Arm joint solutions (7,) for a TCP world pose, closest to `seed` (default: current joints)
        first; [] if unreachable. Self-collision is always checked; obstacles only when `avoid`."""
        self.tick()
        start = self.qpos() if seed is None else np.concatenate([seed, self.qpos()[7:]])
        self._use_cloud(avoid)
        try:
            status, solutions = self.planner.IK(self._vec(self.to_base(world_pose)), start, n_init_qpos=n_init)
        finally:
            self._use_cloud(True)
        if status != "Success" or solutions is None:
            return []
        solutions = [np.asarray(s)[:7] for s in solutions]
        return sorted(solutions, key=lambda s: np.abs(s - start[:7]).sum())

    def in_collision(self, q7, avoid=True):
        """True if arm joints q7 collide with itself or (when avoid) the obstacle cloud."""
        q = np.concatenate([q7, self.qpos()[7:]])
        if self.planner.check_for_self_collision(self.planner.robot, q):
            return True
        return bool(avoid and self.cloud_ready and self.planner.check_for_env_collision(
            self.planner.robot, q, with_point_cloud=True, use_attach=getattr(self, "attached", False)))

    def plan_to_pose(self, world_pose, avoid=True, planning_time=2.0):
        """Collision-free joint trajectory (N, 7) to a TCP world pose, or None. IK goals are computed and
        collision-filtered here (closest to the current joints first), then RRTConnect plans to them."""
        goals = [q for q in self.ik(world_pose, n_init=40, avoid=avoid) if not self.in_collision(q, avoid)]
        if not goals:
            return None
        return self.plan_joints(goals[:6], avoid, planning_time)

    def plan_joints(self, goals, avoid=True, planning_time=2.0):
        """Collision-free joint trajectory to any of the arm joint goals (list of (7,)), or None."""
        fingers = self.qpos()[7:]
        current = self.qpos()[:7]
        for g in goals:
            if np.abs(np.asarray(g) - current).max() < 1e-3:
                return np.asarray([current])
        goals = [np.concatenate([g, fingers]) for g in goals]
        for _ in range(3):
            try:
                result = self.planner.plan_qpos_to_qpos(goals, self.qpos(), time_step=self.dt, planning_time=planning_time,
                                                        use_point_cloud=avoid and self.cloud_ready,
                                                        use_attach=getattr(self, "attached", False) and avoid)
            except RuntimeError:
                continue
            if result["status"] == "Success":
                return np.asarray(result["position"])[:, :7]
        return None

    def fk(self, q7):
        """TCP world pose for arm joints q7 (fingers at their current opening)."""
        model = self.planner.robot.get_pinocchio_model()
        model.compute_forward_kinematics(np.concatenate([q7, self.qpos()[7:]]))
        pose = model.get_link_pose(self.planner.move_group_link_id)
        return self.base_pose() * sapien.Pose(pose[:3], pose[3:])

    def link_positions(self, q7, links=("panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7")):
        """World positions of arm link origins for joints q7 (for cheap clearance checks)."""
        model = self.planner.robot.get_pinocchio_model()
        model.compute_forward_kinematics(np.concatenate([q7, self.qpos()[7:]]))
        base = self.base_pose().to_transformation_matrix()
        out = []
        for name in links:
            p = model.get_link_pose(self.planner.link_name_2_idx[name])[:3]
            out.append(base[:3, :3] @ p + base[:3, 3])
        return np.asarray(out)

    def plan_home(self, q7, avoid=True, planning_time=2.0):
        """Collision-free joint trajectory to arm joints q7, or None."""
        return self.plan_joints([np.asarray(q7, float)], avoid, planning_time)

    def plan_line(self, world_pose, step=0.01, max_jump=0.25):
        """Straight-line TCP motion to a world pose (position interpolated, orientation slerped), as a
        timed joint trajectory, or None if any waypoint is unreachable. IK is chained from the current
        joints so the arm does not flip configuration. No obstacle check: used for short approaches
        where the gripper is meant to touch things."""
        start = self.tcp_pose()
        distance = np.linalg.norm(world_pose.p - start.p)
        angle = 2 * np.arccos(min(1.0, abs(float(np.dot(start.q, world_pose.q)))))
        n = max(2, int(max(distance / step, angle / 0.05)) + 1)
        seed, path = self.qpos()[:7], [self.qpos()[:7]]
        for t in np.linspace(0, 1, n)[1:]:
            pose = sapien.Pose(start.p + (world_pose.p - start.p) * t, _slerp(start.q, world_pose.q, t))
            solutions = self.ik(pose, seed=seed, n_init=6)
            if not solutions or np.abs(solutions[0] - seed).max() > max_jump:
                return None
            seed = solutions[0]
            path.append(seed)
        return self.time_path(np.asarray(path))

    def time_path(self, waypoints):
        """Time-parameterize joint waypoints (M, 7) into a trajectory sampled at the control rate."""
        path = np.asarray(waypoints, dtype=np.float64)
        if len(path) < 2:
            return path
        times, positions, *_ = self.planner.TOPP(path, self.dt)
        return np.asarray(positions)[:, :7]
