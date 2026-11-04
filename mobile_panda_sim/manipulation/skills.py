"""Classical manipulation skills: IK + motion planning + scripted steps, no learning."""
from dataclasses import dataclass, field
import math

import numpy as np
import sapien
import torch

from ..geometry import ARM_REST, MAX_JOINT_DELTA
from .planner import ArmPlanner, pose_from_axes

GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0
MAX_FINGER_SPAN = 0.075
APPROACH = 0.10
LIFT = 0.12
CONTACT_JOINT_STEP = 0.015
HAND_CLEARANCE = 0.008


@dataclass
class SkillResult:
    success: bool
    message: str = ""
    info: dict = field(default_factory=dict)

    def __bool__(self):
        return self.success


class SkillError(RuntimeError):
    def __init__(self, message, info=None):
        super().__init__(message)
        self.info = info or {}


def _rotate(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle)
            + axis * (axis @ vector) * (1 - math.cos(angle)))


_PALM = ((-0.033, -0.105, -0.115), (0.033, 0.105, -0.058))
_FINGERS = {"open": ((-0.013, -0.052, -0.058), (0.013, 0.052, 0.012)),
            "closed": ((-0.013, -0.03, -0.058), (0.013, 0.03, 0.012))}


def _corners(boxes):
    return np.array([[x, y, z] for lo, hi in boxes for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


_HAND = {state: _corners((_PALM, fingers)) for state, fingers in _FINGERS.items()}


def _panel_clearance(tcp, centre, normal, gap, fingers="closed"):
    """Smallest distance from the hand to the panel plane behind a handle bar (negative: penetrating)."""
    pts = _HAND[fingers]
    corners = tcp.to_transformation_matrix() @ np.c_[pts, np.ones(len(pts))].T
    return float(((corners[:3].T - centre) @ normal).min() + gap)


def _densify(path, max_step):
    """Linearly interpolate joint waypoints so no joint moves more than max_step per control step."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, int(np.ceil(np.abs(b - a).max() / max_step)))
        out.extend(a + (b - a) * (i / k) for i in range(1, k + 1))
    return np.asarray(out)


def _rotation_pose(centre, axis, angle):
    """World transform rotating by `angle` about the line through `centre` along `axis`."""
    q = np.r_[math.cos(angle/2), np.asarray(axis) / np.linalg.norm(axis) * math.sin(angle/2)]
    centre = np.asarray(centre, float)
    return sapien.Pose(centre) * sapien.Pose(q=q) * sapien.Pose(-centre)


class ManipulationSkills:
    def __init__(self, env, step_fn=None, on_step=None, on_idle=None):
        """on_step: called after every simulation step. on_idle: called ~30 times a second while the skills
        are planning without stepping (seconds at a time), e.g. env.render to keep a viewer window
        responsive."""
        self.env = env
        self.agent = env.agent
        self.planner = ArmPlanner(env)
        self.planner.idle = on_idle
        self.step_fn = step_fn or env.step
        self.on_step = on_step
        self.gripper = GRIPPER_OPEN
        self.held = None
        self.steps = 0

    def _step(self, q_target=None):
        q = self.planner.qpos()[:7]
        delta = np.zeros(7) if q_target is None else np.clip(q_target - q, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
        action = self.agent.controller.from_action_dict(dict(
            arm=torch.tensor(delta[None] / MAX_JOINT_DELTA, dtype=torch.float32),
            gripper=torch.tensor([[self.gripper]], dtype=torch.float32),
            base=torch.tensor([self.agent.base_action(0.0, 0.0)], dtype=torch.float32)))
        self.step_fn(action)
        self.steps += 1
        if self.on_step:
            self.on_step()

    def _follow(self, trajectory, tol=0.004, settle=40):
        for q in trajectory:
            self._step(q)
        target = trajectory[-1]
        for _ in range(settle):
            if np.abs(self.planner.qpos()[:7] - target).max() < tol:
                break
            self._step(target)

    def _set_gripper(self, value, steps=12):
        self.gripper = value
        hold = self.planner.qpos()[:7]
        for _ in range(steps):
            self._step(hold)

    def _pre_pose(self, grasp, grasp_q, back_off):
        """Collision-free pre-grasp joints `back_off` m behind `grasp` along the approach, in the same IK
        branch as the grasp solution; tries shorter back-offs when the long one collides."""
        approach = grasp.to_transformation_matrix()[:3, 2]
        for distance in (APPROACH, 0.06, 0.04) if back_off is None else (back_off,):
            pre = sapien.Pose(grasp.p - approach * distance, grasp.q)
            for q in self.planner.ik(pre, seed=grasp_q, n_init=10, avoid=True):
                if np.abs(q - grasp_q).max() <= 0.6 and not self.planner.in_collision(q):
                    return pre, q
        return None, None

    def _approach(self, grasp, grasp_q, what, pre_q):
        """Collision-free move to the pre-grasp joints, then a straight line to `grasp`."""
        trajectory = self.planner.plan_joints([pre_q])
        if trajectory is None:
            raise SkillError(f"no collision-free path to the pre-{what} pose")
        self._follow(trajectory)
        line = self.planner.plan_line(grasp)
        if line is None:
            line = _densify(np.stack([self.planner.qpos()[:7], grasp_q]), 0.02)
        self._follow(_densify(line, CONTACT_JOINT_STEP), tol=0.002)
        if np.abs(self.planner.qpos()[:7] - grasp_q).max() > 0.02 and \
                np.linalg.norm(self.planner.fk(grasp_q).p - grasp.p) < 0.005:
            self._follow(_densify(np.stack([self.planner.qpos()[:7], grasp_q]), CONTACT_JOINT_STEP))

    def _back_off(self):
        """Best effort: pull the open gripper straight back along its approach axis (10, 6 or 3 cm)."""
        tcp = self.planner.tcp_pose()
        approach = tcp.to_transformation_matrix()[:3, 2]
        for distance in (APPROACH, 0.06, 0.03):
            line = self.planner.plan_line(sapien.Pose(tcp.p - approach * distance, tcp.q))
            if line is not None:
                self._follow(line)
                return True
        return False

    def _line(self, pose, what="target", slow=False):
        trajectory = self.planner.plan_line(pose)
        if trajectory is None:
            raise SkillError(f"straight-line motion to the {what} is not reachable")
        self._follow(_densify(trajectory, CONTACT_JOINT_STEP) if slow else trajectory)

    def _run(self, name, body):
        """Run a skill body; any SkillError becomes a failed result. Gripper state is kept."""
        start = self.steps
        try:
            info = body() or {}
            return SkillResult(True, f"{name} done", dict(info, steps=self.steps - start))
        except SkillError as error:
            return SkillResult(False, f"{name} failed: {error}", dict(error.info, steps=self.steps - start))

    def _object_points(self, actor):
        meshes = actor.get_collision_meshes(to_world_frame=True)
        mesh = meshes[0] if isinstance(meshes, list) else meshes
        return np.asarray(mesh.vertices)

    def _grasps_for(self, actor):
        """Top-down grasp candidates (TCP poses) for a resting object, best first."""
        pts = self._object_points(actor)
        zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
        xy = pts[:, :2]
        centre = (xy.min(0) + xy.max(0)) / 2
        cov = np.cov((xy - xy.mean(0)).T)
        _, vecs = np.linalg.eigh(cov)
        candidates = []
        for k in range(2):
            axis = np.r_[vecs[:, k], 0.0]
            proj = xy @ vecs[:, k]
            other = xy @ vecs[:, 1 - k]
            width = proj.max() - proj.min()
            if width > MAX_FINGER_SPAN:
                continue
            mid = vecs[:, k] * (proj.max() + proj.min()) / 2 + vecs[:, 1 - k] * (other.max() + other.min()) / 2
            height = zmax - zmin
            z = max(zmax - 0.04, zmin + 0.5 * height, zmin + 0.012)
            for sign in (1, -1):
                candidates.append((width, pose_from_axes(np.r_[mid, z], [0, 0, -1], sign * axis)))
        candidates.sort(key=lambda c: c[0])
        return [pose for _, pose in candidates], (zmin, zmax), centre

    def _handle_frame(self, articulation, joint):
        """Handle bar centre, bar axis and outward normal (world), and the handle record."""
        pose, handle = self.env.handle_pose(articulation, joint)
        m = pose.to_transformation_matrix()
        return pose.p, m[:3, 0], m[:3, 2], handle

    def _handle_grasps(self, articulation, joint):
        """TCP grasp poses on a round handle bar: fingers close across the bar at its centre line,
        approaching into the panel, tilted about the bar axis by 0, +/-20, 40, 60 deg; long (vertical)
        bars are also held near either end."""
        centre, bar, normal, handle = self._handle_frame(articulation, joint)
        slack = max(handle.length / 2 - 0.04, 0.0)
        upward = bar if bar[2] >= 0 else -bar
        grasps = []
        for along in (0.0, 0.5, -0.5, 0.9, -0.9):
            point = centre + upward * along * slack
            for tilt in (0, 20, -20, 40, -40, 60, -60):
                approach = _rotate(-normal, bar, math.radians(tilt))
                closing = np.cross(bar, approach)
                for sign in (1, -1):
                    pose = pose_from_axes(point, approach, sign * closing)
                    if _panel_clearance(pose, centre, normal, handle.standoff, "open") >= 0.003:
                        grasps.append(pose)
            if slack == 0:
                break
        return grasps

    def _link_grasped(self, link):
        """Both fingers press on `link`."""
        scene = self.env.scene
        forces = [scene.get_pairwise_contact_forces(self.agent.finger1_link, link),
                  scene.get_pairwise_contact_forces(self.agent.finger2_link, link)]
        return all(float(torch.linalg.norm(f, dim=-1)[0]) > 0.5 for f in forces)

    def home(self):
        """Return the arm to the rest pose along a collision-free path (gripper unchanged)."""
        def body():
            self.planner.update_obstacles(exclude={self.held} if self.held else ())
            trajectory = self.planner.plan_home(np.asarray(ARM_REST[:7]))
            if trajectory is None:
                trajectory = self.planner.plan_home(np.asarray(ARM_REST[:7]), avoid=False)
                if trajectory is None:
                    raise SkillError("rest pose unreachable")
            self._follow(trajectory)
        return self._run("home", body)

    def pick(self, name):
        """Top-down grasp of a movable object and lift it LIFT m."""
        def body():
            if self.held:
                raise SkillError(f"already holding {self.held}")
            actor = self.env.movable_objects[name]
            if not self._grasps_for(actor)[0]:
                raise SkillError(f"{name} is wider than the gripper in every direction")
            held = False
            for attempt in range(3):
                self.planner.update_obstacles()
                feasible = []
                for grasp in self._grasps_for(actor)[0]:
                    grasp_qs = self.planner.ik(grasp)
                    if grasp_qs:
                        _, pre_q = self._pre_pose(grasp, grasp_qs[0], APPROACH)
                        if pre_q is not None:
                            feasible.append((grasp, grasp_qs[0], pre_q))
                if not feasible:
                    if attempt == 0:
                        raise SkillError(f"{name} is out of the arm's reach from this base pose")
                    break
                grasp, grasp_q, pre_q = feasible[attempt % len(feasible)]
                self._set_gripper(GRIPPER_OPEN, steps=4)
                self._approach(grasp, grasp_q, "grasp", pre_q)
                self._set_gripper(GRIPPER_CLOSED, steps=15)
                if bool(self.agent.is_grasping(actor)[0]):
                    held = True
                    break
                self._set_gripper(GRIPPER_OPEN)
                self._back_off()
            if not held:
                raise SkillError(f"fingers closed but {name} is not held ({attempt + 1} grasps tried)")
            z0 = float(actor.pose.p[0, 2])
            self._line(sapien.Pose(grasp.p + [0, 0, LIFT], grasp.q), what="lift pose", slow=True)
            lifted = float(actor.pose.p[0, 2]) - z0
            if not bool(self.agent.is_grasping(actor)[0]) or lifted < 0.05:
                raise SkillError(f"{name} slipped during the lift")
            self.held = name
            tcp = self.planner.tcp_pose()
            pts = self._object_points(actor)
            lo, hi = pts.min(0), pts.max(0)
            box_centre = sapien.Pose((lo + hi) / 2)
            self.planner.attach_box(hi - lo + 0.01, tcp.inv() * box_centre)
            return dict(object=name, lifted=round(lifted, 3))
        return self._run(f"pick({name})", body)

    def place(self, target, height=None):
        """Put the held object down with its bottom centred on target (x, y), on the surface below it (or
        at `height`, the support height in m), then open the gripper and retreat upwards."""
        def body():
            if not self.held:
                raise SkillError("not holding anything")
            actor = self.env.movable_objects[self.held]
            pts = self._object_points(actor)
            bottom = np.r_[(pts[:, :2].min(0) + pts[:, :2].max(0)) / 2, pts[:, 2].min()]
            tcp = self.planner.tcp_pose()
            offset = tcp.p - bottom
            x, y = float(target[0]), float(target[1])
            support = height if height is not None else self._surface(x, y, exclude=actor)
            if support is None:
                raise SkillError(f"no surface under ({x:.2f}, {y:.2f})")
            goal = sapien.Pose(np.r_[x, y, support + 0.012] + offset, tcp.q)
            self.planner.update_obstacles(exclude={self.held})
            goal_qs = self.planner.ik(goal)
            pre_q = self._pre_pose(goal, goal_qs[0], LIFT)[1] if goal_qs else None
            if pre_q is None:
                raise SkillError("place pose is out of the arm's reach from this base pose")
            self._approach(goal, goal_qs[0], "place", pre_q)
            self._set_gripper(GRIPPER_OPEN, steps=12)
            self.planner.detach()
            name, self.held = self.held, None
            self._line(sapien.Pose(goal.p + [0, 0, LIFT], goal.q), what="retreat pose")
            for _ in range(10):
                self._step(self.planner.qpos()[:7])
            pts = self._object_points(actor)
            placed = np.r_[(pts[:, :2].min(0) + pts[:, :2].max(0)) / 2, pts[:, 2].min()]
            error = float(np.linalg.norm(placed[:2] - [x, y]))
            if error > 0.04 or abs(placed[2] - support) > 0.03:
                raise SkillError(f"{name} ended {error:.3f} m from the target (bottom z {placed[2]:.3f}, "
                                 f"surface {support:.3f})")
            return dict(object=name, position_error=round(error, 3), support=round(float(support), 3))
        return self._run("place", body)

    def _surface(self, x, y, exclude=None, top=2.0):
        """Height of the first surface under (x, y), ignoring the robot and the held object."""
        skip = {link._objs[0].entity for link in self.agent.robot.links}
        if exclude is not None:
            skip.add(exclude._objs[0])
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

    def open(self, articulation, joint, target=None):
        """Open a drawer (prismatic) or door (revolute) by its handle. `target` is the joint position to
        reach (m or rad); default: 0.25 m for drawers, 0.6 rad (35 deg) for doors, capped by the limit."""
        return self._actuate(articulation, joint, target, opening=True)

    def close(self, articulation, joint):
        """Close a drawer or door by its handle (joint back to 0)."""
        return self._actuate(articulation, joint, 0.0, opening=False)

    def _joint_model(self, articulation, joint, target, opening):
        j = self.env.joint(articulation, joint)
        raw = j._objs[0]
        kind = "prismatic" if raw.type == "prismatic" else "revolute"
        lo, hi = (float(v) for v in raw.get_limits()[0])
        art = self.env.articulations[articulation]
        index = [jj.name for jj in art.get_active_joints()].index(joint)
        q0 = float(art.qpos[0, index])
        if opening:
            goal = target if target is not None else (0.25 if kind == "prismatic" else 0.6)
        else:
            goal = 0.0
        goal = float(np.clip(goal, lo, hi))
        frame = raw.get_global_pose()
        axis = frame.to_transformation_matrix()[:3, 0]

        def moved(pose, dq):
            if kind == "prismatic":
                return sapien.Pose(pose.p + axis * dq, pose.q)
            return _rotation_pose(frame.p, axis, dq) * pose
        return dict(joint=j, kind=kind, art=art, index=index, q0=q0, goal=goal, moved=moved, names=(articulation, joint),
                    step=0.01 if kind == "prismatic" else 0.02, frame=frame, axis=axis)

    def _panel(self, articulation, joint):
        """Door panel of a revolute joint: centre, normal (thin axis), vertical and radial (hinge -> free
        edge) axes, half extents along them, from the link's largest collision box."""
        link = self.env.joint(articulation, joint).child_link
        boxes = [s_ for s_ in link._objs[0].get_collision_shapes() if hasattr(s_, "half_size")]
        box = max(boxes, key=lambda b: np.prod(b.half_size))
        m = (link.pose.sp * box.get_local_pose()).to_transformation_matrix()
        half = np.asarray(box.half_size)
        model = self._joint_model(articulation, joint, None, True)
        hinge, axis = model["frame"].p, model["axis"]
        order = np.argsort(half)
        normal_i = order[0]
        vertical_i = max((i for i in range(3) if i != normal_i), key=lambda i: abs(m[:3, i] @ axis))
        radial_i = 3 - normal_i - vertical_i
        radial = m[:3, radial_i] if m[:3, radial_i] @ (m[:3, 3] - hinge) > 0 else -m[:3, radial_i]
        hinge_along = (hinge - m[:3, 3]) @ radial
        return dict(centre=m[:3, 3], normal=m[:3, normal_i], vertical=m[:3, vertical_i], radial=radial,
                    half=(half[normal_i], half[vertical_i], half[radial_i]), hinge_along=hinge_along)

    def _drag_candidates(self, articulation, joint, model=None, offset=0.0):
        """Ways to hold a link while moving it: fingers around the handle bar, approaching at several tilts
        about the bar and (long bars) heights along it; rolls about the bar (TCP x) while dragging.
        Candidates are dicts (grasp pose, roll axis, palm clearance, expected finger gap) so other
        holds, e.g. pushing a door face shut, can be added."""
        candidates = []
        centre, bar, normal, handle = self._handle_frame(articulation, joint)
        for grasp in self._handle_grasps(articulation, joint):
            candidates.append(dict(kind="handle", grasp=grasp, roll_axis=0, clearance=(centre, normal, handle.standoff),
                                   width=2 * handle.radius))
        if offset and model is not None:
            moved = model["moved"]
            for cand in candidates:
                cand["grasp"] = moved(cand["grasp"], offset)
                if cand["clearance"] is not None:
                    c, n_, gap = cand["clearance"]
                    tip = moved(sapien.Pose(c + n_), offset).p
                    c2 = moved(sapien.Pose(c), offset).p
                    cand["clearance"] = (c2, tip - c2, gap)
        return candidates

    def _drag_path(self, model, cand, grasp, dqs, seed, max_jump=0.35, avoid=False):
        """IK chain that moves a held link through joint offsets `dqs`, starting from TCP pose `grasp`. The
        gripper may roll about the candidate's roll axis (within +/-80 deg) to stay reachable. Returns
        (joint waypoints, steps reached)."""
        axis = grasp.to_transformation_matrix()[:3, cand["roll_axis"]]
        centre = grasp.p.copy()
        roll, path = 0.0, []
        min_clear = None if cand["clearance"] is None else \
            min(HAND_CLEARANCE, _panel_clearance(grasp, *cand["clearance"]) - 0.002)
        door = self._panel(*model["names"]) if model["kind"] == "revolute" else None
        if door is not None:
            _, _, handle_normal, _ = self._handle_frame(*model["names"])
            door["out"] = door["normal"] if door["normal"] @ handle_normal > 0 else -door["normal"]
        for dq in dqs:
            best = None
            for delta in (0.0, 0.1, -0.1, 0.2, -0.2, 0.35, -0.35):
                r = roll + delta
                if abs(r) > math.radians(80):
                    continue
                rolled = _rotation_pose(centre, axis, r) * grasp
                if min_clear is not None and _panel_clearance(rolled, *cand["clearance"]) < min_clear:
                    continue
                solutions = self.planner.ik(model["moved"](rolled, dq), seed=seed, n_init=4, avoid=avoid)
                if door is not None:
                    solutions = [q for q in solutions if self._outside_door(q, door, model, dq)]
                if solutions:
                    jump = np.abs(solutions[0] - seed).max()
                    if jump <= max_jump and (best is None or jump < best[0]):
                        best = (jump, solutions[0], r)
                if best is not None and best[0] < 0.08:
                    break
            if best is None:
                break
            _, seed, roll = best
            path.append(seed)
        return path, len(path)

    def _outside_door(self, q7, door, model, dq, margin=0.06):
        """Arm links 3-6 stay >= margin on the outer side of the door (at joint offset dq), or beyond its
        edges."""
        centre = model["moved"](sapien.Pose(door["centre"]), dq).p
        rot = model["moved"](sapien.Pose(), dq).to_transformation_matrix()[:3, :3]
        out, vertical, radial = rot @ door["out"], rot @ door["vertical"], rot @ door["radial"]
        half_n, half_v, half_r = door["half"]
        for p in self.planner.link_positions(q7)[:4]:
            d = p - centre
            if abs(d @ vertical) > half_v + margin or abs(d @ radial) > half_r + margin:
                continue
            if d @ out < half_n + margin:
                return False
        return True

    def _drag_targets(self, model, q_now):
        """Joint offsets from q_now to the goal plus a small overshoot (the grasp gives a little; a closing
        link is pressed shut), one per drag step."""
        over = 0.02 if model["kind"] == "prismatic" else 0.05
        span = model["goal"] - q_now
        final = span + math.copysign(over, span)
        n = max(2, int(abs(final) / model["step"]))
        return np.linspace(0, final, n + 1)[1:], abs(span)

    def _progress(self, model, cand, grasp_q, q_now, avoid=True, coarse=False):
        """Joint distance (m or rad) towards the goal this candidate can drag. By default the same IK chain
        (step size, allowed joint jump) as the real drag, so the choice is not over-optimistic; `coarse`
        (base-pose search) checks every other step with a looser jump limit, ~3x faster."""
        dqs, span = self._drag_targets(model, q_now)
        if coarse and len(dqs) > 3:
            dqs = dqs[1::2]
        _, reached = self._drag_path(model, cand, cand["grasp"], dqs, grasp_q, max_jump=0.5 if coarse else 0.35,
                                     avoid=avoid)
        return min(abs(dqs[reached - 1]) if reached else 0.0, span)

    def _best_drag(self, model, link, q_now, full_cloud, free_cloud, need=None, offset=0.0, skip=(), coarse=False):
        """Best reachable drag candidate from the current base pose: (progress, cand, grasp_q, pre_q).
        `offset` plans for the link `offset` further along than it is now (the full cloud, which has the
        link at its current pose, is then not used)."""
        best = None
        for i, cand in enumerate(self._drag_candidates(*model["names"], model, offset)):
            if i in skip:
                continue
            cand["index"] = i
            self.planner.use_cloud(free_cloud)
            solutions = self.planner.ik(cand["grasp"], avoid=True)
            if not solutions:
                continue
            self.planner.use_cloud(full_cloud if not offset else free_cloud)
            _, pre_q = self._pre_pose(cand["grasp"], solutions[0], None)
            if pre_q is None:
                continue
            self.planner.use_cloud(free_cloud)
            progress = self._progress(model, cand, solutions[0], q_now, coarse=coarse)
            if best is None or progress > best[0] + 1e-6:
                best = (progress, cand, solutions[0], pre_q)
            if progress >= (need if need is not None else abs(model["goal"] - q_now)) - 1e-6:
                break
        return best

    def suggest_base_pose(self, articulation, joint, target=None, opening=True):
        """Search chassis poses (x, y, yaw) facing the handle from which the whole open/close motion is
        reachable. Returns ((x, y, yaw), coverage) with coverage in [0, 1], best first; does not move.
        Footprints of furniture (scene layout) are kept 5 cm clear of the chassis."""
        model = self._joint_model(articulation, joint, target, opening)
        link_name = model["joint"].child_link.name
        handle, _, normal, _ = self._handle_frame(articulation, joint)
        if model["kind"] == "revolute":
            handle = model["moved"](sapien.Pose(handle), (model["goal"] - model["q0"]) / 2).p
        out = np.r_[normal[:2], 0.0]; out /= np.linalg.norm(out)
        side = np.array([-out[1], out[0], 0.0])
        blocked = [f["footprint"] for f in self.env.layout()["furniture"]]
        best = []
        try:
            for distance in np.arange(0.40, 1.05, 0.025):
                for lateral in (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.45, 0.45, -0.6, 0.6):
                    xy = handle[:2] + out[:2] * distance + side[:2] * lateral
                    yaw = math.atan2(handle[1] - xy[1], handle[0] - xy[0])
                    if self._chassis_blocked(xy, yaw, blocked):
                        continue
                    self.planner.virtual_base = self.planner.link0_at(xy[0], xy[1], yaw)
                    if not any(self.planner.ik(c["grasp"], n_init=6) for c in
                               self._drag_candidates(articulation, joint)[:40]):
                        continue
                    full = self.planner.build_cloud()
                    free = self.planner.build_cloud(exclude={link_name})
                    coverage = self._plan_stages(model, full, free) / max(abs(model["goal"] - model["q0"]), 1e-6)
                    if coverage > 0:
                        best.append((min(coverage, 1.0), -distance, (float(xy[0]), float(xy[1]), float(yaw))))
                if best and max(b[0] for b in best) >= 0.999:
                    break
        finally:
            self.planner.virtual_base = None
            self.planner.cloud_ready = False
        if not best:
            return None, 0.0
        coverage, _, pose = max(best)
        return pose, coverage

    @staticmethod
    def _chassis_blocked(xy, yaw, footprints, margin=0.05):
        from ..geometry import BASE_HALF
        c, s_ = math.cos(yaw), math.sin(yaw)
        hx, hy = BASE_HALF[0] + margin, BASE_HALF[1] + margin
        corners = np.array([[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]]) @ np.array([[c, s_], [-s_, c]]) + xy
        lo, hi = corners.min(0), corners.max(0)
        return any(lo[0] < f[1] and hi[0] > f[0] and lo[1] < f[3] and hi[1] > f[2] for f in footprints)

    def _plan_stages(self, model, full_cloud, free_cloud, stages=3):
        """Kinematic estimate (no physics) of how far a sequence of up to `stages` grasp-and-drag stages
        gets the joint towards the goal from the current base pose."""
        q, total = model["q0"], 0.0
        span = abs(model["goal"] - model["q0"])
        sign = math.copysign(1.0, model["goal"] - model["q0"])
        for _ in range(stages):
            best = self._best_drag(model, None, q, full_cloud, free_cloud, need=span - total, offset=q - model["q0"],
                                   coarse=True)
            if best is None or best[0] < 0.01:
                break
            total += best[0]
            q += sign * best[0]
            if total >= span - 1e-6:
                break
        return total

    def _actuate(self, articulation, joint, target, opening, stages=3):
        verb = "open" if opening else "close"

        def body():
            if self.held:
                raise SkillError(f"holding {self.held}; place it first")
            model = self._joint_model(articulation, joint, target, opening)
            kind, art, index, goal = (model[k] for k in ("kind", "art", "index", "goal"))
            if abs(goal - model["q0"]) < 0.01:
                return dict(joint_position=round(model["q0"], 3), note="already there")
            link = model["joint"].child_link
            tolerance = 0.03 if kind == "prismatic" else 0.08
            log, failed = [], set()
            for stage in range(stages + 2):
                model = self._joint_model(articulation, joint, target, opening)
                q_now = model["q0"]
                if abs(goal - q_now) <= tolerance / 2:
                    break
                full, free = self.planner.build_cloud(), self.planner.build_cloud(exclude={link.name})
                best = self._best_drag(model, link, q_now, full, free, skip=failed)
                if best is None or best[0] < 0.01:
                    if stage == 0:
                        raise SkillError(f"{articulation}/{joint} cannot be grasped and moved from this base pose "
                                         f"(see suggest_base_pose)")
                    break
                _, cand, grasp_q, pre_q = best
                self.planner.use_cloud(full)
                self._set_gripper(GRIPPER_OPEN, steps=4)
                try:
                    self._approach(cand["grasp"], grasp_q, "grasp", pre_q)
                except SkillError as error:
                    log.append(dict(grasp=cand["kind"], held=False, approach=str(error)))
                    failed.add(cand["index"])
                    continue
                self._set_gripper(GRIPPER_CLOSED, steps=15)
                gap = float(self.planner.qpos()[7:].sum())
                if not self._link_grasped(link) or abs(gap - cand["width"]) > 0.01:
                    self._set_gripper(GRIPPER_OPEN)
                    self._back_off()
                    log.append(dict(grasp=cand["kind"], held=False, gap=round(gap, 3)))
                    failed.add(cand["index"])
                    continue
                model = self._joint_model(articulation, joint, target, opening)
                dqs, _ = self._drag_targets(model, model["q0"])
                self.planner.use_cloud(self.planner.build_cloud(exclude={link.name},
                                                                clear=(self.planner.tcp_pose().p, 0.08)))
                path, reached = self._drag_path(model, cand, self.planner.tcp_pose(), dqs, self.planner.qpos()[:7], avoid=True)
                checked = True
                if reached < 3:
                    path, reached = self._drag_path(
                        model, cand, self.planner.tcp_pose(), dqs, self.planner.qpos()[:7], avoid=False)
                    checked = False
                if path:
                    self._follow(_densify(np.vstack([self.planner.qpos()[:7], path]), CONTACT_JOINT_STEP), settle=15)
                self._set_gripper(GRIPPER_OPEN, steps=10)
                self.planner.use_cloud(full)
                self._back_off()
                q_after = float(art.qpos[0, index])
                log.append(dict(grasp=cand["kind"], held=True, start=round(model["q0"], 3), to=round(q_after, 3),
                                planned=f"{reached}/{len(dqs)}", predicted=round(best[0], 3), collision_checked=checked))
                if abs(q_after - model["q0"]) < 0.005:
                    failed.add(cand["index"])
                if sum(1 for l in log if l["held"]) >= stages:
                    break
            q_after = float(art.qpos[0, index])
            info = dict(joint_position=round(q_after, 3), goal=round(goal, 3), stages=log)
            if abs(q_after - goal) > tolerance:
                raise SkillError(f"joint stopped at {q_after:.3f} (goal {goal:.3f}) after {len(log)} grasp(s); "
                                 f"the rest is out of reach from this base pose", info)
            return info
        return self._run(f"{verb}({articulation}/{joint})", body)
