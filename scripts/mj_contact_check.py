"""MuJoCo contact check: tabletop cubes and a test drawer, then the kitchen's real drawers, fridge door and groceries.

    python scripts/mj_contact_check.py --seeds 0 1 2 3 4 --report outputs/mj_contact.json
    python scripts/mj_contact_check.py --kitchen-only

The test drawer has the kitchen's handle bar and joint settings. Base poses come from demo_manipulation.py. Arm motion is damped
least-squares IK with collision checks; it measures grasp and drag physics, it is not a skill.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco  # noqa: E402

from mobile_panda_sim.geometry import CUBE_HALF  # noqa: E402
from mobile_panda_sim.mj.sim import MujocoScene  # noqa: E402

PICK_START = (-0.615, -0.08, 0.0)
DRAWER_START = (-0.62, 0.45, 0.0)
HANDLE = np.array([0.10, 0.45, 0.16])
HANDLE_RADIUS, HANDLE_LENGTH, HANDLE_STANDOFF = 0.009, 0.12, 0.04
DRAWER_TRAVEL, DRAWER_TARGET = 0.30, 0.25
LIFT, HOLD_STEPS = 0.12, 60
OPEN, SQUEEZE = 0.04, -0.01
STEP_M = 0.004
DRAG_SAMPLES, DRAG_STEPS, MAX_PITCH = 100, 2, 70
SETTLE_HOLD = 20
SLIDE_OVERSHOOT, HINGE_OVERSHOOT = 0.02, 0.05
READY_POSES = ((0, -0.6, 0, -2.4, 0, 1.9, 0.78), (0.9, -0.5, 0, -2.3, 0, 1.9, 0.78), (-0.9, -0.5, 0, -2.3, 0, 1.9, 0.78))
ROBOT_BODIES = ("chassis_link", "pedestal_link")


def quat_of(matrix):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(-1))
    return quat


def down_quat(yaw):
    """TCP pointing down; fingers close across the cube face at `yaw`."""
    c, s = np.cos(yaw), np.sin(yaw)
    return quat_of([[c, s, 0], [s, -c, 0], [0, 0, -1]])


FORWARD_QUAT = quat_of([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


class Arm:
    def __init__(self, scene):
        self.scene, self.model = scene, scene.model
        self.ik_data = mujoco.MjData(scene.model)
        self.dofs = [self.model.joint(f"panda_joint{i}").dofadr[0] for i in range(1, 8)]
        self.lower, self.upper = self.model.jnt_range[[self.model.joint(f"panda_joint{i}").id for i in range(1, 8)]].T
        self.gripper = OPEN

    def solve(self, seed, position, quat):
        """Damped least-squares IK from seed; returns (joints, position error)."""
        model, data = self.model, self.ik_data
        data.qpos[:] = self.scene.data.qpos
        q = seed.copy()
        for _ in range(60):
            data.qpos[self.scene.arm_q] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            error = np.zeros(6)
            error[:3] = position - data.site_xpos[self.scene.tcp]
            current, inverse = quat_of(data.site_xmat[self.scene.tcp]), np.zeros(4)
            mujoco.mju_negQuat(inverse, current)
            delta = np.zeros(4)
            mujoco.mju_mulQuat(delta, quat, inverse)
            mujoco.mju_quat2Vel(error[3:], delta, 1.0)
            if np.linalg.norm(error) < 1e-4:
                break
            jacobian = np.zeros((6, model.nv))
            mujoco.mj_jacSite(model, data, jacobian[:3], jacobian[3:], self.scene.tcp)
            j = jacobian[:, self.dofs]
            q = np.clip(q + j.T @ np.linalg.solve(j @ j.T + 1e-4*np.eye(6), error), self.lower, self.upper)
        return q, float(np.linalg.norm(error[:3]))

    def solutions(self, position, quat, seed_q, tries=60):
        """Distinct collision-free IK solutions for the pose, nearest to seed_q first."""
        rng = np.random.default_rng(0)
        found = []
        for seed in [seed_q] + [rng.uniform(self.lower, self.upper) for _ in range(tries)]:
            q, error = self.solve(seed, position, quat)
            if error < 1e-3 and not self.collides(q) and not any(np.abs(q - other).max() < 0.1 for other in found):
                found.append(q)
        return sorted(found, key=lambda c: np.abs(c - seed_q).sum())

    def collides(self, q, joint=None, value=0.0):
        """True if the arm at joints q touches anything but itself; `joint` is a qpos address set to `value`."""
        model, data = self.model, self.ik_data
        data.qpos[:] = self.scene.data.qpos
        data.qpos[self.scene.arm_q] = q
        if joint is not None:
            data.qpos[joint] = value
        mujoco.mj_forward(model, data)
        for i in range(data.ncon):
            names = [model.body(model.geom_bodyid[g]).name for g in (data.contact.geom1[i], data.contact.geom2[i])]
            if sum(n.startswith("panda_") for n in names) == 1:
                return True
        return False

    def free_joint_path(self, start, goal, steps=30):
        return not any(self.collides(start + (goal - start)*t) for t in np.linspace(0, 1, steps))

    def route(self, start, goal):
        """Collision-free joint waypoints from start to goal: direct, or through a ready pose."""
        if self.free_joint_path(start, goal):
            return [goal]
        for ready in READY_POSES:
            ready = np.array(ready)
            if not self.collides(ready) and self.free_joint_path(start, ready) and self.free_joint_path(ready, goal):
                return [ready, goal]
        return None

    def joint_move(self, goal, steps=60):
        scene = self.scene
        start = scene.data.qpos[scene.arm_q].copy()
        for t in np.linspace(0, 1, steps)[1:]:
            scene.data.ctrl[scene.arm_ctrl] = start + (goal - start)*t
            scene.step(0, 0, None, self.gripper)
        self.hold(20)

    def reach(self, position, quat, tries=60, seed=0):
        """Joint-space move to the nearest IK solution whose straight joint path is collision free."""
        rng = np.random.default_rng(seed)
        scene = self.scene
        current = scene.data.qpos[scene.arm_q].copy()
        candidates = []
        for _ in range(tries):
            q, error = self.solve(rng.uniform(self.lower, self.upper), position, quat)
            if error < 1e-3:
                candidates.append(q)
        for q in sorted(candidates, key=lambda c: np.abs(c - current).sum()):
            if self.free_joint_path(current, q):
                self.joint_move(q)
                return
        raise RuntimeError(f"no collision-free joint path to {position}")

    def hold(self, steps, gripper=None):
        self.gripper = self.gripper if gripper is None else gripper
        for _ in range(steps):
            self.scene.step(0, 0, None, self.gripper)

    def go(self, position, quat, step_m=STEP_M, settle=12):
        """Straight TCP line from the current pose to (position, quat)."""
        scene = self.scene
        start, q = *scene.tcp_pose()[:1], scene.data.qpos[scene.arm_q].copy()
        start_quat = scene.tcp_pose()[1]
        count = max(int(np.linalg.norm(position - start) / step_m), 20)
        worst = 0.0
        for i in range(1, count + 1):
            t = i / count
            blend = np.zeros(4)
            mujoco.mju_copy(blend, start_quat)
            velocity = np.zeros(3)
            mujoco.mju_subQuat(velocity, quat, start_quat)
            mujoco.mju_quatIntegrate(blend, velocity, t)
            q, error = self.solve(q, start + (position - start)*t, blend)
            worst = max(worst, error)
            scene.data.ctrl[scene.arm_ctrl] = q
            scene.step(0, 0, None, self.gripper)
        self.hold(settle)
        return worst


def pick(seed):
    scene = MujocoScene("tabletop", "wheels", PICK_START)
    scene.reset(seed)
    arm = Arm(scene)
    results = {}
    for name, body in scene.objects.items():
        position, quat = scene.pose(body)
        yaw = 2*np.arctan2(quat[3], quat[0])
        grasp, top_down = position.copy(), down_quat(yaw)
        arm.go(grasp + [0, 0, 0.12], top_down, step_m=0.01)
        ik_error = arm.go(grasp, top_down)
        arm.hold(30, SQUEEZE)
        width = float(scene.joint_positions()[-2:].sum())
        before = scene.pose(body)[0] - scene.tcp_pose()[0]
        tcp_lift = scene.tcp_pose()[0][2]
        arm.go(grasp + [0, 0, LIFT], top_down)
        arm.hold(HOLD_STEPS)
        after = scene.pose(body)[0]
        tracking = float(np.linalg.norm(scene.tcp_pose()[0] - (grasp + [0, 0, LIFT])))
        tilt = float(np.degrees(2*np.arcsin(np.linalg.norm(scene.pose(scene.chassis)[1][1:3]))))
        slip = float(np.linalg.norm(after - scene.tcp_pose()[0] - before))
        results[name] = dict(lifted_m=round(float(after[2] - CUBE_HALF), 4), slip_m=round(slip, 4), finger_gap_m=round(width, 4),
                             ik_error_m=round(ik_error, 5), tcp_start_z=round(float(tcp_lift), 4), tracking_m=round(tracking, 4),
                             chassis_tilt_deg=round(tilt, 2), success=bool(after[2] - CUBE_HALF > LIFT*0.9 and slip < 0.01))
        arm.go(grasp + [0, 0, 0.003], top_down)
        arm.hold(30, OPEN)
        arm.go(grasp + [0, 0, 0.15], top_down, step_m=0.01)
    scene.close()
    return results


def add_drawer(spec):
    box = mujoco.mjtGeom.mjGEOM_BOX
    panel_x = HANDLE[0] + HANDLE_STANDOFF + 0.01
    colour = [0.6, 0.55, 0.45, 1]
    centre = [panel_x + 0.21, HANDLE[1], HANDLE[2]]
    world = spec.worldbody
    for offset, half in (((0, 0.21, 0), (0.2, 0.012, 0.09)), ((0, -0.21, 0), (0.2, 0.012, 0.09)), ((0, 0, -0.1), (0.2, 0.22, 0.012)),
                         ((0, 0, 0.1), (0.2, 0.22, 0.012)), ((0.21, 0, 0), (0.012, 0.22, 0.11))):
        world.add_geom(type=box, size=half, pos=np.add(centre, offset), rgba=colour)
    drawer = world.add_body(name="drawer", pos=[panel_x, HANDLE[1], HANDLE[2]])
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, DRAWER_TRAVEL],
                     damping=2.0, frictionloss=0.3)
    for pos, half, in (((0, 0, 0), (0.008, 0.19, 0.08)), ((0.19, 0, -0.07), (0.19, 0.19, 0.006))):
        drawer.add_geom(type=box, size=half, pos=pos, density=500, rgba=[0.7, 0.62, 0.5, 1])
    bar = drawer.add_geom(name="drawer_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[HANDLE_RADIUS, HANDLE_LENGTH/2, 0],
                          pos=[-HANDLE_STANDOFF - 0.01, 0, 0], quat=[0.7071068, 0.7071068, 0, 0], rgba=[0.7, 0.7, 0.72, 1], density=300)
    bar.friction[0] = 0.8
    for y in (-0.05, 0.05):
        drawer.add_geom(type=box, size=[0.02, 0.005, 0.005], pos=[-0.03, y, 0], density=300, rgba=[0.7, 0.7, 0.72, 1])


def drawer(seed):
    scene = MujocoScene("tabletop", "wheels", DRAWER_START, extend=add_drawer)
    scene.reset(seed)
    model, data = scene.model, scene.data
    arm = Arm(scene)
    slide = model.joint("drawer_slide").qposadr[0]
    base_start = scene.pose(scene.chassis)[0]
    arm.reach(HANDLE - [0.12, 0, 0], FORWARD_QUAT, seed=seed)
    ik_error = arm.go(HANDLE, FORWARD_QUAT)
    arm.hold(30, SQUEEZE)
    gap = float(scene.joint_positions()[-2:].sum())
    bar = model.geom("drawer_handle").id
    worst_lag = 0.0
    pull = ik_error
    for target in np.linspace(0.05, DRAWER_TARGET, 5):
        pull = max(pull, arm.go(HANDLE - [target, 0, 0], FORWARD_QUAT, step_m=0.002, settle=4))
        worst_lag = max(worst_lag, float(np.linalg.norm(data.geom_xpos[bar] - scene.tcp_pose()[0])))
    opened = float(data.qpos[slide])
    open_ok = abs(opened - DRAWER_TARGET) < 0.02
    arm.go(HANDLE - [0.01, 0, 0], FORWARD_QUAT, step_m=0.002)
    closed = float(data.qpos[slide])
    result = dict(opened_m=round(opened, 4), closed_m=round(closed, 4), finger_gap_m=round(gap, 4), handle_offset_m=round(worst_lag, 4),
                  ik_error_m=round(pull, 5), base_drift_m=round(float(np.linalg.norm(scene.pose(scene.chassis)[0] - base_start)), 4),
                  success=bool(open_ok and closed < 0.02))
    scene.close()
    return result


def rotate_about(axis, angle):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*k + (1 - np.cos(angle))*k @ k


def moved_handle(scene, articulation, joint_name):
    """f(q) -> (position, rotation) of the handle after the joint moves by q from now."""
    position, rotation, _ = scene.handle_pose(articulation, joint_name)
    joint = scene.model.joint(f"{articulation}/{joint_name}").id
    axis, anchor = scene.data.xaxis[joint].copy(), scene.data.xanchor[joint].copy()
    if scene.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE:
        return lambda q: (position + axis*q, rotation)
    return lambda q: (anchor + rotate_about(axis, q) @ (position - anchor), rotate_about(axis, q) @ rotation)


def grasp_rotation(rotation, angle):
    """Hand frame on a bar with pose `rotation` (x along the bar, z out of the panel), pitched about the bar."""
    c, s_ = np.cos(angle), np.sin(angle)
    pitched = rotation @ np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
    return np.stack([pitched[:, 0], np.cross(-pitched[:, 2], pitched[:, 0]), -pitched[:, 2]], axis=1)


def follow(arm, moved, target, address, q, pitch):
    """Chain IK along the drag, turning the grasp about the round bar where a fixed grasp cannot follow."""
    path = []
    for value in np.linspace(0, target, DRAG_SAMPLES + 1)[1:]:
        position, rotation = moved(value)
        for change in np.radians([0, 3, -3, 6, -6, 10, -10]):
            if abs(pitch + change) > np.radians(MAX_PITCH):
                continue
            candidate, error = arm.solve(q, position, quat_of(grasp_rotation(rotation, pitch + change)))
            if error < 1e-3 and np.abs(candidate - q).max() < 0.25 and not arm.collides(candidate, address, value):
                q, pitch = candidate, pitch + change
                path.append(q)
                break
        else:
            break
    return path


def plan_drag(arm, moved, target, address):
    """(route to pre-grasp, grasp joints, drag joints) reaching furthest along the drag; fingers open around the bar."""
    scene = arm.scene
    current = scene.data.qpos[scene.arm_q].copy()
    best, best_length = None, -1
    position, rotation = moved(0.0)
    for angle in np.radians([0, 20, -20, 40, -40, 60, -60]):
        hand = grasp_rotation(rotation, angle)
        quat = quat_of(hand)
        for grasp in arm.solutions(position, quat, current):
            pre, error = arm.solve(grasp, position - hand[:, 2]*0.10, quat)
            clear = error < 1e-3 and not arm.collides(pre) and arm.free_joint_path(pre, grasp)
            route = arm.route(current, pre) if clear else None
            if route is None:
                continue
            path = follow(arm, moved, target, address, grasp, angle)
            if len(path) > best_length:
                best, best_length = (route, grasp, path), len(path)
            if best_length == DRAG_SAMPLES:
                return best
    return best


def actuate(start, articulation, joint_name, target, seed=0):
    """Open `joint_name` to `target`, then close it; checks the joint and that nothing else moved."""
    scene = MujocoScene("kitchen", "wheels", start)
    scene.reset(seed)
    data = scene.data
    arm = Arm(scene)
    arm.hold(SETTLE_HOLD)
    address = scene.articulations[articulation][joint_name]
    before = {name: scene.pose(body)[0] for name, body in scene.objects.items()}
    moved = moved_handle(scene, articulation, joint_name)
    hinge = scene.model.jnt_type[scene.model.joint(f"{articulation}/{joint_name}").id] == mujoco.mjtJoint.mjJNT_HINGE
    plan = plan_drag(arm, moved, target + (HINGE_OVERSHOOT if hinge else SLIDE_OVERSHOOT), address)
    result = dict(joint=f"{articulation}/{joint_name}", target=target)
    if plan is None or len(plan[2]) < DRAG_SAMPLES:
        scene.close()
        return dict(result, success=False, reachable=0.0 if plan is None else round(len(plan[2]) / DRAG_SAMPLES, 2),
                    message="no collision-free drag from this base pose")
    route, grasp, path = plan
    for waypoint in route:
        arm.joint_move(waypoint)
    position, rotation = moved(0.0)
    arm.go(position, scene.tcp_pose()[1], step_m=0.003)
    arm.hold(30, SQUEEZE)
    gap = float(scene.joint_positions()[-2:].sum())
    base_start = scene.pose(scene.chassis)[0]
    offset = 0.0

    def drag(waypoints):
        nonlocal offset
        for q in waypoints:
            data.ctrl[scene.arm_ctrl] = q
            for _ in range(DRAG_STEPS):
                scene.step(0, 0, None, arm.gripper)
            offset = max(offset, float(np.linalg.norm(scene.tcp_pose()[0] - scene.handle_pose(articulation, joint_name)[0])))
        arm.hold(15)

    drag(path)
    opened = float(data.qpos[address])
    drag([*path[::-1], grasp])
    closed = float(data.qpos[address])
    gap_end = float(scene.joint_positions()[-2:].sum())
    arm.hold(20, OPEN)
    arm.joint_move(route[-1], steps=40)
    disturbed = {name: round(float(np.linalg.norm(scene.pose(body)[0] - before[name])), 3) for name, body in scene.objects.items()
                 if np.linalg.norm(scene.pose(body)[0] - before[name]) > 0.02}
    tolerance = 0.02 if target < 0.3 else 0.05
    result.update(opened=round(opened, 3), closed=round(closed, 3), finger_gap_m=round(gap, 4), finger_gap_end_m=round(gap_end, 4),
                  handle_offset_m=round(offset, 4), base_drift_m=round(float(np.linalg.norm(scene.pose(scene.chassis)[0] - base_start)), 4),
                  disturbed=disturbed, success=bool(opened > target - tolerance and abs(closed) < 0.03 and not disturbed))
    scene.close()
    return result


def collision_points(scene, body):
    model, data = scene.model, scene.data
    points = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == body and model.geom_contype[geom] and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points.append(data.geom_xpos[geom] + model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T)
    return np.concatenate(points)


def pick_prop(scene, arm, name):
    body = scene.objects[name]
    points = collision_points(scene, body)
    centre = (points.max(0) + points.min(0)) / 2
    low, high = points[:, 2].min(), points[:, 2].max()
    flat = points[:, :2] - points[:, :2].mean(0)
    _, vectors = np.linalg.eigh(flat.T @ flat)
    narrow = vectors[:, 0]
    grasp = np.array([centre[0], centre[1], max(high - 0.04, low + 0.5*(high - low), low + 0.012)])
    before = {n: scene.pose(b)[0] for n, b in scene.objects.items()}
    current = scene.data.qpos[scene.arm_q].copy()
    options = [down_quat(np.arctan2(n[0], -n[1])) for n in (narrow, -narrow)]
    top_down = min(options, key=lambda quat: min((np.abs(q - current).sum() for q in arm.solutions(grasp + [0, 0, 0.12], quat, current, tries=12)),
                                                 default=np.inf))
    arm.reach(grasp + [0, 0, 0.12], top_down, tries=30)
    arm.go(grasp, top_down)
    arm.hold(30, SQUEEZE)
    gap = float(scene.joint_positions()[-2:].sum())
    arm.go(grasp + [0, 0, LIFT], top_down)
    arm.hold(HOLD_STEPS)
    risen = float(scene.pose(body)[0][2] - before[name][2])
    disturbed = {n: round(float(np.linalg.norm(scene.pose(b)[0] - before[n])), 3) for n, b in scene.objects.items()
                 if n != name and np.linalg.norm(scene.pose(b)[0] - before[n]) > 0.02}
    arm.go(grasp + [0, 0, 0.004], top_down)
    arm.hold(30, OPEN)
    arm.go(grasp + [0, 0, 0.14], top_down, step_m=0.01)
    return dict(object=name, lifted_m=round(risen, 4), finger_gap_m=round(gap, 4), disturbed=disturbed, success=bool(risen > LIFT*0.9 and not disturbed))


def kitchen():
    results = []
    scene = MujocoScene("kitchen", "wheels", (-2.95, -1.15, 3.1416))
    scene.reset(0)
    arm = Arm(scene)
    arm.joint_move(np.array(READY_POSES[0]))
    results += [pick_prop(scene, arm, name) for name in ("sugar_box", "pear", "lemon")]
    scene.close()
    results.append(actuate((-1.17, 1.86, 1.989), "kitchen_counter", "middle_slide_top", 0.25))
    results.append(actuate((-2.899, 1.86, 1.153), "kitchen_counter", "left_slide_top", 0.25))
    results.append(actuate((0.249, 1.492, 1.107), "fridge", "top_door_hinge", 0.6))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--kitchen-only", action="store_true")
    args = parser.parse_args()
    report = []
    for seed in [] if args.kitchen_only else args.seeds:
        item = dict(seed=seed, pick=pick(seed), drawer=drawer(seed))
        report.append(item)
        print(json.dumps(item), flush=True)
    picks = [r for item in report for r in item["pick"].values()]
    summary = dict(pick=f"{sum(r['success'] for r in picks)}/{len(picks)}",
                   drawer=f"{sum(i['drawer']['success'] for i in report)}/{len(report)}")
    rooms = kitchen()
    for item in rooms:
        print(json.dumps(item), flush=True)
    summary["kitchen"] = f"{sum(r['success'] for r in rooms)}/{len(rooms)}"
    print(json.dumps(summary))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(summary=summary, runs=report, kitchen=rooms), indent=2), encoding="utf-8")
    return 0 if all(r["success"] for r in [*picks, *rooms]) and all(i["drawer"]["success"] for i in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
