"""MuJoCo contact check: lift the cubes, then open and close a test drawer with the gripper.

    python scripts/mj_contact_check.py --seeds 0 1 2 3 4 --report outputs/mj_contact.json

The drawer is test-only, with the kitchen's handle bar and joint settings. Arm motion is plain damped
least-squares IK with a joint-space approach; it measures grasp and drag physics, it is not a skill.
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

    def collides(self, q):
        """True if the arm at joints q touches anything outside the robot."""
        model, data = self.model, self.ik_data
        data.qpos[:] = self.scene.data.qpos
        data.qpos[self.scene.arm_q] = q
        mujoco.mj_forward(model, data)
        for i in range(data.ncon):
            names = [model.body(model.geom_bodyid[g]).name for g in (data.contact.geom1[i], data.contact.geom2[i])]
            if sum(n.startswith("panda_") for n in names) == 1 and not any(n in ROBOT_BODIES for n in names):
                return True
        return False

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
            if not any(self.collides(current + (q - current)*t) for t in np.linspace(0, 1, 30)):
                for t in np.linspace(0, 1, 60)[1:]:
                    scene.data.ctrl[scene.arm_ctrl] = current + (q - current)*t
                    scene.step(0, 0, None, self.gripper)
                self.hold(20)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = []
    for seed in args.seeds:
        item = dict(seed=seed, pick=pick(seed), drawer=drawer(seed))
        report.append(item)
        print(json.dumps(item), flush=True)
    picks = [r for item in report for r in item["pick"].values()]
    summary = dict(pick=f"{sum(r['success'] for r in picks)}/{len(picks)}",
                   drawer=f"{sum(i['drawer']['success'] for i in report)}/{len(report)}")
    print(json.dumps(summary))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(summary=summary, runs=report), indent=2), encoding="utf-8")
    return 0 if all(r["success"] for r in picks) and all(i["drawer"]["success"] for i in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
