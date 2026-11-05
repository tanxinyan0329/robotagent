"""Mobile pick demo (ported from agent_demo/mobile_panda/demo_drive.py, Linux, new tabletop scene).

Drive from the start pose to a docking pose in front of the chosen cube, stop, then run the official
pick policy through the observation adapter and check the grasp holds for one more second. --backend mujoco
uses the policy trained on MuJoCo (training/mujoco/train_pick.py); the default is ManiSkill's PickCube PPO.
One reset per episode; nothing is teleported. Navigation is a simple go-to-point controller for an
obstacle-free approach, not a planner. Runs in-process (no ROS) so it can be used as a skill test.

    python scripts/demo_mobile_pick.py                         # wheels base, red cube, viewer window
    python scripts/demo_mobile_pick.py --no-render --seeds 0 1 2 3 4 --report outputs/mobile_pick.json
    python scripts/demo_mobile_pick.py --base planar --cube blue_cube --no-render --record outputs/pick.mp4
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobile_panda_sim.geometry import ARM_REST, CUBES, DEFAULT_START, MAX_JOINT_DELTA  # noqa: E402

DOCK_X = -0.615
GOAL_HEIGHT = 0.16


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class ManiSkillRobot:
    def __init__(self, args):
        import gymnasium as gym
        import torch
        import mobile_panda_sim.scenes  # noqa: F401
        from mobile_panda_sim.skills.pick_cube import PickPolicy, encode, in_training_range
        self.torch, self.encode, self.in_training_range = torch, encode, in_training_range
        self.args = args
        self.env = gym.make("MobilePandaTabletop-v1", robot_uids=f"mobile_panda_{args.base}", start=args.start,
                            obs_mode="none", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
                            reward_mode="none", render_mode="rgb_array" if args.no_render else "human")
        self.w = self.env.unwrapped
        names = [j.name for j in self.w.agent.robot.get_active_joints()]
        self.arm_ids = [names.index(n) for n in self.w.agent.arm_joint_names]
        self.policy = PickPolicy()
        self.dt = float(self.w.control_timestep)

    def reset(self, seed, cube):
        self.env.reset(seed=seed)
        self.cube = self.w.cubes[cube]

    def base(self):
        t = self.w.agent.robot.links_map["chassis_link"].pose.to_transformation_matrix()[0]
        return np.array([float(t[0, 3]), float(t[1, 3]), math.atan2(float(t[1, 0]), float(t[0, 0]))])

    def cube_position(self):
        return self.cube.pose.p[0].cpu().numpy()

    def grasped(self):
        return bool(self.w.agent.is_grasping(self.cube)[0])

    def cube_in_range(self):
        return self.in_training_range(self.w, self.cube)

    def step(self, velocity, goal):
        torch = self.torch
        with torch.no_grad():
            if goal is not None:
                out = self.policy(self.encode(self.w, self.cube, torch.tensor([goal], dtype=torch.float32)))
                arm, grip = out[:, :7], out[:, 7:8]
            else:
                q = self.w.agent.robot.get_qpos()[:, self.arm_ids]
                arm = ((torch.tensor(ARM_REST[:7])[None] - q) / MAX_JOINT_DELTA).clamp(-0.7, 0.7)
                grip = torch.ones((1, 1))
            base = torch.tensor([self.w.agent.base_action(*velocity)], dtype=torch.float32)
            self.env.step(self.w.agent.controller.from_action_dict(dict(arm=arm, gripper=grip, base=base)))
        if not self.args.no_render:
            self.w.render()
            time.sleep(0.02)

    def frame(self):
        return self.w.render()[0].cpu().numpy()

    def close(self):
        self.env.close()


class MujocoRobot:
    def __init__(self, args):
        from mobile_panda_sim.geometry import GRIPPER_RANGE
        from mobile_panda_sim.mj.pick_skill import PickObserver, PickPolicy
        from mobile_panda_sim.mj.sim import MujocoScene
        self.observer_type, self.gripper_range = PickObserver, GRIPPER_RANGE
        self.args = args
        self.scene = MujocoScene("tabletop", args.base, args.start, viewer=not args.no_render)
        self.policy = PickPolicy()
        self.dt = self.scene.control_timestep

    def reset(self, seed, cube):
        self.scene.reset(seed)
        self.body = self.scene.objects[cube]
        self.observer = self.observer_type(self.scene, cube)

    def base(self):
        position, q = self.scene.pose(self.scene.chassis)
        return np.array([position[0], position[1], math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))])

    def cube_position(self):
        return self.scene.pose(self.body)[0]

    def grasped(self):
        return self.observer.grasped()

    def cube_in_range(self):
        return self.observer.cube_in_range()

    def step(self, velocity, goal):
        low, high = self.gripper_range
        if goal is not None:
            action = self.policy(self.observer.observation(goal))
            arm, width = action[:7]*MAX_JOINT_DELTA, low + (action[7] + 1)/2*(high - low)
        else:
            q = self.scene.joint_positions()[-9:-2]
            arm, width = np.clip(np.array(ARM_REST[:7]) - q, -0.07, 0.07), high
        self.scene.step(velocity[0], velocity[1], arm, width)
        if not self.args.no_render:
            time.sleep(self.dt)

    def frame(self):
        return self.scene.capture_rgb()["base_camera"]

    def close(self):
        self.scene.close()


class Demo:
    def __init__(self, args):
        self.args = args
        self.robot = (MujocoRobot if args.backend == "mujoco" else ManiSkillRobot)(args)
        self.writer = None

    def base(self):
        return self.robot.base()

    def event(self, phase, **extra):
        self.phase = phase
        record = dict(phase=phase, step=self.steps, base=self.base().round(4).tolist(), **extra)
        self.events.append(record)
        print(json.dumps(record), flush=True)

    def step(self, velocity=(0.0, 0.0), policy=False):
        before = self.base()
        self.robot.step(velocity, self.goal if policy else None)
        self.steps += 1
        after = self.base()
        self.speed = float(np.linalg.norm(after[:2] - before[:2])) / self.robot.dt
        self.yaw_speed = abs(wrap(after[2] - before[2])) / self.robot.dt
        if self.writer:
            self.writer.append_data(self.robot.frame())

    def settle(self, n=25):
        for _ in range(n):
            self.step()

    def drive_to(self, target, yaw=0.0):
        self.event("navigate", target=list(target), target_yaw=yaw)
        wheels = self.args.base == "wheels"
        pos_tol, yaw_tol = (0.025, 0.035) if wheels else (0.012, 0.018)

        def turn(error, gain, limit):
            value = float(np.clip(error * gain, -limit, limit))
            if wheels and abs(error) > yaw_tol:
                value = math.copysign(max(abs(value), 0.12), value)
            return value
        for _ in range(600):
            p = self.base()
            delta = np.asarray(target) - p[:2]
            distance = float(np.linalg.norm(delta))
            if distance < pos_tol:
                error = wrap(yaw - p[2])
                if abs(error) < yaw_tol:
                    self.settle()
                    self.event("parked", position_error=round(distance, 4), yaw_error=round(error, 4),
                               speed=round(self.speed, 4))
                    if self.speed > 0.02 or self.yaw_speed > 0.04:
                        raise RuntimeError("base not stationary after parking")
                    return
                self.step((0.0, turn(error, 1.8, 0.5)))
            else:
                heading = wrap(math.atan2(delta[1], delta[0]) - p[2])
                v = min(0.22, distance * 0.9) if abs(heading) < 0.3 else 0.0
                if wheels and v > 0:
                    v = max(0.035, v)
                self.step((v, turn(heading, 2.0, 0.6)))
        raise RuntimeError("navigation did not reach docking tolerance")

    def pick(self):
        if not self.robot.cube_in_range():
            raise RuntimeError("cube outside the pick policy's training range from this docking pose")
        start = self.base()
        self.event("ppo_pick")
        stable = 0
        for _ in range(160):
            self.step(policy=True)
            at_goal = float(np.linalg.norm(self.robot.cube_position() - self.goal)) <= 0.025
            stable = stable + 1 if self.robot.grasped() and at_goal else 0
            if stable >= 5:
                break
        else:
            raise RuntimeError("policy did not grasp the cube and reach the lift goal")
        for _ in range(20):
            self.step(policy=True)
            if not self.robot.grasped() or float(self.robot.cube_position()[2]) < 0.10:
                raise RuntimeError("cube dropped during sustained-grasp check")
        drift = float(np.linalg.norm(self.base()[:2] - start[:2]))
        if drift > 0.02:
            raise RuntimeError(f"base drifted {drift:.3f} m during the grasp")
        self.event("pick_success", base_drift=round(drift, 4))
        return dict(base_drift=drift)

    def run(self, seed, record=None):
        self.robot.reset(seed, self.args.cube)
        self.steps, self.events, self.speed, self.yaw_speed, self.phase = 0, [], 0.0, 0.0, "start"
        cube_xy = [float(v) for v in self.robot.cube_position()[:2]]
        self.goal = [*cube_xy, GOAL_HEIGHT]
        if record:
            import imageio
            Path(record).parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(str(record), fps=20)
        error, metrics = None, {}
        try:
            self.event("start", seed=seed, base_model=self.args.base, cube=self.args.cube)
            self.settle()
            self.drive_to((DOCK_X, cube_xy[1]), 0.0)
            metrics = self.pick()
        except RuntimeError as exc:
            error = str(exc)
            self.event("failure", error=error)
        finally:
            if self.writer:
                self.writer.close()
                self.writer = None
        return dict(seed=seed, success=error is None, error=error, steps=self.steps, metrics=metrics, events=self.events)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("maniskill", "mujoco"), default="maniskill")
    p.add_argument("--base", choices=("wheels", "planar"), default="wheels")
    p.add_argument("--cube", choices=sorted(CUBES), default="red_cube")
    p.add_argument("--start", nargs=3, type=float, default=list(DEFAULT_START), metavar=("X", "Y", "YAW"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--record", type=Path, help="video of the first episode")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    demo = Demo(args)
    try:
        results = [demo.run(s, args.record if i == 0 else None) for i, s in enumerate(args.seeds)]
    finally:
        demo.robot.close()
    summary = dict(base=args.base, cube=args.cube, start=args.start, successes=sum(r["success"] for r in results),
                   episodes=len(results), results=results)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("base", "cube", "successes", "episodes")}), flush=True)
    return 0 if summary["successes"] == summary["episodes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
