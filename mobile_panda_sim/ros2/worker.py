"""Simulation worker: owns the ManiSkill scene. stdin/stdout carry newline-delimited JSON only."""
import argparse
import base64
import contextlib
import json
import math
import os
import queue
import sys
import threading
import time
import traceback

from .protocol import gripper_action, validate_command

VELOCITY_TIMEOUT = 0.3
SETTLE_SECONDS = 1.0
PUBLISH_PERIOD = 0.1


def emit(stream, kind, **payload):
    stream.write(json.dumps({"type": kind, **payload}, allow_nan=False) + "\n")
    stream.flush()


def pose_dict(pose):
    return {"position": pose.p[0].tolist(), "quaternion_wxyz": pose.q[0].tolist()}


class Simulation:
    def __init__(self, args, send):
        import gymnasium as gym
        import torch
        from mobile_panda_sim.scenes import SCENES

        self.torch, self.args, self.send = torch, args, send
        self.env = gym.make(
            SCENES[args.scene], robot_uids=f"mobile_panda_{args.base}", start=args.start,
            obs_mode="none", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
            reward_mode="none", render_mode="human" if args.viewer else "rgb_array",
        )
        self.uenv = self.env.unwrapped
        self.agent = self.uenv.agent
        self.joint_names = [j.name for j in self.agent.robot.get_active_joints()]
        self.trajectory = 0
        self.reset()

    def reset(self):
        self.env.reset(seed=self.args.seed)
        self.trajectory += 1
        self.steps, self.sim_time = 0, 0.0
        self.velocity, self.velocity_deadline = (0.0, 0.0), 0.0
        self.arm_delta = None
        self.gripper = 1.0
        self.active_until = 0.0
        self.images, self.images_step = None, -1

    def handle(self, cmd):
        validate_command(cmd)
        kind, now = cmd["kind"], time.monotonic()
        if kind == "stop":
            self.velocity, self.velocity_deadline, self.arm_delta = (0.0, 0.0), 0.0, None
            self.send("stopped", trajectory=self.trajectory)
            return
        if kind == "reset":
            self.reset()
            self.send("reset_complete", trajectory=self.trajectory)
            return
        if kind == "velocity":
            self.velocity = (float(cmd.get("linear", 0)), float(cmd.get("angular", 0)))
            self.velocity_deadline = now + VELOCITY_TIMEOUT
        elif kind == "arm":
            self.arm_delta = [float(x) for x in cmd["delta"]]
        elif kind == "gripper":
            self.gripper = gripper_action(cmd["width"])
        self.active_until = now + SETTLE_SECONDS

    def tick(self):
        torch, now = self.torch, time.monotonic()
        if now > self.active_until:
            if self.args.viewer:
                self.uenv.render()
            return False
        linear, angular = self.velocity if now <= self.velocity_deadline else (0.0, 0.0)
        arm = torch.zeros((1, 7))
        if self.arm_delta is not None:
            arm[0] = torch.tensor(self.arm_delta) / 0.1
            self.arm_delta = None
        action = self.agent.controller.from_action_dict(dict(
            arm=arm, gripper=torch.tensor([[self.gripper]]),
            base=torch.tensor([self.agent.base_action(linear, angular)], dtype=torch.float32)))
        self.env.step(action)
        self.steps += 1
        self.sim_time += float(self.uenv.control_timestep)
        if self.args.viewer:
            self.uenv.render()
        return True

    def snapshot(self):
        robot = self.agent.robot
        values = lambda t: t.detach().cpu().reshape(-1).tolist()
        chassis = robot.links_map["chassis_link"].pose
        qpos = values(robot.get_qpos())
        fingers = [qpos[self.joint_names.index(n)] for n in self.agent.gripper_joint_names]
        if self.images_step != self.steps:
            self.images = {name: {"height": int(a.shape[0]), "width": int(a.shape[1]),
                                  "data": base64.b64encode(a.tobytes()).decode("ascii")}
                           for name, a in self.uenv.capture_rgb().items()}
            self.images_step = self.steps
        q = chassis.q[0].tolist()
        yaw = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
        self.send("state", trajectory=self.trajectory, steps=self.steps, sim_time=round(self.sim_time, 4),
                  running=time.monotonic() <= self.active_until, base=self.args.base,
                  base_pose={**pose_dict(chassis), "yaw": yaw},
                  root_pose=pose_dict(robot.pose),
                  tcp_pose=pose_dict(self.agent.tcp.pose),
                  joint_names=self.joint_names, joint_positions=qpos, joint_velocities=values(robot.get_qvel()),
                  gripper_width=fingers,
                  objects={name: pose_dict(actor.pose) for name, actor in self.uenv.movable_objects.items()},
                  articulations={name: dict(zip([j.name for j in art.get_active_joints()], values(art.qpos)))
                                 for name, art in self.uenv.articulations.items()},
                  images=self.images)

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["kitchen", "tabletop"], default="kitchen")
    parser.add_argument("--base", choices=["wheels", "planar"], default="wheels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"))
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if not args.viewer:
        os.environ.setdefault("SAPIEN_NO_DISPLAY", "1")
    protocol_out = sys.stdout
    send = lambda kind, **payload: emit(protocol_out, kind, **payload)
    commands = queue.Queue(maxsize=256)

    def read_commands():
        for line in sys.stdin:
            try:
                commands.put(json.loads(line), timeout=1)
            except (ValueError, queue.Full) as error:
                send("error", reason=str(error))
        commands.put({"kind": "shutdown"})

    sim = None
    with contextlib.redirect_stdout(sys.stderr):
        try:
            sim = Simulation(args, send)
            threading.Thread(target=read_commands, daemon=True).start()
            send("scene", layout=sim.uenv.layout())
            send("ready", scene=args.scene, base=args.base, seed=args.seed, joint_names=sim.joint_names,
                 control_timestep=float(sim.uenv.control_timestep), cameras=sim.uenv.camera_intrinsics())
            last_publish = 0.0
            while True:
                start = time.monotonic()
                shutdown = False
                for _ in range(256):
                    try:
                        cmd = commands.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(cmd, dict) and cmd.get("kind") == "shutdown":
                        shutdown = True
                        break
                    try:
                        sim.handle(cmd)
                    except (ValueError, TypeError) as error:
                        send("rejected", reason=str(error), command=cmd.get("kind") if isinstance(cmd, dict) else None)
                if shutdown:
                    break
                sim.tick()
                if start - last_publish >= PUBLISH_PERIOD:
                    sim.snapshot()
                    last_publish = start
                time.sleep(max(0.0, float(sim.uenv.control_timestep) - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass
        except Exception as error:
            send("fatal", reason=f"{type(error).__name__}: {error}")
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            if sim is not None:
                sim.close()


if __name__ == "__main__":
    main()
