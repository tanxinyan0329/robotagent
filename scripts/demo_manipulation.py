"""Classical manipulation skills in the kitchen scene: pick / place, open / close drawers and doors.

Each scenario parks the robot at a fixed base pose (navigation is not part of these skills) and runs a
skill sequence. Every step is checked: skill success, plus any other object moved > 2 cm.

    python scripts/demo_manipulation.py drawer                 # counter drawers
    python scripts/demo_manipulation.py pickplace              # groceries on the entry console
    python scripts/demo_manipulation.py fridge                 # fridge door, 35 deg
    python scripts/demo_manipulation.py all --repeat 3 --report outputs/manipulation.json
    python scripts/demo_manipulation.py fridge --record outputs/fridge.mp4
    python scripts/demo_manipulation.py drawer --viewer        # watch in the SAPIEN window (close it to stop)
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym  # noqa: E402

from mobile_panda_sim.manipulation.skills import ManipulationSkills  # noqa: E402
from mobile_panda_sim.scenes import SCENES  # noqa: E402

SCENARIOS = {
    "drawer": [
        ((-1.17, 1.86, 1.989), [("open", "kitchen_counter", "middle_slide_top"), ("close", "kitchen_counter", "middle_slide_top"), ("home",)]),
        ((-2.899, 1.86, 1.153), [("open", "kitchen_counter", "left_slide_top"), ("close", "kitchen_counter", "left_slide_top"), ("home",)]),
    ],
    "pickplace": [
        ((-2.95, -1.15, 3.1416), [("pick", "sugar_box"), ("place", -3.40, -1.25), ("home",),
                                  ("pick", "pear"), ("place", -3.40, -1.00), ("home",),
                                  ("pick", "lemon"), ("place", -3.38, -1.42), ("home",)]),
    ],
    "fridge": [
        ((0.249, 1.492, 1.107), [("open", "fridge", "top_door_hinge"), ("close", "fridge", "top_door_hinge"), ("home",)]),
    ],
}
SETTLING = {"frying_pan", "apple", "orange", "fruit_bowl"}


class WindowClosed(Exception):
    pass


def make_env(args):
    return gym.make(SCENES["kitchen"], robot_uids=f"mobile_panda_{args.base}", obs_mode="none",
                    control_mode="pd_joint_delta_pos", sim_backend="physx_cpu", reward_mode="none",
                    render_mode="human" if args.viewer else "rgb_array")


def show(w):
    """Refresh the viewer window (also processes mouse / keyboard input)."""
    viewer = w._viewer
    if viewer is not None and viewer.closed:
        raise WindowClosed
    w.render()


def run(env, start, steps, args, record=None):
    """One scenario on an existing env: park the robot at `start` (reset), run the skill steps."""
    w = env.unwrapped
    w.start = tuple(start)
    env.reset(seed=0)
    frames = []

    def on_step():
        if args.viewer:
            show(w)
        elif record is not None and skills.steps % 2 == 0:
            frames.append(w.render()[0].cpu().numpy())
    skills = ManipulationSkills(w, on_step=on_step, on_idle=(lambda: show(w)) if args.viewer else None)
    results, target = [], None
    for step in steps:
        name, *params = step
        before = {n: a.pose.p[0].numpy().copy() for n, a in w.movable_objects.items()}
        target = params[0] if name == "pick" else target
        t = time.time()
        result = skills.place(params) if name == "place" else getattr(skills, name)(*params)
        moved = {n: round(float(np.linalg.norm(a.pose.p[0].numpy() - before[n])), 3)
                 for n, a in w.movable_objects.items()
                 if n not in SETTLING and n != target and np.linalg.norm(a.pose.p[0].numpy() - before[n]) > 0.02}
        record_ = dict(step=":".join(map(str, step)), success=result.success, message=result.message,
                       disturbed=moved, seconds=round(time.time() - t, 1), info=result.info)
        results.append(record_)
        print(json.dumps({k: v for k, v in record_.items() if k != "info"}), flush=True)
    if record is not None and frames:
        import imageio
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(record), frames, fps=20)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", choices=[*SCENARIOS, "all"])
    p.add_argument("--base", choices=("wheels", "planar"), default="wheels")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--record", type=Path, help="video of the first run of the first scenario")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    report, first = [], True
    env = make_env(args)
    try:
        for name in names:
            for start, steps in SCENARIOS[name]:
                for i in range(args.repeat):
                    results = run(env, start, steps, args, record=args.record if first else None)
                    first = False
                    ok = all(r["success"] and not r["disturbed"] for r in results)
                    report.append(dict(scenario=name, start=start, run=i, ok=ok, steps=results))
        if args.viewer:
            print("done; close the viewer window to exit", flush=True)
            while True:
                show(env.unwrapped)
                time.sleep(1 / 60)
    except (WindowClosed, KeyboardInterrupt):
        print("viewer closed; stopping", flush=True)
    finally:
        env.close()
    summary = {}
    for r in report:
        for s in r["steps"]:
            key = f"{r['scenario']}/{s['step']}"
            done, total = summary.get(key, (0, 0))
            summary[key] = (done + (s["success"] and not s["disturbed"]), total + 1)
    print(json.dumps({k: f"{a}/{b}" for k, (a, b) in summary.items()}, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(summary={k: f"{a}/{b}" for k, (a, b) in summary.items()},
                                               runs=report), indent=2), encoding="utf-8")
    return 0 if all(r["ok"] for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
