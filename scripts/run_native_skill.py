"""Run an official ManiSkill PPO checkpoint in its ORIGINAL task (fixed Panda, native scene).

This checks that a downloaded skill works and shows what it does. It is not the mobile scene: to use
a skill on the mobile Panda its observation must be adapted (see mobile_panda_sim/skills/pick_cube.py).

    python scripts/run_native_skill.py PushCube-v1 --seeds 0 1 2
    python scripts/run_native_skill.py --all --report outputs/native_skills.json
    python scripts/run_native_skill.py PickCube-v1 --record outputs/pickcube.mp4
    python scripts/run_native_skill.py PushCube-v1 --checkpoint training/ppo/runs/pushcube/final_ckpt.pt
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
CKPT_ROOT = REPO / "skills/checkpoints/maniskill"


def actor_from_checkpoint(path):
    import torch
    from torch import nn
    state = torch.load(path, map_location="cpu", weights_only=True)
    weights = [k for k in state if k.startswith("actor_mean.") and k.endswith(".weight")]
    weights.sort(key=lambda k: int(k.split(".")[1]))
    layers = []
    for i, key in enumerate(weights):
        out_dim, in_dim = state[key].shape
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(weights) - 1:
            layers.append(nn.Tanh())
    actor = nn.Sequential(*layers)
    actor.load_state_dict({k[len("actor_mean."):]: v for k, v in state.items() if k.startswith("actor_mean.")})
    return actor.eval(), state[weights[0]].shape[1], state[weights[-1]].shape[0]


def run(task, seeds, record=None, max_steps=None, checkpoint=None):
    import gymnasium as gym
    import torch
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

    ckpt = Path(checkpoint).resolve() if checkpoint else CKPT_ROOT / task / "rl/ppo_pd_joint_delta_pos_ckpt.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} missing; run python scripts/download_checkpoints.py {task}")
    actor, obs_dim, act_dim = actor_from_checkpoint(ckpt)
    env = gym.make(task, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu",
                   render_mode="rgb_array" if record else None, num_envs=1)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    results, frames = [], []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            if obs.shape[-1] != obs_dim:
                raise RuntimeError(f"{task}: env observation {obs.shape[-1]} != checkpoint input {obs_dim}")
            success, steps = False, 0
            limit = max_steps or env.unwrapped.spec.max_episode_steps or 100
            for steps in range(1, limit + 1):
                with torch.no_grad():
                    action = actor(obs.float().cpu()).clamp(-1, 1)
                obs, _, term, trunc, info = env.step(action)
                if record:
                    frames.append(env.render()[0].cpu().numpy())
                if bool(info.get("success", torch.tensor([False]))[0]):
                    success = True
                    break
                if bool(term[0]) or bool(trunc[0]):
                    break
            results.append(dict(seed=seed, success=success, steps=steps))
            print(json.dumps(dict(task=task, **results[-1])), flush=True)
    finally:
        env.close()
    if record and frames:
        import imageio
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(record, frames, fps=20)
    return dict(task=task, checkpoint=str(ckpt.relative_to(REPO) if ckpt.is_relative_to(REPO) else ckpt), obs_dim=obs_dim, act_dim=act_dim,
                successes=sum(r["success"] for r in results), episodes=len(results), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?")
    parser.add_argument("--all", action="store_true", help="every task with a downloaded PPO checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--checkpoint", type=Path, help="your own PPO checkpoint (e.g. training/ppo/runs/<exp>/final_ckpt.pt)")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.all and not args.task:
        parser.error("give a task id or --all")
    tasks = sorted(p.parents[1].name for p in CKPT_ROOT.glob("*/rl/ppo_pd_joint_delta_pos_ckpt.pt")) if args.all else [args.task]
    reports = []
    for task in tasks:
        try:
            reports.append(run(task, args.seeds, args.record if len(tasks) == 1 else None,
                               checkpoint=args.checkpoint if len(tasks) == 1 else None))
        except Exception as error:
            reports.append(dict(task=task, error=f"{type(error).__name__}: {error}"))
            print(json.dumps(reports[-1]), file=sys.stderr, flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps([{k: r.get(k) for k in ("task", "successes", "episodes", "error")} for r in reports]))
    return 0 if all("error" not in r for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
