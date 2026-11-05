"""PPO for the pick-and-carry task on MuJoCo: physics in parallel on the CPU, the network on the GPU if there is one.

Same network and hyperparameters as ManiSkill's PickCube-v1 baseline; the checkpoint has the same actor_mean.* keys.

    python training/mujoco/train_pick.py --total-steps 10000000 --name pick
    python training/mujoco/train_pick.py --evaluate skills/checkpoints/mujoco/pick_cube.pt --episodes 500
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mobile_panda_sim.mj.pick_task import ACTION_SIZE, MAX_STEPS, OBS_SIZE, PickCubeTask  # noqa: E402


def layer(in_size, out_size, std=np.sqrt(2)):
    linear = nn.Linear(in_size, out_size)
    nn.init.orthogonal_(linear.weight, std)
    nn.init.zeros_(linear.bias)
    return linear


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(layer(OBS_SIZE, 256), nn.Tanh(), layer(256, 256), nn.Tanh(), layer(256, 256), nn.Tanh(), layer(256, 1))
        self.actor_mean = nn.Sequential(layer(OBS_SIZE, 256), nn.Tanh(), layer(256, 256), nn.Tanh(), layer(256, 256), nn.Tanh(),
                                        layer(256, ACTION_SIZE, 0.01*np.sqrt(2)))
        self.actor_logstd = nn.Parameter(torch.ones(1, ACTION_SIZE) * -0.5)

    def distribution(self, obs):
        return torch.distributions.Normal(self.actor_mean(obs), torch.exp(self.actor_logstd))


@torch.no_grad()
def evaluate(actor, device, episodes, seed):
    """Share of episodes that succeed within the step limit, with the deterministic policy."""
    task = PickCubeTask(episodes, seed=seed)
    obs = task.observe()
    won = np.zeros(episodes, dtype=bool)
    for _ in range(MAX_STEPS):
        action = actor(torch.from_numpy(obs).to(device)).clamp(-1, 1).cpu().numpy()
        obs, _, _, _, _, success = task.step(action)
        won |= success
    task.close()
    return float(won.mean())


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    run = Path(args.out) / args.name
    run.mkdir(parents=True, exist_ok=True)
    task = PickCubeTask(args.envs, seed=args.seed)
    agent = Agent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    n, length = args.envs, args.rollout
    observations = torch.zeros((length, n, OBS_SIZE), device=device)
    actions = torch.zeros((length, n, ACTION_SIZE), device=device)
    log_probs, rewards, values, ends, boots = (torch.zeros((length, n), device=device) for _ in range(5))
    obs = torch.from_numpy(task.observe()).to(device)
    steps, start, best = 0, time.time(), -1.0
    log = open(run / "log.jsonl", "a")
    iteration = 0
    while steps < args.total_steps:
        iteration += 1
        for t in range(length):
            with torch.no_grad():
                dist = agent.distribution(obs)
                action = dist.sample()
                log_probs[t] = dist.log_prob(action).sum(1)
                values[t] = agent.critic(obs).squeeze(1)
            observations[t], actions[t] = obs, action
            next_obs, reward, terminated, truncated, final_obs, _ = task.step(action.clamp(-1, 1).cpu().numpy())
            rewards[t] = torch.from_numpy(reward).float().to(device)
            done = terminated | truncated
            ends[t] = torch.from_numpy(done.astype(np.float32)).to(device)
            boots[t] = 0
            if done.any():
                # Also after success: valued at zero, hovering by the goal pays more than finishing.
                with torch.no_grad():
                    last = torch.from_numpy(final_obs[done]).to(device)
                    boots[t, torch.from_numpy(done).to(device)] = agent.critic(last).squeeze(1)
            obs = torch.from_numpy(next_obs).to(device)
        steps += n * length
        with torch.no_grad():
            advantage, next_value = torch.zeros(n, device=device), agent.critic(obs).squeeze(1)
            advantages = torch.zeros((length, n), device=device)
            for t in reversed(range(length)):
                keep = 1 - ends[t]
                bootstrap = boots[t] + keep * next_value
                delta = rewards[t] + args.gamma * bootstrap - values[t]
                advantage = delta + args.gamma * args.gae_lambda * keep * advantage
                advantages[t], next_value = advantage, values[t]
            returns = advantages + values
        flat = [x.reshape((n * length,) + x.shape[2:]) for x in (observations, actions, log_probs, advantages, returns)]
        b_obs, b_actions, b_log_probs, b_advantages, b_returns = flat
        minibatch = n * length // args.minibatches
        for _ in range(args.epochs):
            order = torch.randperm(n * length, device=device)
            stop = False
            for i in range(0, n * length, minibatch):
                index = order[i:i + minibatch]
                dist = agent.distribution(b_obs[index])
                new_log_prob = dist.log_prob(b_actions[index]).sum(1)
                ratio = (new_log_prob - b_log_probs[index]).exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - (new_log_prob - b_log_probs[index])).mean()
                if kl > args.target_kl:
                    stop = True
                    break
                adv = b_advantages[index]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(0.8, 1.2)).mean()
                value_loss = 0.5 * ((agent.critic(b_obs[index]).squeeze(1) - b_returns[index]) ** 2).mean()
                loss = policy_loss + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
            if stop:
                break
        if iteration % args.eval_every == 0 or steps >= args.total_steps:
            rate = evaluate(agent.actor_mean, device, args.eval_episodes, args.seed + iteration)
            record = dict(steps=steps, success=round(rate, 4), reward=round(float(rewards.mean()), 4), seconds=round(time.time() - start),
                          steps_per_second=round(steps / (time.time() - start)))
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if rate >= best:
                best = rate
                torch.save(agent.state_dict(), run / "best_ckpt.pt")
    torch.save(agent.state_dict(), run / "final_ckpt.pt")
    task.close()
    print(f"best success {best:.3f}; checkpoints in {run}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="pick")
    p.add_argument("--out", default="training/mujoco/runs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--rollout", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--gae-lambda", type=float, default=0.9)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--target-kl", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--evaluate", type=Path)
    p.add_argument("--episodes", type=int, default=500)
    args = p.parse_args()
    if args.evaluate:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = Agent().to(device)
        agent.load_state_dict(torch.load(args.evaluate, map_location=device, weights_only=True), strict=False)
        print(json.dumps(dict(checkpoint=str(args.evaluate), episodes=args.episodes,
                              success=evaluate(agent.actor_mean, device, args.episodes, args.seed))))
    else:
        train(args)


if __name__ == "__main__":
    main()
