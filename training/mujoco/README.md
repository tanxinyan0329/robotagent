# MuJoCo 上的强化学习

`train_pick.py` 在 MuJoCo 桌面场景里训练抓取策略：抓起方块，送到空中的目标点。只依赖 `mujoco`、`numpy`、`torch`，
物理在 CPU 上多线程并行，网络在 GPU 上（没有 GPU 时用 CPU）。

```bash
python training/mujoco/train_pick.py --name pick --total-steps 10000000     # 权重在 training/mujoco/runs/pick/
python training/mujoco/train_pick.py --evaluate skills/checkpoints/mujoco/pick_cube.pt --episodes 500
python scripts/demo_mobile_pick.py --backend mujoco                         # 底盘开到桌边，再用策略抓取
```

都在装了 `requirements-mujoco.txt` 的环境里运行。仓库里带了一份训练好的权重 `skills/checkpoints/mujoco/pick_cube.pt`。

## 任务

代码在 `mobile_panda_sim/mj/pick_task.py`，定义和 ManiSkill 的 `PickCube-v1` 相同，只有最后一项不同：

| 项目 | 内容 |
|---|---|
| 观测 | 42 维：关节位置和速度、是否夹住、夹爪中心位姿、目标点、方块位姿、两个相对位置 |
| 动作 | 8 维：7 个关节增量（±0.1 rad）+ 夹爪 |
| 奖励 | 靠近方块 + 夹住 + 夹住时靠近目标 + 到位后保持静止，成功时取最大值 |
| 成功 | 方块距目标不超过 2.5 cm，且机械臂基本静止 |
| 回合 | 最多 50 步，方块随机放在 ±10 cm 内，目标随机在上方 |
| "夹住"的判定 | 两根手指都压在方块上，且指间距不小于 3 cm。ManiSkill 判断的是手指受力方向 |

网络结构、超参数和 ManiSkill 官方 PickCube 基线相同。权重的 `actor_mean.*` 键也相同，所以 ManiSkill 的权重可以直接用 `--evaluate` 评估。

## 实测（Ubuntu 22.04，MuJoCo 3.5.0，RTX 5070 Ti Laptop，32 线程 CPU，2026-11-05）

| 项目 | 结果 |
|---|---|
| 训练 1000 万步 | 约 45 分钟（约 3,600 步/秒） |
| 训练好的权重，500 回合 | 97.6% |
| ManiSkill 权重直接用，500 回合 | 56.4% |
| 移动抓取：两种底盘 × 两个方块 × 20 个 seed | 80/80，底盘漂移最大 1.2 mm |

成功率到 300 万步约 80%，500 万步以后在 91–99% 之间波动，所以用的是训练中评估最好的 `best_ckpt.pt`。最后一步的 `final_ckpt.pt` 在同样的 500 回合上是 90.2%。这是一个 seed 的结果，换 seed 可能不同。

移动抓取里方块的位置是固定的，种子只改变红块朝向，蓝块朝向固定，所以蓝块的 40 次其实是同一个场景。方块随机位置的能力看 500 回合那一项。

## 训练时踩过的坑

这两个问题都没有报错，只表现为成功率上不去：

1. **奖励被刷分。** 最初"夹住"只要求两根手指都碰到方块。策略学会了把合拢的指尖压在方块顶上，一直拿这份奖励，成功率停在 0。看奖励曲线是平的，要回放策略才看得出来。
2. **成功后价值算成 0。** 成功时回合结束。如果这时不估计后续价值，一直悬停在目标附近、不完全静止，累计奖励反而比成功一次更高，策略会主动放弃成功：成功率从 43% 掉到 0，奖励却还在涨。解决办法是所有回合结束时都用价值网络估计后续回报，ManiSkill 的 PPO 也是这样做的。

奖励上升而成功率下降，通常说明奖励和目标不一致。

## 能抓什么

只为 4 cm 方块训练。放到厨房玄关台上试过两种没见过的物品，各 9 个停靠位：柠檬 9/9；梨 0/9，会碰倒旁边的盒子，挪走盒子后也夹不牢。抓住宅场景的物品请用 `pick`。

## 换成自己的任务

复制 `pick_task.py`，改 `reset_one`（初始状态）、`observe`（观测）、`reward`（奖励和成功判据）。场景可以用 `build(...)` 的 `extend` 参数加物体。`train_pick.py` 只用到 `reset`、`observe`、`step` 这几个接口。

ManiSkill 的其他 PPO 任务（推、拉、堆叠等）没有迁移，仍然在 [training/ppo](../ppo/README.md)。
