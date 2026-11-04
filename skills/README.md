# 技能库

技能库分三部分：

| 部分 | 内容 | 位置 |
|---|---|---|
| 传统方法技能 | 逆运动学 + 运动规划 + 步骤脚本：抓取、放置、开关抽屉和柜门，不用训练 | [docs/manipulation.md](../docs/manipulation.md)，`catalog.json` |
| 本地技能 | 可以下载权重、在本机直接运行的技能，已在 Linux 上实测 | 本文件，`catalog.json` |
| 外部技能参考 | 网上已有的 Panda 技能、抓取位姿模型、运动规划和 VLA，附权重和许可证 | [EXTERNAL_SKILLS.md](EXTERNAL_SKILLS.md)，`external_skills.json` |
| 训练方法 | 学生自己训练技能的路线，从本平台使用的 ManiSkill 开始 | [TRAINING.md](TRAINING.md) |

权重不放在仓库里，用脚本下载（约 30 MB，Apache-2.0，见 [checkpoints/README.md](checkpoints/README.md)）：

```bash
conda activate mshab
python scripts/download_checkpoints.py            # 10 个 PPO 任务
python scripts/download_checkpoints.py PegInsertionSide-v1 --with-demos   # 只有示范轨迹的任务
```

访问不了 huggingface.co 时先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 0. 传统方法技能（住宅场景，不用训练）

`mobile_panda_sim/manipulation/skills.py`，详细说明、实测结果和适用范围见 [docs/manipulation.md](../docs/manipulation.md)。

| 技能 | 功能 | 实测 |
|---|---|---|
| `pick(name)` | 从上方抓起物体（窄边 < 7.5 cm），抬高 12 cm | 玄关台：糖盒、梨、柠檬 6/6 |
| `place((x, y))` | 把手里的物体放到 (x, y) 处的台面上 | 6/6，误差 0.1–2.3 cm |
| `open` / `close(articulation, joint)` | 抓住把手拉开或关上抽屉、柜门，够不着时重新抓 | 台柜上层抽屉：开 10/10、关 10/10；冰箱上门开 35°：5/5，关上：4/5 |
| `suggest_base_pose(articulation, joint)` | 计算底盘停在哪里能完成开或关 | — |
| `home()` | 机械臂沿无碰撞路径回到收起姿态 | — |

```bash
python scripts/demo_manipulation.py all --repeat 3 --report outputs/manipulation.json
```

## 1. 能在移动 Panda 场景里运行的技能

| ID | 功能 | 代码 | 实测（seed 0–4） |
|---|---|---|---|
| `mobile_pick_cube` | 底盘停在桌边后，抓起红块或蓝块并举到桌面上方 0.16 m | `mobile_panda_sim/skills/pick_cube.py` | 轮式底盘：红 5/5、蓝 5/5；平面底盘：红 5/5、蓝 5/5 |

```bash
python scripts/demo_mobile_pick.py                                  # 带窗口
python scripts/demo_mobile_pick.py --no-render --seeds 0 1 2 3 4 --cube blue_cube --report outputs/pick.json
```

工作原理：权重是 ManiSkill 官方 `PickCube-v1` 的 PPO，训练时 Panda 固定在 `(-0.615, 0, 0)`。
适配器把移动场景里方块、夹爪中心、目标点的位姿换算到训练坐标系，生成同样的 42 维观测，
所以底盘停在哪里都可以用，前提是方块落在训练范围内。

- **前提条件**：底盘停稳；方块在机械臂坐标系下 x = 0.615±0.08 m、y = ±0.10 m，平放在桌面上。
  `in_training_range()` 可以检查这一点；演示脚本的停车点是 `(-0.615, 方块 y, 0)`。
- **动作**：arm(7) + gripper(1)，控制模式 `pd_joint_delta_pos`，策略运行时底盘速度必须为 0。
- **成功判据**：方块被夹住且距目标点 ≤ 2.5 cm（PickCube 原判据）连续 5 步，之后继续夹持 1 s 不掉落，抓取过程中底盘漂移 < 2 cm。
- **局限**：观测用的是仿真真值位姿，不是相机估计；只会"抓起并举高"，不会放置；
  蓝块初始朝向固定为 yaw=0，所以 5 个 seed 下蓝块的场景其实相同；导航只是无障碍的点到点控制。

一个注意点：ManiSkill 的 `Pose` 乘法会统一四元数符号，而训练观测保留的是 SAPIEN 原始符号。
`q` 和 `-q` 在物理上是同一个旋转，但对网络来说是不同的输入，所以适配器手写了四元数乘法。

## 2. 只能在原任务里运行的官方 PPO 技能

这些权重在各自的原生任务中运行（固定底座 Panda，物体真值 state 观测）。
要用到移动场景，需要像 `pick_cube.py` 那样写观测适配器，场景几何也要和原任务一致；推、拉这类技能还依赖原任务里的目标标记。

```bash
python scripts/run_native_skill.py PushCube-v1 --seeds 0 1 2
python scripts/run_native_skill.py --all --report outputs/native_skills.json
```

| 任务 | 功能 | 机器人 | 观测/动作维度 | 实测（seed 0/1/2） |
|---|---|---|---|---|
| PickCube-v1 | 抓取方块举到目标点 | panda | 42 / 8 | 3/3 |
| PushCube-v1 | 把方块推到目标区域 | panda | 35 / 8 | 3/3 |
| PullCube-v1 | 把方块拉回目标区域 | panda | 35 / 8 | 3/3 |
| PokeCube-v1 | 用杆把方块戳到目标位置 | panda | 54 / 8 | 3/3 |
| PushT-v1 | 把 T 形块推到目标位姿 | panda_stick | 31 / 7 | 3/3 |
| RollBall-v1 | 把球推滚到目标区域 | panda | 44 / 8 | **2/3**（seed 0 失败） |
| LiftPegUpright-v1 | 把躺倒的长杆扶正 | panda | 32 / 8 | 3/3 |
| StackCube-v1 | 红块叠到绿块上 | panda | 48 / 8 | 3/3 |
| TwoRobotPickCube-v1 | 双臂协作抓取 | panda ×2 | 66 / 16 | 3/3 |
| TwoRobotStackCube-v1 | 双臂协作堆叠 | panda ×2 | 76 / 16 | 3/3 |

合计 29/30。判据是各任务自己的 `info["success"]`，不代表持续夹持或可以接着做下一步；
3 个 seed 只能说明接入正确，不能当作成功率。每个任务还有 `pd_ee_delta_pos`、`pd_ee_delta_pose` 两种控制模式的权重，
平台统一使用 `pd_joint_delta_pos`，所以没有单独列出。

## 3. 只有示范轨迹的任务（没有策略）

| 任务 | 功能 | 用途 |
|---|---|---|
| PegInsertionSide-v1 | 侧向销钉插孔 | 模仿学习数据 |
| PlugCharger-v1 | 插充电器 | 模仿学习数据 |
| PullCubeTool-v1 | 用 L 形工具勾回方块 | 模仿学习数据 |

这些任务只有运动规划生成的轨迹（`--with-demos` 下载），没有策略网络，不能从新的初始状态完成任务。
可以用 ManiSkill 的 BC / Diffusion Policy 基线训练成技能，见 [TRAINING.md](TRAINING.md)。

没有收录的：`DrawTriangle-v1`、`StackPyramid-v1`（平台固定的 ManiSkill mshab 分支里没有这两个任务）；
`AnymalC-Reach-v1`（四足机器人任务）。

## catalog.json 字段

- `kind`：`ppo_adapted`（已适配到移动场景）、`ppo`（仅原任务）、`demonstrations_only`
- `runs_in`：`mobile_scene` 或 `native_task`
- `validated`：本仓库的实测结果、判据和运行环境
