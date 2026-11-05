# 技能库

| 部分 | 内容 | 位置 |
|---|---|---|
| 传统方法技能 | 逆运动学 + 运动规划 + 步骤脚本：抓取、放置、开关抽屉和柜门，不用训练 | [docs/manipulation.md](../docs/manipulation.md) |
| 强化学习技能 | 拿起并移动、放下，权重在 `skills/policies/` | [training/README.md](../training/README.md) |
| 外部技能参考 | 网上已有的 Panda 技能、抓取位姿模型、运动规划和 VLA，附权重和许可证 | [EXTERNAL_SKILLS.md](EXTERNAL_SKILLS.md) |

机器可读的目录：`catalog.json`、`external_skills.json`。

所有技能都返回 `SkillResult`：`success`、`message`（失败原因）、`info`（步数等细节），可以直接用在 `if` 里。

要抓放物品、开关抽屉和柜门，先用传统方法技能。强化学习技能用来学习和对比训练方法。

## 适用范围

选技能前先看这张表。所有技能都读仿真真值，不经过相机；执行时底盘必须停稳。

| 技能 | 能做 | 不能做 | 实测 |
|---|---|---|---|
| `pick` | 从上方抓窄边小于 7.5 cm、放在够得着的台面上的物体 | 从侧面抓；抓抽屉、冰箱里的东西；台面靠墙一侧太远的物品 | 糖盒、梨、柠檬 15/15 |
| `place` | 把手中物体放到平整台面上 | 放进抽屉或冰箱；放到倾斜或很小的表面 | 15/15 |
| `open` / `close` | 台柜上层左、中抽屉；冰箱上门开到约 35° 再关上 | 台柜右上和所有下层抽屉、冰箱下门、矮柜推拉门（把手低于约 0.55 m）；从 70° 关冰箱门 | 开 15/15，关 15/15 |
| `suggest_base_pose` | 选一个开和关都够得着的停靠位 | 不考虑底盘受力，假设底盘不动 | 两个抽屉和冰箱门，开 12/12，关 12/12 |
| `home` | 手臂沿无碰撞路径回到收起姿态 | 手臂被卡住时 | 每轮都用到 |
| 拿起并移动（`RLSkills.pick_move`） | 玄关台上的柠檬、梨、汤罐，底盘停在 `DOCK`，物体在工作区内 | 其他物体；其他台面；其他停靠位 | 500 回合 83.6%（柠檬 88%、梨 93%、汤罐 70%） |
| 放下（`RLSkills.place`） | 把这三种物体放到玄关台工作区内的指定位置 | 其他物体和其他台面 | 500 回合 86.6%（柠檬 86%、梨 90%、汤罐 83%） |

要抓住宅场景里的物品，用 `pick` 和 `place`，它们更通用，成功率也更高。强化学习技能的用途是教学和训练例程：它们只为这三种物体和这个台面训练，不要指望换物体也能用。

## 传统方法技能

代码在 `mobile_panda_sim/manipulation/skills.py`，用法和实测见 [docs/manipulation.md](../docs/manipulation.md)。

```bash
.venv/bin/python scripts/demo_manipulation.py all --viewer
```

## 强化学习技能

```bash
.venv/bin/python scripts/demo_rl_skills.py --object pear                # 开到玄关台，拿起，移到指定位置，放下
.venv/bin/python scripts/demo_rl_skills.py --no-viewer --episodes 20
```

```python
from mobile_panda_sim.mj.rl_skills import RLSkills
from mobile_panda_sim.mj.rl_tasks import DOCK, TABLE_Z
from mobile_panda_sim.mj.sim import MujocoScene
import numpy as np

scene = MujocoScene("wheels", DOCK)
scene.reset()
skills = RLSkills(scene)
if skills.pick_move("pear", np.array([-3.40, -1.50, TABLE_Z + 0.15])):   # 目标是物体中心的世界坐标
    skills.place((-3.34, -1.56))                                         # 台面上的 (x, y)
```

轮式底盘从充电位出发，导航、拿起、移动、放下全流程的实测，每个物体 20 次，物体和目标位置随机：

| 物体 | 拿起并移动 | 全流程 | 碰动旁边物品的回合 |
|---|---|---|---|
| 柠檬 | 19/20 | 17/20 | 0 |
| 梨 | 19/20 | 16/20 | 1 |
| 汤罐 | 15/20 | 11/20 | 3 |

这是一个训练种子的结果。汤罐最不稳定，训练时成功率波动也较大。

## catalog.json 字段

- `kind`：`classical`（传统方法）或 `rl`（强化学习）
- `call`、`code`、`demo`、`docs`：怎么调用、代码、例程、说明
- `checkpoint`、`training`：权重和训练脚本，只有强化学习技能有
- `applies_to`、`not_for`、`preconditions`：能做的事、不能做的事、前提条件
- `validated`：实测结果、判据和运行环境

`tests/test_catalog.py` 检查目录和代码是否一致，新增技能时要同步更新。
