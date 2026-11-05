# MuJoCo 后端

ROS 话题、消息格式和 ManiSkill 后端相同，学生的 Agent 不用改。切换只需要一个 launch 参数：

```bash
python3 scripts/download_menagerie.py        # 只需一次：下载 Panda 模型（Apache-2.0，不在仓库里）
~/anaconda3/envs/mshab/bin/python scripts/export_kitchen_mujoco.py    # 只需一次：导出厨房，要在 mshab 环境里运行
ros2 launch mobile_panda_bridge simulation.launch.py backend:=mujoco                  # 住宅场景
ros2 launch mobile_panda_bridge simulation.launch.py backend:=mujoco scene:=tabletop  # 桌面场景
```

`sim_python` 默认是 `~/anaconda3/envs/llm/bin/python`。换别的环境时用 `sim_python:=...` 指定，需要 Python 3.10+ 和 `mujoco`（在 3.5.0 上测试）。操作技能另外需要 `mplib==0.1.1`。

目前进度：

| 内容 | 状态 |
|---|---|
| 机器人模型、桌面场景、两种底盘 | 完成 |
| ROS 接口（`teleop_check`）、窗口、两个 RGB 相机 | 完成 |
| 住宅场景：家具、42 个可动物体、台柜、冰箱、矮柜 | 完成 |
| `manipulation/` 操作技能（`pick`、`place`、`open`、`close`、`home`、`suggest_base_pose`） | 完成，和 ManiSkill 共用同一份代码 |
| 强化学习训练和预训练策略 | 未迁移，仍只能在 ManiSkill 上用 |

## 模型

代码在 `mobile_panda_sim/mj/`：`model.py` 组装机器人和桌面场景，`kitchen.py` 加载住宅场景，`sim.py` 负责仿真步进和渲染，`world.py` 是技能用的仿真接口。

- **底盘**直接读 `mobile_panda_{wheels,planar}.urdf`，和 ManiSkill 用同一份尺寸和质量。
- **机械臂和夹爪**来自 MuJoCo Menagerie 的 `franka_emika_panda`，下载时固定了版本，加载时改成 URDF 的连杆和关节名。
- **控制**：物理步长 2 ms，控制周期 50 ms。`arm_joint_delta` 相对测量值，和 ManiSkill 一样；没有新命令时保持当前目标。

和 ManiSkill 模型有意不同的地方，每一处都是测试时发现问题后改的：

| 改动 | 原因 |
|---|---|
| 重力补偿经关节执行器施加 | 手臂不下垂，但重量仍压在底盘上。ManiSkill 里手臂不受重力 |
| 支撑球碰撞半径 4 cm（URDF 是 3.5 cm） | URDF 的支撑球离地 5 mm，车体只靠两个轮子立着，会前后摆动约 ±1.2°，末端跟着晃 2 cm |
| 轮子碰撞体用椭球，外观仍是圆柱 | 圆柱和地面有多个接触点，原地转向时刮地，转向量只有指令的 60% |
| 夹爪增益改成每指 1000 N/m、100 N·s/m | 和 ManiSkill 一致。Menagerie 默认值的夹持力不到 2 N，抓方块够用，拉抽屉会滑脱 |

## 住宅场景

场景不是手写的。`scripts/export_kitchen_mujoco.py` 加载运行中的 ManiSkill 场景，把几何、位姿、质量、关节、把手和贴图导出到 `assets/mujoco_kitchen/`（约 18 MB，不进 git，ReplicaCAD 数据仅限非商业用途）。厨房改了就重新导出一次。

- **碰撞体**：柜体的方盒和胶囊、可动物体的凸分解都直接沿用。12 件固定家具原来是非凸网格，MuJoCo 不支持，现在按连通块处理：凸块用凸包；明显非凸的闭合块（椅子、沙发靠背、桌子侧框）用 2 cm 体素合并成方盒，所以表面有约 1 cm 的台阶；不闭合的块（绿植、垃圾桶）仍用凸包，镂空处是实心的。
- **沉降**：有些物品一开始就和家具重叠（煎锅嵌进台面约 8 cm，ManiSkill 里也一样）。加载时先带阻尼沉降，把结果作为重置位姿，否则物品会被弹开。
- **稳定性**：开启多接触点（`multiccd`）、给可动物体加阻尼、开启休眠（阈值 0.01）。没有这些，平放的物品会摇晃并慢慢移动；开启后物品静止时完全不动，手碰到会立刻醒来，场景约 25 倍实时。
- **加载时间**：约 2.4 秒，含沉降。

## 测试

```bash
~/anaconda3/envs/llm/bin/python -m pytest tests/test_mujoco.py -q
~/anaconda3/envs/llm/bin/python scripts/mj_contact_check.py --seeds 0 1 2 3 4
```

`test_mujoco.py` 把模型和 ManiSkill 的实测值、URDF 正运动学逐项对比，并检查厨房的关节范围、物品稳定性和重置。`mj_contact_check.py` 验证接触物理：桌面上抓方块、开关一个测试抽屉，再到厨房开关真实的抽屉和冰箱门、抓玄关台的物品。它用带碰撞检查的简化 IK，不是正式技能。正式技能的回归是 `scripts/demo_manipulation.py all --backend mujoco --repeat 5`。

## 实测（Ubuntu 22.04，llm 环境，MuJoCo 3.5.0，2026-11-05）

| 项目 | 结果 |
|---|---|
| 正式技能（抓取、放置、开关抽屉和冰箱门） | 轮式底盘 5 轮 90/90，平面底盘 2 轮 36/36，详见 [manipulation.md](manipulation.md) |
| 抓方块（抬高 12 cm，保持 3 s） | 10/10，末端跟踪误差约 0.2 mm，滑移最大 3.8 mm |
| 抽屉：拉开 25 cm 再关回 1 cm | 5/5，拉到 24.9 cm，关到约 1 cm，底盘漂移约 3 mm |
| 厨房：抓糖盒、梨、柠檬 | 3/3，抬高 11.6–12.0 cm，没有碰动其他物体 |
| 厨房：台柜中间、左侧上层抽屉，开 25 cm 再关 | 2/2，开到 26.3 / 28.3 cm（含 2 cm 过冲和夹爪滑动），关到 1.2 / 2.1 cm |
| 厨房：冰箱上门，开 0.6 rad 再关 | 1/1，开到 0.63 rad，关到 0，底盘漂移 3.7 mm |
| 厨房物品静止 60 秒 | 全部移动不超过 0.4 mm |
| 厨房 `/reset` | 通过 |
| `teleop_check`（两种场景，轮式、平面底盘） | 通过 |
| 静止时 TCP 位姿、相机取景和内参 | 和 ManiSkill 一致 |

仿真本身是确定性的，5 个种子只改变红方块朝向和 IK 随机起点，抽屉测试的实际样本少于 5 次独立实验。

## 已知差异

- 画面没有 SAPIEN 的渲染质量：材质更简单，桌面场景地面是棋盘格，厨房没有明显投影。取景相同，但视觉算法不能直接假定两边效果一致。
- `teleop_check` 的前进量比 ManiSkill 大约 5%，转向量大约 10–15%。两者都没超过指令加 0.3 s 超时的上限（前进 0.345 m，转向 0.92 rad），应该是响应更快、停车前多走了一点，还没有逐项对比确认。
- 厨房物品沉降后和 ManiSkill 的位置略有差别：高度最多差 7 mm，水平大多在 1 cm 内，遥控器最大 2.7 cm。
- 在玄关台停靠位 (-2.95, -1.15, π) 上，收起姿态的手会碰到高盒子。ManiSkill 里也是这个姿态。
- 底盘比 ManiSkill 稳：拉抽屉时基本不动，ManiSkill 会被带着移动约 3 cm。依赖这点选的停靠位不能直接搬过来，见 [manipulation.md](manipulation.md)。
- 夹爪压紧时接触略软：夹 4 cm 方块时指间距 3.85 cm，夹 1.8 cm 横杆时 1.69 cm。
