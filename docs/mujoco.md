# MuJoCo 后端

ROS 话题、消息格式和 ManiSkill 后端相同，学生的 Agent 不用改。切换只需要一个 launch 参数：

```bash
python3 scripts/download_menagerie.py        # 只需一次：下载 Panda 模型（Apache-2.0，不在仓库里）
ros2 launch mobile_panda_bridge simulation.launch.py backend:=mujoco scene:=tabletop
```

`sim_python` 默认是 `~/anaconda3/envs/llm/bin/python`。换别的环境时用 `sim_python:=...` 指定，需要 Python 3.10+ 和 `mujoco`（在 3.5.0 上测试）。

目前进度：

| 内容 | 状态 |
|---|---|
| 机器人模型、桌面场景、两种底盘 | 完成 |
| ROS 接口（`teleop_check`）、窗口、两个 RGB 相机 | 完成 |
| 住宅场景 | 未迁移；`backend:=mujoco` 不加 `scene:=tabletop` 会报错 |
| `manipulation/` 技能、强化学习训练和预训练策略 | 未迁移，仍只能在 ManiSkill 上用 |

## 模型

代码在 `mobile_panda_sim/mj/`：`model.py` 组装模型，`sim.py` 负责仿真步进和渲染。

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

## 测试

```bash
~/anaconda3/envs/llm/bin/python -m pytest tests/test_mujoco.py -q
~/anaconda3/envs/llm/bin/python scripts/mj_contact_check.py --seeds 0 1 2 3 4
```

`test_mujoco.py` 把模型和 ManiSkill 的实测值、URDF 正运动学逐项对比。`mj_contact_check.py` 验证接触物理：抓起两个方块，开关一个测试抽屉。抽屉是临时搭的，把手尺寸和关节阻尼取自厨房，不是真实柜体模型。

## 实测（Ubuntu 22.04，llm 环境，MuJoCo 3.5.0，2026-11-04）

| 项目 | 结果 |
|---|---|
| 抓方块（抬高 12 cm，保持 3 s） | 10/10，末端跟踪误差约 0.2 mm，滑移最大 3.8 mm |
| 抽屉：拉开 25 cm 再关回 1 cm | 5/5，拉到 24.9 cm，关到约 1 cm，底盘漂移约 3 mm |
| `teleop_check`（轮式、平面底盘） | 通过 |
| 静止时 TCP 位姿、相机取景和内参 | 和 ManiSkill 一致 |

仿真本身是确定性的，5 个种子只改变红方块朝向和 IK 随机起点，抽屉测试的实际样本少于 5 次独立实验。

## 已知差异

- 画面没有 SAPIEN 的渲染质量：材质更简单，地面是棋盘格。取景相同，但视觉算法不能直接假定两边效果一致。
- `teleop_check` 的前进量比 ManiSkill 大约 5%，转向量大约 10–15%。两者都没超过指令加 0.3 s 超时的上限（前进 0.345 m，转向 0.92 rad），应该是响应更快、停车前多走了一点，还没有逐项对比确认。
- 夹爪压紧时接触略软：夹 4 cm 方块时指间距 3.85 cm，夹 1.8 cm 横杆时 1.69 cm。
