# 移动 Panda 仿真 ROS2 接口

在 ManiSkill 中加载一台**差速底盘 + Franka Panda** 的移动机械臂和一个住宅场景（厨房 + 餐厅 + 客厅），
通过 ROS2 开放控制和感知接口，供学生编写 Agent。附带技能库：传统方法的抓取 / 放置 / 开关柜门、预训练技能、外部技能参考和训练方法。

## 交付内容

| 内容 | 位置 |
|---|---|
| 机器人 URDF 与网格（与仿真加载的是同一份文件） | `ros2_ws/src/mobile_panda_description/urdf/mobile_panda_{wheels,planar}.urdf` |
| launch：只加载场景并建立话题 | `ros2_ws/src/mobile_panda_bridge/launch/simulation.launch.py` |
| ROS2 桥接节点、示例客户端 | `ros2_ws/src/mobile_panda_bridge/` |
| 仿真场景、机器人定义 | `mobile_panda_sim/` |
| 接口说明 | [docs/ros2_interface.md](docs/ros2_interface.md) |
| 技能库 | [skills/README.md](skills/README.md) |

主要话题（命名空间 `/mobile_panda`）：

| 方向 | 话题 | 类型 |
|---|---|---|
| 机械臂控制 | `/mobile_panda/arm_joint_delta` | `std_msgs/Float64MultiArray`（7 个关节增量，rad） |
| 夹爪控制 | `/mobile_panda/gripper` | `std_msgs/Float64`（每指开口，m） |
| 底盘速度控制 | `/mobile_panda/cmd_vel` | `geometry_msgs/Twist`（`linear.x`、`angular.z`） |
| RGB 图像 | `/mobile_panda/base_camera/rgb`、`/mobile_panda/hand_camera/rgb` | `sensor_msgs/Image`（rgb8） |
| 底盘位姿 | `/mobile_panda/base_pose` | `geometry_msgs/PoseStamped`（`map` 坐标系真值） |
| 关节状态 | `/mobile_panda/joint_states` | `sensor_msgs/JointState` |
| 机器人模型、TF | `/robot_description`、`/tf`、`/tf_static` | URDF / TF 树 |

另有相机内参、状态 JSON、场景布局、stop/reset 服务，见接口说明。

launch 启动后加载场景、发布话题，然后等待命令：**不执行任务，也不推进物理时间**，直到收到控制命令。

## 场景

launch 参数 `scene` 选择场景：

| `scene` | 内容 |
|---|---|
| `kitchen`（默认） | 约 8 m × 6 m 的住宅：带抽屉的厨房台柜、冰箱、餐桌椅、储物架、玄关台、沙发茶几等。42 个可移动物品（真实质量），台柜 7 个抽屉、冰箱 2 扇门、矮柜 2 扇推拉门可开合。地面 z=0，机器人从玄关的蓝色充电垫 `(-2.55, -1.95)` 出发，朝北 |
| `tabletop` | 技能测试用的桌面：桌面 z=0、地面 z=-0.92 m，红、蓝两个 4 cm 方块和两个放置区，起点 `(-1.15, -0.30)` |

![住宅场景](docs/kitchen_scene.png)

场景没有任务逻辑和成功判据，任务由 Agent 或任务书定义。物品和家具的位置、抽屉和门的开合程度在 `/mobile_panda/state`，
家具占地范围在 `/mobile_panda/scene_info`。起点可以用 `start_x`、`start_y`、`start_yaw` 改。

两种底盘（launch 参数 `base`）：

| `base` | 底盘运动方式 |
|---|---|
| `wheels`（默认） | 自由根 + 左右驱动轮，靠轮胎摩擦行驶；前后为低摩擦支撑球 |
| `planar` | 根固定，x/y/yaw 虚拟关节；轮子只是外观 |

## 环境要求

- Ubuntu 22.04 + ROS2 Humble（系统 Python 3.10）
- Conda 环境 `mshab`：与 [mshab-agent](https://github.com/tanxinyan0329/mshab-agent) 相同（Python 3.9，ManiSkill mshab 分支 3.0.0b18，PyTorch）。
  本仓库**不需要额外的 Conda 环境，也不需要 pip install**；仿真进程通过 `PYTHONPATH` 加载 `mobile_panda_sim/`。
- 仿真使用 CPU 物理（`physx_cpu`），渲染需要支持 Vulkan 的显卡。

ROS2 节点运行在系统 Python 上，不导入 Torch；仿真进程运行在 Conda Python 上，不导入 rclpy。两者通过进程管道通信。

## 安装

```bash
git clone <本仓库地址> mobile-panda-agent
cd mobile-panda-agent

# 1. Conda 环境 mshab：按 mshab-agent 的 README 安装（ManiSkill mshab 分支，固定提交 17121e3f）
#    住宅场景需要 ReplicaCAD 家具和 YCB 物品（装过 mshab-agent 的机器上已经有了，在 ~/.maniskill/data）
conda activate mshab
python -m mani_skill.utils.download_asset ReplicaCAD -y
python -m mani_skill.utils.download_asset ycb -y
conda deactivate

# 2. ROS2 工作区
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --symlink-install && cd ..

# 3. 技能权重（约 30 MB，只有运行技能时需要）
conda activate mshab
python scripts/download_checkpoints.py
```

访问不了 huggingface.co 时先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 启动

在仓库根目录：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch mobile_panda_bridge simulation.launch.py              # 住宅场景，打开 SAPIEN 窗口
ros2 launch mobile_panda_bridge simulation.launch.py viewer:=false rviz:=true
ros2 launch mobile_panda_bridge simulation.launch.py scene:=tabletop base:=planar seed:=3
ros2 launch mobile_panda_bridge simulation.launch.py backend:=mujoco scene:=tabletop   # MuJoCo，见 docs/mujoco.md
```

不在仓库根目录启动时，加 `project_root:=/path/to/mobile-panda-agent`，或设置环境变量 `MOBILE_PANDA_ROOT`。
Conda 解释器默认是 `~/anaconda3/envs/mshab/bin/python`（`backend:=mujoco` 时是 `llm` 环境），不同时用 `sim_python:=...` 或 `MOBILE_PANDA_SIM_PYTHON` 指定。

检查接口（另开终端，launch 运行中）：

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 run mobile_panda_bridge teleop_check
```

它会接收所有传感器话题，然后让底盘前进、转向，动一个关节，开合夹爪，并检查位姿和关节状态有没有相应变化。

只看 URDF（不启动仿真）：`ros2 launch mobile_panda_description display.launch.py base:=wheels`

## 技能库

详见 [skills/README.md](skills/README.md)。概要：

| 类别 | 内容 | 实测 |
|---|---|---|
| 传统方法技能 | 住宅场景里的抓取、放置、开关抽屉和冰箱门（逆运动学 + 运动规划，不用训练），见 [docs/manipulation.md](docs/manipulation.md) | 抽屉开 10/10、关 10/10；抓放 6/6；冰箱门开 35° 5/5、关 4/5 |
| 移动场景技能 | `mobile_pick_cube`（桌面场景）：停到桌边后用 PickCube PPO 抓起红块或蓝块 | 两种底盘 × 两个方块 × 5 个 seed：20/20 |
| 原任务 PPO | ManiSkill 官方 10 个 Panda 技能（推、拉、戳、推 T、滚球、扶正、堆叠、双臂抓取/堆叠等） | 29/30（RollBall seed 0 失败） |
| 示范轨迹 | 插销、插充电器、用工具拉方块（无策略，可作模仿学习数据） | — |
| 外部参考 | ManipGen、GraspGen、M2T2、cuRobo、RoboCasa 等 18 项 | 见 [EXTERNAL_SKILLS.md](skills/EXTERNAL_SKILLS.md) |
| 训练方法 | ManiSkill PPO（脚本已附带，已实测）、模仿学习、迁移到移动场景 | 见 [TRAINING.md](skills/TRAINING.md) |

技能目前通过 Python 调用（`scripts/demo_manipulation.py`、`scripts/demo_mobile_pick.py`），**还没有接到 ROS 话题上**；学生的 Agent 通过 ROS 控制机器人。

## 修改模型

不要直接改生成的 URDF，改 `mobile_panda_sim/geometry.py`（尺寸、相机位置）或 `scripts/build_description.py`（连杆结构），然后：

```bash
conda activate mshab
python scripts/build_description.py
python tests/test_scene.py                                          # 场景、相机位姿、底盘、夹爪
python scripts/demo_mobile_pick.py --no-render --seeds 0 1 2 3 4   # 结构改动后复测抓取
/usr/bin/python3 -m pytest tests -q                                 # URDF/网格一致性、命令校验（系统 Python，需 pytest）
```

## 已知限制

- MuJoCo 后端目前只有桌面场景，技能和训练还没有迁移，见 [docs/mujoco.md](docs/mujoco.md)。
- ManiSkill 后端沿用 ManiSkill Panda 的理想重力补偿（臂连杆不受重力），底盘连杆受重力。这不是完整的真实动力学模型。
- 轮式底盘的前后支撑是低摩擦球，不是带转向轴的万向轮；质量、惯性、摩擦是仿真初值，未做实机辨识。
- 抓取过程中轮式底盘会漂移约 1–2 cm（演示脚本的门限是 2 cm）。
- 没有障碍物地图话题；家具占地范围在 `/mobile_panda/scene_info` 中。
- 技能只能在 Python 里调用，还没有接到 ROS 话题上；操作技能的适用范围见 [docs/manipulation.md](docs/manipulation.md)。
- RViz 加载 `.glb` 网格时日志里没有报错，但还没有人眼确认显示效果。

## 来源

- MuJoCo 后端的 Panda 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache-2.0），由 `scripts/download_menagerie.py` 下载，不在仓库里。

本仓库整理自本地 `agent_demo/mobile_panda`（Windows 版移动 Panda），改为 Linux + `mshab` 环境，并新增 ROS2 接口。
Panda 网格和 `panda_v2.urdf` 来自 ManiSkill（Apache-2.0）。住宅场景使用 ReplicaCAD（CC BY-NC 4.0，仅限非商业用途）
和 YCB 物品模型，由 ManiSkill 下载，不包含在本仓库中。
