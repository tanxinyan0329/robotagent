# 传统方法操作技能：抓取、放置、开关抽屉和柜门

代码在 `mobile_panda_sim/manipulation/`，不用训练，在 ManiSkill 和 MuJoCo 上是同一份：

| 文件 | 内容 |
|---|---|
| `pose.py` | 位姿运算，不依赖任何仿真器 |
| `planner.py` | 运动规划层：基于 mplib（ManiSkill 自带）的逆运动学、避障路径（RRTConnect）、直线运动。障碍物是从场景碰撞体采样的点云；机械臂下面的立柱和底盘写进了规划用的机器人模型，算作自碰撞 |
| `skills.py` | 技能层：`pick`、`place`、`open`、`close`、`home`，以及选底盘停靠位置的 `suggest_base_pose` |
| `maniskill_world.py` | 技能需要的仿真接口（位姿、关节、步进、碰撞几何、射线、接触）的 ManiSkill 实现；MuJoCo 的实现是 `mj/world.py` |

这些技能读取仿真里的真值（物体网格和位姿、把手位置、关节轴），是**参考实现**，不经过视觉。
底盘必须先停好，技能执行时底盘速度保持为 0；开到哪里、怎么开过去由调用者（学生的导航技能）负责。

## 用法

```python
import gymnasium as gym
import mobile_panda_sim.scenes
from mobile_panda_sim.manipulation.skills import ManipulationSkills

env = gym.make("MobilePandaKitchen-v1", robot_uids="mobile_panda_wheels", start=(-2.95, -1.15, 3.1416),
               obs_mode="none", control_mode="pd_joint_delta_pos", sim_backend="physx_cpu", reward_mode="none")
env.reset(seed=0)
skills = ManipulationSkills(env.unwrapped)

r = skills.pick("lemon")                    # 从上方抓起，抬高 12 cm
r = skills.place((-3.38, -1.42))            # 放到 (x, y) 下方的台面上，松开，抬手
skills.home()                               # 回到收起姿态
print(r.success, r.message, r.info)         # SkillResult：成功与否、原因、细节

pose, coverage = skills.suggest_base_pose("kitchen_counter", "middle_slide_top")   # 底盘该停哪
skills.open("kitchen_counter", "middle_slide_top")       # 抽屉默认拉出 0.25 m
skills.close("kitchen_counter", "middle_slide_top")
skills.open("fridge", "top_door_hinge")                  # 门默认开到 0.6 rad（约 35°），可传 target
```

关节名就是 `/mobile_panda/state` 里 `articulations` 下的名字。

MuJoCo 上只有创建场景的部分不同（环境要求见 [mujoco.md](mujoco.md)）：

```python
from mobile_panda_sim.mj.sim import MujocoScene
from mobile_panda_sim.mj.world import MujocoWorld

scene = MujocoScene("kitchen", "wheels")
scene.reset(0, (-2.95, -1.15, 3.1416))       # 第二个参数是底盘停靠位
skills = ManipulationSkills(MujocoWorld(scene))
```

演示和回归测试：

```bash
conda activate mshab
python scripts/demo_manipulation.py drawer              # 台柜上层两个抽屉：开、关
python scripts/demo_manipulation.py pickplace           # 玄关台上的糖盒、梨、柠檬：抓起、放到别处
python scripts/demo_manipulation.py fridge --record outputs/fridge.mp4
python scripts/demo_manipulation.py all --viewer        # 实时窗口；跑完不自动关闭，关窗口即退出
python scripts/demo_manipulation.py all --repeat 3 --report outputs/manipulation.json

~/anaconda3/envs/llm/bin/python scripts/demo_manipulation.py all --backend mujoco --repeat 5    # MuJoCo，也可以加 --viewer
```

每一步除了看技能本身是否成功，还检查有没有碰动别的物体（位移超过 2 cm 记为失败）。

带窗口运行时要把刷新函数同时传给 `on_idle`：规划一次抓取或拖动可能要算几秒，这期间仿真不步进；
不在这时刷新窗口，拖动视角会卡住，系统还可能把窗口判定为"未响应"：

```python
sk = ManipulationSkills(w, on_step=w.render, on_idle=w.render)
```

## 每个技能怎么做的

**pick(name)**
1. 用物体碰撞网格的水平主轴，选夹爪闭合方向（较窄的一边，必须小于 7.5 cm）。高的物体夹住顶端以下 4 cm，矮的夹中间。
2. 规划无碰撞路径到物体正上方 10 cm，再直线下降，闭合夹爪。
3. 没夹住就退回，换一个抓取姿态，最多 3 次。
4. 慢速抬起 12 cm，确认物体还在手里。之后规划时把它当成挂在手上的方块，避免它撞到东西。

**place((x, y))**
1. 向下射线找 (x, y) 处的支撑面高度，算出让物体底面落在那里的夹爪位置。
2. 规划到正上方，下降，张开夹爪，抬手。
3. 检查物体落点：误差不超过 4 cm、确实落在支撑面上才算成功。

**open / close(articulation, joint)**
1. **生成候选抓法**：在把手（圆柱横杆）上生成多个抓取姿态，包括不同的接近倾角（绕横杆 0、±20°、±40°、±60°）；竖直的长把手还在不同高度。去掉手掌会碰到面板的姿态。
2. **预估再选择**：对每个可达的候选，用 IK 链预估"夹住后能跟着抽屉滑轨或门的转轴走多远"，选最远的那个。
   - 拖动时夹爪可以绕圆柱横杆转动，以保持可达。
   - 机械臂要避开其余家具；开门时还要求机械臂各连杆留在门板外侧，门的位置按每个路点对应的角度算。
3. **执行**：接近，闭合夹爪。用指间距检查是不是真的握住了横杆，而不是捏在边缘；没握住就换下一个候选。然后低速拖动。
4. **够不着就重新抓**：如果拖到一半机械臂够不着了，就松手、重新抓，最多 3 次。

## 实测（Ubuntu 22.04，physx_cpu / MuJoCo 3.5.0，2026-11-05，最终代码）

| 任务 | 底盘位置 (x, y, yaw) | ManiSkill（全部 3 轮） | MuJoCo 轮式（全部 5 轮） | MuJoCo 平面底盘（全部 2 轮） |
|---|---|---|---|---|
| 台柜中间上层抽屉：拉开 0.25 m、关上 | (-1.15, 1.83, 1.989) | 开 3/3，关 3/3 | 开 5/5，关 5/5 | 全过 |
| 台柜左侧上层抽屉：拉开、关上 | (-2.899, 1.86, 1.153) | 开 3/3，关 3/3 | 开 5/5，关 5/5 | 全过 |
| 玄关台：抓糖盒、梨、柠檬，各放到台面另一处 | (-2.95, -1.15, π) | 抓 9/9，放 9/9 | 抓 15/15，放 15/15 | 全过 |
| 冰箱上门：开到 0.6 rad、关上 | (0.249, 1.492, 1.107) | 开 2/3，关 3/3 | 开 5/5，关 5/5 | 全过 |

各项都没有碰动其他物体。ManiSkill 的冰箱门另外单独跑了 6 轮：开 6/6，关 5/6，所以它偶尔失败，这个数和迁移前的代码完全相同。MuJoCo 的样本也只有 5 轮，不代表它比 ManiSkill 更稳。

更早的一次实验：冰箱上门先在 (0.249, 1.492, 1.107) 开到 0.6 rad，再换到 (0.31, 1.40, 1.45) 继续开，到了 1.20 rad，只在 ManiSkill 上试过 1 次。

失败时技能返回 `success=False` 并说明原因（够不着、没夹住、路径规划失败），不会卡住；`info["stages"]` 里有每次抓取的细节。

一次开关动作通常需要 5–15 秒，抓取和放置 1–2 秒（控制频率 20 Hz）。`suggest_base_pose` 对抽屉要几秒到几分钟，MuJoCo 上中间抽屉约 4 分钟（和其他任务同时运行时测的）；对冰箱门更久，因为要预估多次重新抓取。

`suggest_base_pose` 默认只评估打开，不评估关闭。中间抽屉就是这样：它给出的 (-1.17, 1.86, 1.989) 在 MuJoCo 上能开，但关不上，手臂会挤到立柱上。这个位置在 ManiSkill 上能关，是因为拉抽屉时底盘被带着后退了约 3 cm；MuJoCo 的底盘基本不动。上表的位置是后退约 3.6 cm 后实测的。用 `opening=False` 单独评估时，抽屉要处在打开状态才有意义。

## 能用和不能用的

| 对象 | 能否使用 | 原因 |
|---|---|---|
| 台柜上层左、中抽屉（把手 0.59 / 0.66 m） | 能 | |
| 台柜右上抽屉（0.53 m）、所有下层抽屉（0.19–0.42 m） | 不能 | 机械臂底座高 0.93 m，夹爪够不到 0.55 m 以下的把手 |
| 冰箱上门 | 能开到约 35°，关上 | 门把手离铰链 0.66 m，底盘固定时转不过更大角度。要更大角度就换一个底盘位置再调用一次 `open` |
| 冰箱上门从 70° 关回 | 暂不支持 | 门开到这个角度会转到机器人一侧，把手在门的背面；需要从门外侧推门，这个推门动作还没做稳 |
| 冰箱下门（把手 0.42–0.78 m）、东墙矮柜推拉门（0.3 m） | 不能 | 太低 |
| 抓取 | 窄边小于 7.5 cm、放在机械臂够得着的台面上的物体 | 只做了从上方抓 |
| 放置 | 平整台面 | 不能把东西放进抽屉或冰箱：这需要从侧面放入，还没做 |

另外，台面靠墙一侧的物品离台沿 50–80 cm，从上方抓够不着，比如厨房台面里侧的芥末瓶。

## 场景为此做的改动

- **抽屉把手**：ReplicaCAD 的抽屉把手只是画在面板上的，没有碰撞体。现在每个抽屉前面板中间加了一根真实的金属横杆：直径 1.8 cm、长 12 cm，离面板 4 cm，用两根小立柱固定。
- **冰箱门把手**：上门原有的方形把手碰撞体换成了圆柱。下门原来只有外观、没有碰撞体，补了一个同样的圆柱。圆柱让夹爪从任何角度都能夹稳。

## 学生可以接着做的

- 用 `suggest_base_pose` 加上自己的导航，组成"走到柜子前、打开、取物"的完整流程。冰箱门开大角度需要换两次位置。
- 用相机和抓取位姿模型（见 `skills/EXTERNAL_SKILLS.md`）替换真值位姿。
- 把放置扩展成放进抽屉或冰箱，或者实现推门关门。
- 用强化学习训练接触类动作，和这里的传统实现比成功率和耗时。
