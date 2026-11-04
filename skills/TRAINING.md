# 自己训练技能

推荐顺序：先在 **ManiSkill 原生任务**（固定底座 Panda）里训练，再写观测适配器搬到移动场景。
这和 `mobile_pick_cube` 的做法一样：控制模式统一用 `pd_joint_delta_pos`，训练完的动作可以直接通过 ROS 话题发送。

## 1. ManiSkill PPO（首选）

和本平台用的是同一个仿真器。ManiSkill 官方 PPO 脚本已拷贝到 [training/ppo/](../training/ppo/README.md)
（平台固定的 mshab 分支不带 `examples/`）。

```bash
conda activate mshab
cd training/ppo
python ppo.py --env_id=PushCube-v1 --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=2_000_000 --eval_freq=10 --num-steps=20 --no-capture-video --exp-name=pushcube
cd ../..
python scripts/run_native_skill.py PushCube-v1 --checkpoint training/ppo/runs/pushcube/final_ckpt.pt
```

- 默认控制模式就是 `pd_joint_delta_pos`。GPU 并行（`--num_envs` 上千）。
- 本机实测（RTX 5070 Ti Laptop）：PushCube 训练 100 万步约 1 分钟，得到的权重在 `run_native_skill.py` 中 5/5 成功。
  PickCube 等更难的任务需要更多步数，参数见 ManiSkill 仓库 `examples/baselines/ppo/baselines.sh`。
- 输出的 `final_ckpt.pt` 和下载的官方权重结构相同（`actor_mean.*`）。
- 自定义任务：继承 `mani_skill.envs.sapien_env.BaseEnv`，写 `_load_scene`、`_initialize_episode`、`evaluate`、`compute_dense_reward`，参考 `PickCube-v1` 源码。
- 视觉策略：ManiSkill 3.0.1 的 `examples/baselines/ppo/ppo_rgb.py`（**没有拷进本仓库**），观测换成相机图像，训练更慢但不依赖真值位姿。本机 `~/projects/maniskill/ManiSkill-main` 里有一份，在 mshab 环境中试跑过（2026-11-04）。

## 2. 模仿学习：示范数据 + BC / Diffusion Policy

```bash
python scripts/download_checkpoints.py PegInsertionSide-v1 --with-demos
# 示范轨迹在 skills/checkpoints/maniskill/PegInsertionSide-v1/motionplanning/
```

- 用 ManiSkill 的 `mani_skill.trajectory.replay_trajectory` 把轨迹转换成需要的观测和控制模式（如 `pd_joint_delta_pos`）。
- 基线在 [ManiSkill 仓库](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines) 的 `bc/`、`diffusion_policy/`、`act/`。
  这几个没有拷进本仓库，也**没有在 mshab 环境里测试过**，依赖可能需要补装。
- 也可以用 `python -m mani_skill.examples.motionplanning.panda.run -e PickCube-v1` 自己生成运动规划示范。

## 3. 搬到移动场景

训练好的策略在移动场景中要满足两件事：

1. **观测一致**：把移动场景的状态换算成训练时的坐标系。参考 `mobile_panda_sim/skills/pick_cube.py`：
   以 `panda_link0` 为基准做刚体变换，关节量按名称取，四元数不要改符号。
2. **几何一致**：桌面高度、物体尺寸、机械臂到物体的距离要在训练分布内。本场景的桌面 z=0，和 ManiSkill 桌面任务相同；
   停车点 x=-0.615 时，机械臂到桌边物体的距离和训练时一致。

更好的做法是直接在 `MobilePandaTabletop-v1` 上训练（固定底盘位置、只控制机械臂），省掉适配。
这需要给场景加奖励函数和成功判据，并改成 GPU 并行仿真（`sim_backend="physx_cuda"`）。

## 4. 其他框架

| 框架 | Panda 任务 | 训练库 | 备注 |
|---|---|---|---|
| Isaac Lab | 抬方块、开抽屉、堆方块 | rsl_rl、skrl、rl_games | 需要 Isaac Sim；本机有 `isaacsim` 环境 |
| robomimic / MimicGen | Lift、Can、Square 等 | BC-RNN、Diffusion Policy | robosuite（MuJoCo） |
| RoboCasa365 | 移动 Panda 厨房任务 | Diffusion Policy、π0、GR00T、SmolVLA | 数据量大 |
| openpi | DROID Franka | LoRA 微调 | 社区在单张 A100 上跑通过 |

这些框架训出的策略不能直接在 ManiSkill 里用，迁移代价见 [EXTERNAL_SKILLS.md](EXTERNAL_SKILLS.md)。
