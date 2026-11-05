# 技能权重

`mujoco/pick_cube.pt` 是在 MuJoCo 上训练的抓取策略，随仓库提供，见 [training/mujoco](../../training/mujoco/README.md)。其余内容不进 Git，用脚本下载：

```bash
python scripts/download_checkpoints.py
```

下载后的结构：

```text
skills/checkpoints/maniskill/<任务>/rl/ppo_pd_joint_delta_pos_ckpt.pt    # 平台使用的权重
skills/checkpoints/maniskill/<任务>/rl/ppo_pd_ee_delta_pos*_ckpt.pt      # 其他控制模式，未使用
skills/checkpoints/maniskill/<任务>/rl/sample_*.mp4                       # 官方示例视频
skills/checkpoints/maniskill/<任务>/{rl,motionplanning,teleop}/trajectory*.h5   # 仅 --with-demos
```

## 来源与许可

- 来源：Hugging Face 数据集 [haosulab/ManiSkill_Demonstrations](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations)，
  与 `python -m mani_skill.utils.download_demo <任务>` 下载的文件相同。
- 许可证：**Apache-2.0**（数据集页面标注）。
- 这些是 ManiSkill 团队训练的权重，本仓库没有重新训练或修改。
