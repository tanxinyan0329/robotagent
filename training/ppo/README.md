# PPO 训练脚本

`ppo.py` 原样拷贝自 [ManiSkill](https://github.com/haosulab/ManiSkill) 3.0.1 的 `examples/baselines/ppo/ppo.py`，
许可证 Apache-2.0（版权归 ManiSkill 作者）。平台固定的 ManiSkill mshab 分支不带 `examples/`，所以放一份在这里。

已在 `mshab` 环境（ManiSkill 3.0.0b18）中实测可以运行：

```bash
conda activate mshab
cd training/ppo
python ppo.py --env_id=PushCube-v1 --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=2_000_000 --eval_freq=10 --num-steps=20 --no-capture-video --exp-name=pushcube
# 权重：training/ppo/runs/pushcube/final_ckpt.pt
cd ../..
python scripts/run_native_skill.py PushCube-v1 --checkpoint training/ppo/runs/pushcube/final_ckpt.pt
```

各任务的推荐参数见 ManiSkill 仓库的 `examples/baselines/ppo/README.md` 和 `baselines.sh`。
