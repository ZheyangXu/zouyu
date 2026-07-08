# 陆吾

## 安装

```bash
python -m pip install -e source/luwu
```

## AME

### G1

训练

```bash
python scripts/rsl_rl/train.py --task Luwu-AME-G1-29DOF-v0 --max_iterations 15000 --headless --num_envs 1024 --video --video_length 300
```

演示

```bash
python scripts/rsl_rl/play_ame.py --task Luwu-AME-G1-29DOF-Play-v0 --checkpoint /path/to/checkpoint --num_envs 1 --video --video_length 300 --save_attention_weights --vis_attention
```

### GO2

训练

```bash
python scripts/rsl_rl/train.py --task Luwu-AME-Go2-v0 --max_iterations 15000 --headless --num_envs 1024 --video --video_length 300
```

演示

```bash
python scripts/rsl_rl/play_ame.py --task Luwu-AME-Go2-Play-v0 --checkpoint /path/to/checkpoint --num_envs 1 --video --video_length 300 --save_attention_weights --vis_attention
```

### Cyberdog2

训练

```bash
python scripts/rsl_rl/train.py --task Luwu-AME-Cyberdog2-v0 --max_iterations 15000 --headless --num_envs 1024 --video --video_length 300
```

演示

```bash
python scripts/rsl_rl/play_ame.py --task Luwu-AME-Cyberdog2-Play-v0 --checkpoint /path/to/checkpoint --num_envs 1 --video --video_length 300 --save_attention_weights --vis_attention
```

## AMP

训练

```bash
python scripts/rsl_rl/train_amp.py --task Luwu-Unitree-G1-AMP-V0  --headless --num_envs 4096 --video --video_length 300
```

演示

```bash
python scripts/rsl_rl/play_amp.py --task Luwu-Unitree-G1-AMP-Play-V0 --video --video_length 300
```

## DeepMimic

训练

```bash
python scripts/rsl_rl/train_amp.py --task Luwu-Unitree-G1-Deepmimic-V0  --headless --num_envs 4096 --video --video_length 300
```

演示

```bash
python scripts/rsl_rl/play_amp.py --task Luwu-Unitree-G1-Deepmimic-Play-V0 --video --video_length 300
```

## Cyberdog2 Locomotion

训练

```bash
python scripts/rsl_rl/train_amp.py --task Luwu-Rough-Cyberdog2-Velocity --headless --num_envs 4096 --video --video_length 300
```
