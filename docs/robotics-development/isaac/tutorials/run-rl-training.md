# 使用强化学习智能体进行训练（Training with an RL Agent）

在前面的教程中，我们已经学习了如何定义一个强化学习（RL）任务环境、将其注册到 `gymnasium` 的 registry 中，并使用一个随机智能体与环境进行交互。接下来，我们进入下一步：训练一个 RL 智能体来解决该任务。

虽然 `envs.ManagerBasedRLEnv` 遵循 `gymnasium.Env` 的接口，但它并不完全等同于传统意义上的 `gym` 环境：环境的输入与输出并不是 numpy 数组，而是基于 torch 张量（tensor），并且其第一个维度表示环境实例数量（即向量化环境的 batch 维度）。

此外，大多数 RL 库通常会要求其自定义的一套环境接口。例如，[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 期望环境符合其 [VecEnv API](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api)，该 API 通常返回的是“由 numpy 数组组成的列表”，而不是“单个张量”。类似地，[RSL-RL](https://github.com/leggedrobotics/rsl_rl)、[RL-Games](https://github.com/Denys88/rl_games) 和 [SKRL](https://skrl.readthedocs.io) 也都要求不同的接口形式。

由于不存在“一套接口适配所有学习库”的通用方案，Isaac Lab 并没有把 `envs.ManagerBasedRLEnv` 直接绑定到某一个学习库的接口上。相反，Isaac Lab 在 `isaaclab_rl` 模块中提供了一组 wrapper，用于把环境转换成特定学习框架所期望的接口。

在本教程中，我们将使用 Stable-Baselines3 来训练一个 RL 智能体，以解决 cartpole 的平衡任务。

> **Caution**
> 为环境应用“学习框架对应的 wrapper”应当放在最后（即在应用完其它 wrapper 之后）。原因是学习框架的 wrapper 会改变对环境 API 的解释方式，从而可能与 `gymnasium.Env` 的语义不再兼容。

## 代码

本教程使用 `scripts/reinforcement_learning/sb3` 目录下的训练脚本（Stable-Baselines3 workflow）： `train.py` 。

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt

# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import gymnasium as gym
import logging
import numpy as np
import os
import random
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)
# PLACEHOLDER: Extension template (do not remove this comment)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]

    # train the agent
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=None,
        )
    # save the final model
    agent.save(os.path.join(log_dir, "model"))
    print("Saving to:")
    print(os.path.join(log_dir, "model.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
```

## 代码讲解

上面的脚本中，很多内容属于通用样板代码：例如创建日志目录、保存解析后的配置、以及初始化 Stable-Baselines3 的各类组件。对本教程而言，关键在于两件事：

* 创建环境实例（并传入配置）。
* 按顺序为环境套上 wrapper，使其符合 Stable-Baselines3 所期望的接口。

在脚本中，主要使用了三个 wrapper：

1. `gymnasium.wrappers.RecordVideo`：录制环境视频，并保存到指定目录。用于在训练过程中回看智能体行为。
2. `wrappers.sb3.Sb3VecEnvWrapper`：把 Isaac Lab 的环境包装成 Stable-Baselines3 兼容的向量化环境。
3. [`stable_baselines3.common.vec_env.VecNormalize`](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)：对观测与奖励进行归一化（normalization）。

这些 wrapper 采用“层层嵌套”的方式应用，形式上通常就是反复执行 `env = wrapper(env, *args, **kwargs)` 。最终得到的环境实例，会被传给 Stable-Baselines3 的算法（例如 PPO）用于训练。

## 运行方式（The Code Execution）

本教程使用 Stable-Baselines3 的 PPO 算法来训练一个智能体，以解决 cartpole 平衡任务。

### 训练智能体（Training the agent）

训练有三种常见方式，你可以按需求选择各自的取舍。

#### 无界面运行（Headless execution）

设置 `--headless` 后，训练过程中不会渲染仿真画面。这种方式适合在远程服务器上训练，或在本地不需要可视化时使用。通常它会更快，因为只需要执行物理仿真步而无需渲染。

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64 --headless
```

#### 无界面运行 + 离屏渲染（Headless execution with off-screen render）

仅使用 `--headless` 时无法看到智能体的行为。若希望在训练时记录视频用于回看，可以添加 `--enable_cameras` 来启用离屏渲染，并通过 `--video` 录制训练视频。

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64 --headless --video
```

视频会保存到 `logs/sb3/Isaac-Cartpole-v0/<run-dir>/videos/train` 目录下。你可以使用任意视频播放器打开查看。

#### 交互式运行（Interactive execution）

前两种方式适合高效训练，但无法直接交互观察仿真窗口内发生了什么。如果你希望在训练时打开 Isaac Sim 窗口，可以不加 `--headless` ：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0 --num_envs 64
```

这种方式会显著降低训练速度（因为需要实时渲染）。一个常见的折中方法是使用右下角停靠的 "Isaac Lab" 窗口切换不同渲染模式，从而在可视化与速度之间取得平衡。

### 查看日志（Viewing the logs）

你可以在另一个终端中通过 TensorBoard 监控训练过程：

```bash
# execute from the root directory of the repository
./isaaclab.sh -p -m tensorboard.main --logdir logs/sb3/Isaac-Cartpole-v0
```

### 运行训练好的智能体（Playing the trained agent）

训练完成后，你可以运行 `play.py` 来加载并可视化训练好的智能体：

```bash
# execute from the root directory of the repository
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py --task Isaac-Cartpole-v0 --num_envs 32 --use_last_checkpoint
```

上述命令会从 `logs/sb3/Isaac-Cartpole-v0` 目录中加载最新的 checkpoint。你也可以通过 `--checkpoint` 指定要加载的 checkpoint 路径。
