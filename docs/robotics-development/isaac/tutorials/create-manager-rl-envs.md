# 创建基于 Manager 的强化学习环境（Manager-Based RL Environment）

在完成《[创建基于 Manager 的基础环境](create-manager-base-env.md)》之后，我们继续来看如何为强化学习（RL）构建一个基于 Manager 的**任务环境**。

基础环境的设计理念更偏向“感知-行动（sense-act）”闭环：智能体向环境发送动作（或命令），环境返回观测。这种最小接口对许多应用已经足够，例如传统的运动规划与控制。

但在很多场景中，我们还需要更明确的**任务定义（task specification）**，它通常就是学习目标本身。举例来说，在导航任务里，智能体往往需要到达一个目标位置。为此，Isaac Lab 提供了 `envs.ManagerBasedRLEnv` ：它在基础环境之上扩展了任务相关的 RL 组件，从而将“任务是什么、怎么评估、何时结束”等逻辑纳入环境体系。

与 Isaac Lab 里其它组件的设计一致，我们不建议直接修改 `envs.ManagerBasedRLEnv` 这类基类，而是推荐通过编写任务环境对应的配置类 `envs.ManagerBasedRLEnvCfg` 来完成环境定制。这样可以把“任务定义”与“环境实现”解耦，便于复用同一环境的组件去构建不同任务。

在本教程里，我们将使用 `envs.ManagerBasedRLEnvCfg` 来配置 cartpole 环境，构建一个“把杆保持在竖直方向”的 RL 任务。我们会学习如何用奖励项（rewards）、终止条件（terminations）、课程学习（curriculum）与命令（commands）来描述这个任务。

## 代码

本教程使用的 cartpole 环境位于 `isaaclab_tasks.manager_based.classic.cartpole` 模块下。

### cartpole_env_cfg.py

> 说明：原 RST 文档通过 `literalinclude` 引用源码，这里直接按 Markdown 代码块形式展示。你在提问里已经给出了文件内容，因此无需再从仓库拉取。

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip

##
# Scene definition
##

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):

    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

##
# MDP settings
##

@configclass
class ActionsCfg:

    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)

@configclass
class ObservationsCfg:

    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:

    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )

@configclass
class RewardsCfg:

    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )

@configclass
class TerminationsCfg:

    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )

##
# Environment configuration
##

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):

    """Configuration for the cartpole environment."""

    # Scene settings
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
```

### run_cartpole_rl_env.py

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg

def main():

    """Main function."""
    # create environment configuration
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()

if __name__ == "__main__":

    # run the main function
    main()
    # close sim app
    simulation_app.close()
```

## 代码讲解

在《[创建基于 Manager 的基础环境](create-manager-base-env.md)》中，我们已经介绍过如何配置场景（scene）、观测（observations）、动作（actions）与事件（events）。因此，本教程将重点放在 **RL 专属组件** 上：奖励（rewards）、终止条件（terminations）、命令（commands）与课程学习（curriculum）。

在 Isaac Lab 中，许多通用的“MDP 项（term）”实现都放在 `envs.mdp` 模块中。本教程会直接复用其中的一些实现；当然，你也可以编写自己的 term，并把它们放在任务自己的子包里（例如 `isaaclab_tasks.manager_based.classic.cartpole.mdp` ）。

### 定义奖励（Defining rewards）

奖励由 `managers.RewardManager` 统一计算。与其它 manager 一样，Reward manager 的每一项奖励通过 `managers.RewardTermCfg` 来配置。

`RewardTermCfg` 的关键内容包括：

* `func`：实际计算奖励的函数或可调用类
* `weight`：该奖励项的权重（系数）
* `params`：传给 `func` 的参数字典（可选）

在 cartpole 任务中，我们使用以下奖励项：

* **存活奖励（Alive Reward）**：鼓励智能体尽可能长时间保持“未失败”的状态。
* **终止惩罚（Terminating Reward）**：当环境进入终止状态时给予惩罚。
* **杆角度奖励（Pole Angle Reward）**：鼓励杆保持在目标的竖直位置。
* **小车速度惩罚（Cart Velocity Reward）**：鼓励小车速度尽量小。
* **杆角速度惩罚（Pole Velocity Reward）**：鼓励杆角速度尽量小。

对应配置见上方 `RewardsCfg` （代码块中已完整给出）。

### 定义终止条件（Defining termination criteria）

多数学习任务会在有限步数内进行，我们通常称这段交互为一个 episode。以 cartpole 为例，我们希望智能体尽可能长时间维持平衡；但如果系统进入不稳定或不安全状态，就应该提前结束当前 episode。反过来，如果智能体已经平衡了很久，也需要结束并重新开始，以便从新的初始状态继续学习。

终止条件由 `TerminationsCfg` 来配置。本例中，我们希望出现以下任一情况时结束 episode：

* **超时（Episode Length）**：episode 时长超过 `max_episode_length`（或对应的时间长度设置）。
* **越界（Cart out of bounds）**：小车位置超出边界 `[-3, 3]`。

其中， `time_out=True` 用来标记“时间限制导致的截断（truncation）”，而不是任务失败导致的“终止（terminated）”。这对应 Gymnasium 对 time limit 的区分（可参考 [Gymnasium 文档](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)）。

对应配置见上方 `TerminationsCfg` 。

### 定义命令（Defining commands）

在许多“目标条件（goal-conditioned）”任务中，给智能体提供一个目标（或命令）是很自然的需求。这部分由 `managers.CommandManager` 负责：它管理命令的采样、更新，并且也可以把命令作为观测的一部分提供给策略。

但在这个简单的 cartpole 平衡任务里，我们不需要命令输入，因此保持默认（即不配置 command manager，等价于 `None` ）。如果你需要命令机制，可以参考其它 locomotion 或 manipulation 的任务示例。

### 定义课程学习（Defining curriculum）

训练 RL 智能体时，经常会采用“从简单到困难”的策略：先让智能体在更容易的条件下学会基本行为，再逐步增加任务难度，这就是课程学习（curriculum learning）。Isaac Lab 提供了 `managers.CurriculumManager` 来描述这类过程。

本教程为了聚焦核心概念，不引入课程学习；你可以在其它 locomotion / manipulation 任务中找到课程学习的配置范例。

### 串起来：构建完整的环境配置（Tying it all together）

当你定义好上述 RL 组件之后，就可以创建 `ManagerBasedRLEnvCfg` 的配置（本例为 `CartpoleEnvCfg` ）。它与基础环境教程中的 `ManagerBasedEnvCfg` 很相似，只是额外挂上了奖励、终止等 RL 组件。

### 运行仿真循环（Running the simulation loop）

回到 `run_cartpole_rl_env.py` 脚本，主循环整体与基础环境类似；关键差别在于：

* 环境类型从 `envs.ManagerBasedEnv` 换成了 `envs.ManagerBasedRLEnv`
* 因此 `step()` 的返回值会包含额外信号：奖励 `rew`、终止 `terminated`、截断 `truncated`，以及更丰富的 `info`

在 `info` 里通常会维护用于日志与诊断的信息，例如：各奖励项贡献、各终止项是否触发、episode 长度等。

## 运行方式

与前一节类似，你可以通过脚本直接运行环境，例如创建 32 个并行环境：

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32
```

启动后会打开与基础环境类似的仿真窗口。不同的是，这次环境会返回奖励与终止状态；并且每个子环境会根据配置的终止条件，在终止时自动重置。

![run_cartpole_rl_env.py 的运行效果](../../../public/tutorial_create_manager_rl_env.png)

要停止仿真，可以直接关闭窗口，或在启动脚本的终端中按 `Ctrl+C` 。

## 小结

在本教程中，我们学习了如何创建一个用于强化学习的任务环境：通过在基础的 manager-based 环境上增加奖励（rewards）、终止条件（terminations）、命令（commands）与课程学习（curriculum）等组件，来完整描述一个 MDP 任务。

同时，我们也看到 `envs.ManagerBasedRLEnv` 在运行时会提供更丰富的信号（奖励、终止/截断状态以及更详尽的 `info` ），便于训练与调试。

虽然你可以为每个任务手写一个脚本来构造 `ManagerBasedRLEnv` ，但这种方式并不易扩展：任务一多，就会出现大量“专用脚本”。因此，Isaac Lab 通常会进一步借助 `gymnasium.make` 来用统一的 Gym 接口创建环境。下一节教程会继续介绍这一工作流。
