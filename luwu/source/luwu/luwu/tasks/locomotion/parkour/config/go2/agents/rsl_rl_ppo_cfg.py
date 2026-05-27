# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner configurations for Go2 parkour tasks.

Three runner configs corresponding to the three training stages:
- WalkPPORunnerCfg: Basic locomotion (flat + mild roughness)
- FieldPPORunnerCfg: Oracle parkour (BarrierTrack + privileged)
- DistillPPORunnerCfg: Vision-based distillation (depth camera + teacher)
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)


@configclass
class Go2WalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 1: Walking policy PPO runner configuration.

    Uses GRU-based recurrent actor-critic for proprioceptive locomotion.
    Trains from scratch on flat + mild roughness terrain.
    """

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 200
    experiment_name = "parkour_go2_walk"
    empirical_normalization = False
    resume = False

    # Observation groups: actor and critic both use standard observations
    obs_groups: dict[str, list[str]] = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    # GRU-based recurrent actor
    actor = RslRlRNNModelCfg(
        class_name="RNNModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="GaussianDistribution",
            init_std=0.5,
            std_type="scalar",
        ),
    )

    # GRU-based recurrent critic
    critic = RslRlRNNModelCfg(
        class_name="RNNModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        obs_normalization=False,
        distribution_cfg=None,  # Deterministic output for value function
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="adamw",
    )


@configclass
class Go2FieldPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 2: Oracle parkour policy PPO runner configuration.

    Uses GRU-based recurrent actor-critic with privileged critic observations
    (height scan + engaging block info). Resumes from Stage 1 walk checkpoint.
    """

    num_steps_per_env = 24
    max_iterations = 38000
    save_interval = 5000
    experiment_name = "parkour_go2_field"
    empirical_normalization = False
    resume = False  # Set to True and configure load_run for actual training

    obs_groups: dict[str, list[str]] = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    actor = RslRlRNNModelCfg(
        class_name="RNNModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="GaussianDistribution",
            init_std=0.5,
            std_type="scalar",
        ),
    )

    critic = RslRlRNNModelCfg(
        class_name="RNNModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        obs_normalization=False,
        distribution_cfg=None,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,  # No entropy bonus for parkour
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="adamw",
    )


@configclass
class Go2DistillPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 3: Distilled vision-based policy PPO runner configuration.

    Uses CNN-based actor for depth image processing + GRU recurrence.
    Critic retains MLP with privileged height scan access.
    Uses teacher-student distillation (behavior cloning).
    """

    num_steps_per_env = 32
    max_iterations = 60000
    save_interval = 5000
    experiment_name = "parkour_go2_distill"
    empirical_normalization = False
    resume = False  # Set to True and configure load_run for actual training

    obs_groups: dict[str, list[str]] = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    # CNN + MLP + GRU actor: processes depth images
    actor = RslRlRNNModelCfg(
        class_name="RNNModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="GaussianDistribution",
            init_std=0.1,
            std_type="scalar",
        ),
    )

    # MLP critic: uses privileged height scan
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=2,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="adamw",
    )
