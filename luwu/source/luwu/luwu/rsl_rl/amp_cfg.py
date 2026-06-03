from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlAmpCfg:
    disc_obs_buffer_size: int = 1000

    grad_penalty_scale: float = 10.0

    disc_trunk_weight_decay: float = 1.0e-4

    disc_linear_weight_decay: float = 1.0e-2

    disc_learning_rate: float = 1.0e-5

    disc_max_grad_norm: float = 1.0

    @configclass
    class AMPDiscriminatorCfg:
        hidden_dims: list[int] = MISSING

        activation: str = "elu"

        style_reward_scale: float = 1.0

        task_style_lerp: float = 0.0

    amp_discriminator: AMPDiscriminatorCfg = AMPDiscriminatorCfg()

    loss_type: Literal["GAN", "LSGAN", "WGAN"] = "LSGAN"
