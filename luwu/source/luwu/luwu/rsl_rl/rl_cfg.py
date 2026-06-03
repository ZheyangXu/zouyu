from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .amp_cfg import RslRlAmpCfg


@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticConv2d"

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]

    conv_linear_output_size: int = 16


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPOAMP"

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
