from isaaclab.utils import configclass

from .manager_based_animation_env_cfg import ManagerBasedAnimationEnvCfg


@configclass
class ManagerBasedAmpEnvCfg(ManagerBasedAnimationEnvCfg):
    terminal_obs_groups: tuple[str, ...] = ("disc",)
