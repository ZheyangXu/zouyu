import gymnasium as gym

from . import agents

gym.register(
    id="Luwu-Unitree-G1-Deepmimic-v0",
    entry_point="luwu.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="Luwu-Unitree-G1-Deepmimic-Play-v0",
    entry_point="luwu.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DeepMimicPPORunnerCfg",
    },
)

gym.register(
    id="Luwu-Unitree-G1-Deepmimic-Debug-v0",
    entry_point="luwu.envs:ManagerBasedAnimationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_deepmimic_env_cfg:G1DeepMimicEnvCfg_DEBUG",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1DeepMimicPPORunnerCfg",
    },
)
