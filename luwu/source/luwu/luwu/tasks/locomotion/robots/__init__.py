import gymnasium as gym

gym.register(
    id="Luwu-Cyberdog2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "luwu.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        "skrl_cfg_entry_point": "luwu.tasks.locomotion.agents:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": "luwu.tasks.locomotion.agents:sb3_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": "luwu.tasks.locomotion.agents:sb3_sac_cfg.yaml",
    },
)


gym.register(
    id="Luwu-Rough-Cyberdog2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "luwu.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        "skrl_cfg_entry_point": "luwu.tasks.locomotion.agents:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": "luwu.tasks.locomotion.agents:sb3_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": "luwu.tasks.locomotion.agents:sb3_sac_cfg.yaml",
    },
)
