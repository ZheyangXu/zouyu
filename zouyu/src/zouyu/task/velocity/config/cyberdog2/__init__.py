from mjlab.tasks.registry import register_mjlab_task

from zouyu.task.velocity.config.cyberdog2.env_cfgs import (
    zouyu_cyberdog2_flat_env_cfg,
    zouyu_cyberdog2_rough_env_cfg,
)
from zouyu.task.velocity.config.cyberdog2.rl_cfg import zouyu_cyberdog2_ppo_runner_cfg
from zouyu.task.velocity.rl import VelocityOnPolicyRunner

register_mjlab_task(
    task_id="Zouyu-Cyberdog2-Rough",
    env_cfg=zouyu_cyberdog2_rough_env_cfg(),
    play_env_cfg=zouyu_cyberdog2_rough_env_cfg(play=True),
    rl_cfg=zouyu_cyberdog2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Zouyu-Cyberdog2-Flat",
    env_cfg=zouyu_cyberdog2_flat_env_cfg(),
    play_env_cfg=zouyu_cyberdog2_flat_env_cfg(play=True),
    rl_cfg=zouyu_cyberdog2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
