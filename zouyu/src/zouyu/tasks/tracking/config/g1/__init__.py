from mjlab.tasks.registry import register_mjlab_task

from zouyu.tasks.tracking.config.g1.env_cfgs import zouyu_g1_flat_tracking_env_cfg
from zouyu.tasks.tracking.config.g1.rl_cfg import zouyu_g1_tracking_ppo_runner_cfg
from zouyu.tasks.tracking.rl import MotionTrackingOnPolicyRunner

register_mjlab_task(
    task_id="Zouyu-G1-Tracking",
    env_cfg=zouyu_g1_flat_tracking_env_cfg(),
    play_env_cfg=zouyu_g1_flat_tracking_env_cfg(play=True),
    rl_cfg=zouyu_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)


register_mjlab_task(
    task_id="Zouyu-G1-Tracking-No-State-Estimation",
    env_cfg=zouyu_g1_flat_tracking_env_cfg(has_state_estimation=False),
    play_env_cfg=zouyu_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
    rl_cfg=zouyu_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
