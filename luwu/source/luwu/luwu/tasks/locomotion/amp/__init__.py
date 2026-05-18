"""AMP (Adversarial Motion Priors) locomotion tasks."""

from luwu.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from luwu.tasks.locomotion.amp import mdp

# Import config sub-packages to trigger gym environment registration
from luwu.tasks.locomotion.amp.config import g1  # noqa: F401

__all__ = ["LocomotionAmpEnvCfg", "mdp"]
