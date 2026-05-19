"""RSL-RL AMP training configuration for G1."""

from typing import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg

DATA_PATH = MISSING
@configclass
class RslRlAmpDiscriminatorCfg:
    """Configuration for the AMP discriminator network."""

    hidden_dims: list[int] = [1024, 512]

    activation: str = "elu"

    style_reward_scale: float = 5.0

    task_style_lerp: float = 0.4


@configclass
class RslRlAmpCfg:
    """Configuration for AMP training components."""

    # Size of the replay buffer (in number of transitions).
    disc_obs_buffer_size: int = 100

    # Gradient penalty coefficient (lambda in the paper).
    grad_penalty_scale: float = 10.0

    # Discriminator loss type: 'LSGAN', 'BCEWithLogits', or 'Wasserstein'.
    loss_type: str = "LSGAN"

    # Discriminator network configuration.
    amp_discriminator: RslRlAmpDiscriminatorCfg = RslRlAmpDiscriminatorCfg()

    # Dataset configuration (amp_data_path, datasets, slow_down_factor).
    dataset: dict | None = None

    # Whether to use empirical normalization for AMP observations.
    empirical_normalization: bool = False


@configclass
class RslRlPpoAmpAlgorithmCfg:
    """Configuration for the AMP+PPO algorithm."""

    class_name: str = "AmpPPO"

    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()


@configclass
class G1RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL runner configuration for G1 AMP training."""

    class_name = "AmpOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "g1_amp"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
    )

    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="AmpPPO",
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
        amp_cfg=RslRlAmpCfg(
            disc_obs_buffer_size=100000,
            grad_penalty_scale=10.0,
            loss_type="LSGAN",
            empirical_normalization=False,
            dataset={
                "amp_data_path": (
                    DATA_PATH
                ),
                "datasets": {
                    "B10_-__Walk_turn_left_45_stageii": 1.0,
                    "B11_-__Walk_turn_left_135_stageii": 1.0,
                    "B13_-__Walk_turn_right_90_stageii": 1.0,
                    "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
                    "B15_-__Walk_turn_around_stageii": 1.0,
                    "B22_-__side_step_left_stageii": 1.0,
                    "B23_-__side_step_right_stageii": 1.0,
                    "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
                    "B9_-__Walk_turn_left_90_stageii": 1.0,
                    "C11_-_run_turn_left_90_stageii": 1.0,
                    "C12_-_run_turn_left_45_stageii": 1.0,
                    "C13_-_run_turn_left_135_stageii": 1.0,
                    "C14_-_run_turn_right_90_stageii": 1.0,
                    "C15_-_run_turn_right_45_stageii": 1.0,
                    "C16_-_run_turn_right_135_stageii": 1.0,
                    "C17_-_run_change_direction_stageii": 1.0,
                    "C1_-_stand_to_run_stageii": 1.0,
                    "C3_-_run_stageii": 1.0,
                    "C4_-_run_to_walk_a_stageii": 1.0,
                    "C5_-_walk_to_run_stageii": 1.0,
                    "C6_-_stand_to_run_backwards_stageii": 1.0,
                    "C8_-_run_backwards_to_stand_stageii": 1.0,
                    "C9_-_run_backwards_turn_run_forward_stageii": 1.0,
                    "Walk_B10_-_Walk_turn_left_45_stageii": 1.0,
                    "Walk_B13_-_Walk_turn_right_45_stageii": 1.0,
                    "Walk_B15_-_Walk_turn_around_stageii": 1.0,
                    "Walk_B16_-_Walk_turn_change_stageii": 1.0,
                    "Walk_B22_-_Side_step_left_stageii": 1.0,
                    "Walk_B23_-_Side_step_right_stageii": 1.0,
                    "Walk_B4_-_Stand_to_Walk_Back_stageii": 1.0,
                },
                "slow_down_factor": 1,
            },
            amp_discriminator=RslRlAmpDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=5.0,
                task_style_lerp=0.4,
            ),
        ),
    )
