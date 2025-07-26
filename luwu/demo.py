#!/usr/bin/env python3
"""
Luwu Project Demo - Comprehensive demonstration of the robot parkour training framework.

This demo showcases:
1. External configuration system with dynaconf
2. Unified tracking supporting W&B and TensorBoard
3. Domain-driven design architecture
4. CLI interface for training, play, and evaluation
5. Modern Python development practices with PDM, black, ruff, etc.
"""

import sys
from pathlib import Path
import tempfile
import time
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from luwu.infrastructure.config import ConfigManager
from luwu.infrastructure.tracking import TensorboardTracker, TrackerFactory
from luwu.infrastructure.repositories import FileConfigRepository, CachedConfigRepository
from luwu.domain.entities import RobotState, Command, Action, TrainingMetrics, ControlMode
from luwu.application.services import ApplicationService


def demo_configuration_system():
    """Demonstrate the external configuration system."""
    print("🔧 === Configuration System Demo ===")

    # Initialize configuration manager
    config_manager = ConfigManager()

    # Load main configuration
    main_config = config_manager.get_config()
    print(f"✅ Project Name: {main_config.project_name}")
    print(f"✅ Default Environment: {main_config.environment.name}")
    print(f"✅ Default Robot: {main_config.environment.robot}")

    # Load robot-specific configuration
    try:
        robot_config = config_manager.get_robot_config("a1")
        print(f"✅ A1 Robot Config Loaded:")
        print(f"   - Name: {robot_config['ROBOT']['name']}")
        print(f"   - Mass: {robot_config['PHYSICS']['mass']} kg")
        print(f"   - Control Type: {robot_config['CONTROL']['type']}")
        print(f"   - Action Scale: {robot_config['CONTROL']['action_scale']}")
    except Exception as e:
        print(f"❌ Error loading robot config: {e}")

    # Load environment configuration
    try:
        env_config = config_manager.get_environment_config("parkour")
        # Try to access different possible keys
        if "environment" in env_config:
            print(f"✅ Parkour Environment Config Loaded:")
            print(f"   - Name: {env_config['environment']['name']}")
            print(f"   - Terrain Type: {env_config['terrain']['type']}")
            print(f"   - Terrain Size: {env_config['terrain']['size']}")
        else:
            print(f"✅ Parkour Environment Config Loaded (keys: {list(env_config.keys())})")
    except Exception as e:
        print(f"❌ Error loading environment config: {e}")

    # Load training configuration
    try:
        training_config = config_manager.get_training_config("ppo")
        print(f"✅ PPO Training Config Loaded:")
        print(f"   - Algorithm: {training_config['ALGORITHM']['name']}")
        print(f"   - Learning Rate: {training_config['ALGORITHM']['learning_rate']}")
        print(f"   - Batch Size: {training_config['ALGORITHM']['num_mini_batches']}")
    except Exception as e:
        print(f"❌ Error loading training config: {e}")

    print()


def demo_domain_entities():
    """Demonstrate the domain entities."""
    print("🏗️  === Domain Entities Demo ===")

    # Create a robot state
    robot_state = RobotState(
        base_position=np.array([0.0, 0.0, 0.3]),
        base_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        base_linear_velocity=np.array([0.5, 0.0, 0.0]),
        base_angular_velocity=np.array([0.0, 0.0, 0.1]),
        joint_positions=np.array([0.0] * 12),
        joint_velocities=np.array([0.0] * 12),
        joint_torques=np.array([10.0] * 12),
        foot_contacts=np.array([True, True, True, True]),
        contact_forces=np.array([100.0, 100.0, 100.0, 100.0]),
    )
    print(f"✅ Robot State Created: Position={robot_state.base_position}")

    # Create a command
    command = Command(
        linear_velocity=np.array([1.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.5]),
        body_height=0.3,
    )
    print(f"✅ Command Created: Linear Vel={command.linear_velocity}")

    # Create an action
    action = Action(values=np.array([0.1] * 12), control_mode=ControlMode.POSITION)
    print(f"✅ Action Created: Values count={len(action.values)}")

    # Create training metrics
    metrics = TrainingMetrics(
        episode_reward=85.5,
        episode_length=950,
        policy_loss=0.01,
        value_loss=0.02,
        entropy_loss=0.15,
        explained_variance=0.95,
        clipfrac=0.1,
        approx_kl=0.05,
        learning_rate=0.0003,
        reward_components={"forward": 50.0, "height": 20.0, "stability": 15.5},
    )
    print(f"✅ Training Metrics: Reward={metrics.episode_reward}, Length={metrics.episode_length}")
    print()


def demo_tracking_system():
    """Demonstrate the unified tracking system."""
    print("📊 === Tracking System Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create TensorBoard tracker
        tb_tracker = TensorboardTracker(temp_dir)
        print("✅ TensorBoard tracker created")

        # Log some metrics
        test_metrics = {"reward": 75.5, "episode_length": 850, "policy_loss": 0.005}

        try:
            tb_tracker.log_metrics(test_metrics, step=100)
            print("✅ Metrics logged to TensorBoard")
        except Exception as e:
            print(f"⚠️  TensorBoard logging error: {e}")

        try:
            tb_tracker.close()
            print("✅ TensorBoard tracker closed")
        except Exception as e:
            print(f"⚠️  TensorBoard close error: {e}")

    # Demonstrate tracker factory
    try:
        factory_tracker = TrackerFactory.create("tensorboard", log_dir="./demo_runs")
        print("✅ Tracker created via factory")
        factory_tracker.close()
    except Exception as e:
        print(f"⚠️  Factory tracker error: {e}")

    print("💡 Note: W&B tracker would require API keys for actual use")
    print()


def demo_repositories():
    """Demonstrate the repository pattern."""
    print("🗃️  === Repository Pattern Demo ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)

        # Create file repository
        file_repo = FileConfigRepository(config_dir)

        # Create some test data
        test_config = {
            "robot": {"name": "demo_robot", "type": "quadruped"},
            "physics": {"mass": 15.0, "friction": 0.8},
        }

        try:
            # Save configuration
            file_repo.save_robot_config("demo_robot", test_config)
            print("✅ Configuration saved via repository")

            # Load configuration
            loaded_config = file_repo.load_robot_config("demo_robot")
            print(f"✅ Configuration loaded: {loaded_config['robot']['name']}")

            # Test cached repository
            cached_repo = CachedConfigRepository(file_repo)
            cached_config = cached_repo.load_robot_config("demo_robot")
            print("✅ Configuration loaded via cached repository")

        except Exception as e:
            print(f"❌ Repository error: {e}")

    print()


def demo_cli_simulation():
    """Simulate CLI commands without actually running them."""
    print("💻 === CLI Interface Demo ===")

    print("Available CLI commands:")
    print("  luwu-train --robot a1 --environment parkour --algorithm ppo")
    print("  luwu-play --robot a1 --checkpoint model.pt")
    print("  luwu-eval --robot a1 --checkpoint model.pt --num-episodes 100")

    # Simulate a training command
    print("\n🚀 Simulating training command...")
    print("Command: luwu-train --robot a1 --environment parkour --num-envs 16 --max-iterations 10")

    time.sleep(1)
    print("✅ Configuration loaded for A1 robot")
    time.sleep(0.5)
    print("✅ Parkour environment initialized")
    time.sleep(0.5)
    print("✅ PPO algorithm configured")
    time.sleep(0.5)
    print("🏃 Training simulation started...")
    time.sleep(1)
    print("📊 Episode 1/10: Reward=45.2, Success=False")
    time.sleep(0.5)
    print("📊 Episode 5/10: Reward=72.8, Success=True")
    time.sleep(0.5)
    print("📊 Episode 10/10: Reward=89.3, Success=True")
    time.sleep(0.5)
    print("✅ Training completed successfully!")
    print()


def demo_application_service():
    """Demonstrate the application service layer."""
    print("🏢 === Application Service Demo ===")

    try:
        # Initialize application service
        app_service = ApplicationService()
        print("✅ Application service initialized")

        # Validate a setup
        is_valid, errors = app_service.validate_setup("a1", "parkour", "ppo")
        if is_valid:
            print("✅ Setup validation passed")
        else:
            print(f"⚠️  Setup validation issues: {errors}")

        print("✅ Application service demonstration completed")

    except Exception as e:
        print(f"❌ Application service error: {e}")

    print()


def main():
    """Run the complete demo."""
    print("🎯 === Luwu Robot Parkour Framework Demo ===")
    print("Modern, scalable robot training with Isaac Sim/Lab integration")
    print("=" * 60)
    print()

    # Run all demonstrations
    demo_configuration_system()
    demo_domain_entities()
    demo_tracking_system()
    demo_repositories()
    demo_application_service()
    demo_cli_simulation()

    print("🎉 === Demo Complete ===")
    print()
    print("Key Features Demonstrated:")
    print("✅ External configuration with dynaconf (YAML/JSON/TOML)")
    print("✅ Domain-driven design architecture")
    print("✅ Unified tracking (W&B + TensorBoard support)")
    print("✅ Repository pattern for data access")
    print("✅ CLI interface for easy operation")
    print("✅ Modern Python practices (PDM, type hints, etc.)")
    print()
    print("Next Steps:")
    print("🔄 Integrate with Isaac Sim/Lab (requires Python 3.10)")
    print("🔄 Implement actual PPO training loop")
    print("🔄 Add more robot configurations")
    print("🔄 Deploy to production environment")
    print()
    print("Repository: https://github.com/ZheyangXu/zouyu/tree/luwu")


if __name__ == "__main__":
    main()
