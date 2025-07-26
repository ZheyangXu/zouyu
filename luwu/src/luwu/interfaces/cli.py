"""
Command Line Interface for Luwu parkour training framework.
"""

import click
from pathlib import Path
from typing import Optional

from luwu.infrastructure.config import ConfigManager
from luwu.application.services import TrainingService, EvaluationService


@click.group()
@click.version_option(version="0.1.0", prog_name="luwu")
def cli():
    """Luwu - Advanced Legged Robot Parkour Training Framework."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--robot",
    "-r",
    default="a1",
    help="Robot type to use for training",
)
@click.option(
    "--environment",
    "-e",
    default="parkour",
    help="Environment to use for training",
)
@click.option(
    "--algorithm",
    "-a",
    default="ppo",
    help="Training algorithm to use",
)
@click.option(
    "--num-envs",
    "-n",
    default=4096,
    type=int,
    help="Number of parallel environments",
)
@click.option(
    "--max-iterations",
    "-i",
    default=5000,
    type=int,
    help="Maximum number of training iterations",
)
@click.option(
    "--headless",
    is_flag=True,
    default=False,
    help="Run in headless mode (no GUI)",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    help="Directory to save checkpoints",
)
def train(
    config: Optional[Path],
    robot: str,
    environment: str,
    algorithm: str,
    num_envs: int,
    max_iterations: int,
    headless: bool,
    checkpoint_dir: Optional[Path],
):
    """Train a robot using reinforcement learning."""
    click.echo(f"🚀 Starting training with {robot} robot in {environment} environment")
    click.echo(f"Algorithm: {algorithm}")
    click.echo(f"Number of environments: {num_envs}")
    click.echo(f"Max iterations: {max_iterations}")
    click.echo(f"Headless mode: {headless}")

    try:
        # Initialize configuration
        config_manager = ConfigManager(config_dir=config.parent if config else None)

        # Load configurations
        robot_config = config_manager.get_robot_config(robot)
        env_config = config_manager.get_environment_config(environment)
        training_config = config_manager.get_training_config(algorithm)

        click.echo(f"✅ Loaded configuration for {robot} robot")
        click.echo(f"✅ Loaded environment configuration: {environment}")
        click.echo(f"✅ Loaded training configuration: {algorithm}")

        # Initialize training service
        training_service = TrainingService()

        # Start training (placeholder - would integrate with actual training loop)
        click.echo("🏃 Training started...")
        click.echo(
            "⚠️  Note: This is a demonstration. Actual Isaac Sim integration requires Python 3.10"
        )
        click.echo("✅ Training completed successfully!")

    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--robot",
    "-r",
    default="a1",
    help="Robot type to use for play",
)
@click.option(
    "--environment",
    "-e",
    default="parkour",
    help="Environment to use for play",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model checkpoint",
)
@click.option(
    "--num-envs",
    "-n",
    default=1,
    type=int,
    help="Number of parallel environments for visualization",
)
@click.option(
    "--record",
    is_flag=True,
    default=False,
    help="Record the play session",
)
def play(
    config: Optional[Path],
    robot: str,
    environment: str,
    checkpoint: Path,
    num_envs: int,
    record: bool,
):
    """Play/visualize a trained robot policy."""
    click.echo(f"🎮 Starting play mode with {robot} robot")
    click.echo(f"Environment: {environment}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Recording: {record}")

    try:
        # Initialize configuration
        config_manager = ConfigManager(config_dir=config.parent if config else None)

        # Load configurations
        robot_config = config_manager.get_robot_config(robot)
        env_config = config_manager.get_environment_config(environment)

        click.echo(f"✅ Loaded configuration for {robot} robot")
        click.echo(f"✅ Loaded environment configuration: {environment}")

        # Start play mode (placeholder)
        click.echo("🎮 Play mode started...")
        click.echo(
            "⚠️  Note: This is a demonstration. Actual Isaac Sim integration requires Python 3.10"
        )
        click.echo("✅ Play session completed!")

    except Exception as e:
        click.echo(f"❌ Play failed: {e}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--robot",
    "-r",
    default="a1",
    help="Robot type to use for evaluation",
)
@click.option(
    "--environment",
    "-e",
    default="parkour",
    help="Environment to use for evaluation",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model checkpoint",
)
@click.option(
    "--num-episodes",
    "-n",
    default=100,
    type=int,
    help="Number of episodes to evaluate",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Directory to save evaluation results",
)
def evaluate(
    config: Optional[Path],
    robot: str,
    environment: str,
    checkpoint: Path,
    num_episodes: int,
    output_dir: Optional[Path],
):
    """Evaluate a trained robot policy."""
    click.echo(f"📊 Starting evaluation with {robot} robot")
    click.echo(f"Environment: {environment}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Episodes: {num_episodes}")

    try:
        # Initialize configuration
        config_manager = ConfigManager(config_dir=config.parent if config else None)

        # Load configurations
        robot_config = config_manager.get_robot_config(robot)
        env_config = config_manager.get_environment_config(environment)

        click.echo(f"✅ Loaded configuration for {robot} robot")
        click.echo(f"✅ Loaded environment configuration: {environment}")

        # Initialize evaluation service
        evaluation_service = EvaluationService()

        # Start evaluation (placeholder)
        click.echo("📊 Evaluation started...")
        click.echo(
            "⚠️  Note: This is a demonstration. Actual Isaac Sim integration requires Python 3.10"
        )
        click.echo("✅ Evaluation completed successfully!")

    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}", err=True)
        raise click.ClickException(str(e))


# Entry point functions for setup.py
def train():
    """Entry point for luwu-train command."""
    import sys

    sys.argv[0] = "luwu-train"
    cli(["train"] + sys.argv[1:])


def play():
    """Entry point for luwu-play command."""
    import sys

    sys.argv[0] = "luwu-play"
    cli(["play"] + sys.argv[1:])


def evaluate():
    """Entry point for luwu-eval command."""
    import sys

    sys.argv[0] = "luwu-eval"
    cli(["evaluate"] + sys.argv[1:])


if __name__ == "__main__":
    cli()
