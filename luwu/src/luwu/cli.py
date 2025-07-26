"""
Command line interface for luwu.
"""

import sys
from pathlib import Path
from typing import Optional

import click

from luwu.application.services import ApplicationService


@click.group()
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to configuration directory",
)
@click.option(
    "--env",
    default="default",
    help="Environment to use (default, development, production)",
)
@click.pass_context
def cli(ctx: click.Context, config_dir: Optional[Path], env: str) -> None:
    """Luwu - Advanced Legged Robot Parkour Training."""
    # Initialize application service
    app_service = ApplicationService(config_dir)
    app_service.config_manager.set_environment(env)

    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["app_service"] = app_service


@cli.command()
@click.option("--robot", help="Robot type (overrides config)")
@click.option("--environment", help="Environment type (overrides config)")
@click.option("--algorithm", help="Training algorithm (overrides config)")
@click.option("--experiment-name", help="Name of the experiment")
@click.option("--resume", is_flag=True, help="Resume from latest checkpoint")
@click.option("--checkpoint", help="Specific checkpoint to resume from")
@click.pass_context
def train(
    ctx: click.Context,
    robot: Optional[str],
    environment: Optional[str],
    algorithm: Optional[str],
    experiment_name: Optional[str],
    resume: bool,
    checkpoint: Optional[str],
) -> None:
    """Start training a robot."""
    app_service: ApplicationService = ctx.obj["app_service"]

    # Initialize tracking
    app_service.initialize_tracking()

    # Validate configuration
    robot_name = robot or app_service.config.environment.robot
    env_name = environment or app_service.config.environment.name
    algo_name = algorithm or app_service.config.training.algorithm

    is_valid, errors = app_service.validate_configuration(robot_name, env_name, algo_name)
    if not is_valid:
        click.echo("Configuration validation failed:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    # Start training run
    try:
        config = app_service.start_training_run(
            experiment_name=experiment_name,
            robot_name=robot,
            environment_name=environment,
            algorithm_name=algorithm,
        )

        click.echo(f"Training started with configuration:")
        click.echo(f"  Robot: {config['metadata']['robot_name']}")
        click.echo(f"  Environment: {config['metadata']['environment_name']}")
        click.echo(f"  Algorithm: {config['metadata']['algorithm_name']}")

        if resume or checkpoint:
            click.echo("Note: Resume functionality not yet implemented")

        # TODO: Implement actual training loop here
        # This would involve:
        # 1. Creating the environment
        # 2. Creating the policy
        # 3. Running the training loop
        # 4. Logging metrics and checkpoints

        click.echo("Training completed successfully!")

    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)
        sys.exit(1)
    finally:
        app_service.finish_training_run()


@cli.command()
@click.option("--robot", required=True, help="Robot type")
@click.option("--checkpoint", required=True, help="Path to model checkpoint")
@click.option("--environment", help="Environment type (overrides config)")
@click.option("--episodes", default=10, help="Number of episodes to play")
@click.option("--render", is_flag=True, help="Enable rendering")
@click.pass_context
def play(
    ctx: click.Context,
    robot: str,
    checkpoint: str,
    environment: Optional[str],
    episodes: int,
    render: bool,
) -> None:
    """Play/evaluate a trained robot."""
    app_service: ApplicationService = ctx.obj["app_service"]

    # Validate checkpoint exists
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        click.echo(f"Checkpoint not found: {checkpoint}", err=True)
        sys.exit(1)

    # Validate configuration
    env_name = environment or app_service.config.environment.name
    is_valid, errors = app_service.validate_configuration(
        robot, env_name, "ppo"
    )  # Use default algorithm
    if not is_valid:
        click.echo("Configuration validation failed:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    click.echo(f"Playing robot: {robot}")
    click.echo(f"Environment: {env_name}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Episodes: {episodes}")
    click.echo(f"Render: {render}")

    # TODO: Implement actual play functionality
    # This would involve:
    # 1. Loading the model from checkpoint
    # 2. Creating the environment
    # 3. Running episodes
    # 4. Recording/displaying results

    click.echo("Play completed!")


@cli.command()
@click.option("--robot", required=True, help="Robot type")
@click.option("--checkpoint", required=True, help="Path to model checkpoint")
@click.option("--environment", help="Environment type (overrides config)")
@click.option("--episodes", default=100, help="Number of episodes to evaluate")
@click.option("--output", help="Output file for results")
@click.pass_context
def evaluate(
    ctx: click.Context,
    robot: str,
    checkpoint: str,
    environment: Optional[str],
    episodes: int,
    output: Optional[str],
) -> None:
    """Evaluate a trained robot."""
    app_service: ApplicationService = ctx.obj["app_service"]

    # Validate checkpoint exists
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        click.echo(f"Checkpoint not found: {checkpoint}", err=True)
        sys.exit(1)

    # Validate configuration
    env_name = environment or app_service.config.environment.name
    is_valid, errors = app_service.validate_configuration(robot, env_name, "ppo")
    if not is_valid:
        click.echo("Configuration validation failed:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    click.echo(f"Evaluating robot: {robot}")
    click.echo(f"Environment: {env_name}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Episodes: {episodes}")

    # TODO: Implement actual evaluation functionality
    # This would involve:
    # 1. Loading the model from checkpoint
    # 2. Creating the environment
    # 3. Running evaluation episodes
    # 4. Computing statistics
    # 5. Saving results

    results = {
        "mean_reward": 0.0,
        "std_reward": 0.0,
        "mean_episode_length": 0.0,
        "success_rate": 0.0,
        "episodes": episodes,
    }

    click.echo("Evaluation Results:")
    click.echo(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    click.echo(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
    click.echo(f"  Success Rate: {results['success_rate']:.1%}")

    if output:
        import json

        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to: {output}")


@cli.command()
@click.pass_context
def list_configs(ctx: click.Context) -> None:
    """List available configurations."""
    app_service: ApplicationService = ctx.obj["app_service"]

    configs = app_service.list_available_configs()

    click.echo("Available Configurations:")

    if configs["robots"]:
        click.echo("\n  Robots:")
        for robot in configs["robots"]:
            click.echo(f"    - {robot}")
    else:
        click.echo("\n  Robots: None")

    if configs["environments"]:
        click.echo("\n  Environments:")
        for env in configs["environments"]:
            click.echo(f"    - {env}")
    else:
        click.echo("\n  Environments: None")

    if configs["training"]:
        click.echo("\n  Training Algorithms:")
        for algo in configs["training"]:
            click.echo(f"    - {algo}")
    else:
        click.echo("\n  Training Algorithms: None")


@cli.command()
@click.option("--robot", required=True, help="Robot type to validate")
@click.option("--environment", required=True, help="Environment type to validate")
@click.option("--algorithm", required=True, help="Training algorithm to validate")
@click.pass_context
def validate(
    ctx: click.Context,
    robot: str,
    environment: str,
    algorithm: str,
) -> None:
    """Validate a configuration setup."""
    app_service: ApplicationService = ctx.obj["app_service"]

    is_valid, errors = app_service.validate_configuration(robot, environment, algorithm)

    if is_valid:
        click.echo("✓ Configuration is valid!", fg="green")
        click.echo(f"  Robot: {robot}")
        click.echo(f"  Environment: {environment}")
        click.echo(f"  Algorithm: {algorithm}")
    else:
        click.echo("✗ Configuration validation failed:", fg="red", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)


# Entry points for PDM scripts
def train_entry() -> None:
    """Entry point for luwu-train command."""
    cli(["train"])


def play_entry() -> None:
    """Entry point for luwu-play command."""
    cli(["play"])


def evaluate_entry() -> None:
    """Entry point for luwu-eval command."""
    cli(["evaluate"])


if __name__ == "__main__":
    cli()
