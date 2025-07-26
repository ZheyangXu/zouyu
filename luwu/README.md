# Luwu - Advanced Legged Robot Parkour Training

Luwu is a modular and extensible framework for training legged robots to perform parkour tasks using reinforcement learning. Built with Isaac Sim and Isaac Lab, it provides a modern alternative to the archived Isaac Gym.

## Features

* 🤖 **Multi-Robot Support**: Support for various legged robots (A1, Go1, ANYmal, etc.)
* 🏃 **Parkour Training**: Specialized for challenging parkour environments with obstacles
* ⚙️ **Configuration-Driven**: All robot and environment settings are externally configurable using YAML/JSON/TOML
* 📊 **Unified Tracking**: Support for both W&B and TensorBoard experiment tracking
* 🏗️ **Clean Architecture**: Follows Domain-Driven Design (DDD) principles
* 🧪 **High Test Coverage**: Comprehensive test suite with >90% coverage
* 🛠️ **Developer-Friendly**: Modern Python with type hints, Black, isort, and Ruff formatting
* 📦 **PDM Management**: Uses PDM for modern Python dependency management

## Architecture

The project follows Domain-Driven Design (DDD) with clear separation of concerns:

```
src/luwu/
├── domain/          # Core business logic and entities
├── application/     # Application services and use cases  
├── infrastructure/  # External systems (config, tracking, etc.)
└── interfaces/      # External interfaces and adapters
```

## Quick Start

### Installation

```bash
# Clone the repository
cd luwu

# Install dependencies with PDM
pdm install

# Install development dependencies
pdm install -G dev

# Install pre-commit hooks (optional)
pre-commit install
```

### Configuration

Create or modify configuration files in the `configs/` directory:

* `configs/robots/` - Robot-specific configurations
* `configs/environments/` - Environment configurations  
* `configs/training/` - Training algorithm configurations
* `configs/settings.yaml` - Main project settings

### Training

```bash
# Train with default configuration
luwu-train

# Train with specific robot and environment
luwu-train --robot a1 --environment parkour --algorithm ppo

# Train with custom experiment name
luwu-train --experiment-name "my_parkour_experiment"
```

### Playing/Evaluation

```bash
# Play a trained model
luwu-play --robot a1 --checkpoint ./logs/checkpoints/model.pth --render

# Evaluate a model
luwu-eval --robot a1 --checkpoint ./logs/checkpoints/model.pth --episodes 100
```

### Configuration Management

```bash
# List available configurations
luwu list-configs

# Validate a configuration setup
luwu validate --robot a1 --environment parkour --algorithm ppo
```

## Configuration Examples

### Robot Configuration (configs/robots/a1.yaml)

```yaml
robot:
  name: "a1"
  urdf_path: "resources/robots/a1/a1.urdf"

control:
  type: "position"
  action_scale: 0.25
  stiffness:
    hip: 20.0
    thigh: 20.0  
    calf: 20.0

rewards:
  linear_velocity_xy: 1.0
  torques: -1e-5
  orientation: -5.0
```

### Environment Configuration (configs/environments/parkour.yaml)

```yaml
environment:
  name: "parkour"
  
terrain:
  type: "procedural"
  size: [100.0, 100.0]
  obstacles:
    stairs:
      probability: 0.2
      height_range: [0.05, 0.3]
    gaps:
      probability: 0.1
      width_range: [0.2, 0.8]

curriculum:
  enabled: true
  terrain_difficulty:
    initial: 0.0
    final: 1.0
```

### Training Configuration (configs/training/ppo.yaml)

```yaml
algorithm:
  name: "ppo"
  learning_rate: 3.0e-4
  num_learning_iterations: 5000
  
checkpoints:
  save_interval: 100
  
evaluation:
  enabled: true
  interval: 100
```

## Development

### Code Quality

The project uses several tools to ensure code quality:

```bash
# Format code
pdm run black src tests
pdm run isort src tests

# Lint code  
pdm run ruff check src tests

# Type checking
pdm run mypy src

# Run tests
pdm run pytest

# Run tests with coverage
pdm run pytest --cov=luwu --cov-report=html
```

### Pre-commit Hooks

Install pre-commit hooks to automatically format and lint code:

```bash
pdm install -G dev
pre-commit install
```

### Adding New Robots

1. Create a new robot configuration file in `configs/robots/new_robot.yaml`
2. Add URDF and other assets to `resources/robots/new_robot/`
3. The framework will automatically detect and load the new robot

### Adding New Environments

1. Create environment configuration in `configs/environments/new_env.yaml`
2. Implement environment-specific logic if needed
3. The environment will be available for training

## Project Structure

```
luwu/
├── configs/                 # Configuration files
│   ├── robots/             # Robot configurations
│   ├── environments/       # Environment configurations
│   ├── training/           # Training configurations
│   └── settings.yaml       # Main settings
├── src/luwu/               # Source code
│   ├── domain/             # Core business logic
│   ├── application/        # Application services
│   ├── infrastructure/     # External systems
│   └── interfaces/         # External interfaces
├── tests/                  # Test suite
├── resources/              # Robot assets and resources
└── logs/                   # Training logs and checkpoints
```

## Comparison with Original

### Improvements over windranger/legged_gym:

1. **Modern Simulation**: Uses Isaac Sim/Lab instead of archived Isaac Gym
2. **Python Version**: Supports Python 3.10+ instead of being limited to 3.8
3. **Configuration System**: External configuration using dynaconf vs hardcoded values
4. **Clean Architecture**: DDD design vs monolithic structure
5. **Unified Tracking**: Both W&B and TensorBoard vs W&B only
6. **Better Testing**: High test coverage vs minimal tests
7. **Modern Tooling**: PDM, Black, Ruff vs older tools
8. **Type Safety**: Full type hints vs no type annotations

### Migration from windranger:

The framework is designed to be compatible with existing robot configurations and training setups from the original windranger project, but with improved organization and extensibility.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure code quality checks pass
5. Commit your changes (`git commit -am 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* Based on the original legged_gym and rsl_rl projects
* Inspired by the windranger project structure
* Built for modern robotics research and development
