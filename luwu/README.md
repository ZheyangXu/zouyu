# LuWu: Advanced Legged Robot Parkour Training System

LuWu (驱虎) is a unified framework for training legged robots in parkour environments. It supports multiple simulation engines (Isaac Sim/Lab, MuJoCo), provides Gymnasium-compatible APIs, and implements state-of-the-art reinforcement learning algorithms.

## Key Features

* **Multi-Engine Support**: Isaac Sim, Isaac Lab, and MuJoCo backends
* **Gymnasium Compatible**: Standard RL environment interface
* **Configuration-Based**: All robot and training parameters in config files
* **Unified Tracking**: Support for both WandB and TensorBoard
* **RTX 5080 Optimized**: GPU-accelerated training for modern hardware
* **Clean Architecture**: Domain-driven design with comprehensive testing
* **Type Safety**: Full type hints and strict code quality standards

## Architecture

The project follows Domain-Driven Design (DDD) principles:

```
src/luwu/
├── domain/           # Core business logic
│   ├── entities/     # Domain entities (Robot, Environment, etc.)
│   └── services/     # Domain services
├── application/      # Application services
│   ├── algorithms/   # RL algorithms (PPO, etc.)
│   ├── environments/ # Environment implementations
│   └── training/     # Training orchestration
├── infrastructure/   # External concerns
│   ├── config/       # Configuration management
│   ├── simulation/   # Simulation backends
│   └── tracking/     # Logging and tracking
└── interfaces/       # User interfaces (CLI, etc.)
```

## Installation

```bash
# Navigate to luwu directory
cd luwu

# Install dependencies
pdm install

# Install in development mode
pdm install -e .
```

## Configuration

All configurations are stored in YAML files under `configs/` :

### Robot Configuration ( `configs/robots/` )

```yaml
# configs/robots/a1.yaml
name: a1
urdf_path: assets/robots/a1/urdf/a1.urdf
num_joints: 12
joint_names: [FL_hip_joint, FL_thigh_joint, ...]
default_joint_positions: [0.0, 0.9, -1.8, ...]
joint_limits:
  FL_hip_joint: [-1.22, 1.22]
  # ...
mass: 12.0
base_dimensions: [0.366, 0.094, 0.114]
motor_strength: 33.5
```

### Environment Configuration ( `configs/environments/` )

```yaml
# configs/environments/flat_parkour.yaml
name: flat_parkour
terrain_type: flat
terrain_size: [8.0, 8.0]
num_envs: 4096
env_spacing: 2.0
episode_length: 1000
observation_space_dim: 48
action_space_dim: 12

reward_components:
  - name: survival
    weight: 2.0
    enabled: true
  - name: forward_velocity
    weight: 1.5
    enabled: true
  # ...
```

### Training Configuration ( `configs/training/` )

```yaml
# configs/training/basic_ppo.yaml
algorithm: PPO
num_iterations: 5000
num_steps_per_env: 24
mini_batch_size: 4096
num_epochs: 5
learning_rate: 0.0003
gamma: 0.99
lam: 0.95
clip_coef: 0.2
entropy_coef: 0.01
value_loss_coef: 0.5
max_grad_norm: 1.0
save_interval: 100
checkpoint_dir: checkpoints/basic_ppo
```

## Usage

### Training

```bash
# Train with MuJoCo
luwu-train --robot a1 --env flat_parkour --training basic_ppo --engine mujoco

# Train with Isaac Sim
luwu-train --robot go1 --env rough_parkour --training advanced_ppo --engine isaac_sim

# With WandB tracking
luwu-train --robot a1 --env flat_parkour --training basic_ppo --wandb-project my-parkour

# Resume from checkpoint
luwu-train --robot a1 --env flat_parkour --training basic_ppo --resume checkpoints/basic_ppo/checkpoint_1000.pt
```

### Evaluation and Visualization

```bash
# Play trained policy
luwu-play --robot a1 --env flat_parkour --engine mujoco --checkpoint checkpoints/best_model.pt

# Deterministic evaluation
luwu-play --robot a1 --env flat_parkour --engine mujoco --checkpoint checkpoints/best_model.pt --deterministic

# Quantitative evaluation
luwu-eval --robot a1 --env flat_parkour --engine mujoco --checkpoint checkpoints/best_model.pt --num-episodes 100
```

### Python API

```python
from luwu import config_manager
from luwu.domain.entities import RobotConfig, EnvironmentConfig, TrainingConfig
from luwu.application.training import ParkourTrainer
from luwu.infrastructure.simulation import MujocoSimulation
from luwu.application.environments import VectorizedLeggedParkourEnv

# Load configurations
robot_config = RobotConfig(**config_manager.get_robot_config("a1"))
env_config = EnvironmentConfig(**config_manager.get_env_config("flat_parkour"))
training_config = TrainingConfig(**config_manager.get_training_config("basic_ppo"))

# Create simulation backend
simulation = MujocoSimulation(robot_config, env_config.dict())

# Create environment
env = VectorizedLeggedParkourEnv(
    robot_config=robot_config,
    env_config=env_config,
    simulation_backend=simulation,
)

# Create trainer
trainer = ParkourTrainer(
    robot_config=robot_config,
    env_config=env_config,
    training_config=training_config,
    env=env,
)

# Train
trainer.train()
```

## Supported Robots

* **Unitree A1**: 12-DOF quadruped with proprioceptive sensing
* **Unitree Go1**: Lightweight quadruped for agile locomotion
* **Custom Robots**: Add your own via configuration files

## Supported Environments

* **Flat Parkour**: Basic locomotion on flat terrain
* **Rough Parkour**: Navigation through uneven terrain with obstacles
* **Custom Environments**: Define your own via configuration

## Algorithms

* **PPO (Proximal Policy Optimization)**: Stable policy gradient method
* **Extensible**: Easy to add new algorithms following the same interface

## Development

### Code Quality

The project enforces strict code quality standards:

```bash
# Format code
pdm run format

# Lint code
pdm run lint

# Run tests
pdm run test

# Check coverage
pdm run coverage

# Run all checks
pdm run all
```

### Testing

Tests are organized by component with >90% coverage requirement:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=luwu --cov-report=html
```

### Adding New Robots

1. Create URDF/XML model files
2. Add configuration in `configs/robots/your_robot.yaml`
3. Update joint mappings and physical parameters
4. Test with existing environments

### Adding New Environments

1. Define environment in `configs/environments/your_env.yaml`
2. Specify terrain parameters and reward components
3. Implement custom reward functions if needed
4. Test with existing robots

### Adding New Simulation Backends

1. Implement `SimulationBackend` interface
2. Add backend-specific initialization and stepping
3. Register in simulation factory
4. Add tests and documentation

## Hardware Requirements

* **GPU**: RTX 5080 or equivalent CUDA-capable GPU
* **Memory**: 16GB+ RAM recommended for 4096 parallel environments
* **Storage**: 10GB+ for models, logs, and checkpoints

## Dependencies

* **Core**: PyTorch, NumPy, Gymnasium
* **Simulation**: MuJoCo, Isaac Sim/Lab (optional)
* **Configuration**: Dynaconf, Pydantic
* **Tracking**: WandB, TensorBoard
* **Quality**: Black, isort, Ruff, MyPy, pytest

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following code quality standards
4. Add tests for new functionality
5. Submit a pull request

## Citation

If you use LuWu in your research, please cite:

```bibtex
@software{luwu2025,
  title={LuWu: Advanced Legged Robot Parkour Training System},
  author={ZheyangXu},
  year={2025},
  url={https://github.com/ZheyangXu/zouyu}
}
```

## Support

* **Issues**: Report bugs and request features on GitHub
* **Discussions**: Join community discussions
* **Documentation**: Comprehensive docs in `docs/` directory
