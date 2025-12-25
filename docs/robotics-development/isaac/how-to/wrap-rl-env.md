
# 包装环境（Wrapping environments）

环境包装器（environment wrappers）是一种在不修改环境本体的情况下，改变环境行为的方法。
它可用于对观测或奖励应用变换函数、录制视频、强制时间步上限等。
关于该 API 的详细说明，请参阅 `gymnasium.Wrapper` 类。

目前，所有继承自 `envs.ManagerBasedRLEnv` 或 `envs.DirectRLEnv` 的强化学习环境都兼容 `gymnasium.Wrapper` ，因为其基类实现了 `gymnasium.Env` 接口。
要包装一个环境，你需要先初始化基础环境；随后即可通过反复调用 `env = wrapper(env, *args, **kwargs)` ，按需叠加任意数量的 wrapper。

例如，以下示例展示了如何包装一个环境，以强制在调用 step 或 render 之前必须先调用 reset：

```python
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

# create base environment
cfg = load_cfg_from_registry("Isaac-Reach-Franka-v0", "env_cfg_entry_point")
env = gym.make("Isaac-Reach-Franka-v0", cfg=cfg)
# wrap environment to enforce that reset is called before step
env = gym.wrappers.OrderEnforcing(env)
```

## 用于录制视频的包装器

`gymnasium.wrappers.RecordVideo` 包装器可用于录制环境视频。
该包装器接收一个 `video_dir` 参数，用于指定保存视频的位置。
视频会按指定间隔、针对指定数量的环境 step 或 episode，以 [mp4](https://en.wikipedia.org/wiki/MP4_file_format) 格式保存。

使用该包装器前，你需要先安装 `ffmpeg` 。在 Ubuntu 上，可以运行以下命令安装：

```bash
sudo apt-get install ffmpeg
```

> 注意：
> 默认情况下，当以 headless 模式运行环境时，Omniverse viewport 会被禁用。
> 这样做是为了避免不必要的渲染，从而提升性能。
>
> 我们在 RTX 3090 GPU 上使用 `Isaac-Reach-Franka-v0` 环境、在不同渲染模式下观察到如下性能：
>
> - 未启用 off-screen rendering 的无 GUI 执行：约 65, 000 FPS
> - 启用 off-screen 的无 GUI 执行：约 57, 000 FPS
> - 启用完整渲染的 GUI 执行：约 13, 000 FPS

用于渲染的 viewport 相机是场景中的默认相机，名称为 `"/OmniverseKit_Persp"` 。
相机的位姿与图像分辨率可以通过 `envs.ViewerCfg` 类进行配置。

<details>
<summary>ViewerCfg 类的默认参数：</summary>

```python
@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target position (in m). Default is (0.0, 0.0, 0.0)."""

    cam_prim_path: str = "/OmniverseKit_Persp"
    """The camera prim path to record images from. Default is "/OmniverseKit_Persp",
    which is the default camera in the viewport.
    """

    resolution: tuple[int, int] = (1280, 720)
    """The resolution (width, height) of the camera specified using :attr:`cam_prim_path`.
    Default is (1280, 720).
    """

    origin_type: Literal["world", "env", "asset_root", "asset_body"] = "world"
    """The frame in which the camera position (eye) and target (lookat) are defined in. Default is "world".

    Available options are:

    * ``"world"``: The origin of the world.
    * ``"env"``: The origin of the environment defined by :attr:`env_index`.
    * ``"asset_root"``: The center of the asset defined by :attr:`asset_name` in environment :attr:`env_index`.
    * ``"asset_body"``: The center of the body defined by :attr:`body_name` in asset defined by :attr:`asset_name` in environment :attr:`env_index`.
    """

    env_index: int = 0
    """The environment index for frame origin. Default is 0.

    This quantity is only effective if :attr:`origin` is set to "env" or "asset_root".
    """

    asset_name: str | None = None
    """The asset name in the interactive scene for the frame origin. Default is None.

    This quantity is only effective if :attr:`origin` is set to "asset_root".
    """

    body_name: str | None = None
    """The name of the body in :attr:`asset_name` in the interactive scene for the frame origin. Default is None.

    This quantity is only effective if :attr:`origin` is set to "asset_body".
    """
```

</details>

在调整参数之后，你可以通过使用 `gymnasium.wrappers.RecordVideo` 包装环境并启用 off-screen rendering 标志来录制视频。
此外，你需要将环境的渲染模式指定为 `"rgb_array"` 。

例如，下面的代码会对 `Isaac-Reach-Franka-v0` 环境录制 200 个 step 的视频，并以 1500 step 的间隔保存到 `videos` 文件夹中。

```python
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode with off-screen rendering
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

# adjust camera resolution and pose
env_cfg.viewer.resolution = (640, 480)
env_cfg.viewer.eye = (1.0, 1.0, 1.0)
env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
# create isaac-env instance
# set render mode to rgb_array to obtain images on render calls
env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
# wrap for video recording
video_kwargs = {
    "video_folder": "videos/train",
    "step_trigger": lambda step: step % 1500 == 0,
    "video_length": 200,
}
env = gym.wrappers.RecordVideo(env, **video_kwargs)
```

## 面向学习框架的包装器

每个学习框架都有自己与环境交互的 API。
例如，[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 库使用 [gym. Env](https://gymnasium.farama.org/api/env/) 接口与环境交互。
但像 [RL-Games](https://github.com/Denys88/rl_games)、[RSL-RL](https://github.com/leggedrobotics/rsl_rl) 或 [SKRL](https://skrl.readthedocs.io) 这类库，则使用自己的 API 与学习环境进行交互。
由于不存在“适用于所有情况”的通用方案，我们并不将 `envs.ManagerBasedRLEnv` 与 `envs.DirectRLEnv` 基类建立在任何特定学习框架的环境定义之上。
相反，我们实现 wrapper，使其能够兼容对应学习框架的环境定义。

下面给出一个示例，展示如何将 RL 任务环境与 Stable-Baselines3 一起使用：

```python
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

# create isaac-env instance
env = gym.make(task_name, cfg=env_cfg)
# wrap around environment for stable baselines
env = Sb3VecEnvWrapper(env)
```

> 警告：
> 使用对应学习框架的 wrapper 来包装环境，应当放在最后一步，也就是在应用完所有其他 wrapper 之后再进行。
> 这是因为学习框架的 wrapper 会改变对环境 API 的解释方式，从而可能不再兼容 `gymnasium.Env` 。

## 添加新的包装器

所有新的 wrapper 都应添加到 `isaaclab_rl` 模块中。
在应用 wrapper 之前，它们应当检查底层环境是否为 `isaaclab.envs.ManagerBasedRLEnv` 或 `envs.DirectRLEnv` 的实例。
这可以通过使用 `unwrapped` 属性实现。

我们在该模块中提供了一组 wrapper，可作为实现自定义 wrapper 的参考。
如果你实现了新的 wrapper，请考虑通过提交 pull request 的方式将其贡献回框架。
