# Classes and Configs

首先, 前往任务目录: `source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial` , 查看 `isaac_lab_tutorial_env_cfg.py` 的内容. 你会看到类似如下的代码:

```python
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):

    # Some useful fields
    .
    .
    .

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Some more useful fields
    .
    .
    .
```

这是模板附带的简单 cartpole 环境的默认配置, 用来定义在对应环境中你所做一切的 self 作用域.

首先注意 `@configclass` 装饰器. 它将一个类定义为配置类, 这在 Isaac Lab 中有特殊作用. 配置类是 Isaac Lab 在克隆环境以扩展训练规模时判断“需要关心什么”的重要部分. Isaac Lab 根据你的目标提供不同的基础配置类. 在这里, 我们使用 `DirectRLEnvCfg` , 因为我们想在 direct 工作流中进行强化学习.

其次注意配置类的内容. 作为作者, 你可以按需指定字段, 但通常这里会有三项总是需要定义: `sim` , `scene` 和 `robot` . 留意这些字段本身也都是配置类！这种组合式的设计是为了克隆任意复杂的环境.

`sim` 是 `SimulationCfg` 的实例, 控制我们构建的仿真现实的性质. 该字段属于基类 `DirecRLEnvCfg` , 并有默认配置, 因此在技术上可选. `SimulationCfg` 决定时间步长（dt）, 重力方向以及物理仿真方式. 这里我们只指定时间步长和渲染间隔: 前者表示每次时间推进模拟 $1/120$ 秒, 后者表示每隔多少步渲染一帧（值为 2 意味着每隔一帧渲染一次）.

`scene` 是 `InteractiveSceneCfg` 的实例. Scene 描述要放到 stage 上的内容, 并管理要在多个环境间克隆的 simulation entities. Scene 同样属于基类 `DirectRLEnvCfg` , 但与 sim 不同, 它没有默认值, 必须在每个 DirectRLEnvCfg 中定义. InteractiveSceneCfg 描述为训练要创建的 Scene 副本数量, 以及这些副本在 stage 上的间距.

最后是 `robot` 定义, 它是 `ArticulationCfg` 的实例. 一个环境可以有多个 `articulation` , 因此定义 `DirectRLEnv` 时并不强制需要 `ArticulationCfg` . 常见的做法是为机器人定义一个正则路径, 并替换基础配置中的 `prim_path` . 这里的 `CARTPOLE_CFG` 定义在 `isaaclab_assets.robots.cartpole` , 通过将 `prim` 路径替换为 `/World/envs/env_.*/Robot` , 我们就隐含地声明: Scene 的每个副本都包含一个名为 `Robot` 的机器人.

## The Environment

接下来, 看看任务目录下另一个 Python 文件: `isaac_lab_tutorial_env.py`

```python
# imports
.
.
.
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg

class IsaacLabTutorialEnv(DirectRLEnv):
    cfg: IsaacLabTutorialEnvCfg

    def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        . . .

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        . . .

    def _apply_action(self) -> None:
        . . .

    def _get_observations(self) -> dict:
        . . .

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(...)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        . . .

    def _reset_idx(self, env_ids: Sequence[int] | None):
        . . .

@torch.jit.script
def compute_rewards(...):
    . . .
    return total_reward
```

部分代码为便于讨论已省略. 这里是 direct 工作流的“核心”, 我们在按需调整模板时, 大多数修改都会发生在此. 当前 `IsaacLabTutorialEnv` 的成员函数都直接继承自 `DirectRLEnv` . 这一已知接口是 Isaac Lab 以及其支持的 RL 框架与环境交互的方式.

环境初始化时, 会接收自身的配置作为参数, 并立即传递给 `super` 来初始化 `DirectRLEnv` . 这一 `super` 调用也会触发 `_setup_scene` , 实际构建 scene 并进行克隆. 值得注意的是, 机器人在 `_setup_scene` 中被创建并注册到 scene 中. 首先, 机器人 `articulation` 通过我们在 `IsaacLabTutorialEnvCfg` 中定义的 `robot_config` 创建——在此之前它尚不存在！创建 `articulation` 后, 机器人会出现在 stage 的 `/World/envs/env_0/Robot` . 随后调用 `scene.clone_environments` 适当地复制 `env_0` . 此时机器人已经在 stage 上存在多个副本, 剩下的就是告诉 scene 需要跟踪这个 `articulation` . Scene 的 `articulations` 存放在一个字典中, 因此 `scene.articulations["robot"] = self.robot` 会在 `articulations` 字典中创建一个名为 `robot` 的元素, 并将其值设为 `self.robot` .

还要注意, 剩余函数除了 `_reset_idx` 外都不再接收额外参数. 这是因为环境只负责将动作应用到被仿真的智能体, 并随后更新仿真. 这正是 `_pre_physics_step` 和 `_apply_action` 的目的: 我们将驱动命令设置给机器人, 这样在仿真向前推进时, 这些动作就会被应用, 关节驱动到新的目标. 将流程拆分成这些步骤, 确保了对环境执行顺序的系统化控制, 尤其在 `manager` 工作流中尤为重要. 类似的关系也存在于 `_get_dones` 与 `_reset_idx` 之间: 前者判断各个环境是否处于终止状态, 并填充布尔张量以指示哪些环境因终止状态而结束, 哪些因超时结束（函数返回的两个张量）；后者 `_reset_idx` 接收环境索引列表（整数）, 并实际重置这些环境. 确保诸如更新驱动目标或重置环境等操作不会在物理或渲染步骤期间发生, 这种接口拆分方式有助于避免问题.
