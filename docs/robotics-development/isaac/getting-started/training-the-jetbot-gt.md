# 训练 Jetbot

在环境已定义的基础上, 我们现在可以开始调整观测与奖励, 以训练一个策略来作为 Jetbot 的控制器. 作为用户, 我们希望能够指定 Jetbot 行驶的目标方向, 并让车轮转动, 使机器人尽可能快速地朝该方向行驶. 如何用 **Reinforcement Learning (RL)** 实现这一点？如果你想直接查看本阶段的最终结果, 可以访问[教程仓库的这个分支](https://github.com/isaac-sim/IsaacLabTutorial/tree/jetbot-intro-1-2). 

## 扩展环境

首先需要为 stage 上的每个 Jetbot 创建设置指令的逻辑. 每条指令都是一个单位向量, 且我们需要为 stage 上每个机器人克隆各准备一条指令, 也就是一个形状为 `[num_envs, 3]` 的张量. 尽管 Jetbot 只在 2D 平面上运动, 但使用 3D 向量可以利用 Isaac Lab 提供的全部数学工具. 

同样, 设置可视化也很有帮助, 这样在训练和推理时更容易看出策略在做什么. 这里我们定义两个箭头 `VisualizationMarkers` ：一个表示机器人的 “forward” 方向, 一个表示指令方向. 策略训练充分后, 这两支箭头应当对齐！在早期就加上这些可视化, 有助于避免“静默的 bug”（不会导致崩溃但行为错误的问题）. 

首先, 需要定义 `marker` 的配置, 并用该配置实例化 markers. 将以下内容加入 `isaac_lab_tutorial_env.py` 的全局作用域：

```python
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)
```

`VisualizationMarkersCfg` 定义了用作 “marker” 的 USD prim. 任何 prim 都可以, 但通常应尽量保持简单, 因为 `marker` 的克隆会在每个时间步的运行时发生. 原因在于这些 `marker` 仅用于 *debug visualization only*, 并不参与仿真：用户可完全控制何时何地绘制多少 `marker` . NVIDIA 在公开的 nucleus 服务器（路径 `ISAAC_NUCLEUS_DIR` ）上提供了一些简单的 `mesh` , 这里我们选择 `arrow_x.usd` . 

若需更详细的 `VisualizationMarkers` 使用示例, 可查看 `markers.py` demo！

接下来, 扩展初始化和 setup 步骤, 构建用于记录指令以及 marker 位置、旋转的数据. 将 `_setup_scene` 的内容替换为：

```python
def _setup_scene(self):
    self.robot = Articulation(self.cfg.robot_cfg)
    # add ground plane
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    # clone and replicate
    self.scene.clone_environments(copy_from_source=False)
    # add articulation to scene
    self.scene.articulations["robot"] = self.robot
    # add lights
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    self.visualization_markers = define_markers()

    # setting aside useful variables for later
    self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
    self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
    self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
    self.commands[:,-1] = 0.0
    self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)

    # offsets to account for atan range and keep things on [-pi, pi]
    ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
    gzero = torch.where(self.commands > 0, True, False)
    lzero = torch.where(self.commands < 0, True, False)
    plus = lzero[:,0]*gzero[:,1]
    minus = lzero[:,0]*lzero[:,1]
    offsets = torch.pi*plus - torch.pi*minus
    self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

    self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
    self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
    self.marker_offset[:,-1] = 0.5
    self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
    self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
```

大部分代码都是为指令和 marker 做记录, 但指令初始化和偏航角计算值得展开. 指令通过 `torch.randn` 从多元正态分布采样, z 分量固定为 0, 再归一化为单位向量. 为了让指令箭头沿这些向量指向, 需要适当地旋转基础箭头 mesh, 也就是定义一个 `quaternion <https://en.wikipedia.org/wiki/Quaternion>` _ 来让箭头 prim 围绕 z 轴按指令角度旋转. 按惯例, 绕 z 轴的旋转称为 “yaw” 旋转（类似 roll、pitch）. 

幸运的是, Isaac Lab 提供了根据旋转轴和角度生成四元数的工具函数：:func: `isaaclab.utils.math.quat_from_axis_angle` , 所以真正棘手的只有确定角度. 

![Useful vector definitions for training](../../_static/setup/walkthrough_training_vectors.svg)

yaw 是围绕 z 轴定义的, yaw 为 0 时指向 x 轴, 正角度逆时针. 指令向量的 x、y 分量给出了该角的正切, 因此需要该比值的 *arctangent* 来得到 yaw. 

考虑两条指令：A 在第二象限 (-x, y), B 在第四象限 (x, -y). 两者的 y/x 之比相同. 如果不处理这一点, 部分指令箭头会指向与指令相反的方向！本质上, 指令定义在 `[-pi, pi]` , 但 `arctangent` 只在 `[-pi/2, pi/2]` 上定义. 

为解决这一问题, 我们根据指令所在象限为 yaw 加上或减去 `pi` ：

```python
ratio = self.commands[:,1]/(self.commands[:,0]+1E-8) #in case the x component is zero
gzero = torch.where(self.commands > 0, True, False)
lzero = torch.where(self.commands < 0, True, False)
plus = lzero[:,0]*gzero[:,1]
minus = lzero[:,0]*lzero[:,1]
offsets = torch.pi*plus - torch.pi*minus
self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)
```

涉及张量的布尔表达式可能存在歧义, pytorch 会因此报错. pytorch 提供了多种办法让定义更明确； `torch.where` 会生成与输入同形状的张量, 元素由对单个元素评估表达式得到. 处理张量布尔运算的可靠方式是生成布尔索引张量, 再用代数方式表示运算, `AND` 用乘法, `OR` 用加法, 上述实现即如此. 这等价于：

```python
yaws = torch.atan(ratio)
yaws[commands[:,0] < 0 and commands[:,1] > 0] += torch.pi
yaws[commands[:,0] < 0 and commands[:,1] < 0] -= torch.pi
```

接下来是实际绘制 markers 的方法. 请记住, 这些 marker 不是 scene entities！想看到它们时需要“画”出来：

```python
def _visualize_markers(self):
    # get marker locations and orientations
    self.marker_locations = self.robot.data.root_pos_w
    self.forward_marker_orientations = self.robot.data.root_quat_w
    self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

    # offset markers so they are above the jetbot
    loc = self.marker_locations + self.marker_offset
    loc = torch.vstack((loc, loc))
    rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

    # render the markers
    all_envs = torch.arange(self.cfg.scene.num_envs)
    indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
    self.visualization_markers.visualize(loc, rots, marker_indices=indices)
```

`VisualizationMarkers` 的 `visualize` 方法就像这个 “draw” 函数. 它接收 marker 的空间变换张量, 以及 `marker_indices` 张量来指定每个 marker 使用哪种原型. 只要这些张量的第一维一致, 就会按指定变换绘制对应 marker, 这就是为什么要将位置、旋转、索引堆叠在一起. 

现在只需在 pre physics step 中调用 `_visualize_markers` 让箭头可见. 将 `_pre_physics_step` 改为：

``python
def _pre_physics_step(self, actions: torch. Tensor) -> None:

    self.actions = actions.clone()
    self._visualize_markers()

``

在深入 RL 训练前, 最后一次主要修改是更新 `_reset_idx` 以处理指令和 markers. 每当重置环境, 都需要生成新的指令并重置 markers. 逻辑如前所述, 将 `_reset_idx` 替换为：

```python
def _reset_idx(self, env_ids: Sequence[int] | None):
    if env_ids is None:
        env_ids = self.robot._ALL_INDICES
    super()._reset_idx(env_ids)

    # pick new commands for reset envs
    self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
    self.commands[env_ids,-1] = 0.0
    self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)

    # recalculate the orientations for the command markers with the new commands
    ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
    gzero = torch.where(self.commands[env_ids] > 0, True, False)
    lzero = torch.where(self.commands[env_ids]< 0, True, False)
    plus = lzero[:,0]*gzero[:,1]
    minus = lzero[:,0]*lzero[:,1]
    offsets = torch.pi*plus - torch.pi*minus
    self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

    # set the root state for the reset envs
    default_root_state = self.robot.data.default_root_state[env_ids]
    default_root_state[:, :3] += self.scene.env_origins[env_ids]

    self.robot.write_root_state_to_sim(default_root_state, env_ids)
    self._visualize_markers()
```

就这样！我们现在既能生成指令, 也能可视化 Jetbot 的朝向, 可以开始微调观测和奖励了. 

![Visualization of the command markers](../../_static/setup/walkthrough_1_2_arrows.jpg)
