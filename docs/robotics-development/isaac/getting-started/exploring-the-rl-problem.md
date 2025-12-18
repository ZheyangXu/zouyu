## 强化学习问题

Jetbot 的控制命令是一个单位向量，指定期望的驾驶方向。我们必须让智能体感知到这个指令，从而能够相应地调整动作。实现方式有很多，最简单的“零阶”方法是直接修改观测空间，将指令纳入观测。首先，**edit the `IsaacLabTutorialEnvCfg` to set the observation space to 9**：机器人在世界坐标系下的速度向量包含线速度和角速度，共 6 个维度，将命令向量附加到这个速度向量后，总共 9 个维度。

接下来，只需在获取观测时完成这个拼接。同时，我们还需要计算后续要用到的前向向量。Jetbot 的前向向量是 x 轴，将 `root_link_quat_w` 应用于 `[1,0,0]` ，即可得到世界坐标系下的前向向量。将 `_get_observations` 方法替换为如下代码：

```python
def _get_observations(self) -> dict:
    self.velocity = self.robot.data.root_com_vel_w
    self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
    obs = torch.hstack((self.velocity, self.commands))
    observations = {"policy": obs}
    return observations
```

那么奖励应该怎么设计？

当机器人表现符合预期时，会以全速朝命令方向前进。如果同时奖励“向前行驶”和“与命令对齐”，那么最大化二者之和是否就会让机器人朝命令方向行驶呢？

试试看！将 `_get_rewards` 方法替换为：

```python
def _get_rewards(self) -> torch.Tensor:
    forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
    alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
    total_reward = forward_reward + alignment_reward
    return total_reward
```

`forward_reward` 是机器人在机体系下线速度 x 分量，即前向速度。我们知道 x 方向即资产的前向方向，因此它应当等价于前向向量与世界坐标系线速度的内积。 `alignment_reward` 是前向向量与命令向量的内积：同向时为 1，反向时为 -1。两者相加得到总奖励，然后就可以开始训练了！

```bash
python scripts/skrl/train.py --task=Template-Isaac-Lab-Tutorial-Direct-v0
```

![Naive results](https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/walkthrough_naive_webp.webp)

我们当然可以做得更好！

## 奖励和观察调试

在调优训练环境时，有一个经验法则：尽量保持观测空间足够小。这样可以减少模型参数数量（奥卡姆剃刀的直观含义），并提升训练速度。本例中需要同时编码与命令的对齐程度和前向速度。一种方法是利用线性代数中的点积和叉积！将 `_get_observations` 的内容替换为：

```python
def _get_observations(self) -> dict:
    self.velocity = self.robot.data.root_com_vel_w
    self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

    dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
    cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
    forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
    obs = torch.hstack((dot, cross, forward_speed))

    observations = {"policy": obs}
    return observations
```

同时需要 **edit the `IsaacLabTutorialEnvCfg` to set the observation space back to 3**，包含点积、叉积 z 分量以及前向速度。

点积（内积）用一个标量量化两个向量的对齐程度：同向且高度对齐时为大的正值，反向对齐时为大的负值，垂直时为 0。因此前向向量与命令向量的内积可以告诉我们“面对命令”的程度，但无法说明应该向哪一侧转向以改善对齐。

叉积同样衡量对齐程度，但结果是一个向量。两个向量的叉积定义了一个垂直于它们所处平面的轴，其方向由坐标系的手性决定。由于我们在 2D 平面内，可以只关注 :math: $\vec{forward} \times \vec{command}$ 的 z 分量：共线时为 0，命令在前向左侧为正，在右侧为负。

最后，质心线速度的 x 分量给出前向速度，正为向前、负为向后。将三者沿维度 1 拼接即可生成每个 Jetbot 的观测。这一步就能显著提升性能！

![Improved results](https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/walkthrough_improved_webp.webp)

表现更好了，Jetbot 也在往前挪动……但还可以更好！

另一个经验法则是尽量简化奖励函数。奖励项的组合类似逻辑“OR”。此处我们将“向前行驶”和“与命令对齐”相加，等于允许智能体只要满足其中之一就可获得奖励。要强制学习“朝命令方向行驶”，应只在“向前 AND 对齐”时奖励。逻辑 AND 联想到乘法，因此奖励函数可改为：

```python
def _get_rewards(self) -> torch.Tensor:
    forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
    alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
    total_reward = forward_reward*alignment_reward
    return total_reward
```

现在只有在对齐奖励不为零时，向前行驶才会得到奖励。看看效果如何！

![Tuned results](https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/walkthrough_tuned_webp.webp)

训练速度确实更快了，但当命令在身后时，Jetbot 学会了倒车前进。对本例或许可以接受，但也揭示了策略行为高度依赖奖励函数。这里存在**退化解**：奖励在“向前且对齐”时最大，但若倒车朝命令方向行驶，前向项为负，对齐项也为负，乘积却为正！设计环境时经常会遇到类似退化解，大量的奖励工程就是通过修改奖励函数抑制或鼓励这些行为。

假设我们不希望这种行为。对齐项的取值范围是 `[-1, 1]` ，但我们更希望将其映射到正值区间。我们不想完全去掉符号，而是希望大的负值变得接近 0，使得严重未对齐时不产生奖励。指数函数正好满足这一需求！

```python
def _get_rewards(self) -> torch.Tensor:
    forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
    alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
    total_reward = forward_reward*torch.exp(alignment_reward)
    return total_reward
```

在这种设计下，Jetbot 会先转向，始终以前向姿态朝命令方向行驶！

![Directed results](https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/walkthrough_directed_webp.webp)
