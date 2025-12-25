# 编写资产配置

本指南将带你了解如何创建 `assets.ArticulationCfg` 。
`assets.ArticulationCfg` 是一个配置对象，用于定义 Isaac Lab 中 `assets.Articulation` 的各项属性。

> **注意：**
> 虽然本指南只覆盖 `assets.ArticulationCfg` 的创建过程，但创建其他资产配置对象的流程也是类似的。

我们将使用 Cartpole 示例来演示如何创建 `assets.ArticulationCfg` 。
Cartpole 是一个简单的机器人：由一台小车和一根连接在小车上的杆组成。小车可以沿着轨道自由移动，杆可以绕着小车自由转动。
该配置示例对应的文件是 `source/isaaclab_assets/isaaclab_assets/robots/cartpole.py` 。

<details>
<summary>Cartpole 配置代码</summary>

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

CARTPOLE_CFG = ArticulationCfg(
	spawn=sim_utils.UsdFileCfg(
		usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
		rigid_props=sim_utils.RigidBodyPropertiesCfg(
			rigid_body_enabled=True,
			max_linear_velocity=1000.0,
			max_angular_velocity=1000.0,
			max_depenetration_velocity=100.0,
			enable_gyroscopic_forces=True,
		),
		articulation_props=sim_utils.ArticulationRootPropertiesCfg(
			enabled_self_collisions=False,
			solver_position_iteration_count=4,
			solver_velocity_iteration_count=0,
			sleep_threshold=0.005,
			stabilization_threshold=0.001,
		),
	),
	init_state=ArticulationCfg.InitialStateCfg(
		pos=(0.0, 0.0, 2.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
	),
	actuators={
		"cart_actuator": ImplicitActuatorCfg(
			joint_names_expr=["slider_to_cart"],
			effort_limit_sim=400.0,
			stiffness=0.0,
			damping=10.0,
		),
		"pole_actuator": ImplicitActuatorCfg(
			joint_names_expr=["cart_to_pole"], effort_limit_sim=400.0, stiffness=0.0, damping=0.0
		),
	},
)
"""Configuration for a simple Cartpole robot."""
```

</details>

## 定义 spawn 配置

如 `tutorial-spawn-prims` 教程（原 RST 交叉引用： `tutorial-spawn-prims` ）所述，spawn 配置定义了要生成（spawn）的资产属性。该生成过程既可以是过程式（procedural）的，也可以基于已有资产文件（例如 USD 或 URDF）。在本示例中，我们将从 USD 文件生成 Cartpole。

当从 USD 文件生成资产时，我们会定义其 `sim.spawners.from_files.UsdFileCfg` 。
该配置对象包含以下参数：

* `sim.spawners.from_files.UsdFileCfg.usd_path`：用于生成的 USD 文件路径
* `sim.spawners.from_files.UsdFileCfg.rigid_props`：articulation 根节点（root）的属性
* `sim.spawners.from_files.UsdFileCfg.articulation_props`：articulation 所有 link 的属性

最后两个参数是可选项。如果未指定，它们会保持 USD 文件中的默认值。

```python
spawn=sim_utils.UsdFileCfg(
	usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
	rigid_props=sim_utils.RigidBodyPropertiesCfg(
		rigid_body_enabled=True,
		max_linear_velocity=1000.0,
		max_angular_velocity=1000.0,
		max_depenetration_velocity=100.0,
		enable_gyroscopic_forces=True,
	),
	articulation_props=sim_utils.ArticulationRootPropertiesCfg(
		enabled_self_collisions=False,
		solver_position_iteration_count=4,
		solver_velocity_iteration_count=0,
		sleep_threshold=0.005,
		stabilization_threshold=0.001,
	),
),
```

如果希望从 URDF 文件而不是 USD 文件导入 articulation，可以将 `sim.spawners.from_files.UsdFileCfg` 替换为 `sim.spawners.from_files.UrdfFileCfg` 。
更多细节请参阅 API 文档。

## 定义初始状态

每个资产都需要通过其配置来定义在仿真中的初始（initial）或 *默认*（default）状态。
该配置会写入资产的默认状态缓冲区（default state buffers），当资产需要被 reset 时可以访问这些缓冲区。

> **注意：**
> 资产的初始状态是相对于其本地环境坐标系（local environment frame）定义的。
> 当重置资产状态时，需要将其转换到全局仿真坐标系（global simulation frame）。
> 更多细节请参阅 `tutorial-interact-articulation` 教程（原 RST 交叉引用： `tutorial-interact-articulation` ）。

对于 articulation， `assets.ArticulationCfg.InitialStateCfg` 对象定义了 articulation 根节点的初始状态，以及所有关节的初始状态。
在本示例中，我们会将 Cartpole 生成在 XY 平面原点处，并将 Z 高度设置为 2.0 米；同时，将关节位置与速度都设置为 0.0。

```python
init_state=ArticulationCfg.InitialStateCfg(
	pos=(0.0, 0.0, 2.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
),
```

## 定义执行器（actuator）配置

执行器是 articulation 的关键组件。通过该配置，可以定义要使用的执行器模型类型。
我们既可以使用物理引擎提供的内部执行器模型（即隐式执行器模型，implicit actuator model），也可以使用由用户自定义方程系统所控制的自定义执行器模型（即显式执行器模型，explicit actuator model）。
关于执行器的更多细节，请参阅 `overview-actuators` （原 RST 交叉引用： `overview-actuators` ）。

Cartpole 的 articulation 有两个执行器，分别对应两个关节： `cart_to_pole` 和 `slider_to_cart` 。
这里我们以示例方式为它们使用了两个不同的执行器模型配置。然而，由于它们实际上使用的是相同的执行器模型，也可以将它们合并为一个执行器模型。

<details>
<summary>使用分别的执行器模型配置（separate actuator models）的执行器模型配置</summary>

```python
actuators={
	"cart_actuator": ImplicitActuatorCfg(
		joint_names_expr=["slider_to_cart"],
		effort_limit_sim=400.0,
		stiffness=0.0,
		damping=10.0,
	),
	"pole_actuator": ImplicitActuatorCfg(
		joint_names_expr=["cart_to_pole"], effort_limit_sim=400.0, stiffness=0.0, damping=0.0
	),
},
```

</details>

<details>
<summary>使用单一执行器模型配置（single actuator model）的执行器模型配置</summary>

```python
actuators={
   "all_joints": ImplicitActuatorCfg(
	  joint_names_expr=[".*"],
	  effort_limit=400.0,
	  velocity_limit=100.0,
	  stiffness={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
	  damping={"slider_to_cart": 10.0, "cart_to_pole": 0.0},
   ),
},
```

</details>

## ActuatorCfg 速度/力矩限制的注意事项

在 IsaacLab v1.4.0 中，朴素的 `velocity_limit` 与 `effort_limit` 属性并**不会**被一致地写入物理求解器（physics solver）：

* **隐式执行器（Implicit actuators）**
  + `velocity_limit` 会被忽略（在仿真中从未设置）
  + `effort_limit` 会被写入仿真

* **显式执行器（Explicit actuators）**
  + `velocity_limit` 与 `effort_limit` 都只被驱动模型（drive model）使用，而不会被求解器使用

在 v2.0.1 中我们曾意外改变了这一点：无论是隐式还是显式执行器，所有 `velocity_limit` 与 `effort_limit` 都会被应用到求解器。这导致许多在旧默认（solver 限制未封顶/uncapped）条件下的训练出现问题。

为了恢复原有行为，同时仍然让用户能够完全控制求解器层面的限制，我们引入了两个新标志：

* **velocity_limit_sim**
  在仿真中设置物理求解器的关节最大速度上限。
* **effort_limit_sim**
  在仿真中设置物理求解器的关节最大 effort 上限。

它们会在仿真层面显式设置求解器的关节速度与关节 effort 上限。

另一方面， `velocity_limit` 与 `effort_limit` 用于建模所有显式执行器在力矩计算时的电机硬件层约束，而不是用于限制仿真层面的约束。
对于隐式执行器，由于它们不建模电机硬件限制， `velocity_limit` 在 v2.1.1 中被移除并标记为 deprecated。这保留了它们在 v1.4.0 中的相同行为。
最终，隐式执行器的 `velocity_limit` 与 `effort_limit` 也会被 deprecated，仅保留 `velocity_limit_sim` 与 `effort_limit_sim` 。

**Limit Options Comparison**

| **Attribute** | **Implicit Actuator** | **Explicit Actuator** |
|---|---|---|
| `velocity_limit` | Deprecated（ `velocity_limit_sim` 的别名） | 由模型（例如 DC motor）使用，不写入仿真 |
| `effort_limit` | Deprecated（ `effort_limit_sim` 的别名） | 由模型使用，不写入仿真 |
| `velocity_limit_sim` | 写入仿真 | 写入仿真 |
| `effort_limit_sim` | 写入仿真 | 写入仿真 |

如果用户希望调节底层物理求解器的限制，应当设置带 `_sim` 后缀的标志。

## USD 与 ActuatorCfg 不一致时的解析规则

USD 可能带有默认值，而 ActuatorCfg 又允许设置为 `None` ，或者指定覆盖值；这有时会让“最终到底写入仿真的是哪个值”变得难以判断。
解析规则很简单：对每个关节（per joint）与每个属性（per property），遵循如下规则：

**Resolution Rules for USD vs. ActuatorCfg**

| **Condition** | **ActuatorCfg Value** | **Applied** |
|---|---|---|
| No override provided | Not Specified | USD Value |
| Override provided | User's ActuatorCfg | Same as ActuatorCfg |

深入查看 USD 有时不太方便。为了帮助澄清最终写入仿真的具体数值，我们设计了一个标志： `isaaclab.assets.ArticulationCfg.actuator_value_resolution_debug_print` ，用于帮助用户确认仿真中实际采用的参数值。

当某个执行器参数在用户的 ActuatorCfg 中被覆盖（或保持未指定）时，我们会将其与从 USD 定义中读取的值进行对比，并记录任何差异。对每个关节与每个属性，如果发现不匹配的值，我们会记录解析结果：

1. **USD Value**
   从 USD 资产中解析得到的默认 limit 或 gain。
2. **ActuatorCfg Value**
   用户提供的覆盖值（若未提供则为 “Not Specified”）。
3. **Applied**
   仿真最终实际使用的数值：如果用户没有覆盖，则与 USD 值一致；否则反映用户的设置。

只有当确实存在不一致时，这些解析信息才会以 warning 表格的形式输出。
下面是你将看到的示例：

```text
+----------------+--------------------+---------------------+----+-------------+--------------------+----------+
|     Group      |      Property      |         Name        | ID |  USD Value  | ActuatorCfg Value  | Applied  |
+----------------+--------------------+---------------------+----+-------------+--------------------+----------+
| panda_shoulder | velocity_limit_sim |    panda_joint1     |  0 |    2.17e+00 |   Not Specified    | 2.17e+00 |
|                |                    |    panda_joint2     |  1 |    2.17e+00 |   Not Specified    | 2.17e+00 |
|                |                    |    panda_joint3     |  2 |    2.17e+00 |   Not Specified    | 2.17e+00 |
|                |                    |    panda_joint4     |  3 |    2.17e+00 |   Not Specified    | 2.17e+00 |
|                |     stiffness      |    panda_joint1     |  0 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
|                |                    |    panda_joint2     |  1 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
|                |                    |    panda_joint3     |  2 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
|                |                    |    panda_joint4     |  3 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
|                |      damping       |    panda_joint1     |  0 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
|                |                    |    panda_joint2     |  1 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
|                |                    |    panda_joint3     |  2 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
|                |                    |    panda_joint4     |  3 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
|                |      armature      |    panda_joint1     |  0 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    |    panda_joint2     |  1 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    |    panda_joint3     |  2 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    |    panda_joint4     |  3 |    0.00e+00 |   Not Specified    | 0.00e+00 |
| panda_forearm  | velocity_limit_sim |    panda_joint5     |  4 |    2.61e+00 |   Not Specified    | 2.61e+00 |
|                |                    |    panda_joint6     |  5 |    2.61e+00 |   Not Specified    | 2.61e+00 |
|                |                    |    panda_joint7     |  6 |    2.61e+00 |   Not Specified    | 2.61e+00 |
|                |     stiffness      |    panda_joint5     |  4 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
|                |                    |    panda_joint6     |  5 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
|                |                    |    panda_joint7     |  6 |    2.29e+04 |      8.00e+01      | 8.00e+01 |
|                |      damping       |    panda_joint5     |  4 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
|                |                    |    panda_joint6     |  5 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
|                |                    |    panda_joint7     |  6 |    4.58e+03 |      4.00e+00      | 4.00e+00 |
|                |      armature      |    panda_joint5     |  4 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    |    panda_joint6     |  5 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    |    panda_joint7     |  6 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |      friction      |    panda_joint5     |  4 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    |    panda_joint6     |  5 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    |    panda_joint7     |  6 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|  panda_hand    | velocity_limit_sim | panda_finger_joint1 |  7 |    2.00e-01 |   Not Specified    | 2.00e-01 |
|                |                    | panda_finger_joint2 |  8 |    2.00e-01 |   Not Specified    | 2.00e-01 |
|                |     stiffness      | panda_finger_joint1 |  7 |    1.00e+06 |      2.00e+03      | 2.00e+03 |
|                |                    | panda_finger_joint2 |  8 |    1.00e+06 |      2.00e+03      | 2.00e+03 |
|                |      armature      | panda_finger_joint1 |  7 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    | panda_finger_joint2 |  8 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |      friction      | panda_finger_joint1 |  7 |    0.00e+00 |   Not Specified    | 0.00e+00 |
|                |                    | panda_finger_joint2 |  8 |    0.00e+00 |   Not Specified    | 0.00e+00 |
+----------------+--------------------+---------------------+----+-------------+--------------------+----------+
```

为保持日志输出的整洁性， `isaaclab.assets.ArticulationCfg.actuator_value_resolution_debug_print` 默认为 False；需要时记得将其开启。
