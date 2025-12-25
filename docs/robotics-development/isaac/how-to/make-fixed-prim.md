# 在仿真中将物理 prim 固定（不随物理运动）

当一个 USD prim 应用了物理（physics）相关的 schema 后，它就会受到物理仿真的影响。这意味着该 prim 在仿真世界中可以移动、旋转，并与其他 prim 发生碰撞。
然而，在一些情况下，我们希望让某些 prim 在仿真世界中保持静态（static）：也就是说，prim 仍应参与碰撞，但它的位置与朝向不应发生改变。

以下各节将介绍如何生成（spawn）一个带物理 schema 的 prim，并使其在仿真世界中保持静态。

## 静态碰撞体（Static colliders）

静态碰撞体是不受物理影响、但可以与仿真世界中其他 prim 碰撞的 prim。它们不包含任何刚体（rigid body）属性。
但这也意味着，你无法通过 physics tensor API 来访问它们（即无法通过 `assets.RigidObject` 类访问）。

例如，要在仿真世界中生成一个静态的圆锥（cone），可以使用以下代码：

```python
import isaaclab.sim as sim_utils

cone_spawn_cfg = sim_utils.ConeCfg(
    radius=0.15,
    height=0.5,
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
)
cone_spawn_cfg.func(
    "/World/Cone", cone_spawn_cfg, translation=(0.0, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
)
```

## 刚体对象（Rigid object）

刚体对象（即对象只有单一刚体 body）可以通过将 `sim.schemas.RigidBodyPropertiesCfg.kinematic_enabled` 设置为 True 来变为静态。
这会使对象变为运动学（kinematic）对象，从而不受物理影响。

例如，要在仿真世界中生成一个静态圆锥，但该圆锥仍带有 rigid body schema，可以使用以下代码：

```python
import isaaclab.sim as sim_utils

cone_spawn_cfg = sim_utils.ConeCfg(
    radius=0.15,
    height=0.5,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
)
cone_spawn_cfg.func(
    "/World/Cone", cone_spawn_cfg, translation=(0.0, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
)
```

## 关节体（Articulation）

要固定一个 articulation 的根部（root），需要在该 articulation 的根刚体 link 上与世界坐标系之间存在一个固定关节（fixed joint）。
这可以通过将 `sim.schemas.ArticulationRootPropertiesCfg.fix_root_link` 设置为 True 来实现。
根据该参数的取值，会出现以下几种情况：

* 如果设置为 `None`，则不会修改 root link。
* 如果该 articulation 已经存在固定 root link，则该标志会启用或禁用该固定关节。
* 如果该 articulation 不存在固定 root link，则该标志会在世界坐标系与 root link 之间创建一个固定关节。
  该关节会以名称 "FixedJoint" 创建在 root link 之下。

例如，要生成一个 ANYmal 机器人并使其在仿真世界中保持静态，可以使用以下代码：

```python
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

anymal_spawn_cfg = sim_utils.UsdFileCfg(
    usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
        fix_root_link=True,
    ),
)
anymal_spawn_cfg.func(
    "/World/ANYmal", anymal_spawn_cfg, translation=(0.0, 0.0, 0.8), orientation=(1.0, 0.0, 0.0, 0.0)
)
```

这会在世界坐标系与 ANYmal 机器人 root link 之间创建一个固定关节。
由于 root link 位于 `"/World/ANYmal/base"` ，因此固定关节会被创建在 prim path `"/World/ANYmal/base/FixedJoint"` 。

## 进一步说明

鉴于 USD 资产设计的灵活性，通常会遇到以下几种场景：

1. **rigid body prim 上有 articulation root schema，但没有 fixed joint**

   这是浮动基座（floating-base）articulation 最常见且推荐的场景。root prim 同时具有 rigid body 与 articulation root 属性。
   在这种情况下，articulation root 会被解析为 floating-base，且 articulation 的 root prim 为 `Link0Xform` 。

```text
   ArticulationXform
       └── Link0Xform  (RigidBody and ArticulationRoot schema)
   ```

2. **父 prim 上有 articulation root schema，并且存在 fixed joint**

   这是固定基座（fixed-base）articulation 期望的组织方式。root prim 只有 rigid body 属性，而 articulation root 属性应用在其父 prim 上。
   在这种情况下，articulation root 会被解析为 fixed-base，且 articulation 的 root prim 为 `Link0Xform` 。

```text
   ArticulationXform (ArticulationRoot schema)
       └── Link0Xform  (RigidBody schema)
       └── FixedJoint (connecting the world frame and Link0Xform)
   ```

3. **父 prim 上有 articulation root schema，但没有 fixed joint**

   该场景下，root prim 只有 rigid body 属性，而 articulation root 属性应用在其父 prim 上；但世界坐标系与 root link 之间并未创建固定关节。
   在这种情况下，articulation 会被解析为 floating-base 系统。但 PhysX 解析器会使用其自身的启发式规则（例如按字母顺序）来确定 articulation 的 root prim。
   它可能选择 `Link0Xform` 作为 root prim，也可能选择其他 prim 作为 root prim。

```text
   ArticulationXform (ArticulationRoot schema)
       └── Link0Xform  (RigidBody schema)
   ```

4. **rigid body prim 上有 articulation root schema，并且存在 fixed joint**

   虽然这是一个合法场景，但不推荐，因为它可能导致非预期行为。
   在这种情况下，articulation 仍会被解析为 floating-base 系统。然而，世界坐标系与 root link 之间创建的 fixed joint 会被认为是最大坐标树（maximal coordinate tree）的一部分。
   这不同于 PhysX 将 articulation 视为 fixed-base 系统的方式，因此仿真行为可能不符合预期。

```text
   ArticulationXform
       └── Link0Xform  (RigidBody and ArticulationRoot schema)
       └── FixedJoint (connecting the world frame and Link0Xform)
   ```

对于浮动基座 articulation，root prim 通常同时具有 rigid body 与 articulation root 属性。
但如果直接将该 prim 与世界坐标系相连，会导致仿真将 fixed joint 视为最大坐标树的一部分，这与 PhysX 将 articulation 视为 fixed-base 系统的方式不同。

在内部实现中，当 `sim.schemas.ArticulationRootPropertiesCfg.fix_root_link` 设置为 True，且 articulation 被检测为 floating-base 系统时，会在世界坐标系与 articulation 的 root rigid body link 之间创建固定关节。
但为了让 PhysX 解析器将 articulation 视为 fixed-base 系统，会将 articulation root 属性从 root rigid body prim 上移除，并改为应用到其父 prim 上。

> **注意：**
> 在 Isaac Sim 的未来版本中，PhysX 将在 articulation root schema 中添加一个显式标志，用于在 fixed-base 与 floating-base 系统之间切换。
> 这将消除上述 workaround 的需求。
