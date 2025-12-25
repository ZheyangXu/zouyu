# 录制仿真动画

当前模块： `isaaclab`

Isaac Lab 支持两种录制物理仿真动画的方法：**Stage Recorder** 与 **OVD Recorder**。
两者都会生成可在 Omniverse 中回放的 USD 输出，但它们的工作方式以及适用场景有所不同。

[Stage Recorder](https://docs.omniverse.nvidia.com/extensions/latest/ext_animation_stage-recorder.html) 扩展会在仿真过程中监听 stage 中所有运动与 USD 属性变化，并将其记录为**时间采样数据（time-sampled data）**。
其结果是一个 USD 文件：只捕获发生动画的变化——**而不是**完整场景——并且在录制时刻与原始 stage 的层级结构保持一致。
这使得它很容易作为 sublayer 添加，用于回放或渲染。

该方法通过 `isaaclab.envs.ui.BaseEnvWindow` 集成在 Isaac Lab 的 UI 中。
不过，要录制仿真动画，你需要禁用 [Fabric](https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html)，以允许将所有变化（例如运动与 USD 属性）读写到 USD stage。

**OVD Recorder** 面向更可扩展或自动化的工作流。
它使用 OmniPVD 从播放中的 stage 捕获仿真物理，然后将其直接**烘焙（bake）**成一个带动画的 USD 文件。
它可以在 Fabric 启用的情况下工作，并通过 CLI 参数运行。
生成的动画 USD 可以通过拖动时间线窗口快速回放与检查，而无需再次执行昂贵的物理仿真。

> 注意：
> Omniverse 在同一个 USD prim 上只支持**二选一**：物理仿真**或**动画回放，不能同时启用。
> 请对你希望做动画回放的 prim 禁用物理。

## Stage Recorder

在 Isaac Lab 中，Stage Recorder 集成在 `isaaclab.envs.ui.BaseEnvWindow` 类中。
这是最容易通过可视化方式捕获物理仿真的方法，并且可以直接通过 UI 使用。

要进行录制，必须禁用 Fabric——这样 recorder 才能跟踪 USD 的变化并将其写出。

### Stage Recorder 设置

Isaac Lab 在 `base_env_window.py` 中为 Stage Recorder 设置了合理的默认值。
如有需要，你可以在 Omniverse Create 中直接使用 Stage Recorder 扩展来覆盖或检查这些设置。

<details>
<summary>base_env_window.py 中使用的设置</summary>

```python
def _toggle_recording_animation_fn(self, value: bool):
    """Toggles the animation recording."""
    if value:
        # log directory to save the recording
        if not hasattr(self, "animation_log_dir"):
            # create a new log directory
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.animation_log_dir = os.path.join(os.getcwd(), "recordings", log_dir)
        # start the recording
        _ = omni.kit.commands.execute(
            "StartRecording",
            target_paths=[("/World", True)],
            live_mode=True,
            use_frame_range=False,
            start_frame=0,
            end_frame=0,
            use_preroll=False,
            preroll_frame=0,
            record_to="FILE",
            fps=0,
            apply_root_anim=False,
            increment_name=True,
            record_folder=self.animation_log_dir,
            take_name="TimeSample",
        )
    else:
        # stop the recording
        _ = omni.kit.commands.execute("StopRecording")
        # save the current stage
        source_layer = self.stage.GetRootLayer()
        # output the stage to a file
        stage_usd_path = os.path.join(self.animation_log_dir, "Stage.usd")
        source_prim_path = "/"
        # creates empty anon layer
        temp_layer = Sdf.Find(stage_usd_path)
        if temp_layer is None:
            temp_layer = Sdf.Layer.CreateNew(stage_usd_path)
        temp_stage = Usd.Stage.Open(temp_layer)
        # update stage data
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.GetStageUpAxis(self.stage))
        UsdGeom.SetStageMetersPerUnit(temp_stage, UsdGeom.GetStageMetersPerUnit(self.stage))
        # copy the prim
        Sdf.CreatePrimInLayer(temp_layer, source_prim_path)
        Sdf.CopySpec(source_layer, source_prim_path, temp_layer, source_prim_path)
        # set the default prim
        temp_layer.defaultPrim = Sdf.Path(source_prim_path).name
        # remove all physics from the stage
        for prim in temp_stage.TraverseAll():
            # skip if the prim is an instance
            if prim.IsInstanceable():
                continue
            # if prim has articulation then disable it
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            # if prim has rigid body then disable it
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            # if prim is a joint type then disable it
            if prim.IsA(UsdPhysics.Joint):
                prim.GetAttribute("physics:jointEnabled").Set(False)
        # resolve all paths relative to layer path
        omni.usd.resolve_paths(source_layer.identifier, temp_layer.identifier)
        # save the stage
        temp_layer.Save()
        # print the path to the saved stage
        print("Recording completed.")
        print(f"\tSaved recorded stage to    : {stage_usd_path}")
        print(f"\tSaved recorded animation to: {os.path.join(self.animation_log_dir, 'TimeSample_tk001.usd')}")
        print("\nTo play the animation, check the instructions in the following link:")
        print(
            "\thttps://docs.omniverse.nvidia.com/extensions/latest/ext_animation_stage-recorder.html#using-the-captured-timesamples"
        )
        print("\n")
        # reset the log directory
        self.animation_log_dir = None
```

</details>

### 使用示例

在独立运行的 Isaac Lab 环境中，传入 `--disable_fabric` 标志：

```bash
./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 8 --device cpu --disable_fabric
```

启动后，Isaac Lab 的 UI 窗口会显示一个 “Record Animation” 按钮。
点击一次开始录制；再次点击停止录制。

以下文件会被保存到 `recordings/` 文件夹：

* `Stage.usd` —— 禁用了物理的原始 stage
* `TimeSample_tk001.usd` —— 动画（时间采样）层

要回放：

```bash
./isaaclab.sh -s  # Opens Isaac Sim
```

在 Layers 面板中，将 `Stage.usd` 与 `TimeSample_tk001.usd` 同时作为 sublayer 插入。
现在当你点击播放按钮时，动画将开始回放。

关于 layer 的更多用法，请参考 [tutorial on layering in Omniverse](https://www.youtube.com/watch?v=LTwmNkSDh-c&ab_channel=NVIDIAOmniverse)。

## OVD Recorder

OVD Recorder 使用 OmniPVD 记录仿真数据，并将其直接烘焙到一个新的 USD stage 中。
这种方法更具扩展性，也更适用于大规模训练场景（例如 multi-env RL）。

它不通过 UI 控制——整个流程通过 CLI 标志启用，并自动运行。

### 工作流概览

1. 用户通过 CLI 启用动画录制后运行 Isaac Lab
2. Isaac Lab 启动仿真
3. 仿真运行期间记录 OVD 数据
4. 到达指定停止时间后，将仿真烘焙到输出 USD 文件，并关闭 IsaacLab
5. 最终结果是一个完全烘焙、可自包含的 USD 动画

### 使用示例

要录制动画：

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/run_cartpole_rl_env.py \
  --anim_recording_enabled \
  --anim_recording_start_time 1 \
  --anim_recording_stop_time 3
```

> 注意：
> 提供的 `--anim_recording_stop_time` 应当大于仿真时间。

> 警告：
> 目前，最终录制步骤可能会输出来自 [omni.usd] 的大量 warning 日志。
> 这是一个已知问题，这些 warning 信息可以忽略。

当到达停止时间后，文件会保存到：

```text
anim_recordings/<timestamp>/baked_animation_recording.usda
```
