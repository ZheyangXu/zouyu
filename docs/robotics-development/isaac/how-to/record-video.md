# 在训练期间录制视频片段

Isaac Lab 支持在训练期间使用 [gymnasium.wrappers. RecordVideo](https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/) 类录制视频片段。

你可以先安装 `ffmpeg` ，并在训练脚本中使用以下命令行参数来启用该功能：

* `--video`：在训练期间启用视频录制
* `--video_length`：每段录制视频的长度（以 step 为单位）
* `--video_interval`：每次录制之间的间隔（以 step 为单位）

在 headless 模式运行时，也请确保添加 `--enable_cameras` 参数。
请注意，启用录制等同于在训练期间启用渲染，这会降低启动与运行时性能。

示例用法：

```sh
python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video --video_length 100 --video_interval 500
```

录制的视频会保存在与训练 checkpoint 相同的目录下，路径为：
`IsaacLab/logs/<rl_workflow>/<task>/<run>/videos/train` 。
