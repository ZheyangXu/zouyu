# 添加你自己的学习库

Isaac Lab 预先集成了若干学习库（例如 RSL-RL、RL-Games、SKRL、Stable Baselines 等）。
不过，你可能希望将自己的学习库与 Isaac Lab 集成，或使用与 Isaac Lab 默认安装版本不同的库版本。
只要该库以 Python 包的形式提供，并且支持底层模拟器所使用的 Python 版本，这就是可行的。
例如，如果你使用的是 Isaac Sim 4.0.0 及以上版本，则需要确保该库支持 Python 3.11。

## 使用与默认不同的库版本

如果你希望使用与 Isaac Lab 默认安装版本不同的库版本，可以通过从源码构建安装，或安装 PyPI 上提供的其他版本。

例如，如果你希望使用自己修改过的 [rsl-rl](https://github.com/leggedrobotics/rsl_rl) 库版本，可以按以下步骤操作：

1. 按照安装 Isaac Lab 的说明进行安装。这会安装默认版本的 `rsl-rl` 库。
2. 从 GitHub 仓库克隆 `rsl-rl` 库：

```bash
    git clone git@github.com:leggedrobotics/rsl_rl.git
    ```

3. 在你的 Python 环境中安装该库：
    

```bash
    # Assuming you are in the root directory of the Isaac Lab repository
    cd IsaacLab

    # Note: If you are using a virtual environment, make sure to activate it before running the following command
    ./isaaclab.sh -p -m pip install -e /path/to/rsl_rl
    ```

在这种情况下， `rsl-rl` 库会被安装到 Isaac Lab 使用的 Python 环境中。
现在你就可以在实验中使用 `rsl-rl` 库。
要检查该库的版本与其他信息，可以使用以下命令：

```bash
./isaaclab.sh -p -m pip show rsl-rl-lib
```

此时输出应当会将 `rsl-rl` 库的位置显示为你克隆该库的目录。
例如，如果你将库克隆到 `/home/user/git/rsl_rl` ，那么上述命令的输出应当类似于：

```bash
Name: rsl_rl
Version: 3.0.1
Summary: Fast and simple RL algorithms implemented in pytorch
Home-page: https://github.com/leggedrobotics/rsl_rl
Author: ETH Zurich, NVIDIA CORPORATION
Author-email:
License: BSD-3
Location: /home/user/git/rsl_rl
Requires: torch, torchvision, numpy, GitPython, onnx
Required-by:
```

## 集成新的库

向 Isaac Lab 添加一个新库，与使用一个不同版本的库类似：你可以在 Python 环境中安装该库，并在实验中使用它。
但是，如果你希望将该库与 Isaac Lab 进行集成，那么首先需要为该库编写一个 wrapper，如 [包装环境](wrap-rl-env.md#how-to-env-wrappers) 中所述。

可按以下步骤将新库与 Isaac Lab 集成：

1. 在扩展 `isaaclab_rl` 的 `setup.py` 中，将你的库添加为一个额外依赖（extra-dependency）。
    这会确保在你安装 Isaac Lab 时该库会被安装；否则在该库未安装或不可用时，安装过程会提示错误。

2. 在 Isaac Lab 使用的 Python 环境中安装你的库。你可以按照上一节提到的步骤来完成。
3. 为该库创建一个 wrapper。你可以查看 `isaaclab_rl` 模块，参考其中针对不同库的 wrapper 示例。
    你可以为你的库创建一个新的 wrapper 并将其添加到该模块中；如果你愿意，也可以为 wrapper 单独创建一个新模块。

4. 为你的库创建用于训练与评估智能体的工作流脚本。你可以查看 `scripts/reinforcement_learning` 目录下的现有工作流脚本作为参考。
    你可以为你的库创建新的工作流脚本并将其添加到该目录中。

可选地，你还可以为该 wrapper 添加一些测试与文档。这有助于确保 wrapper 按预期工作，并指导用户如何使用该 wrapper。

* 添加一些测试，以确保 wrapper 能够按预期工作，并持续与该库保持兼容。
  这些测试可以添加到 `source/isaaclab_rl/test` 目录中。
* 为 wrapper 添加一些文档。你可以将 API 文档添加到 `isaaclab_rl` 模块的 API documentation（api-isaaclab-rl）中。

## 配置 RL 智能体

在你将新库与 Isaac Lab 集成完成后，你可以配置示例环境以使用该新库。
你可以参考教程 `tutorial-configure-rl-training` ，了解如何配置训练流程以使用不同的库。
