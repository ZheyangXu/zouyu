# MPC 四足机器人运动控制

## 1. 简介

模型预测控制（MPC）是四足机器人运动控制的主流方法——它在滚动时域上规划地面反作用力，同时满足摩擦锥和接触约束。然而，在每个控制周期（30–50 Hz）求解二次规划（QP）带来了巨大的计算负担，尤其对于机载嵌入式处理器而言。

本项目实现了Sun等人（2025）提出的**PID-DE-MPC**框架，通过以下三个思路来降低在线计算量，同时保持鲁棒性：

1. **事件触发MPC** — 仅当状态预测误差超过阈值时才重新求解QP；否则将上一时刻的解平移后复用。
2. **扩张状态观测器（ESO）** — 实时估计作用在质心上的外部扰动，将估计值作为前馈补偿施加到地面反作用力上。
3. **PID型动态触发机制** — 事件触发阈值利用状态误差的比例、积分和微分项进行自适应调节，使触发条件比静态阈值更加智能。

同时实现了两个简化基线算法用于对比：

| 控制器 | 事件触发机制 | 扰动观测器 |
| --- | --- | --- |
| **MPC**（基线） | 无 — 每步求解 | 无 |
| **EMPC** | 静态 `‖e‖ > 0.8` | 无 |
| **DEMPC** | 静态 `‖e‖ > 0.8` | ESO + 前馈补偿 |
| **PID-DE-MPC** | PID自适应 | ESO + 前馈补偿 |

## 四足机器人运动控制的凸MPC

控制器基于[[Di Carlo et al., 2018]](https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf)提出的凸MPC框架，该框架最初在MIT Cheetah 3上验证。

### 简化动力学

将机器人建模为受接触点地面反作用力作用的单刚体。状态向量包含12个变量：

$$
\mathbf{x} = [\mathbf{\Theta}, \ \mathbf{p}, \ \boldsymbol{\omega}, \ \dot{\mathbf{p}}]^T \in \mathbb{R}^{12}
$$

其中 $\mathbf{\Theta} = [\phi, \theta, \psi]^T$ 为Z-Y-X欧拉角（滚转、俯仰、偏航），$\mathbf{p}$ 为质心位置，$\boldsymbol{\omega}$ 为角速度，$\dot{\mathbf{p}}$ 为线速度。

在较小滚转和俯仰角的情况下，姿态动力学近似为：

$$
\dot{\mathbf{\Theta}} \approx \mathbf{R}_z(\psi)^T \boldsymbol{\omega}, \qquad
\frac{d}{dt}(\mathbf{I}\boldsymbol{\omega}) \approx \mathbf{I}\dot{\boldsymbol{\omega}}
$$

最终得到连续时间的线性时变动力学：

$$
\dot{\mathbf{x}}(t) = \mathbf{A}_c(\psi)\, \mathbf{x}(t) + \mathbf{B}_c(\mathbf{r}_1, \ldots, \mathbf{r}_4, \psi)\, \mathbf{u}(t)
$$

其中 $\mathbf{u} = [\mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3, \mathbf{f}_4]^T \in \mathbb{R}^{12}$ 为每条腿的3D地面反作用力，$\mathbf{r}_i$ 为质心到足端 $i$ 的向量。

### 力约束

每条着地腿需满足摩擦金字塔约束：

$$
f_{z} \geq f_{\min}, \qquad -\mu f_z \leq f_x \leq \mu f_z, \qquad -\mu f_z \leq f_y \leq \mu f_z
$$

摆动腿的所有力分量均被约束为零。

### QP形式

MPC被表述为在 horizon $N$ 上的紧凑二次规划：

$$
\min_{\mathbf{U}} \ \frac{1}{2}\mathbf{U}^T\mathbf{H}\mathbf{U} + \mathbf{U}^T\mathbf{g}
\quad \text{s.t.} \quad \underline{\mathbf{c}} \leq \mathbf{C}\mathbf{U} \leq \overline{\mathbf{c}}
$$

其中 $\mathbf{U} \in \mathbb{R}^{12N}$ 堆叠了整个预测时域上的所有接触力。紧凑形式消除了状态变量，将问题规模降至 $12N$ 个决策变量。OSQP通过CasADi的锥优化接口求解QP。

### 3.4 控制架构

![控制系统框图](../public/control-system-block-diagram.png)

整体架构采用分层结构：
* **高层操作员**通过摇杆/脚本提供速度指令
* **参考轨迹生成器**将指令转换为步态周期内的12自由度状态参考
* **MPC**以约48 Hz的频率计算最优地面反作用力
* **腿部控制器**通过雅可比转置以200 Hz将接触力映射为关节力矩
* **状态估计器**以1 kHz融合IMU和关节编码器数据

---

## Unitree Go2机器人

### 物理参数

Go2是一款12自由度的电驱动四足机器人。URDF模型位于 `models/URDF/go2_description/` 。通过Pinocchio提取的关键参数：

| 参数 | 符号 | 数值 |
| --- | --- | --- |
| 总质量 | $m$ | ~12 kg |
| 基座惯量 (xx) | $I_{xx}$ | 0.0245 kg·m² |
| 基座惯量 (yy) | $I_{yy}$ | 0.0981 kg·m² |
| 基座惯量 (zz) | $I_{zz}$ | 0.107 kg·m² |
| 摩擦系数 | $\mu$ | 0.8 |
| 腿数 | — | 4 |
| 每条腿关节数 | — | 3 (髋侧摆、髋前摆、膝) |

### 状态空间模型

质心动力学以零阶保持器在时间步长 $\Delta t$ 下离散化：

$$
\mathbf{x}_{k+1} = \mathbf{A}_d\, \mathbf{x}_k + \mathbf{B}_{d, k}\, \mathbf{u}_k + \mathbf{g}_d
$$

其中 $\mathbf{A}_d \in \mathbb{R}^{12\times 12}$ 为（常值）离散时间状态矩阵，$\mathbf{B}_{d, k} \in \mathbb{R}^{12\times 12}$ 为时变输入矩阵（依赖于足端位置和偏航角），$\mathbf{g}_d \in \mathbb{R}^{12}$ 为离散重力向量。

**状态向量**（12自由度）：

$$
\mathbf{x} = [p_x, p_y, p_z, \ \phi, \theta, \psi, \ v_x, v_y, v_z, \ \omega_x, \omega_y, \omega_z]^T
$$

**控制输入**（12个力，每条腿3个）：

$$
\mathbf{u} = [f_{FL, x}, f_{FL, y}, f_{FL, z}, \ f_{FR, x}, f_{FR, y}, f_{FR, z}, \ f_{RL, x}, f_{RL, y}, f_{RL, z}, \ f_{RR, x}, f_{RR, y}, f_{RR, z}]^T
$$

**腿部构型**（站立姿态）：每条腿初始关节角度为 `[0.0, 0.9, -1.8]` rad（髋侧摆、髋前摆、膝）。

## 实验结果

我们在MuJoCo物理仿真中对四种控制器在三种运动场景下进行了对比。每个场景运行10秒，使用3 Hz小跑步态（0.6占空比）。MPC以48 Hz运行（步态周期 / 16），腿部控制器以200 Hz运行，物理仿真以1000 Hz运行。仿真中施加随机脉冲踢击扰动（2–3次，70–100 N 侧向力，0.03 s 持续时间），用于测试抗扰能力。

### 前向小跑 (ex11) — 0.5 m/s

| 控制器 | 求解次数 | 求解率 | 缩减 | RMSE_vx | RMSE_vy | RMSE_yaw | RMSE_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **MPC** | 500 | 100.0% | — | 0.1152 | 0.1051 | 0.0885 | 0.1793 |
| **EMPC** | 421 | 84.2% | 15.8% | 0.1190 | 0.1008 | 0.0993 | 0.1849 |
| **DEMPC** | 439 | 87.8% | 12.2% | 0.0962 | 0.1390 | 0.1073 | 0.2003 |
| **PID-DE-MPC** | 382 | 76.4% | 23.6% | 0.0804 | 0.1096 | 0.1047 | 0.1716 |

![前向小跑滚动优化对比](../public/projects/mpc/ex11_mpc_rollout_compare.png)

PID-DE-MPC 以最低的RMSE（0.1716）将QP求解次数减少23.6%。

<video width="320" controls src="../public/projects/mpc/ex11_pid_de_mpc.mp4"></video>

![扰动图](../public/projects/mpc/ex11_disturbance.png)

![速度跟踪](../public/projects/mpc/ex11_velocity_tracking.png)

![RMSE对比](../public/projects/mpc/ex11_rmse_comparison.png)

### 侧向小跑 (ex12) — 0.4 m/s 侧向

| 控制器 | 求解次数 | 求解率 | 缩减 | RMSE_vx | RMSE_vy | RMSE_yaw | RMSE_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **MPC** | 500 | 100.0% | — | 0.1692 | 0.3551 | 0.9616 | 1.0389 |
| **EMPC** | 494 | 98.8% | 1.2% | 0.2655 | 0.4745 | 0.9777 | 1.1187 |
| **DEMPC** | 490 | 98.0% | 2.0% | 0.1683 | 0.2791 | 0.1763 | 0.3705 |
| **PID-DE-MPC** | 451 | 90.2% | 9.8% | 0.1087 | 0.1385 | 0.1208 | 0.2135 |

![侧向小跑滚动优化对比](../public/projects/mpc/ex12_mpc_rollout_compare.png)

PID-DE-MPC 以最低的RMSE（0.2135）将QP求解次数减少9.8%。

<video width="320" controls src="../public/projects/mpc/ex12_pid_de_mpc.mp4"></video>

![扰动图](../public/projects/mpc/ex12_disturbance.png)

![速度跟踪](../public/projects/mpc/ex12_velocity_tracking.png)

![RMSE对比](../public/projects/mpc/ex12_rmse_comparison.png)

### 旋转小跑 (ex13) — 4.0 rad/s 偏航

| 控制器 | 求解次数 | 求解率 | 缩减 | RMSE_vx | RMSE_vy | RMSE_yaw | RMSE_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **MPC** | 500 | 100.0% | — | 0.0507 | 0.0898 | 0.4916 | 0.5023 |
| **EMPC** | 494 | 98.8% | 1.2% | 0.0463 | 0.0844 | 0.5128 | 0.5218 |
| **DEMPC** | 492 | 98.4% | 1.6% | 0.0758 | 0.1614 | 0.5158 | 0.5458 |
| **PID-DE-MPC** | 471 | 94.2% | 5.8% | 0.0840 | 0.1374 | 0.5219 | 0.5462 |

![旋转小跑滚动优化对比](../public/projects/mpc/ex13_mpc_rollout_compare.png)

PID-DE-MPC 将QP求解次数减少5.8%，RMSE基本持平（0.5462 vs 0.5023）。

<video width="320" controls src="../public/projects/mpc/ex13_pid_de_mpc.mp4"></video>

![扰动图](../public/projects/mpc/ex13_disturbance.png)

![速度跟踪](../public/projects/mpc/ex13_velocity_tracking.png)

![RMSE对比](../public/projects/mpc/ex13_rmse_comparison.png)
