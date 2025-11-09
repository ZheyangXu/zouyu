# 模型预测控制

## 模型预测控制形式

考虑线性系统

$$
\begin{equation}
\begin{array}{l}
    x_{k+1} = Ax_k + B u_k \\
    z_k = Cx_k
\end{array}
\end{equation}
$$

其中, $x_k \in \mathbb{R}^n$ 为状态(state), $u_k \in \mathbb{R}^m$ 为控制输入(control input), $z_k \in \mathbb{R}^p$ 为输出(output). $A$, $B$, $C$ 分别为系统矩阵(system matrix), 输入矩阵(input matrix), 输出矩阵(output matrix). 在当前的问题表述中, 输出 $\mathbf{z}_{k}$ 并不一定是观测输出, 也就是说，它不必等于测量得到的输出.

在我们的建模中, $\mathbf{z}_k$ 实际上是一组应当跟踪给定参考轨迹的状态变量。我们假设状态向量 $\mathbf{x}_{k}$ 是完全已知的。换言之，我们假定已经精确重建了状态向量.在后续的教程中, 我们将考虑状态向量不可直接观测的情形。在那种情况下, 我们会将状态重建器（也称为观测器）与 MPC 算法相结合.
