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

在我们的建模中, $\mathbf{z}_k$ 实际上是一组应当跟踪给定参考轨迹的状态变量。我们假设状态向量 $\mathbf{x}_{k}$ 是完全已知的。换言之，我们假定已经精确重建了状态向量. 在后续的教程中, 我们将考虑状态向量不可直接观测的情形。在那种情况下, 我们会将状态重建器（也称为观测器）与 MPC 算法相结合.

接下来, 我们需要引入下列记号, 图中亦有说明.

![alt text](../public/mpc-1.png)

* 预测时域的长度记为 $f$, 它表示未来用于预测状态轨迹行为的时域, 由用户选择, 在某些情况下也可视为调节参量.

* 控制时域的长度记为 $v$, 它表示未来允许控制输入变化的时域, 我们引入限制 $v \le f$, 控制时域之后的控制输入保持常值并等于控制时域内的最后一个取值, 在后文中将更加清楚, 控制时域由用户选择, 在某些情况下也可视为调节参量.

* $\mathbf{u}_{k+i|k}, \ i=0, 1, 2, 3, \ldots, v-1$ 表示在时刻 $k$ 计算并用于时刻 $k+i$ 的控制输入, 该控制输入由 MPC 问题的解确定, 在此需注意控制输入向量仅在 $k+v$ 之前变化, 此后保持常值并等于 $\mathbf{u}_{k+v-1|k}$, 后文会进一步说明.

* $\mathbf{x}_{k+i|k}, \ i=0, 1, 2, 3, \ldots, f$ 表示在时刻 $k$ 得到的对应时刻 $k+i$ 的状态向量预测.

* $\mathbf{z}_{k+i|k}, \ i=0, 1, 2, 3, \ldots, f$ 表示在时刻 $k$ 得到的对应时刻 $k+i$ 的输出向量预测.

模型预测控制器的目标是在时刻 $k$ 确定一组控制输入 $\mathbf{u}_{k+i|k}, \ i=0, 1, 2, 3, \ldots, v-1$, 使系统输出 $\mathbf{z}_{k+i}, \ i=1, 2, 3, \ldots, f$ 在预测时域内跟随给定输出轨迹（期望控制轨迹）. 这是通过利用当前状态 $\mathbf{x}_{k}$ 以及系统模型矩阵 $A, B, C$ 预测输出轨迹 $\mathbf{z}_{k+i|k}, \ i=0, 1, 2, 3, \ldots, f$, 并最小化期望轨迹与预测轨迹之间的差异来实现的. 后续将进一步阐明.

设当前处于时刻 $k$, 希望对直到时刻 $k+f$ 的状态与输出进行预测. 另设时刻 $k$ 的状态完全已知, 即 $\mathbf{x}_{k}$ 已知. 由式 (1) 可得在时刻 $k+1$

$$
\begin{equation}
\begin{array}{c}

    x_{k+1|k} = A x_{k} + B u_{k|k} \\
    \mathbf{z}_{k+1|k} = C x_{k+1|k} = C A x_{k} + C B u_{k|k}

\end{array}
\end{equation}
$$

随后, 利用上一式, 可以写出时刻 $k + 2$ 的状态与输出预测

$$
\begin{equation}
\begin{array}{c}
x_{k+2|k} = A x_{k+1|k} + B u_{k+1|k} = A^2 x_{k} + A B u_{k|k} + B u_{k+1|k} \\
\mathbf{z}_{k+2|k} = C x_{k+2|k} = C A^2 x_{k} + C A B u_{k|k} + C B u_{k+1|k}
\end{array}
\end{equation}
$$

同理, 对于时刻 $k + 3$ 可得

$$
\begin{equation}
\begin{array}{c}
x_{k+3|k} = A x_{k+2|k} + B u_{k+2|k} = A^3 x_{k} + A^2 B u_{k|k} + A B u_{k+1|k} + B u_{k+2|k} \\
\mathbf{z}_{k+3|k} = C x_{k+3|k} = C A^3 x_{k} + C A^2 B u_{k|k} + C A B u_{k+1|k} + C B u_{k+2|k}
\end{array}
\end{equation}
$$

上述等式可写成紧凑形式

$$
\begin{equation}
\begin{array}{c}
\mathbf{z}_{k+3|k} = C A^3 x_{k} + \left[ C A^2 B \quad C A B \quad C B \right] \left[ \begin{array}{c} u_{k|k} \\ u_{k+1|k} \\ u_{k+2|k} \end{array} \right]
\end{array}
\end{equation}
$$

将该过程持续至时间索引 $v$ ($v$ 为控制时域), 可得

$$
\begin{equation}
\begin{array}{c}
\mathbf{z}_{k+v|k} = C A^{v} x_{k} + \left[ C A^{v-1} B \quad C A^{v-2} B \quad \ldots \quad C B \right] \left[ \begin{array}{c} u_{k|k} \\ u_{k+1|k} \\ \vdots \\ u_{k+v-1|k} \end{array} \right]
\end{array}
\end{equation}
$$

索引 $v$ 之所以重要, 主要是因为在该索引之后, 我们有

$$
\begin{equation}
\mathbf{u}_{k+v-1|k} = \mathbf{u}_{k+v|k} = \mathbf{u}_{k+v+1|k} = \ldots = \mathbf{u}_{k+f-1|k}
\end{equation}
$$

这是我们对控制器施加的约束。利用该等式, 可以得到时刻 $k+v+1$ 的预测。

$$
\begin{equation}
\begin{array}{c}
\mathbf{z}_{k+v+1|k} = C A^{v+1} x_{k} + \left[ C A^{v} B \quad C A^{v-1} B \quad \ldots \quad CA^{2}B \quad C(A+I) B \right] \left[ \begin{array}{c} u_{k|k} \\ u_{k+1|k} \\ \vdots \\ u_{k+v-1|k} \\ u_{k+v-1|k} \end{array} \right]
\end{array}
\end{equation}
$$

采用相同步骤并对时间索引进行平移, 可得到最终时刻 $k+f$ 的预测

$$
\begin{equation}
\begin{array}{c}
\mathbf{z}_{k+f|k} = C A^{f} x_{k} \\
* \left[ C A^{f-1} B \quad C A^{f-2} B \quad \ldots \quad CA^{f-v+1}B \quad C(A^{f+1} + A^{f-v-1} + \ldots A + I) B \right] \left[ \begin{array}{c} u_{k|k} \\ u_{k+1|k} \\ \vdots \\ u_{k+v-1|k} \\ u_{k+v-1|k} \\ \vdots \\ u_{k+v-1|k} \end{array} \right]
\end{array}
\end{equation}
$$

上述等式可写成紧凑形式

$$
\begin{equation}
\begin{array}{c}
\mathbf{z}_{k+f|k} = C A^{f} x_{k} + \left[ C A^{f-1} B \quad C A^{f-2} B \quad \ldots \quad CA^{f-v+1}B \quad C\bar{A}_{f, v} B \right] \left[ \begin{array}{c} u_{k|k} \\ u_{k+1|k} \\ \vdots \\ u_{k+v-1|k} \end{array} \right]
\end{array}
\end{equation}
$$

其中

$$
\begin{equation}
\bar{A}_{f, v} = A^{f-v} + A^{f-v-1} + \ldots + A + I
\end{equation}
$$

在一般情形中, 对于 $i>v$, 我们引入下列记号

$$
\begin{equation}
\bar{A}_{i, v} = A^{i-v} + A^{i-v-1} + \ldots + A + I
\end{equation}
$$

将所有预测等式合并为单一等式, 可得

$$
\begin{equation}
\begin{array}{c}
\left[ \begin{array}{c} \mathbf{z}_{k+1|k} \\ \mathbf{z}_{k+2|k} \\ \mathbf{z}_{k+3|k} \\ \vdots \\ \mathbf{z}_{k+v|k} \\ \mathbf{z}_{k+v+1|k} \\ \vdots \\ \mathbf{z}_{k+f|k} \end{array} \right] = \\
\left[ \begin{array}{c} C A \\ C A^{2} \\ C A^{3} \\ \vdots \\ CA^v \\ CA^{v+1} \\ \vdots \\ C A^{f} \end{array} \right] x_{k} +
\left[ \begin{array}{cccccccc} C B & 0 & 0 & 0 & \ldots & 0 \\ C A B & C B & 0 & 0 & \ldots & 0 \\ C A^{2} B & C A B & CB & 0 & \ldots & 0 \\ \vdots & \vdots & \vdots & \ldots & \ddots & \vdots \\  CA^{v-1}B & CA^{v-2}B & CA^{v-3}B & \ldots & CAB & CB \\ CA^{v}B & CA^{v-1}B & CA^{v-2}B & \ldots & CA^2B & C\bar{A}_{v+1, v}B \\ \vdots & \vdots & \vdots & \ldots & \ddots & \vdots \\ C A^{f} B & C A^{f-1} B & CA^{f-2}B & \ldots & C A^{f-v+1} B & C\bar{A}_{f, v}B \end{array} \right] \left[ \begin{array}{c} u_{k|k} \\ u_{k+1|k} \\ u_{k+2|k} \\ \vdots \\ u_{k+v-2|k} \\ u_{k+v-1|k} \end{array} \right]
\end{array}
\end{equation}
$$

其中 $\bar{A}_{v+1, v}$ 在等式 (12) 中以 $i=v+1$ 的情形定义, 而 $\bar{A}_{f, v}$ 在等式 (11) 中定义。

上述等式可进一步写成紧凑形式如下。

$$
\begin{equation}
\mathbf{z} = O x_{k} + M u
\end{equation}
$$

其中

$$
\begin{equation}
\begin{array}{l}
\mathbf{z} = \left[ \begin{array}{c} \mathbf{z}_{k+1|k} \\ \mathbf{z}_{k+2|k} \\ \mathbf{z}_{k+3|k} \\ \vdots \\ \mathbf{z}_{k+f|k} \end{array} \right], \quad
u = \left[ \begin{array}{c} u_{k|k} \\ u_{k+1|k} \\ u_{k+2|k} \\ \vdots \\ u_{k+v-1|k} \end{array} \right], \quad
O = \left[ \begin{array}{c} C A \\ C A^{2} \\ C A^{3} \\ \vdots \\ C A^{f} \end{array} \right], \\

M = \left[ \begin{array}{cccccccc} C B & 0 & 0 & 0 & \ldots & 0 \\ C A B & C B & 0 & 0 & \ldots & 0 \\ C A^{2} B & C A B & CB & 0 & \ldots & 0 \\ \vdots & \vdots & \vdots & \ldots & \ddots & \vdots \\  CA^{v-1}B & CA^{v-2}B & CA^{v-3}B & \ldots & CAB & CB \\ CA^{v}B & CA^{v-1}B & CA^{v-2}B & \ldots & CA^2B & C\bar{A}_{v+1, v}B \\ \vdots & \vdots & \vdots & \ldots & \ddots & \vdots \\ C A^{f} B & C A^{f-1} B & CA^{f-2}B & \ldots & C A^{f-v+1} B & C\bar{A}_{f, v}B \end{array} \right]
\end{array}
\end{equation}
$$

模型预测控制问题可表述如下。目标是跟踪参考输出轨迹。设这些期望输出记为

$$
\begin{equation}
z_{k+1}^{d}, \quad z_{k+2}^{d}, \quad z_{k+3}^{d}, \quad \ldots, \quad z_{k+f}^{d}
\end{equation}
$$

其中, 上标 $d$ 表示该输出为期望值。令向量 $z^d$ 定义为

$$
\begin{equation}
z^d = \left[ \begin{array}{c} z_{k+1}^{d} \\ z_{k+2}^{d} \\ z_{k+3}^{d} \\ \vdots \\ z_{k+f}^{d} \end{array} \right]
\end{equation}
$$

一种自然的建模方式是确定在等式 (15) 中定义的控制输入向量 $u$, 使下列代价函数达到最小值

$$\begin{equation}
\min\limits_{u} \left\| z^d - z \right\|_{2} = \min\limits_{u} (z^d - z)^{T} (z^d - z)
\end{equation}
$$

其中, $z$ 在等式 (15) 中定义。然而, 该作法的问题在于我们无法约束控制输入 $u$ 的幅值。控制输入可能非常大, 因而在实际中无法施加, 或者导致执行器饱和, 从而在反馈控制环中造成严重后果。因此, 我们需要在代价函数中引入对控制输入的惩罚。此外, 需要在代价函数 (18) 中引入权重, 以便更好地控制算法的收敛性。

对控制输入实施惩罚的代价函数部分为

$$
\begin{equation}
\begin{array}{l}
   J_u = \mathbf{u}_{k|k}^{T} Q_0 \mathbf{u}_{k|k} + \\
   (\mathbf{u}_{k+1|k} - \mathbf{u}_{k|k})^{T} Q_1 (\mathbf{u}_{k+1|k} - \mathbf{u}_{k|k}) + \\
   (\mathbf{u}_{k+2|k} - \mathbf{u}_{k+1|k})^{T} Q_2 (\mathbf{u}_{k+2|k} - \mathbf{u}_{k+1|k}) + \ldots + \\ 
   (\mathbf{u}_{k+v-1|k} - \mathbf{u}_{k+v-2|k})^{T} Q_{v-1} (\mathbf{u}_{k+v-1|k} - \mathbf{u}_{k+v-2|k}) 
\end{array}

\end{equation}
$$

其中 $Q_i, \quad i=1, 2, \ldots, v-1$ 为用户选取的对称权重矩阵。引入该代价函数的思想在于惩罚首个施加的控制输入 $\mathbf{u}_{k|k}$ 的幅值以及后续控制输入之间的差异。权重矩阵 $Q_i, \quad i = 1, 2, \ldots, v-1$ 用于对这些输入施加惩罚。接下来需要将代价函数 (19) 写成向量形式。为此可引入下式

$$
\begin{equation}
\begin{bmatrix}
I & 0 & 0 & 0 & \ldots & 0 \\
- I & I & 0 & 0 & \ldots & 0 \\
0 & - I & I & 0 & \ldots & 0 \\
\vdots & \vdots & \vdots & \ldots & \ddots & \vdots \\
0 & 0 & 0 & \ldots & - I & I 
\end{bmatrix} \begin{bmatrix}
\mathbf{u}_{k|k} \\
\mathbf{u}_{k+1|k} \\
\mathbf{u}_{k+2|k} \\
\vdots \\
\mathbf{u}_{k+v-2|k} \\
\mathbf{u}_{k+v-1|k} \\
\end{bmatrix} = 

\begin{bmatrix}
\mathbf{u}_{k|k} \\
\mathbf{u}_{k+1|k} - \mathbf{u}_{k|k} \\
\mathbf{u}_{k+2|k} - \mathbf{u}_{k+1|k} \\
\vdots \\
\mathbf{u}_{k+v-2|k} - \mathbf{u}_{k+v-3|k} \\
\mathbf{u}_{k+v-1|k} - \mathbf{u}_{k+v-2|k}
\end{bmatrix}

\end{equation}
$$

上述表达式可写成紧凑形式

$$
\begin{equation}
W_1 u
\end{equation}
$$

其中

$$
\begin{equation}
W_1 = \begin{bmatrix}
I & 0 & 0 & 0 & \ldots & 0 \\
- I & I & 0 & 0 & \ldots & 0 \\
0 & - I & I & 0 & \ldots & 0 \\
\vdots & \vdots & \vdots & \ldots & \ddots & \vdots \\
0 & 0 & 0 & \ldots & - I & I 
\end{bmatrix}
\end{equation}
$$

在等式 (15) 中定义。于是代价函数 (19) 可写为

$$
\begin{equation}
J_u = (W_1 u)^{T} W_2 (W_1 u) = u^{T} W_1^{T} W_2 W_1 u = u^{T} W_3 u
\end{equation}
$$

其中

$$
\begin{equation}
\begin{array}{l}
    W_3 = W_1^{T} W_2 W_1 \\
    W_2 = \begin{bmatrix}
    Q_0 & 0 & 0 & 0 & \ldots & 0 \\
    0 & Q_1 & 0 & 0 & \ldots & 0 \\
    0 & 0 & Q_2 & 0 & \ldots & 0 \\
    \vdots & \vdots & \vdots & \ldots & \ddots & \vdots \\
    0 & 0 & 0 & \ldots & 0 & Q_{v-1}
    \end{bmatrix}
\end{array}
\end{equation}
$$

其中矩阵 $W_2$ 为块对角矩阵, 主对角块为 $Q_i, \quad i=1, 2, \ldots, v-1$.

接下来引入与跟踪误差对应的代价函数部分

$$
\begin{equation}
\begin{array}{l}
   J_z = (z^d_{k+1} - z_{k+1|k})^{T} P_1 (z_{k+1}^d - z_{k+1|k}) + (z^d_{k+2} - z_{k+2|k})^{T} P_2 (z_{k+2}^d - z_{k+2|k}) \\ + \ldots + (z^d_{k+f} - z_{k+f|k})^{T} P_f (z_{k+f}^d - z_{k+f|k})
\end{array}
\end{equation}
$$

通过引入

$$
\begin{equation}
W_4 = \begin{bmatrix}
P_1 & 0 & 0 & 0 & \ldots & 0 \\
0 & P_2 & 0 & 0 & \ldots & 0 \\
0 & 0 & P_3 & 0 & \ldots & 0 \\
\vdots & \vdots & \vdots & \ldots & \ddots & \vdots \\
0 & 0 & 0 & \ldots & 0 & P_f
\end{bmatrix}

\end{equation}
$$

其中 $P_i, \quad i=1,2,3,\ldots,f$ 为用户选取的对称权重矩阵。利用等式 (14), (15), (17) 与 (26), 代价函数可写为

$$
\begin{equation}
\begin{array}{l}
J_{z} &= (z^d - z)^{T} W_4 (z^d - z) \\ 
 & = (z^d - O x_{k} - M u)^{T} W_4 (z^d - O x_{k} - M u) \\
 & = (s - M u)^{T} W_4 (s - M u) 
    
\end{array}
\end{equation}
$$

其中

$$
\begin{equation}
s = z^d - O x_{k}
\end{equation}
$$


该部分代价函数用于惩罚期望轨迹与受控轨迹之间的差异。我们分析代价函数 (27) 的这一部分。由于假定状态向量 $x_k$ 已知, 且期望轨迹向量 $z^d$ 给定, 因此可计算等式 (28) 中定义的向量 $s$. 我们的目标是确定向量 $u$. 该向量将通过最小化下列代价函数得到

$$
\begin{equation}
\min\limits_{u} J_{z} + J_{u}
\end{equation}
$$

将等式 (23) 与 (27) 代入 (29), 得到

$$
\begin{equation}
\min\limits_{u}((s - M u)^{T} W_4 (s - M u) + u^{T} W_3 u)
    
\end{equation}
$$

接下来需要最小化代价函数 (30)。为此写成

$$
\begin{equation}
\begin{array}{l}
J = (s - M u)^{T} W_4 (s - M u) + u^{T} W_3 u \\
J = s^{T} W_4 s - s^{T} W_4 M u - u^{T} M^{T} W_4 s + u^{T} M^{T} W_4 M u + u^{T} W_3 u \\
\end{array}
\end{equation}
$$

随后需要最小化表达式 $J$. 可以回顾关于标量函数对向量求导的公式。令 $w$ 为向量, $a$ 为常向量, $H$ 为常对称矩阵

$$
\begin{equation}
\begin{array}{l}
\frac{\partial W^T a}{\partial w} = \frac{\partial a^Tw}{w} = a \\
\frac{\partial w^T H w}{\partial w} = 2 H w
\end{array}
\end{equation}
$$

可对等式 (31) 中的各项求导:

$$
\begin{equation}
\begin{array}{l}
\frac{\partial (s^{T} W_4 s)}{\partial u} = 0 \\
\frac{\partial (s^{T} W_4 M u)}{\partial u} = M^{T} W_4 s \\
\frac{\partial (u^{T} M^{T} W_4 s)}{\partial u} = M^{T} W_4 s \\
\frac{\partial (u^{T} M^{T} W_4 M u)}{\partial u} = 2 M^{T} W_4 M u \\
\frac{\partial (u^{T} W_3 u)}{\partial u} = 2 W_3 u \\
\end{array}
\end{equation}
$$

借助等式 (33)-(37), 可计算等式 (31) 中定义的代价函数 $J$ 的导数

$$
\begin{equation}
\frac{\partial J}{\partial u} = - 2 M^{T} W_4 s + 2 M^{T} W_4 M u + 2 W_3 u
\end{equation}
$$

为了在 $u$ 上找到代价函数的最小值, 令

$$
\begin{equation}
\frac{\partial J}{\partial u} = 0
\end{equation}
$$

由此可得

$$
\begin{equation}
\begin{array}{l}
   \frac{\partial J}{u} = - 2 M^{T} W_4 s + 2 M^{T} W_4 M u + 2 W_3 u = 0 \\
   -2 M^{T} W_4 s + 2 (M^{T} W_4 M + W_3) u = 0 \\
   (M^{T} W_4 M + W_3) u = M^{T} W_4 s 
\end{array}

\end{equation}
$$

根据最后一个等式, 可得到模型预测控制问题的解:

$$
\begin{equation}
\hat{u} = (M^{T} W_4 M + W_3)^{-1} M^{T} W_4 s
\end{equation}
$$

可以证明该解确实将代价函数 $J$ 最小化。可通过计算 Hessian 矩阵并证明其正定性来完成。解 $\hat{u}$ 可写为

$$
\begin{equation}
\hat{u} = \begin{bmatrix}
\hat{u}_{k|k} \\
\hat{u}_{k+1|k} \\
\hat{u}_{k+2|k} \\
\vdots \\
\hat{u}_{k+v-1|k}
\end{bmatrix}
\end{equation}
$$

我们只需其中的首个分量, 即 $\hat{u}_{k|k}$。将该输入施加到系统上后, 等待时刻 $k+1$, 观测状态 $x_{k+1}$ , 为时刻 $k+1$ 构建新的代价函数, 计算其解并将 $\hat{u}_{k+1∣k+1}$ 施加到系统上, 然后对时刻 $k+2$ 重复该过程。


推导得到的模型预测控制算法可概括如下.

步骤 1: 在时刻 $k$, 基于已知状态向量 $x_k$ 以及系统矩阵 $A,B,C$, 构造提升矩阵并计算式 (42) 所给出的解 $\hat{u}$.

步骤 2: 取该解的首个分量 $\hat{u}_{k|k}$ 并施加于系统.

步骤 3: 等待系统响应并获得状态测量 $x_{k+1}$ 将时间索引平移至 $k+1$, 返回步骤 1.