# 宇树人形机器人 G1 基于 A* 与 DWA 的路径规划与避障

## 1. 概述

在 Gazebo 仿真环境中，为宇树 G1 人形机器人部署了一套完整的路径规划与避障系统。该系统由两阶段组成：首先使用 **A\* 算法**在已知静态地图中规划出一条从起点到终点的全局最优路径，然后由 **DWA（Dynamic Window Approach）局部路径规划器**在机器人沿全局路径行进过程中实时感知动态障碍物并进行避障。

---

## 2. A\* 全局路径规划

A\* 算法是一种经典的启发式图搜索算法，融合了 Dijkstra 算法的最优性保证与贪心最佳优先搜索的效率。

### 2.1 算法原理

A\* 算法通过维护 open list 和 closed list，在栅格地图中搜索代价最小的路径。对于每个被探索的节点 $n$，计算其代价函数：

$$
f(n) = g(n) + h(n)
$$

其中：

- $g(n)$：从起点到节点 $n$ 的实际代价
- $h(n)$：从节点 $n$ 到终点的启发式估计代价（采用欧几里得距离或曼哈顿距离）
- $f(n)$：通过节点 $n$ 的估计总代价

算法每次从 open list 中选取 $f(n)$ 最小的节点进行扩展，直至终点被加入 closed list 或 open list 为空。

### 2.2 启发式函数选择

在 G1 的路径规划中，启发式函数 $h(n)$ 使用欧几里得距离：

$$
h(n) = \sqrt{(x_n - x_{goal})^2 + (y_n - y_{goal})^2}
$$

欧几里得距离满足可采纳性（admissible）和一致性（consistent）的条件，保证 A\* 搜索到的是全局最优路径。

### 2.3 算法步骤

1. 将起点加入 open list，初始化 $g(start) = 0$
2. 从 open list 中取出 $f(n)$ 最小的节点作为当前节点
3. 若当前节点为终点，回溯路径并返回
4. 将当前节点移入 closed list
5. 遍历当前节点的所有八邻域邻居，对每个邻居：
   - 若不可通行或已在 closed list 中，跳过
   - 计算试探性 $g$ 值，若更优则更新 $g$、$f$ 和父节点
   - 若邻居不在 open list 中，加入 open list
6. 重复步骤 2-5，直至找到终点或 open list 为空

### 2.4 算法伪代码

```text
Algorithm: A* Search
Input: start, goal, grid_map
Output: path (list of nodes from start to goal)

 1:  OPEN  ← {start}
 2:  CLOSED ← ∅
 3:  g[start] ← 0
 4:  f[start] ← h(start, goal)
 5:  parent[start] ← None
 6:
 7:  while OPEN ≠ ∅ do
 8:      current ← node in OPEN with minimum f
 9:      if current = goal then
10:          return ReconstructPath(parent, current)
11:      end if
12:
13:      Remove current from OPEN
14:      Add current to CLOSED
15:
16:      for each neighbor ∈ GetNeighbors(current, grid_map) do
17:          if neighbor ∈ CLOSED or IsObstacle(neighbor) then
18:              continue
19:          end if
20:
21:          tentative_g ← g[current] + Distance(current, neighbor)
22:
23:          if neighbor ∉ OPEN then
24:              OPEN ← OPEN ∪ {neighbor}
25:          else if tentative_g ≥ g[neighbor] then
26:              continue
27:          end if
28:
29:          parent[neighbor] ← current
30:          g[neighbor] ← tentative_g
31:          f[neighbor] ← g[neighbor] + h(neighbor, goal)
32:      end for
33:  end while
34:
35:  return ∅  // no path found
```

---

## 3. DWA 局部路径规划与避障

DWA（Dynamic Window Approach）是一种基于速度空间的局部路径规划方法，通过考虑机器人的运动学约束，在速度空间 $(v, \omega)$ 中实时搜索使目标函数最优的控制指令。

### 3.1 动态窗口

机器人的线速度 $v$ 和角速度 $\omega$ 受以下约束限制：

**运动学约束**（由最大加减速度决定）：

$$
V_d = \left\{ (v, \omega) \mid v \in [v_c - \dot{v}_{max} \cdot \Delta t, \ v_c + \dot{v}_{max} \cdot \Delta t], \ \omega \in [\omega_c - \dot{\omega}_{max} \cdot \Delta t, \ \omega_c + \dot{\omega}_{max} \cdot \Delta t] \right\}
$$

**硬件约束**（由最大速度决定）：

$$
V_s = \left\{ (v, \omega) \mid v \in [v_{min}, \ v_{max}], \ \omega \in [\omega_{min}, \ \omega_{max}] \right\}
$$

**制动约束**（确保在碰到障碍物前能停下）：

$$
V_a = \left\{ (v, \omega) \mid v \leq \sqrt{2 \cdot dist(v, \omega) \cdot \dot{v}_{max}}, \ \omega \leq \sqrt{2 \cdot dist(v, \omega) \cdot \dot{\omega}_{max}} \right\}
$$

动态窗口即为三者的交集：

$$
V_r = V_s \cap V_d \cap V_a
$$

### 3.2 轨迹评价函数

在动态窗口 $V_r$ 内采样多组 $(v, \omega)$，对每组速度模拟一段短时轨迹，通过评价函数选出最优轨迹：

$$
G(v, \omega) = \alpha \cdot heading(v, \omega) + \beta \cdot dist(v, \omega) + \gamma \cdot velocity(v, \omega)
$$

其中：

- $heading(v, \omega)$：机器人朝向与目标方向的一致性
- $dist(v, \omega)$：轨迹与最近障碍物的距离（越大越安全）
- $velocity(v, \omega)$：线速度大小（鼓励快速到达目标）

### 3.3 算法伪代码

```text
Algorithm: DWA (Dynamic Window Approach)
Input: current_pose, goal, laser_scan, current_v, current_ω, dt
Output: optimal control (v, ω)

 1:  V_s ← [v_min, v_max] × [ω_min, ω_max]                   // hardware limits
 2:  V_d ← [v - a_max·dt, v + a_max·dt] × [ω - α_max·dt, ω + α_max·dt]  // dynamics
 3:  V_r ← V_s ∩ V_d
 4:
 5:  best_score  ← -∞
 6:  best_cmd    ← (0, 0)
 7:
 8:  for each (v, ω) ∈ SampleVelocitySpace(V_r, Δv, Δω) do
 9:      if not IsAdmissible(v, ω, laser_scan) then           // braking constraint
10:          continue
11:      end if
12:
13:      traj ← SimulateTrajectory(current_pose, v, ω, dt, T_sim)
14:
15:      heading ← ComputeHeading(traj, goal)                  // angle to goal
16:      clearance ← MinDistToObstacle(traj, laser_scan)       // obstacle distance
17:      speed ← |v|
18:
19:      score ← α · heading + β · clearance + γ · speed       // evaluate
20:
21:      if score > best_score then
22:          best_score ← score
23:          best_cmd   ← (v, ω)
24:      end if
25:  end for
26:
27:  return best_cmd
```

### 3.4 G1 上的适配

对于 G1 人形机器人，DWA 的控制指令 $(v, \omega)$ 会被送入全向运动控制器，转换为机器人基座坐标系下的速度指令，再通过逆运动学解算为各关节的目标位置，驱动机器人在避障的同时沿全局路径前进。

---

## 4. Gazebo 仿真部署

系统整体架构如下：

1. **地图构建**：通过 FAST-LIO 完成环境的三维点云建图，将三维点云投影为二维栅格地图
2. **全局规划**：在栅格地图上运行 A\* 算法，生成从起点到终点的最优路径
3. **局部避障**：G1 沿全局路径行进，DWA 实时接收激光雷达数据，感知动态障碍物并调整局部轨迹
4. **运动控制**：DWA 输出的速度指令经全向运动控制器转换为关节指令，驱动机器人

---

## 5. 演示

<video width="1080" controls src="../../public/projects/humanoid/astar.mp4"></video>
