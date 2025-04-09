# Lecture 5

## Inequality Constrained Minimization

$$\min\limits_{x}f(x) \\ s.t. \quad C(x) \leq 0$$

KKT Conditions:

$$KKT \: Conditions \begin{cases}\triangledown{f} + (\frac{\partial{C}}{\partial{x}})^{\tau}\lambda & = 0 & \leftarrow stationarity\\ C(x) & \leq 0 & \leftarrow primal \: feasibility \\ \lambda & \geq 0 & \leftarrow dual \: feasibility \\ \lambda \odot C(x) = \lambda^T C(x) & = 0 & \leftarrow complementarity \end{cases}$$

Optimization Algorithms

* Active-Set Method
  + Gives active/inactive constraints
  + Solve equality constrained problem
  + Used when you have a good hemistic for active set
* Barricr/Interior-Point
  + Replace inequlities with barrier function in objective:
  
  $$\begin{rcases}\min\limits_{x}f(x) \\s.t. \: C(x) \leq 0 \end{rcases} \rightarrow \min\limits_{x} f(x) + \frac{p}{2}[\max(0, c(x))]^2$$

  + Easy to implement
  + Has issues with numerical ill-conditioning
  + Can't archieve high accuracy
* Augmented Lagrangian
  + Add Lagrange multiplier estimate to penalty method:
  
  $$\min\limits_{x} f(x) + \tilde{\lambda}^T c(x) + \frac{p}{2}[\max(0, c(x))]^2$$
  
  + Update $\tilde{\lambda}$ by "offlanding" pealty term into $\tilde{\lambda}$ at each itteration:
  
  $$\frac{\partial{f}}{\partial{x}} + \tilde{\lambda}^T\frac{\partial{c}}{\partial{x}} + pc(x)^T \frac{\partial{c}}{\partial{x}} =\frac{\partial{f}}{\partial{x}} + [\tilde{\lambda} + pc(x)]^T\frac{\partial{c}}{\partial{x}} = 0 \\ \Rightarrow \tilde{\lambda} \leftarrow \tilde{\lambda} + pc(X)$$
