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

  * Repeat until convergence:
    * $\min\limits_{x}L_p(x, \tilde{\lambda})$
    - $\tilde{\lambda} \leftarrow \max(0, \tilde{\lambda} + pc(x))$
    - $p \leftarrow \alpha p$
    - Fixes ill-conditioning of penalty method
    - Converges with finite $p$
    - Works well on non-convex problem

Quadratic Program

$$ \min\limits_{x} \frac{1}{2}x^TQx + q^T x, Q \gt 0 \\ \begin{align} s.t. \quad &Ax \leq b \\ \quad &Cx=d \end{align}$$

* Super useful in control
* Can be solved very fast (~ kHz)

* Example
  + Try with penalty, full AL, just $\lambda$ updates

Reqularization

* Given:

$$\min\limits_{x} f(x) \\ s.t. \quad c(x)=0$$

* We might like to turn this into:

$$\min\limits_{x}\max\limits_{\lambda}f(x) + \lambda^{T}C(x)$$

* Whenever $C(x)\neq 0$, inner max problem blow up
* Similar for inequlities

$$\begin{rcases}\min\limits_{x}f(x) \\ s.t. \quad C(x) \leq 0 \end{rcases} \rightarrow \min\limits_{x}f(x) + p_{\infin}^{+}(C(x))$$

$$\Rightarrow \min\limits_{x}\max\limits_{\lambda \geq 0} f(x) + \lambda^T C(x)$$

where $p_{\infin}^{+} = \begin{cases}0, & C(x) \leq 0 \\ +\infin, & C(x) \gt 0\end{cases}$

* Interpretation: KKT conditions define a saddle point in $(x, \lambda)$
* KKT system should have $dim(x)$ pos. eigenvalues and $dim(\lambda)$ neg. eigenvalues at an optimum called "quasi definite"

$$\begin{bmatrix}H + \beta I & C^T \\ C & -\beta I\end{bmatrix} \begin{bmatrix}\triangle{x} \\ \triangle{\lambda}\end{bmatrix} = \begin{bmatrix}-\nabla_{x}L \\ -C(x)\end{bmatrix}, \quad \beta \gt 0$$

* Examples:
  + Still overshoot $\Rightarrow$ need line search.
