# Lecture 4

## Linear Search

* Often $\triangle x$ step from Newton overshoots the minimum
* To fix this, check $f(x + \triangle x)$ and bachtrack until we get a good reduction
* Many Strategics
* A simple effecfive one is Armijo Rule

$$\alpha = 1$$
$$while f(x + \alpha\triangle{x}) \gt f(x) + b\alpha\triangledown f(x)^T\triangle x$$
  $$\alpha \leftarrow c\alpha$$
end

* Intuition:
  + Make sure step agress with linearization within some tolerance $b$
* Typical values: $c=\frac{1}{2}$, $b = 10^{-4} ~ 0.1$
* Take Away:
  + Newton with sinple cheap modifications(globalization strategics) is extremely effecfive at finding local optimn
  

## Equality Constraints

$$\min\limits_x f(x) \leftarrow f(x): \mathcal{R}^n \rightarrow \mathcal{R}$$
$$s.t. C(x) = 0 \leftarrow C(x): \mathcal{R}^n \rightarrow \mathcal{R}^m$$

* First-Order Necessary Conditions:
  + Need $\triangledown f(x)= 0$ in free direction
  + Need $C(x) = 0$

* Any non-zero component of $\triangle f$ must be normal to the constraint surface/manifold
  
$$\Rightarrow \triangledown f + \lambda\triangledown C = 0$$

* In general:

$$\frac{\partial{f}}{\partial{x}} + \lambda^T\frac{\partial{c}}{\partial{x}} = 0, \quad \lambda \in \mathcal{R}^m$$

* Based on this gradient condition, we define:

$$L(x, \lambda) = f(x) + \lambda^TC(x)$$

* Such that:

$$\bigtriangledown_{x}L(x, \lambda) = \bigtriangledown{f} + (\frac{\partial{C}}{\partial{x}})^T\lambda = 0$$
$$\bigtriangledown_{\lambda}L(x, \lambda) = C(x) = 0$$

* We can solve this with Newton:

$$\bigtriangledown_{x}L(x + \triangledown{x}, \lambda + \triangledown\lambda) \approx \bigtriangledown_{x}L(x, \lambda) + \frac{\partial^2L}{\partial{x^2}}\triangle{x} + \frac{\partial^2{L}}{\partial{x}\partial{y}}\triangle\lambda = 0$$

$$\bigtriangledown_{\lambda}L(x + \triangle{x}, \lambda + \triangle{\lambda}) \approx C(x) + \frac{\partial{c}}{\partial{x}}\triangle{x} = 0$$

$$\begin{bmatrix}\frac{\partial^2L}{\partial{x^2}} & (\frac{\partial{c}}{\partial{x}})^T \\ \frac{\partial{c}}{\partial{x}} & 0\end{bmatrix}\begin{bmatrix}\triangle{x} \\ \triangle{\lambda}\end{bmatrix} = \begin{bmatrix}-\triangledown_{x}L(x, \lambda) \\ -C(x)\end{bmatrix}$$

where is KKT System.

* Gauss-Newton Method:
    $$\frac{\partial^2{L}}{\partial{x^2}} = \triangledown^2f + \frac{\partial}{\partial{x}}[(\frac{\partial{c}}{\partial{x}})^T\lambda]$$

  + We often drop the $2^{nd}$ constraint cornture term
  + Called Gauss - Newton
  + Slightly slower convergence than full Newton(more itterations) but: Herations are cheaper $\Rightarrow$ often wins in wall-clock time
  
* Example:
  + start at $[-1, -1], \quad [-3, 2]$

* Take Aways:
  + May still need to reguliarize $\frac{\partial^2{L}}{\partial{x^2}}$ even if $\bigtriangledown^2f \gt 0$
  + Gauss-Newton is often used in practice

## Inequality Constraints

$$\min\limits_{x} f(x) \\ s.t. C(X) \leq 0$$

* We'll just look at inequalities for now
* Just combine with previews methods to handle both kinds of constraints

### First-Order Necessary Conditions:

1. $\bigtriangledown f = 0$ in the free directions
2. $C(x) \leq 0$

$$KKT \: Conditions \begin{cases}\triangledown{f} + (\frac{\partial{C}}{\partial{x}})^{\tau}\lambda & = 0 & \leftarrow stationarity\\ C(x) & \leq 0 & \leftarrow primal \: feasibility \\ \lambda & \geq 0 & \leftarrow dual \: feasibility \\ \lambda \odot C(x) = \lambda^T C(x) & = 0 & \leftarrow complementarity \end{cases}$$

### Intuition

* If constraint is active $(C(x) = 0 \Rightarrow \lambda \gt 0$
* If constraint is inactive $(C(x) \lt 0) \Rightarrow \lambda = 0$
* Complementarity corcodes onlett switching

$$\begin{array}{l}\min\limits_{x} \frac{1}{2}x^TQx + q^T x, Q \gt 0 \\ s.t. \quad Ax \leq b \\ \quad Cx=d\end{array}$$
