# Lecture 3

## Some Notation:

Given $f(x): \mathcal{R^n} \rightarrow \mathcal{R}$, $\frac{\partial{f}}{\partial{x}}\in\mathcal{R^{1\times n}}$ is a row vector

This is because $\frac{\partial{f}}{\partial{x}}$ is the linear opertator mapping $\triangle X$ into $\triangle{f}$:

$$f(x+\triangle x) \approx f(x) + \frac{\partial{f}}{\partial{x}}\triangle{x}$$

Similarity $g(y): \mathcal{R^m}\rightarrow\mathcal{R^n}$

$$\frac{\partial{g}}{\partial{y}} \in \mathcal{R^{n\times m}}$$

because:

$$g(y + \triangle{y}) \approx g(y) + \frac{\partial{g}}{\partial{y}}\triangle{y}$$

This conventions make the chain rule work:

$$f(g(y + \triangle{y})) \approx f(g(y)) + \frac{\partial{f}}{\partial{x}} \vert_{g(y)} \frac{\partial{g}}{\partial{y}}\vert_{y}\triangle{y}$$

For convenience, we will define:

$$\nabla f(x) = (\frac{\partial{f}}{\partial{x}})^T \in \mathcal{R}^{n \times 1}$$
$$\nabla^2f(x) = \frac{\partial}{\partial{x}}(\nabla{f(x)}) = \frac{\partial^2f}{\partial{x^2}} \in \mathcal{R}^{n\times m}$$
$$f(x+\triangle{x}) \approx f(x) + \frac{\partial{f}}{\partial{x}} + \frac{1}{2}\triangle{x}^T\frac{\partial^2{f}}{\partial{x^2}}\triangle{x}$$

## Root Finding

* Given $f(x)$, find $x^{\star}$ such that $f(x^{\star}) = 0$
* Example: equilibrivm of a continuous-time dynamics
* Closely related: find point such that

$$f(x^{\star})=x^{\star}$$

* Example: equilibrivm of discrete-time dynamics
* Fixed-point Itteration
  + Simplest solution method
  + If fixed point is stable, just "itteractive the dynamics" until it converges to $x^{\star}$
  + Only worse for stable $x^{\star}$ and has slow convergence

## Newton's Method

* Fit a linear approximation to $f(x)$:

$$f(x+\triangle{x}) \approx f(x) + \frac{\partial{f}}{\partial{x}}\vert_{x}\triangle_{x}$$

* Set approximation to zero and solve for $\triangle{x}$:

$$f(x) + \frac{\partial{f}}{\partial{x}}\triangle{x} = 0 \Rightarrow \triangle{x} = -(\frac{\partial{f}}{\partial{x}})^{-1}f(x)$$

* Apply correction

$$x \leftarrow x + \triangle{x}$$

* Repear until convergence
* Example: Backword Euler
  + Very fast convergence with Newton(Quadratic)
  + Can get machine precision
  + Most expensive part is solving a linear system $O(n^3)$
  + Can improve complexity by taking advantage of problem structure(more later)

## Minimization:

$$\min\limits_x f(x), \ f(x): \mathcal{R}^n \rightarrow \mathcal{R}$$

* If $f$ is smooth, $\frac{\partial{f}}{\partial{x}}|_{x^{\star}} = 0$ at a local minimum
* Now we have a root-finding problem $\triangledown f(x) = 0$ $\Rightarrow$ Apply Newtown!

$$\triangledown f(x + \triangle x) \approx \triangledown f(x) + \frac{\partial}{\partial{x}}(\triangledown f(x))\triangle x = 0$$

$$\Rightarrow \triangle x = -(\triangledown^2f(x))^{-1}\triangledown f(x)$$

$$x \leftarrow x + \triangledown x$$

repeat until convergence

* Intiuition:
  + Fit a quadratic approximation to $f(x)$
  + Exactly minimize approximation

* Example
  
$$\min\limits_x f(x) = x^4 + x^3 - ^2 - x$$

start at: 1.0, -1.3 0.03

## Take Away Messages

* Newton is a local root-finding method will coverge to the closest fixed point to the initial guess(min, max, saddle).
* Suffient Conditions
  + $\triangledown f(x) = 0$ "first-order necessary condition" for a minimum. Not a sufficient condition.
  + Let's look at scalar case:
  
  $$\triangle x = -(\triangledown^2 f)^{-1}\triangledown f$$
  $$\triangledown^2 f \geq 0 \Rightarrow descent (minimization)$$
  $$\triangledown^2 f \le 0 \Rightarrow ascent (maximization)$$

  + IN $\mathcal{R}^n$, $\triangledown^2 f \ge 0$, $\triangledown^2 f \in \mathcal{S}^n_{++}$ $\Rightarrow descent$
  + If $\triangledown^2 f(x) \ge 0, \quad \forall x \Leftrightarrow f(x)$ is strongly convex $\Rightarrow$ can always solve with Newton
  + Usually not free for hard/nonlinear problems

* Regularization:
  + Pratical solution tom make sure whire always minimizing
  
$$H \leftarrow \triangledown^2f$$
while $H \nsucc 0$ 
  $H \leftarrow H + \beta I(\beta \gt 0)$
end
$$\triangle x = -H^{-1}\triangledown f$$
$$x \leftarrow x + \triangledown x$$

* Also called damped Newton
* Gvararities descent + shrinks step
