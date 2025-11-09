# Lecture 6

## Merit Functions

How do we do a line search on a root finding problem?

find $x^{\star}$ such that $C(x^{\star}) = 0$

  Defiene scalar merit function $P(x)$ that measures distance to solution

Standard Choices
  
$$P(x) = \frac{1}{2}C(x)^{T}C(x) - \frac{1}{2} \lVert C(x)\rVert_{2}^{2}$$

$$P(x) = \lVert C(x) \rVert_{1}$$

Now just do Armijo on $P(x)$:

  $$\begin{array}{l}

    \alpha = 1 \\
    \text{while } P(x+\alpha\triangle x) > P(x) + b\alpha\triangledown P(x)^{T}\triangle x \\
    \quad \alpha \leftarrow \theta\alpha \\
    end \\
    x \leftarrow x + \alpha \triangle x

  \end{array}$$

How about constrained optimization?

$$
\begin{rcases}
\begin{array}{l}
\min\limits_{x} f(x) \\
\text{s.t. } C(x) \leq 0 \\
\quad d(x) = 0
\end{array}
\end{rcases}
h(x, \lambda, \mu) = f(x) + \lambda^{T}C(x) + \mu^{T}d(x)
$$

Let's of options for merit functions:

$$P(x, \lambda, \mu) = \frac{1}{2}\lVert r_{kkt}(x, \lambda, \mu)\rVert_{2}^{2}$$

where $r_{kkt}(x, \lambda, \mu) = \begin{bmatrix}
\triangledown_{x} L \\
\min(\theta, C(x)) \\
d(x)  
\end{bmatrix}$

$$
P(x, \lambda, \mu) = f(x) + \rho \lVert \begin{bmatrix}
\min(0, C(A)) \\
dx
\end{bmatrix}\rVert_1
$$

where $\rho$ is scalar weight.

$$P(x, \lambda, \mu) = L_{\rho}(x, \lambda, \mu)$$
