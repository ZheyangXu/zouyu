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

