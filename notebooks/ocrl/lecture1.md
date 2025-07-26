# Lecture 1

## Continuous Time Dynamics

The most general/generic for a smooth system:

$$\dot{x} = f(x, u)$$

where $x$ is state $x \in \mathbb{R}^n$, $\dot{x}$ is state time derivative, $u$ is input $u \in \mathbb{R}^m$, $f$ is dynamics.

For a mechanical system: 

$$x = \begin{bmatrix} q \\ v \end{bmatrix}$$

Example Pendulum

$$ml^2 \ddot{\theta} + mgl\sin(\theta) = \tilde{l}$$

where $q=\theta$, $v=\dot{\theta}$, $u=\tilde{l}$

$$x=\begin{bmatrix}\theta \\ \dot{\theta} \end{bmatrix} \Rightarrow \dot{x}=\begin{bmatrix}\dot{\theta} \\ \ddot{\theta} \end{bmatrix} = \begin{bmatrix}\dot{\theta} \\ -\frac{q}{l}\sin(\theta) + \frac{1}{ml^2}u \end{bmatrix}$$

where $q \in S^\prime(circle)$, $x \in S^\prime \times \mathbb{R}(cylinder)$

## Control Affine Systems

$$\dot{x} = f_0(x) + B(x)u$$

where $f_0(x)$ is drift, $B(x)u$ is input Jacobian.

* most systems can be put in this form.
* Pendulum:
    $$f_0(x) = \begin{bmatrix}\dot\theta \\ -\frac{q}{l}\sin(\theta) \end{bmatrix}$$

## Manipulator Dynamics

$$M(q)\dot{v} + C(q, v) = B(q)u \cdot F$$

where $M(q)$ is Mass Matrix, $C(q, v)$ is Dynamic Bias, $B(q)$ is Input Jacobian.

$$ \dot{q} = G(q)v $$ 

Where this is Velocity Kinematics

$$\dot{x} = f(x, u) =  \begin{bmatrix} G(q)v \\ M(q)^{-1}(B(q)u + F - C)\end{bmatrix}$$

### Pendulum

$$M(q) = ml^2, \ C(q, v) = mgl\sin(\theta), B = I, G = I$$

* All mechanical systems can be written like this
* This is just a way of rewriting the Euler-Lagrange equation for:

$$L = \frac{1}{2}v^{+}M(q)v - V(q)$$

where item 1 is kinetic energy, item 2 is potential energy.

## Linear Systems

$$\dot{x} = A(t)(x) + B(t)u$$

* Called `time invariant` if $A(t) = A$, $B(t) = B$
* Called `time varying` otherwise
* Super important in control
* We often appraximate nonlinear system with liner ones:
  
  $$\dot{x} = f(x, u) \Rightarrow A=\frac{\partial{f}}{\partial{x}}, \  B = \frac{\partial{f}}{\partial u}$$

## Equilibria

* A point where a system will remain at rest

   $$\Rightarrow \dot{x} = f(x, u) = 0$$

* Algebraically, roots of the dynamics
* Pendulum:
    $$\dot{x} = \begin{bmatrix}\dot{\theta} \\ -\frac{q}{l}\sin(\theta)\end{bmatrix} = \begin{bmatrix}0 \\ 0 \end{bmatrix} \Rightarrow \begin{matrix}\dot{\theta} = 0 \\ \theta=0, \pi\end{matrix}$$

## First Control Problem

* Can I move the equilibria
    $$\dot{x} = \begin{bmatrix}\dot{\theta} \\ -\frac{q}{l}\sin(\frac{\pi}{2}) + \frac{1}{ml^2}u\end{bmatrix} = \begin{bmatrix}0 \\ 0\end{bmatrix}$$

    $$\frac{1}{ml^2}u = \frac{q}{l}\sin(\frac{\pi}{2}) \Rightarrow u = mgl$$

* In general, we get a root-finding problem in $u$:
    $$f(x, u) = 0$$

## Stability of Equilibria

* When will we stay "near" an equilibrium point under pertorbutions?
* Look at a 1D system($x \in \mathbb{R}$)
  $$\frac{\partial{f}}{\partial{x}} < 0 \Rightarrow stable, \  \frac{\partial{f}}{\partial{x}} > 0 \Rightarrow unstable$$
* In higher dimensions: $\frac{\partial{f}}{\partial{x}}$ is a Jacobian Matrix.
* Take an eigendecmposition $\Rightarrow$ decomple into n 1D systems
    $$\Re[eigvals(\frac{\partial{f}}{\partial{x}})] < 0 \Rightarrow stable \\ otherwise \Rightarrow unstable$$

* Pendulum:
    $$f(x) = \begin{bmatrix}\dot{\theta} \\ -\frac{q}{l}\sin(\theta)\end{bmatrix} \Rightarrow \frac{\partial{f}}{\partial{x}} = \begin{bmatrix}0 & 1 \\ -\frac{q}{l}\cos(\theta) & 0\end{bmatrix}$$

    $$\frac{\partial{f}}{\partial{x}}\lvert_{\theta=\pi} = \begin{bmatrix}0 \ 1 \\ -\frac{q}{l} \ 0\end{bmatrix} \Rightarrow eigvals(\frac{\partial{f}}{\partial{x}}) = \pm \dot{l}\sqrt{\frac{q}{l}} \\ \Rightarrow undamped \; oscillations$$

* Add damping (e.g. $u=-k_d\dot{\theta}$) results in strictly negative real part.
