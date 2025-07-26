# Lecture 2

## Motivation

* In general, we can't solve $\dot{x} = f(x)$ for $x(t)$
* Computationally, need to represent $x(t)$ with discrete $X_n$
* Discrete-time models can capture some effects that cantinaous ODEs, can't

## Discret-Time Dynamics

Explicit Form:

$$x_{n+1} = f_d(x_u, u_n)$$

* Simplest discritization:

Forward Euler Integration

$$x_{n+1} = x_n + hf(x_u, u_n)$$

where $h$ is time step

* Pendulum sim:

$$l = m = 1, \ h =0.1, 0.01, 0.001$$

## Stability of Discrete-Time Systems

* In continuous time:

$$Re[eigvals(\frac{\partial{f}}{\partial{x}})] < 0 \Rightarrow stable$$

* In discrete time, dynamics is an iterated map:

$$x_n = f_d(f_d(\dots f_d(x_0)))$$

* Linearize + apply chain rule:

$$\frac{\partial{x_N}}{\partial{x_0}} = \frac{\partial{f_d}}{\partial{x}}\vert_{x_0}\frac{\partial{f_d}}{\partial{x}}\vert_{x_0} \dots \frac{\partial{f_d}}{\partial{x}}\vert_{x_0} = A_d^N$$

* Assume $x=0$ is an equilibrisun

$$ stable \Rightarrow \lim\limits_{k \to \infty}A_d^kx_0 = 0 \; \forall x_0 \\ \Rightarrow \lim\limits_{k \to \infty} A_d^n = 0 \\ \Rightarrow \vert eigvals(A_d) \vert < 1$$

* Let's do this for the pendulum + forward Euler

$$x_{n+1} = x_n + hf(x_n)$$

$$A_d = \frac{\partial{f_d}}{\partial{x_n}} = I + hA = I + h \begin{bmatrix}0 & 1 \\ -\frac{q}{l}\cos{x} & 0\end{bmatrix}$$

$$eigvals(A_a\vert_{0:0}) \approx 1 + 0.313i$$

* Plot $\vert eigvals(A_d) \vert$ vs. $h$

$$\Rightarrow Only \, (marginally) stable \  as \  h \rightarrow 0$$

* Take-away messages:

  + Be careful when discreting ODEs
  + Sanity check eg energy, momentum behavior
  + Don't use forward Euler integration

* A better explicit integrator:
  + 4Th - Order Runge-Kutta Method
  + Intvitior
    - Euler fits a line segment over each time step
    - RK4 fits a cubic polynomial $\Rightarrow$ much better accuracy
* Pseudo Code:

$$X_{n+1} = f_d(X_n)$$
$$K_1 = f(X_n)$$
$$K_2 = f(X_n + \frac{h}{2}K_1)$$
$$K_3 = f(X_n + \frac{h}{2}K_2)$$
$$K_4 = f(X_n + \frac{h}{2}K_3)$$
$$X_{n+1} = X_n + \frac{h}{6}(K_1 + 2K_2 + 2K_3 + 2K_4)$$

* Take Away:

* Accuracy win $\gg$ addition compute cose
* Even good integrators have issues -> always sanity check
* Implicit Form: $f_d(X_{n+1}, X_n, u_m)=0$$
* Simplest version: $X_{n+1} = X_n + hf(X_{n+1})$
* How do we simulate
  
  write as:
  $$f_d(X_{n+1}, X_n, u_n) = X_n + hf(X_{n+1}) - X_{n+1} = 0$$
  + solve root-finding problem for $X_{n+1}$

* Pendulum Sim:
  + Opposite energy behavior from forward Euler
  + Artificial dumping
  + While unphysical, this allows simulators to take big steps and is often convenient
    $\Rightarrow$ Very common in low-fi similators in graphics/robotics

Take-Aways:

* Implicit methods typically "more stable" then explicit
* For forward sim, implicit methods are generally more exensive
* In many trajectery optimization methods they're not more expenssive
