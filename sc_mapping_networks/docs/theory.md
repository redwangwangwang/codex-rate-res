# Theory sketch: Subspace-Certified Mapping Networks

Original Mapping Networks justify latent-to-weight training by assuming that good network weights lie near a low-dimensional smooth manifold and then showing the existence of a smooth map into the full parameter space. SC-Mapping makes the claim diagnosable by using an explicit data-aligned tangent subspace.

Let the empirical risk be \(\hat R(\theta)\), the full target parameter be \(\theta\in\mathbb{R}^P\), and the low-dimensional model be

\[
\theta(z)=\theta_0 + U z, \qquad U^\top U=I_d,\quad z\in\mathbb{R}^d.
\]

The basis \(U\) is constructed from the top left singular vectors of a gradient sketch

\[
G=[\nabla_\theta \ell_{B_1}(\theta_0),\ldots,\nabla_\theta \ell_{B_K}(\theta_0)].
\]

## Excess-risk decomposition

Assume \(\hat R\) is \(\beta\)-smooth around the relevant path and satisfies a \(\mu\)-PL condition inside \(\theta_0+\mathrm{span}(U)\). Let

\[
\theta_U^* = \arg\min_{z}\hat R(\theta_0+Uz),
\qquad
\epsilon_U=\|(I-UU^\top)(\theta^*-\theta_0)\|.
\]

Gradient descent in latent space,

\[
z_{t+1}=z_t-\eta U^\top \nabla_\theta \hat R(\theta_0+Uz_t),
\qquad 0<\eta\le 1/\beta,
\]

satisfies

\[
\hat R(\theta_0+Uz_T)-\hat R(\theta_U^*)
\le (1-\eta\mu)^T
\left[\hat R(\theta_0+Uz_0)-\hat R(\theta_U^*)\right].
\]

By smoothness, the approximation penalty is controlled by the projection residual:

\[
\hat R(\theta_U^*)-\hat R(\theta^*) \le \frac{\beta}{2}\epsilon_U^2.
\]

With a bounded/Lipschitz loss and \(\|z\|\le R\), the population risk obeys a finite-sample bound of the form

\[
R(\theta_0+Uz_T)-R(\theta^*)
\lesssim
\underbrace{\frac{\beta}{2}\epsilon_U^2}_{\text{projection error}}
+
\underbrace{(1-\eta\mu)^T\Delta_0}_{\text{optimization error}}
+
\underbrace{\mathcal{O}\!\left(\sqrt{\frac{d\log n+\log(1/\delta)}{n}}\right)}_{\text{generalization error}}.
\]

This gives a cleaner and more testable claim than an unconstrained existence proof: the method succeeds when the task update is well captured by the measured subspace, and failure is predicted by a high projection residual.
