# Equations and Concepts

Please note that since we assumed the homogeneous braket form, the density matrix is symmetric, and some factors of two is applicable to `SIGMA` and `LAPL`.

- Orbital grids are defined as

    $$
    \begin{aligned}
    \phi_{g i} &= \sum_{\mu} \phi_{g \mu} C_{\mu i}
    \\\\
    \phi_{g i, t} &= \sum_{\mu} \phi_{g \mu, t} C_{\mu i}
    \end{aligned}
    $$

- `RHO`:

    $$
    \rho_g = \sum_{i} \phi_{g i}^2
    $$

- `SIGMA`:

    $$
    \sigma_{g t} = 2 \sum_{i} \phi_{g i} \phi_{g i, t}
    $$

- `TAU`:

    $$
    \tau_g = \frac{1}{2} \sum_{i} \sum_{t} \phi_{g i, t}^2
    $$

- `LAPL`:

    $$
    \nabla^2 \rho_g
    = 4 \tau_g + 2 \sum_{i} \sum_{t} \phi_{g i} \phi_{g i, tt}
    $$
