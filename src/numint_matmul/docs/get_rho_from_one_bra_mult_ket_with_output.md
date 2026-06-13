# Equations and Concepts

Since bra and ket may differ, the density matrix $D_{\mu\nu} = \sum_i \mathrm{bra}_{\mu i} \mathrm{ket}_{\nu i}$ is not necessarily symmetric, so both cross terms must be computed explicitly.

- Orbital grids are defined as

    $$
    \begin{aligned}
    \phi_{g i}^{\mathrm{bra}} &= \sum_{\mu} \phi_{g \mu} \mathrm{bra}_{\mu i}
    \\\\
    \phi_{g i}^{\mathrm{ket}} &= \sum_{\mu} \phi_{g \mu} \mathrm{ket}_{\mu i}
    \\\\
    \phi_{g i, t}^{\mathrm{bra}} &= \sum_{\mu} \phi_{g \mu, t} \mathrm{bra}_{\mu i}
    \\\\
    \phi_{g i, t}^{\mathrm{ket}} &= \sum_{\mu} \phi_{g \mu, t} \mathrm{ket}_{\mu i}
    \end{aligned}
    $$

- `RHO`:

    $$
    \rho_g = \sum_{i} \phi_{g i}^{\mathrm{bra}} \phi_{g i}^{\mathrm{ket}}
    $$

- `SIGMA`:

    $$
    \sigma_{g t} = \sum_{i} \left( \phi_{g i}^{\mathrm{bra}} \phi_{g i, t}^{\mathrm{ket}} + \phi_{g i, t}^{\mathrm{bra}} \phi_{g i}^{\mathrm{ket}} \right)
    $$

    Unlike the symmetric case, the two terms are not equal in general so both must be computed.

- `TAU`:

    $$
    \tau_g = \frac{1}{2} \sum_{i} \sum_{t} \phi_{g i, t}^{\mathrm{bra}} \phi_{g i, t}^{\mathrm{ket}}
    $$

- `LAPL`:

    $$
    \nabla^2 \rho_g
    = 4 \tau_g + \sum_{i} \sum_{t} \left( \phi_{g i}^{\mathrm{bra}} \phi_{g i, tt}^{\mathrm{ket}} + \phi_{g i, tt}^{\mathrm{bra}} \phi_{g i}^{\mathrm{ket}} \right)
    $$
