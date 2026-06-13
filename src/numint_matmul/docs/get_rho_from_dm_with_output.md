## Equations and Concepts

- `RHO`:

    $$
    \rho_g = \sum_{\mu \nu} \phi_{g \mu} D_{\mu \nu} \phi_{g \nu}
    $$

- `SIGMA`:

    $$
    \begin{aligned}
    \sigma_{g t} &= \sum_{\mu \nu} \left( \phi_{g \mu, t} D_{\mu \nu} \phi_{g \nu} + \phi_{g \mu} D_{\mu \nu} \phi_{g \nu, t} \right)
    \\\\
    &= 2 \sum_{\mu \nu} \phi_{g \mu} D_{\mu \nu} \phi_{g \nu, t} \quad \text{(symm applied)}
    \end{aligned}
    $$

    Note we have already assumed the symmetry of the density matrix, so the two terms in the summation are equal. In the actual code, we only compute one of them and multiply by 2.

- `TAU`:

    $$
    \tau_g = \frac{1}{2} \sum_{\mu \nu} \sum_{t} \phi_{g \mu, t} D_{\mu \nu} \phi_{g \nu, t}
    $$

- `LAPL`:

    $$
    \begin{aligned}
    \nabla^2 \rho_g
    &= \sum_{\mu \nu} \left( \sum_{t} \phi_{g \mu, tt} D_{\mu \nu} \phi_{g \nu} + 2 \sum_{t} \phi_{g \mu, t} D_{\mu \nu} \phi_{g \nu, t} + \sum_{t} \phi_{g \mu} D_{\mu \nu} \phi_{g \nu, tt}  \right)
    \\\\
    &= 4 \tau_g + 2 \sum_{\mu \nu} \sum_{t} \phi_{g \mu} D_{\mu \nu} \phi_{g \nu, tt}
    \quad \text{(symm applied)}
    \end{aligned}
    $$

    Similar to `SIGMA`, we have already assumed the symmetry of the density matrix, so the first and third terms in the summation are equal. In the actual code, we only compute one of them and multiply by 2.

## Usage tips

For MatMul driver, we are not going to exploit the sparsity of atomic grids or density matrix. It is better to use `braket` versions to exploit the low-rank nature of the density matrix.

Usually, in only the following three cases, use this function:
- density comes from post-SCF reduced density matrix, which is usually non-zero-definite.
- prototype validation.
- basis set is small, number of occupation is large.
