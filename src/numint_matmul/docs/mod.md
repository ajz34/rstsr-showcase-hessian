# Introduction of `MatMul` DFT grid driver

This driver is the naive DFT driver. The code effort is minimized, yet still be efficient to small systems.

As the name suggests, we fully utilize BLAS3 GEMM to perform the DFT grid operations.

## `ao`: atomic orbital values

The AO tensor `ao` has shape `[ngrids, nao, ncomp]` where the component dimension is ordered as:

| Index | Deriv | Component |
|--|--|--|
| 0     | 0 | $\phi_{g \mu}$                  |
| 1–3   | 1 | $\phi_{g \mu, x}, \phi_{g \mu, y}, \phi_{g \mu, z}$ |
| 4–9   | 2 | $\phi_{g \mu,xx}, \phi_{g \mu,xy}, \phi_{g \mu,xz}, \phi_{g \mu,yy}, \phi_{g \mu,yz}, \phi_{g \mu,zz}$ |

Additional to the above table, for notation simplicity, we denote

- orbital notation

    $$
    \phi_{g \mu} = \phi_{\mu} (\bm{r}_g)
    $$

    Where subscript $g$ denotes grid, $\bm{r}_g$ denotes the coordinate of grid point $g$, and $\mu$ denotes the AO index.

- orbital derivative notation

    $$
    \phi_{g \mu, x} = \partial_x \phi_{\mu} (\bm{r}_g)
    $$

    The partial derivative is taken with respect to the electron coordinate.

- We usually use $t, s, r \in \{ x, y, z \}$ to denote the subscript of the coordinate. 

## `rho`: density values

We will introduce 4 types of density values:

| property | [`RHO`] | [`SIGMA`] | [`TAU`] | [`LAPL`] |
|--|--|--|--|--|
| notation | $\rho$ | $\sigma$ | $\tau$ | $\nabla^2 \rho$ |
| [`num_rho_comp`](NIDenType::num_rho_comp) | 1 | 4 | 5 | 6 |
| [`num_ao_deriv`](NIDenType::num_ao_deriv) | 0 | 1 | 1 | 2 |
| [`num_ao_comp`](NIDenType::num_ao_comp) | 1 | 4 | 4 | 10 |
| usual XC type | LDA | GGA | mGGA | mGGA |

The density to be input in this driver is `[ngirds, num_rho_comp]` for spin unpolarized case, and `[ngrids, num_rho_comp, 2]` for spin polarized case. The density is computed by contracting the AO tensor with the density matrix, which is the same as the usual DFT grid driver.

The density components are ordered as:

$$
\rho, \rho_x, \rho_y, \rho_z, \tau, \nabla^2 \rho
$$

For laplacian density functional, whatever the xc functional uses tau, currently we enforce the evaluation of $\tau$ to be the 4th component of the density.

## Important pure functions

**Density Grid Evaluation**

- [`get_rho_from_dm_with_output`]
- [`get_rho_from_homogeneous_braket_with_output]
