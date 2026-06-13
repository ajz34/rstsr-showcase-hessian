# Notes on LibXC/XCFun convention transformation

## Explanation by specific example

LibXC and XCFun have different conventions on how to handle the spin-polarized cases.

We will use GGA spin-polarized second-order derivative (fxc) as example to explain that.

As the first step, we state that for the energy part (zk) and first-order derivative part (vxc), LibXC and XCFun have the same conventions. The order is

| Index | Notation |
|--|--|
| 0 | zk |
| 1 | r_u |
| 2 | r_d |
| 3 | s_uu |
| 4 | s_ud |
| 5 | s_dd |

For the notation in above table,
- Rho is spin-polarization two-component, `r_u` means $\rho^\uparrow$, `r_d` means $\rho^\downarrow$.
- Sigma is spin-polarization three-component, `s_uu` means $\sigma^{\uparrow \uparrow}$, `s_ud` means $\sigma^{\uparrow \downarrow}$, `s_dd` means $\sigma^{\downarrow \downarrow}$.

For the fxc part, there will involve two densities as variables. The LibXC and XCFun have fundamental difference at

- LibXC respect which type of density first (sorted by rho, sigma), then its spin;
- XCFun respect the spin-component first (sorted by r_u, r_d, s_uu, s_ud, s_dd).

It is more favorable to use XCFun-style for future DFT evaluation. However, LibXC supports more functionals, with better API design and popularity. So an index transform mapping is required. For this specific task,

| Index | Notation<br>LibXC | Notation<br>XCFun | Map |
|-:|--|--|-:|
|  6 | r_u  / r_u  | r_u  / r_u  |  6 |
|  7 | r_u  / r_d  | r_u  / r_d  |  7 |
|  8 | r_d  / r_d  | r_u  / s_uu |  9 |
|  9 | r_u  / s_uu | r_u  / s_ud | 10 |
| 10 | r_u  / s_ud | r_u  / s_dd | 11 |
| 11 | r_u  / s_dd | r_d  / r_d  |  8 |
| 12 | r_d  / s_uu | r_d  / s_uu | 12 |
| 13 | r_d  / s_ud | r_d  / s_ud | 13 |
| 14 | r_d  / s_dd | r_d  / s_dd | 14 |
| 15 | s_uu / s_uu | s_uu / s_uu | 15 |
| 16 | s_uu / s_ud | s_uu / s_ud | 16 |
| 17 | s_uu / s_dd | s_uu / s_dd | 17 |
| 18 | s_ud / s_ud | s_ud / s_ud | 18 |
| 19 | s_ud / s_dd | s_ud / s_dd | 19 |
| 20 | s_dd / s_dd | s_dd / s_dd | 20 |

We can see that

- LibXC will first category `r/r` (6--8), then `r/s` (9--14), then `s/s` (16--20). For each category, sort by spin.
- XCFun will first category `r_u` (6--10), then `r_d` (11--14), then `s_uu` (15--17), then `s_ud` (18--19), then `s_dd` (20). For each category, sort by the same way in first category.

## Extension to the example

- **Higher derivative**: We may encounter more higher derivatives (usually up to 4th derivative, but can be more).
- **More kinds of density**: We may use more density types. The priority is RHO > SIGMA > TAU > LAPL.
  - TAU: `t_u`, `t_d`
  - LAPL: `l_u`, `l_d`
- We assume the RHO (LDA) only inputs RHO; SIGMA (GGA) inputs both RHO and SIGMA; TAU (some meta-GGA) inputs RHO SIGMA TAU, and LAPL (some meta-GGA) inputs all RHO SIGMA TAU LAPL (though some LAPL meta-GGAs does not actually input tau, but we still require the TAU to be available for simplicity).

## Code for Automatic Index Map Generation

```python
from itertools import combinations_with_replacement
from math import comb


def libxc_to_xcfun_indices_map(den_type: str, deriv: int) -> list[int]:
    """Spin-Polarized Indices Map from LibXC to XCFun.

    Parameters
    ----------
    den_type : str
        Density Type. Supports `rho`, `sigma`, `tau`, `lapl`.
    deriv : int
        Derivative level.

    Example
    -------
    >>> libxc_to_xcfun_indices_map("sigma", 2)
    [6, 7, 9, 10, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    """
    # Each variable: (type_priority, spin_index)
    # RHO has 2 spin components, SIGMA has 3, TAU has 2, LAPL has 2
    group_specs = [
        ("rho", 0, 2),
        ("sigma", 1, 3),
        ("tau", 2, 2),
        ("lapl", 3, 2),
    ]

    type_map = {
        "rho": ["rho"],
        "sigma": ["rho", "sigma"],
        "tau": ["rho", "sigma", "tau"],
        "lapl": ["rho", "sigma", "tau", "lapl"],
    }

    if den_type not in type_map:
        raise ValueError(f"Unknown den_type: {den_type}")

    active_groups = set(type_map[den_type])

    # Build variable list: each variable is (type_priority, spin_index)
    variables = []
    for group_name, priority, n_spin in group_specs:
        if group_name in active_groups:
            for spin in range(n_spin):
                variables.append((priority, spin))

    d = len(variables)

    # Generate all non-decreasing multi-indices of length deriv
    # combinations_with_replacement yields them in lexicographic order = XCFun order
    xcfun_order = list(combinations_with_replacement(range(d), deriv))

    # LibXC order: sort by density type signature first, then by variable indices
    def libxc_key(mi):
        return tuple(variables[i][0] for i in mi) + mi

    libxc_order = sorted(xcfun_order, key=libxc_key)

    # Build reverse lookup: multi-index -> LibXC position
    libxc_pos = {mi: pos for pos, mi in enumerate(libxc_order)}

    # Base offset = sum of outputs for all previous derivative levels
    base_offset = sum(comb(d + i - 1, i) for i in range(deriv))

    return [base_offset + libxc_pos[mi] for mi in xcfun_order]
```
