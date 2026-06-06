import numpy as np


def get_dm0_restricted(mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
    """Generate the density matrix for current SCF component.

    Parameters
    ----------
    mo_coeff : np.ndarray
        Molecular orbital coefficients, shape [nao, nmo].
    mo_occ : np.ndarray
        Molecular orbital occupation numbers, shape [nmo].

    Returns
    -------
    dm0 : np.ndarray
        The density matrix for current SCF component, shape [nao, nao].
    """
    occidx = mo_occ > 1e-15
    return mo_coeff[:, occidx] * mo_occ[occidx] @ mo_coeff[:, occidx].T


def get_dm0_unrestricted(mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
    """Generate per-spin density matrices for UHF.

    Parameters
    ----------
    mo_coeff : np.ndarray
        Molecular orbital coefficients, shape ``[2, nao, nmo]``.
    mo_occ : np.ndarray
        Molecular orbital occupation numbers, shape ``[2, nmo]``. Occupied entries are 1.

    Returns
    -------
    dm0 : np.ndarray
        Per-spin density matrices, shape ``[2, nao, nao]``.
    """
    return np.array([get_dm0_restricted(mo_coeff[s], mo_occ[s]) for s in range(2)])


def get_dme0_unrestricted(mo_coeff: np.ndarray, mo_occ: np.ndarray, mo_energy: np.ndarray) -> np.ndarray:
    """Generate per-spin energy-weighted density matrices for UHF.

    Returns
    -------
    dme0 : np.ndarray
        Per-spin energy-weighted density matrices, shape ``[2, nao, nao]``.
    """
    return np.array([get_dme0_restricted(mo_coeff[s], mo_occ[s], mo_energy[s]) for s in range(2)])


def get_dme0_restricted(mo_coeff: np.ndarray, mo_occ: np.ndarray, mo_energy: np.ndarray) -> np.ndarray:
    """Generate the energy-weighted density matrix for current SCF component.

    Parameters
    ----------
    mo_coeff : np.ndarray
        Molecular orbital coefficients, shape [nao, nmo].
    mo_occ : np.ndarray
        Molecular orbital occupation numbers, shape [nmo].
    mo_energy : np.ndarray
        Molecular orbital energies, shape [nmo].

    Returns
    -------
    dme0 : np.ndarray
        The energy-weighted density matrix for current SCF component, shape [nao, nao].
    """
    occidx = mo_occ > 1e-15
    return mo_coeff[:, occidx] * (mo_occ[occidx] * mo_energy[occidx]) @ mo_coeff[:, occidx].T


def pack_uhf_mo_pair(arr_pair: list[np.ndarray]) -> np.ndarray:
    """Flatten a UHF pair ``[arr_alpha, arr_beta]`` into a single 2D array.

    Each entry is expected to have shape ``[nset, ..., nmo_or_nocc, nocc_sigma]``;
    the leading ``nset`` dimension is preserved and the trailing dimensions are
    flattened. The resulting shape is ``[nset, size_alpha + size_beta]``.
    """
    n = arr_pair[0].shape[0]
    assert arr_pair[1].shape[0] == n
    return np.hstack([arr_pair[0].reshape(n, -1), arr_pair[1].reshape(n, -1)])


def unpack_uhf_mo_pair(flat: np.ndarray, shape_alpha: tuple, shape_beta: tuple) -> list[np.ndarray]:
    """Inverse of `pack_uhf_mo_pair`.

    Parameters
    ----------
    flat : np.ndarray
        Shape ``[nset, size_alpha + size_beta]``.
    shape_alpha, shape_beta : tuple
        Trailing shape (without the leading ``nset``) for each spin block.
    """
    n = flat.shape[0]
    size_a = int(np.prod(shape_alpha))
    size_b = int(np.prod(shape_beta))
    assert flat.shape[1] == size_a + size_b
    a = flat[:, :size_a].reshape((n,) + tuple(shape_alpha))
    b = flat[:, size_a:].reshape((n,) + tuple(shape_beta))
    return [a, b]
