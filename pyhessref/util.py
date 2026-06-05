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
