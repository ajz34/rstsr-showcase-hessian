import numpy as np


def get_dm(mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
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
