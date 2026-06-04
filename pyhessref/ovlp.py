from pyscf import gto
import numpy as np


def get_hess_ovlp(mol: gto.Mole, dme0: np.ndarray) -> np.ndarray:
    """Hessian contribution from overlap matrix derivative.

    Notes
    -----
    Please be aware that the overlap matrix derivative is **NOT skeleton derivative**.
    It's true origin is the application of Hellmann-Feynman theorem, that converts part of the response of density matrix to the response of basis functions.

    Parameters
    ----------
    mol : gto.Mole
        The molecule object.
    dme0 : np.ndarray
        The energy-weighted density matrix for current SCF component, shape [nao, nao].

    Returns
    -------
    de_ovlp : np.ndarray
        The overlap matrix derivative Hessian, shape [natm, natm, 3, 3].
    """
    # definitions
    natm = mol.natm
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()
    de_ovlp = np.zeros([natm, natm, 3, 3])

    # sanity check
    assert dme0.shape == (nao, nao)

    s2_aa = mol.intor("int1e_ipipovlp").reshape(3, 3, nao, nao)
    s2_ab = mol.intor("int1e_ipovlpip").reshape(3, 3, nao, nao)
    for A in range(natm):
        _, _, p0A, p1A = aoslices[A]
        slcA = slice(p0A, p1A)
        # de_ovlp[A, A] -= 2 * np.einsum('tsuv, uv -> ts', s2_aa[:, :, slcA], dme0[slcA])
        de_ovlp[A, A] -= 2 * (s2_aa[:, :, slcA, :] * dme0[slcA, :]).sum(axis=(-1, -2))
        for B in range(natm):
            _, _, p0B, p1B = aoslices[B]
            slcB = slice(p0B, p1B)
            # de_ovlp[A, B] -= 2 * np.einsum('tsuv, uv -> ts', s2_ab[:, :, slcA, slcB], dme0[slcA, slcB])
            de_ovlp[A, B] -= 2 * (s2_ab[:, :, slcA, slcB] * dme0[slcA, slcB]).sum(
                axis=(-1, -2)
            )
        for B in range(A):
            de_ovlp[B, A] = de_ovlp[A, B].T
    return de_ovlp


class HessOvlp:
    """Hessian contribution from overlap matrix derivative.

    Note that overlap is special to the SCF part, in that
    - The contribution of hessian from overlap is not skeleton, so we do not derive this class from `HessCoreAPI`.
    - The CP-HF requires both first order derivative of hcore and ovlp, but their roles are different.

    Due to these reasons, although it has the similar interface to `HessCoreAPI`,
    `HessOvlp` is designed as a standalone class, without inheriting from any abstract class.
    """

    def __init__(self, mol: gto.Mole):
        self.mol = mol

    def make_hess(self, dme0: np.ndarray) -> np.ndarray:
        return get_hess_ovlp(self.mol, dme0)
