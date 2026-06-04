from pyscf import gto
import numpy as np

from pyhessref.hess_trait import HessRscfAPI


def get_hess_ovlp(mol: gto.Mole, dme0: np.ndarray) -> np.ndarray:
    """Hessian contribution from overlap matrix derivative.

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


class HessOvlp(HessRscfAPI):
    """Hessian contribution from overlap matrix derivative."""

    def __init__(self, mol: gto.Mole):
        self.mol = mol

    def make_hess(
        self, mo_coeff: np.ndarray, mo_occ: np.ndarray, mo_energy: np.ndarray, **kwargs
    ) -> np.ndarray:
        dme0 = self.get_dme0(mo_coeff, mo_occ, mo_energy)
        return get_hess_ovlp(self.mol, dme0)
