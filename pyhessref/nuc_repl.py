import numpy as np
from pyscf import gto

from pyhessref.hess_trait_restricted import RHessCoreAPI


def get_nuc_repl_hess(mol: gto.Mole) -> np.ndarray:
    """Hessian contribution from nuclear repulsion.

    Parameters
    ----------
    mol : gto.Mole
        The molecule object.

    Returns
    -------
    de_nuc : np.ndarray
        The nuclear repulsion Hessian, shape [natm, natm, 3, 3].
    """
    natm = mol.natm
    de_nuc = np.zeros([natm, natm, 3, 3])

    qs = np.asarray([mol.atom_charge(i) for i in range(natm)])
    rs = np.asarray([mol.atom_coord(i) for i in range(natm)])
    for i in range(natm):
        r12 = rs[i] - rs  # shape (natm, 3)
        s12 = np.sqrt(np.sum(r12 * r12, axis=1))  # einsum: 'ki,ki->k'
        s12[i] = np.inf  # avoid division by zero
        tmp1 = qs[i] * qs / s12**3  # shape [natm]
        prefactor = -3 * qs[i] * qs / s12**5  # shape [natm]
        tmp2 = prefactor[:, None, None] * r12[:, :, None] * r12[:, None, :]

        # Diagonal block h[i,i]
        de_nuc[i, i, 0, 0] = de_nuc[i, i, 1, 1] = de_nuc[i, i, 2, 2] = -tmp1.sum()
        de_nuc[i, i] -= np.sum(tmp2, axis=0)  # einsum: 'kij->ij'

        # Off-diagonal blocks h[i,:] for all k
        de_nuc[i, :, 0, 0] += tmp1
        de_nuc[i, :, 1, 1] += tmp1
        de_nuc[i, :, 2, 2] += tmp1
        de_nuc[i, :] += tmp2
    return de_nuc


class HessNucRepl(RHessCoreAPI):
    """Hessian contribution from nuclear repulsion."""

    def __init__(self, mol: gto.Mole):
        self.mol = mol

    def make_skeleton_hess(self, *args, **kwargs) -> np.ndarray:
        return get_nuc_repl_hess(self.mol)

    def generator_deriv1(self) -> callable:
        return None
