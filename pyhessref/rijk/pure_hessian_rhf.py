import numpy as np
from pyscf import gto


def get_nuc_repl_hess(mol: gto.Mole):
    """Hessian contribution from nuclear repulsion."""
    natm = mol.natm
    hess = np.zeros([natm, natm, 3, 3])

    qs = np.asarray([mol.atom_charge(i) for i in range(natm)])
    rs = np.asarray([mol.atom_coord(i) for i in range(natm)])
    for i in range(natm):
        r12 = rs[i] - rs  # shape (natm, 3)
        s12 = np.sqrt(np.sum(r12 * r12, axis=1))  # einsum: 'ki,ki->k'
        s12[i] = np.inf  # avoid division by zero
        tmp1 = qs[i] * qs / s12**3  # shape (natm,)
        prefactor = -3 * qs[i] * qs / s12**5  # shape (natm,)
        tmp2 = prefactor[:, None, None] * r12[:, :, None] * r12[:, None, :]

        # Diagonal block h[i,i]
        hess[i, i, 0, 0] = hess[i, i, 1, 1] = hess[i, i, 2, 2] = -tmp1.sum()
        hess[i, i] -= np.sum(tmp2, axis=0)  # einsum: 'kij->ij'

        # Off-diagonal blocks h[i,:] for all k
        hess[i, :, 0, 0] += tmp1
        hess[i, :, 1, 1] += tmp1
        hess[i, :, 2, 2] += tmp1
        hess[i, :] += tmp2
    return hess
