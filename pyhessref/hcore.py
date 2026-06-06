import numpy as np
from pyscf import gto

from pyhessref.hess_trait_restricted import RHessCoreAPI
from pyhessref.hess_trait_unrestricted import UHessCoreAPI
from pyhessref.util import get_dm0_restricted, get_dm0_unrestricted


def generator_hcore_deriv2(mol) -> callable:
    """Generator for second derivatives of the core Hamiltonian (skeleton derivative).

    Parameters
    ----------
    mol : gto.Mole
        The molecule object.

    Returns
    -------
    get_hcore_deriv_at_atoms : function(A: int, B: int) -> np.ndarray
        A function that computes the second derivative of the core Hamiltonian with respect to the nuclear coordinates.
        Input is the pair of atom indices (A, B).
        The returned array has shape [3, 3, nao, nao].
    """

    # preparation
    nao = mol.nao
    nbas = mol.nbas
    aoslices = mol.aoslice_by_atom()
    ecp_atoms = set(mol._ecpbas[:, gto.ATOM_OF])
    # we need to prepare some integrals, to somehow avoid redundant calculations in the loop
    # - aa: Hamiltonian derivative to only the first basis
    # - ab: Hamiltonian derivative to the first and second basis

    h2_aa = mol.intor("int1e_ipipkin").reshape(3, 3, nao, nao)
    h2_ab = mol.intor("int1e_ipkinip").reshape(3, 3, nao, nao)
    h2_aa += mol.intor("int1e_ipipnuc").reshape(3, 3, nao, nao)
    h2_ab += mol.intor("int1e_ipnucip").reshape(3, 3, nao, nao)
    if mol.has_ecp():
        h2_aa += mol.intor("ECPscalar_ipipnuc").reshape(3, 3, nao, nao)
        h2_ab += mol.intor("ECPscalar_ipnucip").reshape(3, 3, nao, nao)

    def get_hcore_deriv_at_atoms(A, B):
        sh0A, sh1A, p0A, p1A = aoslices[A]
        sh0B, sh1B, p0B, p1B = aoslices[B]
        slcA = slice(p0A, p1A)
        slcB = slice(p0B, p1B)
        zi = mol.atom_charge(A)
        hcore_deriv = np.zeros((3, 3, nao, nao))

        if A == B:
            with mol.with_rinv_at_nucleus(A):
                rinv_aa = -zi * mol.intor("int1e_ipiprinv").reshape(3, 3, nao, nao)
                rinv_ab = -zi * mol.intor("int1e_iprinvip").reshape(3, 3, nao, nao)
                if A in ecp_atoms:
                    rinv_aa += mol.intor("ECPscalar_ipiprinv").reshape(3, 3, nao, nao)
                    rinv_ab += mol.intor("ECPscalar_iprinvip").reshape(3, 3, nao, nao)
                hcore_deriv += rinv_aa + rinv_ab
                hcore_deriv[:, :, slcA, :] += h2_aa[:, :, slcA, :]
                hcore_deriv[:, :, slcA, slcA] += h2_ab[:, :, slcA, slcA]
                hcore_deriv[:, :, slcA, :] -= rinv_aa[:, :, slcA, :]
                hcore_deriv[:, :, slcA, :] -= rinv_ab[:, :, slcA, :]
                hcore_deriv[:, :, :, slcA] -= rinv_aa[:, :, slcA, :].swapaxes(-1, -2)
                hcore_deriv[:, :, :, slcA] -= rinv_ab[:, :, :, slcA]
        else:
            hcore_deriv[:, :, slcA, slcB] += h2_ab[:, :, slcA, slcB]
            # handle rinv@i, basis@j
            zi = mol.atom_charge(A)
            with mol.with_rinv_at_nucleus(A):
                shls_slice = (sh0B, sh1B, 0, nbas)
                rinv_atom_aa = -zi * mol.intor("int1e_ipiprinv", shls_slice=shls_slice).reshape(3, 3, -1, nao)
                rinv_atom_ab = -zi * mol.intor("int1e_iprinvip", shls_slice=shls_slice).reshape(3, 3, -1, nao)
                if A in ecp_atoms:
                    rinv_atom_aa += mol.intor("ECPscalar_ipiprinv", shls_slice=shls_slice).reshape(3, 3, -1, nao)
                    rinv_atom_ab += mol.intor("ECPscalar_iprinvip", shls_slice=shls_slice).reshape(3, 3, -1, nao)
            hcore_deriv[:, :, slcB, :] -= rinv_atom_aa
            hcore_deriv[:, :, slcB, :] -= rinv_atom_ab.swapaxes(0, 1)
            # handle rinv@j, basis@i
            zj = mol.atom_charge(B)
            with mol.with_rinv_at_nucleus(B):
                shls_slice = (sh0A, sh1A, 0, nbas)
                rinv_atom_aa = -zj * mol.intor("int1e_ipiprinv", shls_slice=shls_slice).reshape(3, 3, -1, nao)
                rinv_atom_ab = -zj * mol.intor("int1e_iprinvip", shls_slice=shls_slice).reshape(3, 3, -1, nao)
                if B in ecp_atoms:
                    rinv_atom_aa += mol.intor("ECPscalar_ipiprinv", shls_slice=shls_slice).reshape(3, 3, -1, nao)
                    rinv_atom_ab += mol.intor("ECPscalar_iprinvip", shls_slice=shls_slice).reshape(3, 3, -1, nao)
            hcore_deriv[:, :, slcA, :] -= rinv_atom_aa
            hcore_deriv[:, :, slcA, :] -= rinv_atom_ab

        hcore_deriv += hcore_deriv.swapaxes(-1, -2)
        return hcore_deriv

    return get_hcore_deriv_at_atoms


def generator_hcore_deriv1(mol) -> callable:
    """Generator for first derivatives of the core Hamiltonian (skeleton derivative).

    Parameters
    ----------
    mol : gto.Mole
        The molecule object.

    Returns
    -------
    get_hcore_deriv_at_atoms : function(A: int) -> np.ndarray
        A function that computes the first derivative of the core Hamiltonian with respect to the nuclear coordinates.
        Input is the atom index A.
        The returned array has shape [3, nao, nao].
    """

    h1 = -mol.intor("int1e_ipkin") - mol.intor("int1e_ipnuc")
    if mol.has_ecp():
        h1 -= mol.intor("ECPscalar_ipnuc")
    ecp_atoms = set(mol._ecpbas[:, gto.ATOM_OF])
    aoslices = mol.aoslice_by_atom()

    def get_hcore_deriv_at_atoms(A):
        _, _, p0, p1 = aoslices[A]
        z = mol.atom_charge(A)
        with mol.with_rinv_at_nucleus(A):
            h1ao = -z * mol.intor("int1e_iprinv")
            if A in ecp_atoms:
                h1ao += mol.intor("ECPscalar_iprinv")
        h1ao[:, p0:p1] += h1[:, p0:p1]
        return h1ao + h1ao.swapaxes(-1, -2)

    return get_hcore_deriv_at_atoms


def get_hess_hcore(mol: gto.Mole, dm0: np.ndarray) -> np.ndarray:
    """Hessian contribution from the core Hamiltonian.

    Parameters
    ----------
    mol : gto.Mole
        The molecule object.
    dm0 : np.ndarray
        The density matrix, shape [nao, nao].

    Returns
    -------
    de_hcore : np.ndarray
        The Hessian contribution from the core Hamiltonian, shape [natm, natm, 3, 3].
    """
    gen_hcore_deriv2 = generator_hcore_deriv2(mol)
    natm = mol.natm
    de_hcore = np.zeros([natm, natm, 3, 3])

    for A in range(natm):
        for B in range(A + 1):
            hcore_deriv2 = gen_hcore_deriv2(A, B)  # shape [3, 3, nao, nao]
            # de_hcore[A, B] += np.einsum('tsuv, uv -> ts', hcore_deriv2, dm0)
            de_hcore[A, B] += (hcore_deriv2 * dm0).sum(axis=(-1, -2))
        for B in range(A):
            de_hcore[B, A] = de_hcore[A, B].T
    return de_hcore


class RHessHcore(RHessCoreAPI):
    """Hessian contribution from the core Hamiltonian."""

    def __init__(self, mol: gto.Mole):
        self.mol = mol

    def make_skeleton_hess(self, mo_coeff: np.ndarray, mo_occ: np.ndarray, dm0: np.ndarray = None) -> np.ndarray:
        dm0 = dm0 if dm0 is not None else get_dm0_restricted(mo_coeff, mo_occ)
        return get_hess_hcore(self.mol, dm0)

    def generator_deriv1(self) -> callable:
        return generator_hcore_deriv1(self.mol)


class UHessHcore(UHessCoreAPI):
    """UHF version of core Hamiltonian Hessian contribution.

    The skeleton derivative only depends on the *total* density matrix, so we reuse
    the restricted pure function ``get_hess_hcore`` after summing the per-spin density
    matrices. The first-order derivative generator is spin-independent and identical
    to the restricted case.
    """

    def __init__(self, mol: gto.Mole):
        self.mol = mol

    def make_skeleton_hess(self, mo_coeff: np.ndarray, mo_occ: np.ndarray, dm0: np.ndarray = None) -> np.ndarray:
        if dm0 is None:
            dm0 = get_dm0_unrestricted(mo_coeff, mo_occ).sum(axis=0)
        return get_hess_hcore(self.mol, dm0)

    def generator_deriv1(self) -> callable:
        return generator_hcore_deriv1(self.mol)
