import numpy as np
from pyscf import gto
from functools import partial
from pyscf.df.grad.rhf import _int3c_wrapper

from pyhessref.hess_trait_restricted import RHessElecInteractAPI
from pyhessref.util import get_dm0_restricted

# override einsum for some efficiency
einsum = partial(np.einsum, optimize=True)


def get_decomposed_rij_skeleton_deriv2_naive(
    mol: gto.Mole,
    aux: gto.Mole,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
) -> dict[str, np.ndarray]:
    """Get the skeleton of the second derivative of the Coulomb interaction.

    This is naive implementation:
    - computes all integrals in full, not optimized for memory;
    - extensively use einsum, easy for equation-code translation but not fully efficient;
    - not extensively combined contribution terms;
    - in principle, RI-J should be evaluated alongwith RI-K, but here we only compute RI-J part;
    - we evaluated all auxiliary basis derivative contributions, which is sometimes not necessary for hessian computation.

    This function not only returns the summed hessian, but also all the separated contributions, useful for debugging and understanding the code.

    Parameters
    ----------
    mol : gto.Mole
        The molecule object.
    aux : gto.Mole
        The auxiliary basis set molecule object.
    mo_coeff : np.ndarray
        The molecular orbital coefficients, shape [nao, nmo].
    mo_occ : np.ndarray
        The molecular orbital occupation numbers, shape [nmo].
    auxbasis_response : int
        The derivative level of the auxiliary basis set. For hessian computation, it should be 0/1/2.

    Returns
    -------
    de_J_skeleton : dict[str, np.ndarray]
        The skeleton of the second derivative of the Coulomb interaction, separated by different contributions.
        Each contribution is [natm, natm, 3, 3] array.
        The contributions are denoted as `de_J<bas_deriv><aux_deriv>_<contrib_idx>`, e.g. `de_J20_2`.
        Sometimes the contribution idx will be number with alphabet like `2a` and `2b`.
        Meaning of these contribution may not be fully documented in returned keys. See code comments for details.
    """
    # some elementary information
    nao = mol.nao
    naux = aux.nao
    dm0 = get_dm0_restricted(mo_coeff, mo_occ)
    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    auxslices = aux.aoslice_by_atom()

    # integrals we need
    int2c2e = aux.intor("int2c2e")
    int2c2e_inv = np.linalg.inv(int2c2e)
    int3c2e = _int3c_wrapper(mol, aux, "int3c2e", "s1")()
    int3c2e_ip1 = _int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")().reshape([3, nao, nao, naux])
    int3c2e_ip2 = _int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")().reshape([3, nao, nao, naux])
    int3c2e_ipip1 = _int3c_wrapper(mol, aux, "int3c2e_ipip1", "s1")().reshape([3, 3, nao, nao, naux])
    int3c2e_ipvip1 = _int3c_wrapper(mol, aux, "int3c2e_ipvip1", "s1")().reshape([3, 3, nao, nao, naux])
    int3c2e_ip1ip2 = _int3c_wrapper(mol, aux, "int3c2e_ip1ip2", "s1")().reshape([3, 3, nao, nao, naux])
    int3c2e_ipip2 = _int3c_wrapper(mol, aux, "int3c2e_ipip2", "s1")().reshape([3, 3, nao, nao, naux])
    int2c2e_ip1 = aux.intor("int2c2e_ip1")
    int2c2e_ipip1 = aux.intor("int2c2e_ipip1").reshape([3, 3, naux, naux])
    int2c2e_ip1ip2 = aux.intor("int2c2e_ip1ip2").reshape([3, 3, naux, naux])

    # --- J20 (basis deriv 2, aux deriv 0) --- #

    # (10|0)(0|10)
    dbas_J20_1 = einsum(
        "tuvP, PQ, sklQ, uv, kl -> tsuk",
        int3c2e_ip1,
        int2c2e_inv,
        int3c2e_ip1,
        dm0,
        dm0,
    )
    de_J20_1 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(aoslices):
            de_J20_1[A, B] += 4 * einsum("tsuv -> ts", dbas_J20_1[:, :, p0A:p1A, p0B:p1B])

    # (11|0)(0|00)
    dbas_J20_2 = einsum("tsuvP, PQ, klQ, kl -> tsuv", int3c2e_ipvip1, int2c2e_inv, int3c2e, dm0)
    de_J20_2 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(aoslices):
            de_J20_2[A, B] += 2 * einsum(
                "tsuv, uv -> ts",
                dbas_J20_2[:, :, p0A:p1A, p0B:p1B],
                dm0[p0A:p1A, p0B:p1B],
            )

    # (20|0)(0|00)
    dbas_J20_3 = einsum("tsuvP, PQ, klQ, kl -> tsuv", int3c2e_ipip1, int2c2e_inv, int3c2e, dm0)
    de_J20_3 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        de_J20_3[A, A] += 2 * einsum("tsuv, uv -> ts", dbas_J20_3[:, :, p0A:p1A], dm0[p0A:p1A])

    de_J20 = de_J20_1 + de_J20_2 + de_J20_3

    # --- J11 (basis deriv 1, aux deriv 1) --- #

    # (10|1)(0|0)(0|00)
    dbas_J11_1 = einsum("tsuvP, PQ, klQ, uv, kl -> tsuP", int3c2e_ip1ip2, int2c2e_inv, int3c2e, dm0, dm0)
    de_J11_1 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J11_1[A, B] += 2 * einsum("tsuP -> ts", dbas_J11_1[:, :, p0A:p1A, p0B:p1B])
    de_J11_1 += de_J11_1.transpose(1, 0, 3, 2)

    # (10|0)(0|1)(0|00)
    dbas_J11_2 = einsum(
        "tuvP, PQ, sQR, RS, klS, uv, kl -> tsuR",
        int3c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J11_2 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J11_2[A, B] += 2 * einsum("tsuR -> ts", dbas_J11_2[:, :, p0A:p1A, p0B:p1B])
    de_J11_2 += de_J11_2.transpose(1, 0, 3, 2)

    # (10|0)(1|0)(0|00)
    dbas_J11_3 = einsum(
        "tuvP, PQ, sQR, RS, klS, uv, kl -> tsuQ",
        int3c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J11_3 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J11_3[A, B] += -2 * einsum("tsuQ -> ts", dbas_J11_3[:, :, p0A:p1A, p0B:p1B])
    de_J11_3 += de_J11_3.transpose(1, 0, 3, 2)

    # (10|0)(0|0)(1|00)
    dbas_J11_4 = einsum(
        "tuvP, PQ, sklQ, uv, kl -> tsuQ",
        int3c2e_ip1,
        int2c2e_inv,
        int3c2e_ip2,
        dm0,
        dm0,
    )
    de_J11_4 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J11_4[A, B] += 2 * einsum("tsuQ -> ts", dbas_J11_4[:, :, p0A:p1A, p0B:p1B])
    de_J11_4 += de_J11_4.transpose(1, 0, 3, 2)

    de_J11 = de_J11_1 + de_J11_2 + de_J11_3 + de_J11_4

    # --- J02 (basis deriv 0, aux deriv 2) --- #

    # (00|2)(0|00)
    dbas_J02_1 = einsum("tsuvP, PQ, klQ, uv, kl -> tsP", int3c2e_ipip2, int2c2e_inv, int3c2e, dm0, dm0)
    de_J02_1 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        de_J02_1[A, A] += einsum("tsP -> ts", dbas_J02_1[:, :, p0A:p1A])

    # (00|0)(2|0)(0|00)
    dbas_J02_2 = einsum(
        "uvP, PQ, tsQR, RS, klS, uv, kl -> tsQ",
        int3c2e,
        int2c2e_inv,
        int2c2e_ipip1,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J02_2 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        de_J02_2[A, A] += -1 * einsum("tsQ -> ts", dbas_J02_2[:, :, p0A:p1A])
    de_J02_2 = de_J02_2

    # (00|0)(1|1)(0|00)
    dbas_J02_3a = einsum(
        "uvP, PQ, tsQR, RS, klS, uv, kl -> tsQR",
        int3c2e,
        int2c2e_inv,
        int2c2e_ip1ip2,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J02_3a = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_3a[A, B] += -0.5 * einsum("tsQR -> ts", dbas_J02_3a[:, :, p0A:p1A, p0B:p1B])
    de_J02_3a += de_J02_3a.transpose(1, 0, 3, 2)

    # (00|0)(1|0)(0|1)(0|00)
    dbas_J02_3b = einsum(
        "uvP, PQ, tQR, RS, sST, TU, klU, uv, kl -> tsQT",
        int3c2e,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J02_3b = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_3b[A, B] += -0.5 * einsum("tsQT -> ts", dbas_J02_3b[:, :, p0A:p1A, p0B:p1B])
    de_J02_3b += de_J02_3b.transpose(1, 0, 3, 2)

    # (00|1)(1|0)(0|00)
    dbas_J02_4 = einsum(
        "tuvP, PQ, sQR, RS, klS, uv, kl -> tsPQ",
        int3c2e_ip2,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J02_4 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_4[A, B] += -1 * einsum("tsPQ -> ts", dbas_J02_4[:, :, p0A:p1A, p0B:p1B])
    de_J02_4 += de_J02_4.transpose(1, 0, 3, 2)

    # (00|1)(1|00)
    dbas_J02_5 = einsum(
        "tuvP, PQ, sklQ, uv, kl -> tsPQ",
        int3c2e_ip2,
        int2c2e_inv,
        int3c2e_ip2,
        dm0,
        dm0,
    )
    de_J02_5 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_5[A, B] += 0.5 * einsum("tsPQ -> ts", dbas_J02_5[:, :, p0A:p1A, p0B:p1B])
    de_J02_5 += de_J02_5.transpose(1, 0, 3, 2)

    # (00|0)(0|1)(1|0)(0|00)
    dbas_J02_6 = einsum(
        "uvP, PQ, tRQ, RS, sST, TU, klU, uv, kl -> tsRS",
        int3c2e,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J02_6 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_6[A, B] += 0.5 * einsum("tsRS -> ts", dbas_J02_6[:, :, p0A:p1A, p0B:p1B])
    de_J02_6 += de_J02_6.transpose(1, 0, 3, 2)

    # (00|1)(0|1)(0|00)
    dbas_J02_7 = einsum(
        "tuvP, PQ, sRQ, RS, klS, uv, kl -> tsPR",
        int3c2e_ip2,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J02_7 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_7[A, B] += -1 * einsum("tsPR -> ts", dbas_J02_7[:, :, p0A:p1A, p0B:p1B])
    de_J02_7 += de_J02_7.transpose(1, 0, 3, 2)

    # (00|0)(1|0)(1|0)(0|00)
    dbas_J02_8 = einsum(
        "uvP, PQ, tQR, RS, sST, TU, klU, uv, kl -> tsRT",
        int3c2e,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
    de_J02_8 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_8[A, B] += 1 * einsum("tsRT -> ts", dbas_J02_8[:, :, p0A:p1A, p0B:p1B])
    de_J02_8 += de_J02_8.transpose(1, 0, 3, 2)

    de_J02 = de_J02_1 + de_J02_2 + de_J02_3a + de_J02_3b + de_J02_4 + de_J02_5 + de_J02_6 + de_J02_7 + de_J02_8

    de_J_skeleton = {
        # de_J20
        "de_J20_1": de_J20_1,
        "de_J20_2": de_J20_2,
        "de_J20_3": de_J20_3,
        # de_J11
        "de_J11_1": de_J11_1,
        "de_J11_2": de_J11_2,
        "de_J11_3": de_J11_3,
        "de_J11_4": de_J11_4,
        # de_J02
        "de_J02_1": de_J02_1,
        "de_J02_2": de_J02_2,
        "de_J02_3a": de_J02_3a,
        "de_J02_3b": de_J02_3b,
        "de_J02_4": de_J02_4,
        "de_J02_5": de_J02_5,
        "de_J02_6": de_J02_6,
        "de_J02_7": de_J02_7,
        "de_J02_8": de_J02_8,
        # total
        "de_J20": de_J20,
        "de_J11": de_J11,
        "de_J02": de_J02,
    }
    return de_J_skeleton


def get_decomposed_rik_skeleton_deriv2_naive(
    mol: gto.Mole,
    aux: gto.Mole,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
) -> dict[str, np.ndarray]:
    """Get the skeleton of the second derivative of the exchange interaction.

    This is naive implementation, see `get_decomposed_rij_skeleton_deriv2_naive` for details and returned keys documentation.

    Parameters
    ----------
    mol : gto.Mole
        The molecule object.
    aux : gto.Mole
        The auxiliary basis set molecule object.
    mo_coeff : np.ndarray
        The molecular orbital coefficients, shape [nao, nmo].
    mo_occ : np.ndarray
        The molecular orbital occupation numbers, shape [nmo].
    auxbasis_response : int
        The derivative level of the auxiliary basis set. For hessian computation, it should be 0/1/2.

    Returns
    -------
    de_K_skeleton : dict[str, np.ndarray]
        The skeleton of the second derivative of the exchange interaction, separated by different contributions.
        Each contribution is [natm, natm, 3, 3] array.
        The contributions are denoted as `de_K<bas_deriv><aux_deriv>_<contrib_idx>`, e.g. `de_K20_2`.
        Sometimes the contribution idx will be number with alphabet like `2a` and `2b`.
        Meaning of these contribution may not be fully documented in returned keys. See code comments for details.
    """
    # some elementary information
    nao = mol.nao
    naux = aux.nao
    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    auxslices = aux.aoslice_by_atom()

    # occupation
    occidx = mo_occ > 0
    mocc = mo_coeff[:, occidx]
    occ = mo_occ[occidx]
    mocc_2 = mocc * np.sqrt(occ)

    # integrals we need
    int2c2e = aux.intor("int2c2e")
    int2c2e_inv = np.linalg.inv(int2c2e)
    int3c2e = _int3c_wrapper(mol, aux, "int3c2e", "s1")()
    int3c2e_ip1 = _int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")().reshape([3, nao, nao, naux])
    int3c2e_ip2 = _int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")().reshape([3, nao, nao, naux])
    int3c2e_ipip1 = _int3c_wrapper(mol, aux, "int3c2e_ipip1", "s1")().reshape([3, 3, nao, nao, naux])
    int3c2e_ipvip1 = _int3c_wrapper(mol, aux, "int3c2e_ipvip1", "s1")().reshape([3, 3, nao, nao, naux])
    int3c2e_ip1ip2 = _int3c_wrapper(mol, aux, "int3c2e_ip1ip2", "s1")().reshape([3, 3, nao, nao, naux])
    int3c2e_ipip2 = _int3c_wrapper(mol, aux, "int3c2e_ipip2", "s1")().reshape([3, 3, nao, nao, naux])
    int2c2e_ip1 = aux.intor("int2c2e_ip1")
    int2c2e_ipip1 = aux.intor("int2c2e_ipip1").reshape([3, 3, naux, naux])
    int2c2e_ip1ip2 = aux.intor("int2c2e_ip1ip2").reshape([3, 3, naux, naux])

    # --- K20 (basis deriv 2, aux deriv 0) --- #

    # (10|0)(0|10), part a
    dbas_K20_1a = einsum(
        "tuvP, PQ, sklQ, ui, vj, ki, lj -> tsuk",
        int3c2e_ip1,
        int2c2e_inv,
        int3c2e_ip1,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K20_1a = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(aoslices):
            de_K20_1a[A, B] += 2 * einsum("tsuk -> ts", dbas_K20_1a[:, :, p0A:p1A, p0B:p1B])

    # (10|0)(0|10), part b
    dbas_K20_1b = einsum(
        "tuvP, PQ, sklQ, ui, vj, kj, li -> tsuk",
        int3c2e_ip1,
        int2c2e_inv,
        int3c2e_ip1,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K20_1b = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(aoslices):
            de_K20_1b[A, B] += 2 * einsum("tsuk -> ts", dbas_K20_1b[:, :, p0A:p1A, p0B:p1B])

    # (11|0)(0|00)
    dbas_K20_2 = einsum(
        "tsuvP, PQ, klQ, ui, vj, ki, lj -> tsuv",
        int3c2e_ipvip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K20_2 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(aoslices):
            de_K20_2[A, B] += 2 * einsum("tsuv -> ts", dbas_K20_2[:, :, p0A:p1A, p0B:p1B])

    # (20|0)(0|00)
    dbas_K20_3 = einsum(
        "tsuvP, PQ, klQ, ui, vj, ki, lj -> tsuv",
        int3c2e_ipip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K20_3 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        de_K20_3[A, A] += 2 * einsum("tsuv -> ts", dbas_K20_3[:, :, p0A:p1A])

    de_K20 = de_K20_1a + de_K20_1b + de_K20_2 + de_K20_3

    # --- K11 (basis deriv 1, aux deriv 1) --- #

    # (10|1)(0|0)(0|00)
    dbas_K11_1 = einsum(
        "tsuvP, PQ, klQ, vi, li, uj, kj -> tsuP",
        int3c2e_ip1ip2,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K11_1 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K11_1[A, B] += 2 * einsum("tsuP -> ts", dbas_K11_1[:, :, p0A:p1A, p0B:p1B])
    de_K11_1 += de_K11_1.transpose(1, 0, 3, 2)

    # (10|0)(0|1)(0|00)
    dbas_K11_2 = einsum(
        "tuvP, PQ, sQR, RS, klS, ui, vj, ki, lj -> tsuR",
        int3c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K11_2 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K11_2[A, B] += 2 * einsum("tsuR -> ts", dbas_K11_2[:, :, p0A:p1A, p0B:p1B])
    de_K11_2 += de_K11_2.transpose(1, 0, 3, 2)

    # (10|0)(1|0)(0|00)
    dbas_K11_3 = einsum(
        "tuvP, PQ, sQR, RS, klS, ui, vj, ki, lj -> tsuQ",
        int3c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K11_3 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K11_3[A, B] += -2 * einsum("tsuQ -> ts", dbas_K11_3[:, :, p0A:p1A, p0B:p1B])
    de_K11_3 += de_K11_3.transpose(1, 0, 3, 2)

    # (10|0)(0|0)(1|00)
    dbas_K11_4 = einsum(
        "tuvP, PQ, sklQ, ui, vj, ki, lj -> tsuQ",
        int3c2e_ip1,
        int2c2e_inv,
        int3c2e_ip2,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K11_4 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K11_4[A, B] += 2 * einsum("tsuQ -> ts", dbas_K11_4[:, :, p0A:p1A, p0B:p1B])
    de_K11_4 += de_K11_4.transpose(1, 0, 3, 2)

    de_K11 = de_K11_1 + de_K11_2 + de_K11_3 + de_K11_4

    # --- K02 (basis deriv 0, aux deriv 2) --- #

    # (00|2)(0|00)
    dbas_K02_1 = einsum(
        "tsuvP, PQ, klQ, ui, vj, ki, lj -> tsP",
        int3c2e_ipip2,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_1 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        de_K02_1[A, A] += einsum("tsP -> ts", dbas_K02_1[:, :, p0A:p1A])

    # (00|0)(2|0)(0|00)
    dbas_K02_2 = einsum(
        "uvP, PQ, tsQR, RS, klS, ui, vj, ki, lj -> tsQ",
        int3c2e,
        int2c2e_inv,
        int2c2e_ipip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_2 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        de_K02_2[A, A] += -1 * einsum("tsQ -> ts", dbas_K02_2[:, :, p0A:p1A])
    de_K02_2 = de_K02_2

    # (00|0)(1|1)(0|00)
    dbas_K02_3a = einsum(
        "uvP, PQ, tsQR, RS, klS, ui, vj, ki, lj -> tsQR",
        int3c2e,
        int2c2e_inv,
        int2c2e_ip1ip2,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_3a = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K02_3a[A, B] += -0.5 * einsum("tsQR -> ts", dbas_K02_3a[:, :, p0A:p1A, p0B:p1B])
    de_K02_3a += de_K02_3a.transpose(1, 0, 3, 2)

    # (00|0)(1|0)(0|1)(0|00)
    dbas_K02_3b = einsum(
        "uvP, PQ, tQR, RS, sST, TU, klU, ui, vj, ki, lj -> tsQT",
        int3c2e,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_3b = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K02_3b[A, B] += -0.5 * einsum("tsQT -> ts", dbas_K02_3b[:, :, p0A:p1A, p0B:p1B])
    de_K02_3b += de_K02_3b.transpose(1, 0, 3, 2)

    # (00|1)(1|0)(0|00)
    dbas_K02_4 = einsum(
        "tuvP, PQ, sQR, RS, klS, ui, vj, ki, lj -> tsPQ",
        int3c2e_ip2,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_4 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K02_4[A, B] += -1 * einsum("tsPQ -> ts", dbas_K02_4[:, :, p0A:p1A, p0B:p1B])
    de_K02_4 += de_K02_4.transpose(1, 0, 3, 2)

    # (00|1)(1|00)
    dbas_K02_5 = einsum(
        "tuvP, PQ, sklQ, ui, vj, ki, lj -> tsPQ",
        int3c2e_ip2,
        int2c2e_inv,
        int3c2e_ip2,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_5 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K02_5[A, B] += 0.5 * einsum("tsPQ -> ts", dbas_K02_5[:, :, p0A:p1A, p0B:p1B])
    de_K02_5 += de_K02_5.transpose(1, 0, 3, 2)

    # (00|0)(0|1)(1|0)(0|00)
    dbas_K02_6 = einsum(
        "uvP, PQ, tRQ, RS, sST, TU, klU, ui, vj, ki, lj -> tsRS",
        int3c2e,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_6 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K02_6[A, B] += 0.5 * einsum("tsRS -> ts", dbas_K02_6[:, :, p0A:p1A, p0B:p1B])
    de_K02_6 += de_K02_6.transpose(1, 0, 3, 2)

    # (00|1)(0|1)(0|00)
    dbas_K02_7 = einsum(
        "tuvP, PQ, sRQ, RS, klS, ui, vj, ki, lj -> tsPR",
        int3c2e_ip2,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_7 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K02_7[A, B] += -1 * einsum("tsPR -> ts", dbas_K02_7[:, :, p0A:p1A, p0B:p1B])
    de_K02_7 += de_K02_7.transpose(1, 0, 3, 2)

    # (00|0)(1|0)(1|0)(0|00)
    dbas_K02_8 = einsum(
        "uvP, PQ, tQR, RS, sST, TU, klU, ui, vj, ki, lj -> tsQS",
        int3c2e,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int2c2e_ip1,
        int2c2e_inv,
        int3c2e,
        mocc_2,
        mocc_2,
        mocc_2,
        mocc_2,
    )
    de_K02_8 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K02_8[A, B] += 1 * einsum("tsQS -> ts", dbas_K02_8[:, :, p0A:p1A, p0B:p1B])
    de_K02_8 += de_K02_8.transpose(1, 0, 3, 2)

    de_K02 = de_K02_1 + de_K02_2 + de_K02_3a + de_K02_3b + de_K02_4 + de_K02_5 + de_K02_6 + de_K02_7 + de_K02_8

    de_K_skeleton = {
        # de_K20
        "de_K20_1a": de_K20_1a,
        "de_K20_1b": de_K20_1b,
        "de_K20_2": de_K20_2,
        "de_K20_3": de_K20_3,
        # de_K11
        "de_K11_1": de_K11_1,
        "de_K11_2": de_K11_2,
        "de_K11_3": de_K11_3,
        "de_K11_4": de_K11_4,
        # de_K02
        "de_K02_1": de_K02_1,
        "de_K02_2": de_K02_2,
        "de_K02_3a": de_K02_3a,
        "de_K02_3b": de_K02_3b,
        "de_K02_4": de_K02_4,
        "de_K02_5": de_K02_5,
        "de_K02_6": de_K02_6,
        "de_K02_7": de_K02_7,
        "de_K02_8": de_K02_8,
        # total
        "de_K20": de_K20,
        "de_K11": de_K11,
        "de_K02": de_K02,
    }
    return de_K_skeleton


class RHessRIJKNaive(RHessElecInteractAPI):
    def __init__(self, mol: gto.Mole, aux: gto.Mole):
        self.mol = mol
        self.aux = aux
        self.result = dict()

    def make_skeleton_hess(self, mo_coeff, mo_occ, dm0=None):
        de_J_skeleton = get_decomposed_rij_skeleton_deriv2_naive(self.mol, self.aux, mo_coeff, mo_occ)
        de_K_skeleton = get_decomposed_rik_skeleton_deriv2_naive(self.mol, self.aux, mo_coeff, mo_occ)

        self.result.update(de_J_skeleton)
        self.result.update(de_K_skeleton)

        de_J = de_J_skeleton["de_J20"] + de_J_skeleton["de_J11"] + de_J_skeleton["de_J02"]
        de_K = de_K_skeleton["de_K20"] + de_K_skeleton["de_K11"] + de_K_skeleton["de_K02"]
        de_JK = de_J - 0.5 * de_K
        return de_JK
