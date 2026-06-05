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
    dbas_J11_1 = einsum(
        "tsuvP, PQ, klQ, uv, kl -> tsuP",
        int3c2e_ip1ip2,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
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
    dbas_J02_1 = einsum(
        "tsuvP, PQ, klQ, uv, kl -> tsP",
        int3c2e_ipip2,
        int2c2e_inv,
        int3c2e,
        dm0,
        dm0,
    )
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


def get_rij_deriv1_ao_naive(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray
) -> dict[str, np.ndarray]:
    """Get the first derivative of the Coulomb interaction in AO basis.

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

    Returns
    -------
    j1ao : dict[str, np.ndarray]
        The first derivative components of the Coulomb interaction in AO basis, shape [natm, 3, nao, nao].
    """
    nao = mol.nao
    natm = mol.natm
    dm0 = get_dm0_restricted(mo_coeff, mo_occ)
    aoslices = mol.aoslice_by_atom()
    auxslices = aux.aoslice_by_atom()

    int2c2e = aux.intor("int2c2e")
    int2c2e_inv = np.linalg.inv(int2c2e)
    int2c2e_ip1 = aux.intor("int2c2e_ip1")
    int3c2e = _int3c_wrapper(mol, aux, "int3c2e", "s1")()
    int3c2e_ip1 = _int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")()
    int3c2e_ip2 = _int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")()

    scr1 = einsum("tuvP, PQ, klQ, kl -> tuv", int3c2e_ip1, int2c2e_inv, int3c2e, dm0)

    # --- aux derivative 0 --- #

    j1ao_aux0 = np.zeros([natm, 3, nao, nao])
    for A in range(natm):
        _, _, p0, p1 = aoslices[A]
        slc = slice(p0, p1)
        # (10|0)(0|00)
        j1ao_aux0[A, :, slc, :] -= scr1[:, slc, :]
        # (01|0)(0|00) (can be symmetrized)
        j1ao_aux0[A, :, :, slc] -= scr1[:, slc, :].swapaxes(-1, -2)
        # (00|0)(0|10), (00|0)(0|01)
        scr2 = einsum(
            "tklP, PQ, uvQ, kl -> tuv",
            int3c2e_ip1[:, slc],
            int2c2e_inv,
            int3c2e,
            dm0[slc],
        )
        j1ao_aux0[A] -= 2 * scr2

    # --- aux derivative 1 --- #

    j1ao_aux1 = np.zeros([natm, 3, nao, nao])
    for A in range(natm):
        _, _, p0, p1 = auxslices[A]
        slc = slice(p0, p1)
        # (00|1)(0|00)
        j1ao_aux1[A] -= einsum(
            "tuvP, PQ, klQ, kl -> tuv",
            int3c2e_ip2[:, :, :, slc],
            int2c2e_inv[slc, :],
            int3c2e,
            dm0,
        )
        # (00|0)(1|00)
        j1ao_aux1[A] -= einsum(
            "uvP, PQ, tklQ, kl -> tuv",
            int3c2e,
            int2c2e_inv[:, slc],
            int3c2e_ip2[:, :, :, slc],
            dm0,
        )
        # (00|0)(1|0)(0|00)
        j1ao_aux1[A] += einsum(
            "uvP, PQ, tQR, RS, klS, kl -> tuv",
            int3c2e,
            int2c2e_inv[:, slc],
            int2c2e_ip1[:, slc],
            int2c2e_inv,
            int3c2e,
            dm0,
        )
        # (00|0)(0|1)(0|00)
        j1ao_aux1[A] += einsum(
            "uvP, PQ, tRQ, RS, klS, kl -> tuv",
            int3c2e,
            int2c2e_inv,
            int2c2e_ip1[:, slc],
            int2c2e_inv[slc, :],
            int3c2e,
            dm0,
        )

    return {"j1ao_aux0": j1ao_aux0, "j1ao_aux1": j1ao_aux1}


def get_rik_deriv1_ao_naive(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray
) -> dict[str, np.ndarray]:
    """Get the first derivative of the exchange interaction in AO basis.

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

    Returns
    -------
    j1ao : dict[str, np.ndarray]
        The first derivative components of the exchange interaction in AO basis, shape [natm, 3, nao, nao].
    """
    nao = mol.nao
    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    auxslices = aux.aoslice_by_atom()

    occidx = mo_occ > 1e-15
    mocc = mo_coeff[:, occidx]
    occ = mo_occ[occidx]
    mocc_2 = mocc * np.sqrt(occ)

    int2c2e = aux.intor("int2c2e")
    int2c2e_inv = np.linalg.inv(int2c2e)
    int2c2e_ip1 = aux.intor("int2c2e_ip1")
    int3c2e = _int3c_wrapper(mol, aux, "int3c2e", "s1")()
    int3c2e_ip1 = _int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")()
    int3c2e_ip2 = _int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")()

    # --- aux derivative 0 --- #

    scr1 = einsum("tuvP, PQ, klQ, vi, li -> tuk", int3c2e_ip1, int2c2e_inv, int3c2e, mocc_2, mocc_2)

    k1ao_aux0 = np.zeros([natm, 3, nao, nao])
    for A in range(natm):
        _, _, p0, p1 = aoslices[A]
        slc = slice(p0, p1)
        # (10|0)(0|00)
        k1ao_aux0[A, :, slc, :] -= scr1[:, slc, :]
        # (01|0)(0|00)
        k1ao_aux0[A, :, :, slc] -= scr1[:, slc, :].swapaxes(-1, -2)
        # (00|0)(0|10), (00|0)(0|01)
        scr2 = einsum("tklP, PQ, uvQ, ki, ui -> tlv", int3c2e_ip1[:, slc], int2c2e_inv, int3c2e, mocc_2[slc], mocc_2)
        k1ao_aux0[A] -= scr2 + scr2.swapaxes(-1, -2)

    # --- aux derivative 1 --- #

    k1ao_aux1 = np.zeros([natm, 3, nao, nao])
    for A in range(natm):
        _, _, p0, p1 = auxslices[A]
        slc = slice(p0, p1)
        # (00|1)(0|00)
        k1ao_aux1[A] -= einsum(
            "tuvP, PQ, klQ, vi, li -> tuk", int3c2e_ip2[:, :, :, slc], int2c2e_inv[slc, :], int3c2e, mocc_2, mocc_2
        )
        # (00|0)(1|00)
        k1ao_aux1[A] -= einsum(
            "uvP, PQ, tklQ, vi, li -> tuk", int3c2e, int2c2e_inv[:, slc], int3c2e_ip2[:, :, :, slc], mocc_2, mocc_2
        )
        # (00|0)(1|0)(0|00)
        k1ao_aux1[A] += einsum(
            "uvP, PQ, tQR, RS, klS, vi, li -> tuk",
            int3c2e,
            int2c2e_inv[:, slc],
            int2c2e_ip1[:, slc],
            int2c2e_inv,
            int3c2e,
            mocc_2,
            mocc_2,
        )
        # (00|0)(0|1)(0|00)
        k1ao_aux1[A] += einsum(
            "uvP, PQ, tRQ, RS, klS, vi, li -> tuk",
            int3c2e,
            int2c2e_inv,
            int2c2e_ip1[:, slc],
            int2c2e_inv[slc, :],
            int3c2e,
            mocc_2,
            mocc_2,
        )

    return {"k1ao_aux0": k1ao_aux0, "k1ao_aux1": k1ao_aux1}


def get_rijk_response_bra_naive(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray, bra: np.ndarray
) -> np.ndarray:
    """Compute the response of RI-JK by given bra (perturbed coefficients).

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
    bra : np.ndarray
        The bra vector (perturbed coefficients), shape [..., nao, nocc].

    Returns
    -------
    resp_half_trans : np.ndarray
        The response of RI-JK with half transformation, shape [..., nao, nocc].

    Notes
    -----
    This function only works in restricted case, not designed for fractional charge or ROHF.
    We also will check if `bra` have the correct occupation number.
    """
    nao = mol.nao
    occidx = mo_occ > 1e-15
    mocc = mo_coeff[:, occidx]
    nocc = mocc.shape[-1]

    int2c2e = aux.intor("int2c2e")
    int2c2e_inv = np.linalg.inv(int2c2e)
    int3c2e = _int3c_wrapper(mol, aux, "int3c2e", "s1")()

    # reshape bra to (-1, nao, nocc)
    bra_shape = bra.shape
    assert bra_shape[-2] == nao
    assert bra_shape[-1] == nocc
    bra = bra.reshape(-1, nao, nocc)

    resp_bra_j = 4 * einsum("uvP, PQ, klQ, Akj, lj, vi -> Aui", int3c2e, int2c2e_inv, int3c2e, bra, mocc, mocc)
    resp_bra_k0 = einsum("uvP, PQ, klQ, Avj, lj, ki -> Aui", int3c2e, int2c2e_inv, int3c2e, bra, mocc, mocc)
    resp_bra_k1 = einsum("uvP, PQ, klQ, Akj, vj, li -> Aui", int3c2e, int2c2e_inv, int3c2e, bra, mocc, mocc)
    resp_bra = resp_bra_j - resp_bra_k0 - resp_bra_k1

    # restore original shape
    resp_bra = resp_bra.reshape(bra_shape)
    return resp_bra


class RHessRIJKNaive(RHessElecInteractAPI):
    def __init__(self, mol: gto.Mole, aux: gto.Mole):
        self.mol = mol
        self.aux = aux
        self.scale_j = 1.0
        self.scale_k = 0.5
        self.mo_coeff = None
        self.mo_occ = None
        self.result = dict()

    def make_skeleton_hess(self, mo_coeff, mo_occ):
        de_J_skeleton = get_decomposed_rij_skeleton_deriv2_naive(self.mol, self.aux, mo_coeff, mo_occ)
        de_K_skeleton = get_decomposed_rik_skeleton_deriv2_naive(self.mol, self.aux, mo_coeff, mo_occ)

        self.result.update(de_J_skeleton)
        self.result.update(de_K_skeleton)

        de_J = de_J_skeleton["de_J20"] + de_J_skeleton["de_J11"] + de_J_skeleton["de_J02"]
        de_K = de_K_skeleton["de_K20"] + de_K_skeleton["de_K11"] + de_K_skeleton["de_K02"]
        de_JK = self.scale_j * de_J - self.scale_k * de_K
        return de_JK

    def get_deriv1_ao(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        j1ao_dict = get_rij_deriv1_ao_naive(self.mol, self.aux, mo_coeff, mo_occ)
        k1ao_dict = get_rik_deriv1_ao_naive(self.mol, self.aux, mo_coeff, mo_occ)

        self.result.update(j1ao_dict)
        self.result.update(k1ao_dict)

        j1ao = j1ao_dict["j1ao_aux0"] + j1ao_dict["j1ao_aux1"]
        k1ao = k1ao_dict["k1ao_aux0"] + k1ao_dict["k1ao_aux1"]
        deriv_ao = self.scale_j * j1ao - self.scale_k * k1ao
        return deriv_ao

    def prepare_response(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

    def get_response_bra(self, bra: np.ndarray) -> np.ndarray:
        return get_rijk_response_bra_naive(self.mol, self.aux, self.mo_coeff, self.mo_occ, bra)
