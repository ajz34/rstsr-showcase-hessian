"""
Prototype of restricted Hessian optimization using JK skeletonization.
"""

import numpy as np
import scipy
from pyscf import gto, ao2mo, lib
from functools import partial
from pyscf.df.grad.rhf import _int3c_wrapper

from pyhessref.util import get_dm0_restricted

# override einsum for some efficiency
einsum = partial(np.einsum, optimize=True)


def gen_solve_by_j2c(int2c):
    int2c_l = scipy.linalg.cholesky(int2c, lower=True)

    def solve_by_j2c(v, flip=False, left=True):
        res = None
        shape = v.shape
        if left and not flip:
            v = v.reshape(shape[0], -1)
            res = scipy.linalg.solve_triangular(int2c_l, v, lower=True).reshape(shape)
        elif left and flip:
            v = v.reshape(shape[0], -1)
            res = scipy.linalg.solve_triangular(int2c_l.T, v, lower=False).reshape(shape)
        elif not left and not flip:
            v = v.reshape(-1, shape[-1])
            res = scipy.linalg.solve_triangular(int2c_l.T, v.T, lower=False).T.reshape(shape)
        elif not left and flip:
            v = v.reshape(-1, shape[-1])
            res = scipy.linalg.solve_triangular(int2c_l, v.T, lower=True).T.reshape(shape)
        return res

    return solve_by_j2c


def get_decomposed_skeleton(
    mol: gto.Mole,
    aux: gto.Mole,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    cderi: np.ndarray,
    nbatch_aux: int,
    atm_list: list[int] | None = None,
) -> dict[str, np.ndarray]:
    # === 1. basic preparation === #

    # --- 1.1 really basic --- #

    nao = mol.nao
    naux = aux.nao
    result = {}

    # --- 1.2 occupation --- #

    # - mocc                [nao, nocc]
    # - mocc_2              [nao, nocc]
    # - occ_invsqrt         [nocc]
    occidx = mo_occ > 0
    mocc = mo_coeff[:, occidx]
    occ = mo_occ[occidx]
    nocc = len(occ)
    mocc_2 = mocc * np.sqrt(occ)
    occ_invsqrt = occ**-0.5
    dm0 = get_dm0_restricted(mo_coeff, mo_occ)
    # dm0_tp: packed density matrix, off-diagonal multiplied by 2
    dm0_ = 2 * dm0
    for i in range(nao):
        dm0_[i, i] *= 0.5
    dm0_tp = lib.pack_tril(dm0_)

    # --- 1.3 handle aoslices --- #

    aoslices = mol.aoslice_by_atom()
    auxslices = aux.aoslice_by_atom()
    aoslices = aoslices if atm_list is None else [aoslices[A] for A in atm_list]
    auxslices = auxslices if atm_list is None else [auxslices[A] for A in atm_list]
    natm = len(aoslices)

    # --- 1.4 partition (without atom) --- #

    # [shl_start, shl_end, p1-p0]
    aux_ranges_ = ao2mo.outcore.balance_partition(aux.ao_loc, nbatch_aux)
    # [shl_start, shl_end, p0, p1]
    aux_ranges = []
    p0 = 0
    for sh0, sh1, size in aux_ranges_:
        p1 = p0 + size
        aux_ranges.append((sh0, sh1, p0, p1))
        p0 = p1
    print(aux_ranges)

    # --- 1.5 integral generator --- #
    # NOTE: this is prototype optimize implementation, we use full integrals instead
    #       the 3c full integrals are not supposed to be stored in memory in real applications

    FULL3c_ip1 = _int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")().reshape([3, nao, nao, naux])
    FULL3c_ip2 = _int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")().reshape([3, nao, nao, naux])
    FULL3c_ipip1 = _int3c_wrapper(mol, aux, "int3c2e_ipip1", "s1")().reshape([3, 3, nao, nao, naux])
    FULL3c_ipvip1 = _int3c_wrapper(mol, aux, "int3c2e_ipvip1", "s1")().reshape([3, 3, nao, nao, naux])
    FULL3c_ip1ip2 = _int3c_wrapper(mol, aux, "int3c2e_ip1ip2", "s1")().reshape([3, 3, nao, nao, naux])
    FULL3c_ipip2 = _int3c_wrapper(mol, aux, "int3c2e_ipip2", "s1")().reshape([3, 3, nao, nao, naux])

    FULL3c_ip1 = np.ascontiguousarray(einsum("tuvP -> tPvu", FULL3c_ip1))
    FULL3c_ip2 = np.ascontiguousarray(einsum("tuvP -> tPvu", FULL3c_ip2))
    FULL3c_ipip1 = np.ascontiguousarray(einsum("tsuvP -> tsPvu", FULL3c_ipip1))
    FULL3c_ipvip1 = np.ascontiguousarray(einsum("tsuvP -> tsPvu", FULL3c_ipvip1))
    FULL3c_ip1ip2 = np.ascontiguousarray(einsum("tsuvP -> tsPvu", FULL3c_ip1ip2))
    FULL3c_ipip2 = np.ascontiguousarray(einsum("tsuvP -> tsPvu", FULL3c_ipip2))

    # === 2. common tensor preparation === #

    # --- 2.1 solve_by_j2c --- #

    j2c = aux.intor("int2c2e")
    solve_by_j2c = gen_solve_by_j2c(j2c)

    # --- 2.2 cderi related --- #

    # TODO: lcd_eri_bra, lcd_eri_occ are K-only

    # cderi                 [naux, nao_tp]
    # lcd_eri_aux           [naux]                          itm_j
    # lcd_eri_occ           [naux, nocc, nocc]              itm_k_occ
    # lcd_eri_bra           [naux, nocc, nao]               cderi_xob
    lcd_eri_aux = cderi @ dm0_tp
    lcd_eri_occ = np.empty([naux, nocc, nocc])
    lcd_eri_bra = np.empty([naux, nocc, nao])
    for p in range(naux):  # PAR-ITER
        tmp1 = lib.unpack_tril(cderi[p])
        lcd_eri_bra[p] = mocc.T @ tmp1
        lcd_eri_occ[p] = lcd_eri_bra[p] @ mocc

    # llcd_eri_aux          [naux]                          solved_itm_j
    # llcd_eri_occ          [naux, nocc, nocc]              solved_itm_k_occ
    # llcd_eri_bra          [naux, nocc, nao]               solved_cderi_xob
    llcd_eri_aux = solve_by_j2c(lcd_eri_aux, left=True, flip=True)
    llcd_eri_occ = solve_by_j2c(lcd_eri_occ, left=True, flip=True)
    llcd_eri_bra = solve_by_j2c(lcd_eri_bra, left=True, flip=True)

    # --- 2.3 2c related --- #

    # TODO: this part is aux1/aux2-only

    # j2c_ip1               [3, naux, naux]
    # j2c_ipip1             [3, 3, naux, naux]
    # j2c_ip1ip2            [3, 3, naux, naux]
    j2c_ip1 = aux.intor("int2c2e_ip1")
    j2c_ipip1 = aux.intor("int2c2e_ipip1").reshape([3, 3, naux, naux])
    j2c_ip1ip2 = aux.intor("int2c2e_ip1ip2").reshape([3, 3, naux, naux])
    j2c_ip1 = np.ascontiguousarray(einsum("tPQ -> tQP", j2c_ip1))
    j2c_ipip1 = np.ascontiguousarray(einsum("tsPQ -> tsQP", j2c_ipip1))
    j2c_ip1ip2 = np.ascontiguousarray(einsum("tsPQ -> tsQP", j2c_ip1ip2))

    # === 3. evaluation: ip2 === #

    # --- 3.1 J02_2 --- #

    # dbas_J02_2 = np.einsum("P, tsPQ, Q -> tsP", llcd_eri_aux, j2c_ipip1, llcd_eri_aux)
    dbas_J02_2 = (j2c_ipip1 * llcd_eri_aux).sum(axis=-1) * llcd_eri_aux

    de_J02_2 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0, p1) in enumerate(auxslices):
        # de_J02_2[A, A] += -1 * np.einsum("tsQ -> ts", dbas_J02_2[:, :, p0:p1])
        de_J02_2[A, A] += -1 * dbas_J02_2[..., p0:p1].sum(axis=-1)
    result["de_J02_2"] = de_J02_2

    return result
