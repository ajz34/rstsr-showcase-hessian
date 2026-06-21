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
np.einsum = partial(np.einsum, optimize=True)


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
    # === TASKS TO DO === #

    """
    | TASK        | J | K |
    |-------------|---|---|
    | 20-1        | x | x |
    | 20-2        | x | x |
    | 20-3        | x | x |
    | 11-1        | x | x |
    | 11-2        | x | x |
    | 11-3        | x | x |
    | 11-4        | x | x |
    | 02-1        | x | x |
    | 02-2        | x | x |
    | 02-3a       | x | x |
    | 02-3b       | x | x |
    | 02-4        | x | x |
    | 02-5        | x | x |
    | 02-6        | x | x |
    | 02-7        | x | x |
    | 02-8        | x | x |
    | f1-aux0-1/2 |   |   |
    | f1-aux0-3/4 |   |   |
    | f1-aux1-1/2 | x | x |
    | f1-aux1-3/4 | x | x |
    """

    # region 1. basic preparation

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
    #
    # Memory (units: #floats; naux ~ 3*nao, nocc ~ 6*natm, nao > nocc):
    #   Per-batch production cost (the real bound): nbatch_aux * nao^2 per 1st-deriv slice,
    #   3 * nbatch_aux * nao^2 per 2nd-deriv slice (the t / t,s component is NOT batched).

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

    # endregion 1

    # region 2. common tensor preparation
    #
    # Memory (units: #floats; naux ~ 3*nao, nocc ~ 6*natm):
    #   cderi          [naux, nao_tp]         ~ 0.5 * nao^2 * naux    (input, kept)
    #   lcd_eri_aux    [naux]                 ~ naux                  (transient -> llcd_eri_aux)
    #   lcd_eri_occ    [naux, nocc, nocc]     ~ naux * nocc^2         (transient -> llcd_eri_occ)
    #   lcd_eri_bra    [naux, nocc, nao]      ~ naux * nocc * nao     (transient -> llcd_eri_bra)
    #   llcd_eri_*     same shapes as lcd_*                            (kept through region 5)
    #   fold_eri_aux   [naux, naux]           ~ naux^2                (kept through 3k2)
    #   j2c_ip1 [3,naux,naux], j2c_ipip1/j2c_ip1ip2 [3,3,naux,naux]  ~ 3*naux^2 / 9*naux^2
    #   lcd/llcd_j2c_ip1 [3,naux,naux] each                          ~ 3*naux^2 each
    #   j2c_inv, j2c_l_inv [naux,naux] each                          ~ naux^2 each
    #   region-2 carried peak ~ naux^2 + 2*naux*nocc*nao (lcd_eri_bra x2) + small;
    #   after llcd_* are built, lcd_* are freed.

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
        lcd_eri_bra[p] = mocc_2.T @ tmp1
        lcd_eri_occ[p] = lcd_eri_bra[p] @ mocc_2

    # llcd_eri_aux          [naux]                          solved_itm_j
    # llcd_eri_occ          [naux, nocc, nocc]              solved_itm_k_occ
    # llcd_eri_bra          [naux, nocc, nao]               solved_cderi_xob
    # fold_eri_aux          [naux, naux]                    solved_itm_k_aux
    llcd_eri_aux = solve_by_j2c(lcd_eri_aux, left=True, flip=True)
    llcd_eri_occ = solve_by_j2c(lcd_eri_occ, left=True, flip=True)
    llcd_eri_bra = solve_by_j2c(lcd_eri_bra, left=True, flip=True)
    del lcd_eri_aux, lcd_eri_occ, lcd_eri_bra
    # fold_eri_aux = np.einsum("Pij, Qij -> PQ", llcd_eri_occ, llcd_eri_occ)
    fold_eri_aux = llcd_eri_occ.reshape(naux, -1) @ llcd_eri_occ.reshape(naux, -1).T

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

    rcd_j2c_ip1 = np.asarray([solve_by_j2c(m, left=False, flip=True) for m in j2c_ip1])
    rrcd_j2c_ip1 = np.asarray([solve_by_j2c(m, left=False, flip=False) for m in rcd_j2c_ip1])
    lcd_j2c_ip1 = np.asarray([solve_by_j2c(m, left=True, flip=False) for m in j2c_ip1])
    llcd_j2c_ip1 = np.asarray([solve_by_j2c(m, left=True, flip=True) for m in lcd_j2c_ip1])

    assert np.allclose(rrcd_j2c_ip1, -llcd_j2c_ip1.swapaxes(-1, -2))
    assert np.allclose(rcd_j2c_ip1, -lcd_j2c_ip1.swapaxes(-1, -2))
    # we should try disable one of the cholesky solve, since for ip1, solve left/right is asymmetric
    del rcd_j2c_ip1, rrcd_j2c_ip1

    # j2c_inv = solve_by_j2c(solve_by_j2c(np.eye(naux), left=True, flip=True), left=False, flip=False)
    j2c_l_inv = solve_by_j2c(np.eye(naux), left=True, flip=True)
    j2c_inv = solve_by_j2c(j2c_l_inv.T, left=True, flip=True)

    # endregion 2

    # region 3j2. evaluation: non cderi derivative (j part)
    #
    # Memory (units: #floats): all dbas_J02_* are aux-pair tensors.
    #   dbas_J02_2   [3,3,naux]            ~ 9*naux
    #   dbas_J02_3a/3b/6/8 [3,3,naux,naux] ~ 9*naux^2 each  (4 of them)
    #   region-3j2 additional peak ~ 4 * 9 * naux^2 = 36 * naux^2  (+ transient tmp1/tmp2 ~ naux^2)
    # Inputs j2c_ipip1, j2c_ip1ip2 are shared with 3k2 -> freed after 3k2.

    # --- J02-2 --- #

    # dbas_J02_2 = np.einsum("P, tsPQ, Q -> tsP", llcd_eri_aux, j2c_ipip1, llcd_eri_aux)
    dbas_J02_2 = (j2c_ipip1 * llcd_eri_aux).sum(axis=-1) * llcd_eri_aux
    dbas_J02_2 *= -1

    # --- J02-3a --- #

    # dbas_J02_3a = np.einsum("P, tsPQ, Q -> tsPQ", llcd_eri_aux, j2c_ip1ip2, llcd_eri_aux)
    dbas_J02_3a = j2c_ip1ip2 * llcd_eri_aux[:, None] * llcd_eri_aux[None, :]
    dbas_J02_3a *= -0.5

    # --- J02-3b --- #

    # dbas_J02_3b = np.einsum("P, tRP, sRQ, Q -> tsPQ", llcd_eri_aux, lcd_j2c_ip1, lcd_j2c_ip1, llcd_eri_aux)
    tmp1 = lcd_j2c_ip1 * llcd_eri_aux
    dbas_J02_3b = tmp1[:, None].swapaxes(-1, -2) @ tmp1[None, :]
    dbas_J02_3b *= 0.5

    # --- J02-6 --- #

    # dbas_J02_6 = einsum("R, tRP, PQ, sSQ, S -> tsPQ", llcd_eri_aux, j2c_ip1, j2c_inv, j2c_ip1, llcd_eri_aux)
    tmp1 = (j2c_ip1 * llcd_eri_aux).sum(axis=-1)
    dbas_J02_6 = tmp1[:, None, :, None] * j2c_inv * tmp1[None, :, None, :]
    dbas_J02_6 *= 0.5

    # --- J02-8 --- #

    # dbas_J02_8 = np.einsum("R, tPR, sPQ, Q -> tsPQ", llcd_eri_aux, j2c_ip1, llcd_j2c_ip1, llcd_eri_aux)
    tmp1 = (j2c_ip1 * llcd_eri_aux).sum(axis=-1)
    tmp2 = llcd_j2c_ip1 * llcd_eri_aux
    dbas_J02_8 = tmp1[:, None, :, None] * tmp2
    dbas_J02_8 *= -1

    # --- skeleton j2 sum --- #

    de_J02_2 = np.zeros((natm, natm, 3, 3))
    de_J02_3a = np.zeros((natm, natm, 3, 3))
    de_J02_3b = np.zeros((natm, natm, 3, 3))
    de_J02_6 = np.zeros((natm, natm, 3, 3))
    de_J02_8 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        de_J02_2[A, A] = dbas_J02_2[..., p0A:p1A].sum(axis=-1)
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_3a[A, B] = dbas_J02_3a[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_J02_3b[A, B] = dbas_J02_3b[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_J02_6[A, B] = dbas_J02_6[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_J02_8[A, B] = dbas_J02_8[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
    de_J02_3a += de_J02_3a.transpose(1, 0, 3, 2)
    de_J02_3b += de_J02_3b.transpose(1, 0, 3, 2)
    de_J02_6 += de_J02_6.transpose(1, 0, 3, 2)
    de_J02_8 += de_J02_8.transpose(1, 0, 3, 2)
    result["de_J02_2"] = de_J02_2
    result["de_J02_3a"] = de_J02_3a
    result["de_J02_3b"] = de_J02_3b
    result["de_J02_6"] = de_J02_6
    result["de_J02_8"] = de_J02_8
    del dbas_J02_2, dbas_J02_3a, dbas_J02_3b, dbas_J02_6, dbas_J02_8

    # endregion 3j2

    # region 3j1. evaluation: non cderi derivative (j1ao part)
    #
    # Memory (units: #floats): j1ao_aux1_3/4 outputs are [natm,3,nao,nao] ~ natm*nao^2 each.
    #   per-t transients: tmp1 [naux,nocc,nao] or [naux,nocc,nocc] ~ naux*nocc*nao
    #   tmp2 [natm,3,naux], j1ao_aux1_*_tp [natm*3, nao_tp] ~ natm*nao^2
    #   region-3j1 additional peak ~ natm*nao^2 (output) + naux*nocc*nao (tmp1) + natm*nao^2 (tp)

    # temporary area for j1 aux1-3
    # tmp1 = np.einsum("tRQ, R -> tQ", j2c_ip1, llcd_eri_aux)
    tmp1 = -(j2c_ip1 * llcd_eri_aux).sum(axis=-1)
    tmp2 = np.zeros((natm, 3, naux))
    for A in range(mol.natm):
        _, _, p0, p1 = auxslices[A]
        slcA = slice(p0, p1)
        tmp2[A, :, slcA] = tmp1[:, slcA]
    tmp3 = solve_by_j2c(tmp2, left=False, flip=True)
    j1ao_aux1_3_tp = tmp3.reshape(natm * 3, naux) @ cderi
    j1ao_aux1_3 = lib.unpack_tril(j1ao_aux1_3_tp).reshape(natm, 3, nao, nao)
    result["j1ao_aux1_3"] = j1ao_aux1_3

    # temporary area for j1 aux1-4
    tmp1 = lcd_j2c_ip1 * llcd_eri_aux
    tmp2 = np.zeros((natm, 3, naux))
    for A in range(mol.natm):
        _, _, p0, p1 = auxslices[A]
        slcA = slice(p0, p1)
        tmp2[A] = tmp1[:, :, slcA].sum(axis=-1)
    j1ao_aux1_4_tp = tmp2.reshape(natm * 3, naux) @ cderi
    j1ao_aux1_4 = lib.unpack_tril(j1ao_aux1_4_tp).reshape(natm, 3, nao, nao)
    result["j1ao_aux1_4"] = j1ao_aux1_4

    # endregion 3j1

    # region 3k2. evaluation: non cderi derivative (k part)
    #
    # Memory (units: #floats): same aux-pair shapes as 3j2.
    #   dbas_K02_2   [3,3,naux]            ~ 9*naux
    #   dbas_K02_3a/3b/6/8 [3,3,naux,naux] ~ 9*naux^2 each  (4 of them)
    #   region-3k2 additional peak ~ 36 * naux^2 (+ transient ~ naux^2)
    # Inputs j2c_ipip1, j2c_ip1ip2 (shared with 3j2), fold_eri_aux, lcd_j2c_ip1 freed at end of 3k2.

    # --- K02-2 --- #

    # dbas_K02_2 = np.einsum("PQ, tsPQ -> tsQ", fold_eri_aux, j2c_ipip1)
    # dbas_K02_2 = np.einsum("PQ, tsPQ -> tsP", fold_eri_aux, j2c_ipip1)
    dbas_K02_2 = (j2c_ipip1 * fold_eri_aux).sum(axis=-1)
    dbas_K02_2 *= -1

    # --- K02-3a --- #

    # dbas_K02_3a = np.einsum("PQ, tsPQ -> tsPQ", fold_eri_aux, j2c_ip1ip2)
    dbas_K02_3a = j2c_ip1ip2 * fold_eri_aux
    dbas_K02_3a *= -0.5

    # --- K02-3b --- #

    # dbas_K02_3b = np.einsum("PQ, tRP, sRQ -> tsPQ", fold_eri_aux, lcd_j2c_ip1, lcd_j2c_ip1)
    dbas_K02_3b = lcd_j2c_ip1[:, None].swapaxes(-1, -2) @ lcd_j2c_ip1[None, :] * fold_eri_aux
    dbas_K02_3b *= 0.5

    # --- K02-6 --- #

    # dbas_K02_6 = - np.einsum("RS, tRP, PQ, sSQ -> tsPQ", fold_eri_aux, j2c_ip1, j2c_inv, j2c_ip1)
    dbas_K02_6 = j2c_ip1[:, None] @ fold_eri_aux @ j2c_ip1[None, :] * j2c_inv
    dbas_K02_6 *= -0.5

    # --- K02-8 --- #

    # dbas_K02_8 = np.einsum("PS, tQP, sQS -> tsPQ", fold_eri_aux, llcd_j2c_ip1, j2c_ip1)
    dbas_K02_8 = fold_eri_aux @ j2c_ip1[None, :].swapaxes(-1, -2) * llcd_j2c_ip1[:, None].swapaxes(-1, -2)
    dbas_K02_8 *= -1

    # --- skeleton k2 sum --- #

    de_K02_2 = np.zeros((natm, natm, 3, 3))
    de_K02_3a = np.zeros((natm, natm, 3, 3))
    de_K02_3b = np.zeros((natm, natm, 3, 3))
    de_K02_6 = np.zeros((natm, natm, 3, 3))
    de_K02_8 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        de_K02_2[A, A] = dbas_K02_2[..., p0A:p1A].sum(axis=-1)
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_K02_3a[A, B] = dbas_K02_3a[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_K02_3b[A, B] = dbas_K02_3b[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_K02_6[A, B] = dbas_K02_6[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_K02_8[A, B] = dbas_K02_8[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
    de_K02_3a += de_K02_3a.transpose(1, 0, 3, 2)
    de_K02_3b += de_K02_3b.transpose(1, 0, 3, 2)
    de_K02_6 += de_K02_6.transpose(1, 0, 3, 2)
    de_K02_8 += de_K02_8.transpose(1, 0, 3, 2)
    result["de_K02_2"] = de_K02_2
    result["de_K02_3a"] = de_K02_3a
    result["de_K02_3b"] = de_K02_3b
    result["de_K02_6"] = de_K02_6
    result["de_K02_8"] = de_K02_8
    del dbas_K02_2, dbas_K02_3a, dbas_K02_3b, dbas_K02_6, dbas_K02_8
    del j2c_ipip1, j2c_ip1ip2, fold_eri_aux

    # endregion 3k2

    # region 3k1. evaluation: non cderi derivative (k1bra part)
    #
    # Memory (units: #floats): k1bra_aux1_3/4 outputs [natm,3,nocc,nao] ~ natm*nocc*nao each.
    #   per-t transients: tmp1 [naux,nocc,nao] or [naux,nocc,nocc] ~ naux*nocc*nao
    #   region-3k1 additional peak ~ 2*natm*nocc*nao (outputs) + naux*nocc*nao (tmp1)

    # additional memory: tmp1 (aux * basis * occ)
    k1bra_aux1_3 = np.zeros([natm, 3, nocc, nao])
    for t in range(3):
        # tmp1 = np.einsum("RQ, Rjk -> Qjk", j2c_ip1[t], llcd_eri_bra)
        tmp1 = j2c_ip1[t].T @ llcd_eri_bra.reshape(naux, nocc * nao)
        tmp1 = tmp1.reshape(naux, nocc, nao)
        for A in range(mol.natm):
            _, _, p0, p1 = auxslices[A]
            slcA = slice(p0, p1)
            # k1bra_aux1_3[A, t] = np.einsum("Qji, Qjk -> ik", llcd_eri_occ[slcA], tmp1[slcA])
            k1bra_aux1_3[A, t] = llcd_eri_occ[slcA].reshape(-1, nocc).T @ tmp1[slcA].reshape(-1, nao)
    k1bra_aux1_3 *= occ_invsqrt[None, None, :, None]
    result["k1bra_aux1_3"] = k1bra_aux1_3

    # additional memory: tmp1 (aux * occ * occ)
    k1bra_aux1_4 = np.zeros([natm, 3, nocc, nao])
    for t in range(3):
        # tmp1 = np.einsum("Qij, QR -> Rij", llcd_eri_occ, j2c_ip1[t])
        tmp1 = (j2c_ip1[t].T @ llcd_eri_occ.reshape(naux, nocc * nocc)).reshape(naux, nocc, nocc)
        for A in range(mol.natm):
            _, _, p0, p1 = auxslices[A]
            slcA = slice(p0, p1)
            # k1bra_aux1_4[A, t] = np.einsum("Rji, Rjk -> ik", tmp1[slcA], llcd_eri_bra[slcA])
            k1bra_aux1_4[A, t] = tmp1[slcA].reshape(-1, nocc).T @ llcd_eri_bra[slcA].reshape(-1, nao)
    k1bra_aux1_4 *= occ_invsqrt[None, None, :, None]
    result["k1bra_aux1_4"] = k1bra_aux1_4

    # endregion 3k1

    # region 4. evaluation: one-shot derivative
    #
    # Memory (units: #floats): the 3c 2nd-deriv integrals are consumed per aux batch (size
    # nbatch_aux); only the batch slice lives in memory in production.
    #   per-batch slice j3c_*_batch [3,3,nbatch_aux,nao,nao] ~ 9 * nbatch_aux * nao^2
    #   tmp1 [3,3,nao,nao] ~ 9*nao^2  (per-batch transient); tmp_k_ao [nbatch_aux,nao,nao]
    #   accumulated dbas_J20_2/3, dbas_K20_2/3   [3,3,nao,nao] ~ 9*nao^2 each  (4 of them)
    #   accumulated dbas_J11_1, dbas_K11_1       [3,3,nao,naux] ~ 9*nao*naux each  (2 of them)
    #   accumulated dbas_J02_1, dbas_K02_1       [3,3,naux] ~ 9*naux each
    #   region-4 production peak ~ 9*nbatch_aux*nao^2 (slice) + 4*9*nao^2 + 2*9*nao*naux (dbas)
    # The 4 FULL3c_* 2nd-deriv integrals are freed at the end of region 4 (region 5 needs only ip2).

    dbas_J20_2 = np.zeros((3, 3, nao, nao))
    dbas_J20_3 = np.zeros((3, 3, nao, nao))
    dbas_J11_1 = np.zeros((3, 3, nao, naux))
    dbas_J02_1 = np.zeros((3, 3, naux))
    dbas_K20_2 = np.zeros((3, 3, nao, nao))
    dbas_K20_3 = np.zeros((3, 3, nao, nao))
    dbas_K11_1 = np.zeros((3, 3, nao, naux))
    dbas_K02_1 = np.zeros((3, 3, naux))

    for _sh0, _sh1, p0, p1 in aux_ranges:

        # --- J20-2 --- #
        j3c_ipvip1_batch = FULL3c_ipvip1[:, :, p0:p1]  # use generator in real application
        tmp1 = (j3c_ipvip1_batch * llcd_eri_aux[p0:p1, None, None]).sum(axis=-3)
        # dbas_J20_2 += einsum("tsvu, uv -> tsuv", tmp1, dm0)
        dbas_J20_2 += tmp1.swapaxes(-1, -2) * dm0

        # --- J20-3 --- #
        j3c_ipip1_batch = FULL3c_ipip1[:, :, p0:p1]  # use generator in real application
        tmp1 = (j3c_ipip1_batch * llcd_eri_aux[p0:p1, None, None]).sum(axis=-3)
        # dbas_J20_3 += einsum("tsvu, uv -> tsuv", tmp1, dm0)
        dbas_J20_3 += tmp1.swapaxes(-1, -2) * dm0

        # --- J11-1 --- #
        j3c_ip1ip2_batch = FULL3c_ip1ip2[:, :, p0:p1]  # use generator in real application
        # dbas_J11_1[..., p0:p1] = einsum("tsPvu, uv, P -> tsuP", j3c_ip1ip2, dm0, llcd_eri_aux[p0:p1])
        tmp1 = (j3c_ip1ip2_batch * dm0).sum(axis=-2).swapaxes(-1, -2)
        dbas_J11_1[..., p0:p1] = tmp1 * llcd_eri_aux[p0:p1]

        # --- J02-1 --- #
        j3c_ipip2_batch = FULL3c_ipip2[:, :, p0:p1]  # use generator in real application
        tmp1 = (j3c_ipip2_batch * dm0).sum(axis=(-1, -2))
        dbas_J02_1[..., p0:p1] = tmp1 * llcd_eri_aux[p0:p1]

        # --- K preparation --- #
        # tmp_k_ao = np.einsum("Pij, vj, ui -> Pvu", llcd_eri_occ[p0:p1], mocc_2, mocc_2)
        # NOTE: tmp_k_ao is symmetric in its AO pair (v,u); this symmetry is what lets the
        # K20/K11/K02 contractions below use plain elementwise `*` (the AO-pair order of the
        # asymmetric 3c integrals does not need explicit swapping against tmp_k_ao).
        tmp_k_ao = mocc_2 @ llcd_eri_occ[p0:p1] @ mocc_2.T

        # --- K20-2 --- #
        # dbas_K20_2 += einsum("tsPvu, Pvu -> tsuv", j3c_ipvip1, tmp_k_ao)
        dbas_K20_2 += (j3c_ipvip1_batch * tmp_k_ao).sum(axis=-3).swapaxes(-1, -2)

        # --- K20-3 --- #
        # dbas_K20_3 += einsum("tsPvu, Pvu -> tsuv", j3c_ipip1, tmp_k_ao)
        dbas_K20_3 += (j3c_ipip1_batch * tmp_k_ao).sum(axis=-3).swapaxes(-1, -2)

        # --- K11-1 --- #
        # dbas_K11_1[..., p0:p1] = einsum("tsPvu, Puv -> tsuP", j3c_ip1ip2, tmp_k_ao)
        tmp1 = (j3c_ip1ip2_batch * tmp_k_ao).sum(axis=-2).swapaxes(-1, -2)
        dbas_K11_1[..., p0:p1] = tmp1

        # --- K02-1 --- #
        # dbas_K02_1[..., p0:p1] = einsum("tsPvu, Pvu -> tsP", j3c_ipip2, tmp_k_ao[p0:p1])
        dbas_K02_1[..., p0:p1] = (j3c_ipip2_batch * tmp_k_ao).sum(axis=(-1, -2))

    de_J20_2 = np.zeros((natm, natm, 3, 3))
    de_J20_3 = np.zeros((natm, natm, 3, 3))
    de_J11_1 = np.zeros((natm, natm, 3, 3))
    de_J02_1 = np.zeros((natm, natm, 3, 3))
    de_K20_2 = np.zeros((natm, natm, 3, 3))
    de_K20_3 = np.zeros((natm, natm, 3, 3))
    de_K11_1 = np.zeros((natm, natm, 3, 3))
    de_K02_1 = np.zeros((natm, natm, 3, 3))
    # dbas_J20_2/3 and dbas_K20_2/3 are [t, s, u, v]; A -> u (axis -2), B -> v (axis -1).
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        de_J20_3[A, A] = 2 * dbas_J20_3[..., p0A:p1A, :].sum(axis=(-1, -2))
        de_K20_3[A, A] = 2 * dbas_K20_3[..., p0A:p1A, :].sum(axis=(-1, -2))
        for B, (_, _, p0B, p1B) in enumerate(aoslices):
            de_J20_2[A, B] = 2 * dbas_J20_2[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_K20_2[A, B] = 2 * dbas_K20_2[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
    for A, (_, _, p0A, p1A) in enumerate(aoslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J11_1[A, B] = 2 * dbas_J11_1[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_K11_1[A, B] = 2 * dbas_K11_1[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
    de_J11_1 += de_J11_1.transpose(1, 0, 3, 2)
    de_K11_1 += de_K11_1.transpose(1, 0, 3, 2)
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        de_J02_1[A, A] = dbas_J02_1[..., p0A:p1A].sum(axis=-1)
        de_K02_1[A, A] = dbas_K02_1[..., p0A:p1A].sum(axis=-1)
    result["de_J20_2"] = de_J20_2
    result["de_J20_3"] = de_J20_3
    result["de_J11_1"] = de_J11_1
    result["de_J02_1"] = de_J02_1
    result["de_K20_2"] = de_K20_2
    result["de_K20_3"] = de_K20_3
    result["de_K11_1"] = de_K11_1
    result["de_K02_1"] = de_K02_1
    del FULL3c_ipvip1, FULL3c_ipip1, FULL3c_ip1ip2, FULL3c_ipip2

    # endregion 4

    # region 5. evaluation: ip2-only derivative
    #
    # Memory (units: #floats): the only 3c-2e derivative integral here is int3c2e_ip2, consumed
    # per aux batch; only the batch slice lives in memory in production.
    #   per-batch slice j3c_ip2_batch [3,nbatch_aux,nao,nao] ~ 3 * nbatch_aux * nao^2
    #   j3c_ip2_aux   [3,naux]                       ~ 3*naux
    #   j3c_ip2_occ   [3,naux,nocc,nocc]             ~ 3*naux*nocc^2
    #   j1ao_aux1_1   [natm,3,nao,nao]               ~ natm*nao^2
    #   k1bra_aux1_2  [natm,3,nocc,nao]              ~ natm*nocc*nao
    #   tmp_k1        [naux,nocc,nao]                ~ naux*nocc*nao
    #   dbas_J02_4/5/7, dbas_K02_4/5/7 [3,3,naux,naux] ~ 9*naux^2 each (6 of them)
    #   region-5 production peak ~ 3*nbatch_aux*nao^2 (slice) + 6*9*naux^2 (dbas, dominates)
    #     + 3*naux*nocc^2 (j3c_ip2_occ) + natm*nao^2 (j1ao_aux1_1)
    # After region 5, all remaining carried inputs (cderi, dm0, j2c_*, llcd_*, FULL3c_ip2) are
    # freed -- the result dict holds only the small [natm,...] / [natm,natm,3,3] outputs.

    # --- shared ip2 intermediate --- #
    # j3c_ip2_aux[t, P] (J; AO-contracted):
    #   j3c_ip2_aux = np.einsum("tPvu, vu -> tP", FULL3c_ip2, dm0)
    # j3c_ip2_occ[t, P, i, j] (K; occ-contracted):
    #   j3c_ip2_occ = np.einsum("tPvu, vj, ui -> tPij", FULL3c_ip2, mocc_2, mocc_2)
    j3c_ip2_aux = np.zeros((3, naux))
    j3c_ip2_occ = np.zeros((3, naux, nocc, nocc))
    j1ao_aux1_1 = np.zeros((natm, 3, nao, nao))
    # k1bra_aux1_2 needs the *full* int3c2e_ip2 (tPvu), so it is fused into the same batched
    # generator pass as j1ao_aux1_1 to avoid a second ip2 evaluation. Precompute the
    # P-shared factor A[P, i, l] once:
    #   tmp_k1 = np.einsum("Pij, lj -> Pil", llcd_eri_occ, mocc_2)   (l = u AO index)
    k1bra_aux1_2 = np.zeros((natm, 3, nocc, nao))
    tmp_k1 = llcd_eri_occ @ mocc_2.T  # [naux, nocc, nao]  (l = u AO index)
    for _sh0, _sh1, p0, p1 in aux_ranges:
        j3c_ip2_batch = FULL3c_ip2[:, p0:p1]  # use generator in real application
        # j3c_ip2_aux[:, p0:p1] = np.einsum("tPvu, vu -> tP", j3c_ip2_batch, dm0)
        j3c_ip2_aux[:, p0:p1] = (j3c_ip2_batch * dm0).sum(axis=(-1, -2))
        for t in range(3):
            # j3c_ip2_occ[t, p0:p1] = np.einsum("Pvu, vj, ui -> Pij", j3c_ip2_batch[t], mocc_2, mocc_2)
            j3c_ip2_occ[t, p0:p1] = mocc_2.T @ j3c_ip2_batch[t] @ mocc_2  # [P, i, j]

        # --- j1ao_aux1_1 / k1bra_aux1_2 --- #
        # Both consume the full j3c_ip2_batch on atom A's aux slice, so share the overlap slice.
        #   j1ao_aux1_1[A] = - np.einsum("tPvu, P -> tvu", j3c_ip2_batch[:, slc_batch], llcd_eri_aux[slc_full])
        #   k1bra_aux1_2[A, t] = - np.einsum("Pil, Pkl, i -> ik", tmp_k1[slc_full], j3c_ip2_batch[t, slc_batch], occ_invsqrt)
        #     (k=v, l=u AO indices)
        for A in range(natm):
            _, _, p0A, p1A = auxslices[A]
            start = max(p0, p0A)
            end = min(p1, p1A)
            if start >= end:
                continue
            slc_batch = slice(start - p0, end - p0)
            slc_full = slice(start, end)
            j1ao_aux1_1[A] += -(j3c_ip2_batch[:, slc_batch] * llcd_eri_aux[slc_full, None, None]).sum(axis=1)
            # k1bra_aux1_2: contract (P, l) of tmp_k1[slc_full] and j3c_ip2_batch[t, slc_batch]
            # via batched matmul over Ps (tmp_k1 stays contiguous; j3c_ip2_batch[t, slc_batch]
            # used as-is -- int3c2e_ip2 is symmetric in its AO pair (k,l), so no swapaxes needed).
            for t in range(3):
                # np.einsum("Pil, Pkl -> ik", tmp_k1[slc_full], j3c_ip2_batch[t, slc_batch]) == (tmp_k1[slc_full] @ j3c_ip2_batch[t, slc_batch]).sum(axis=0)
                k1bra_aux1_2[A, t] += (
                    -(tmp_k1[slc_full] @ j3c_ip2_batch[t, slc_batch]).sum(axis=0) * occ_invsqrt[:, None]
                )
    result["j1ao_aux1_1"] = j1ao_aux1_1
    result["k1bra_aux1_2"] = k1bra_aux1_2

    # --- J02-4 --- #
    # tmp1 = np.einsum("sRQ, R -> sQ", j2c_ip1, llcd_eri_aux)
    tmp1 = -(j2c_ip1 * llcd_eri_aux).sum(axis=-1)
    # dbas_J02_4 = np.einsum("tP, PQ, sQ -> tsPQ", j3c_ip2_aux, j2c_inv, tmp1)
    dbas_J02_4 = j3c_ip2_aux[:, None, :, None] * tmp1[None, :, None, :] * j2c_inv
    dbas_J02_4 *= -1

    # --- J02-5 --- #
    # dbas_J02_5 = np.einsum("tP, PQ, sQ -> tsPQ", j3c_ip2_aux, j2c_inv, j3c_ip2_aux)
    dbas_J02_5 = j3c_ip2_aux[:, None, :, None] * j3c_ip2_aux[None, :, None, :] * j2c_inv
    dbas_J02_5 *= 0.5

    # --- J02-7 --- #
    # dbas_J02_7 = np.einsum("tP, sPR, R -> tsPR", j3c_ip2_aux, llcd_j2c_ip1, llcd_eri_aux)
    tmp1 = llcd_j2c_ip1 * llcd_eri_aux[None, None, :]
    dbas_J02_7 = j3c_ip2_aux[:, None, :, None] * tmp1[None, :, :, :]
    dbas_J02_7 *= -1

    # --- K02-4 --- #
    # tmp1: contract j2c_ip1 with llcd_eri_occ. Production contracts the *contiguous* Q axis
    # (last axis of j2c_ip1) via matmul, exploiting j2c_ip1's antisymmetry in (R,Q); the labeled
    # reference einsum("sRQ, Rij -> sQij", ...) is sign-flipped relative to the matmul below, so
    # the matmul form is kept as the active implementation.
    # tmp1 = np.einsum("sRQ, Rij -> sQij", j2c_ip1, llcd_eri_occ)
    tmp1 = (j2c_ip1 @ llcd_eri_occ.reshape(naux, -1)).reshape(3, naux, nocc, nocc)  # [s, Q, i, j]
    # dbas_K02_4 = np.einsum("tPij, sQij -> tsPQ", j3c_ip2_occ, tmp1) * j2c_inv
    # contract the shared (i, j) pair; loop over s-component, each a [t, P, ij] @ [ij, Q] matmul.
    dbas_K02_4 = np.empty((3, 3, naux, naux))
    j3c_ip2_occ_2d = j3c_ip2_occ.reshape(3, naux, nocc * nocc)  # [t, P, ij]
    for s in range(3):
        tmp1_s = tmp1[s].reshape(naux, nocc * nocc).T  # [ij, Q]
        dbas_K02_4[:, s] = (j3c_ip2_occ_2d @ tmp1_s) * j2c_inv  # [t, P, Q]
    dbas_K02_4 *= 1

    # --- K02-5 --- #
    # dbas_K02_5 = np.einsum("tPij, sQij -> tsPQ", j3c_ip2_occ, j3c_ip2_occ) * j2c_inv
    # contract the shared (i, j) pair of two copies of j3c_ip2_occ; loop over s-component.
    dbas_K02_5 = np.empty((3, 3, naux, naux))
    for s in range(3):
        tmp1_s = j3c_ip2_occ[s].reshape(naux, nocc * nocc).T  # [ij, Q]
        dbas_K02_5[:, s] = (j3c_ip2_occ_2d @ tmp1_s) * j2c_inv  # [t, P, Q]
    dbas_K02_5 *= 0.5

    # --- K02-7 --- #
    # tmp1 = np.einsum("tPij, Rij -> tPR", j3c_ip2_occ, llcd_eri_occ)
    # dbas_K02_7 = np.einsum("tPR, sPR -> tsPR", tmp1, llcd_j2c_ip1)
    llcd_eri_occ_2d = llcd_eri_occ.reshape(naux, nocc * nocc).T  # [ij, R]
    tmp1 = np.empty((3, naux, naux))  # [t, P, R]
    for t in range(3):
        # tmp1[t] = np.einsum("Pij, Rij -> PR", j3c_ip2_occ[t], llcd_eri_occ)
        tmp1[t] = j3c_ip2_occ_2d[t] @ llcd_eri_occ_2d  # [P, ij] @ [ij, R] -> [P, R]
    dbas_K02_7 = tmp1[:, None, :, :] * llcd_j2c_ip1[None, :, :, :]  # [t, s, P, R]
    dbas_K02_7 *= -1

    # --- skeleton j2/k2 (ip2-only) sum --- #
    de_J02_4 = np.zeros((natm, natm, 3, 3))
    de_J02_5 = np.zeros((natm, natm, 3, 3))
    de_J02_7 = np.zeros((natm, natm, 3, 3))
    de_K02_4 = np.zeros((natm, natm, 3, 3))
    de_K02_5 = np.zeros((natm, natm, 3, 3))
    de_K02_7 = np.zeros((natm, natm, 3, 3))
    for A, (_, _, p0A, p1A) in enumerate(auxslices):
        for B, (_, _, p0B, p1B) in enumerate(auxslices):
            de_J02_4[A, B] = dbas_J02_4[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_J02_5[A, B] = dbas_J02_5[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_J02_7[A, B] = dbas_J02_7[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_K02_4[A, B] = dbas_K02_4[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_K02_5[A, B] = dbas_K02_5[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
            de_K02_7[A, B] = dbas_K02_7[..., p0A:p1A, p0B:p1B].sum(axis=(-1, -2))
    de_J02_4 += de_J02_4.transpose(1, 0, 3, 2)
    de_J02_5 += de_J02_5.transpose(1, 0, 3, 2)
    de_J02_7 += de_J02_7.transpose(1, 0, 3, 2)
    de_K02_4 += de_K02_4.transpose(1, 0, 3, 2)
    de_K02_5 += de_K02_5.transpose(1, 0, 3, 2)
    de_K02_7 += de_K02_7.transpose(1, 0, 3, 2)
    result["de_J02_4"] = de_J02_4
    result["de_J02_5"] = de_J02_5
    result["de_J02_7"] = de_J02_7
    result["de_K02_4"] = de_K02_4
    result["de_K02_5"] = de_K02_5
    result["de_K02_7"] = de_K02_7
    del dbas_J02_4, dbas_J02_5, dbas_J02_7, dbas_K02_4, dbas_K02_5, dbas_K02_7

    # --- j1ao aux1-2 --- #
    # Scatter j3c_ip2_aux into a per-atom buffer (derivative on aux center A), solve by j2c
    # (right, flip), then contract with the triangular-packed cderi (same pattern as
    # j1ao_aux1_3/4). The final contraction, if cderi were unpacked to cderi_ao[P,u,v], is:
    #   j1ao_aux1_2 = - np.einsum("Puv, AtP -> Atuv", cderi_ao, tmp2)
    # production uses the packed cderi + lib.unpack_tril to avoid the full [naux,nao,nao].
    tmp1 = np.zeros((natm, 3, naux))
    for A in range(natm):
        _, _, p0, p1 = auxslices[A]
        slcA = slice(p0, p1)
        tmp1[A, :, slcA] = j3c_ip2_aux[:, slcA]
    tmp2 = solve_by_j2c(tmp1, left=False, flip=True)
    j1ao_aux1_2_tp = tmp2.reshape(natm * 3, naux) @ cderi
    j1ao_aux1_2 = -lib.unpack_tril(j1ao_aux1_2_tp).reshape(natm, 3, nao, nao)
    result["j1ao_aux1_2"] = j1ao_aux1_2

    # --- k1bra aux1-1 --- #
    # k1bra_aux1_1[A] = - np.einsum("tPij, Pjk, i -> tik", j3c_ip2_occ[:, slcA], llcd_eri_bra[slcA], occ_invsqrt)
    # contract (P, j) via batched matmul over Ps, loop t.
    k1bra_aux1_1 = np.zeros((natm, 3, nocc, nao))
    for A in range(natm):
        _, _, p0, p1 = auxslices[A]
        slcA = slice(p0, p1)
        for t in range(3):
            # j3c_ip2_occ[t, slcA]: [Ps, i, j]; llcd_eri_bra[slcA]: [Ps, j, k];
            # batched @ -> [Ps, i, k], sum over Ps.
            k1bra_aux1_1[A, t] = -(j3c_ip2_occ[t, slcA] @ llcd_eri_bra[slcA]).sum(axis=0)
    k1bra_aux1_1 *= occ_invsqrt[None, None, :, None]
    result["k1bra_aux1_1"] = k1bra_aux1_1

    # endregion 5

    # region 6. evaluation: ip1

    # this term is K only
    # fold_j3c_bra = np.einsum("Pji, ui -> Pju", llcd_eri_occ, mocc_2)
    fold_j3c_bra = llcd_eri_occ @ mocc_2.T  # in production code, use reshape

    j3c_ip1_aux = np.zeros((3, naux, nao))
    j3c_ip1_bra = np.zeros((3, naux, nocc, nao))
    j3c_ip1_j1ao_tmp1 = np.zeros((3, nao, nao))
    j3c_ip1_k1ao_tmp1 = np.zeros((3, nao, nao))
    k1bra_aux0_4 = np.zeros((natm, 3, nocc, nao))
    # j3c_ip1_j1ao_tmp = np.einsum("tPvu, P -> tvu", FULL3c_ip1, llcd_eri_aux)
    # j3c_ip1_j1ao_tmp = np.einsum("tPiu, Pik -> tuk", j3c_ip1_bra, llcd_eri_bra)
    for _sh0, _sh1, p0, p1 in aux_ranges:
        j3c_ip1_batch = FULL3c_ip1[:, p0:p1]
        # j3c_ip1_aux[:, p0:p1] = np.einsum("tPvu, vu -> tPu", j3c_ip1_batch, dm0)
        j3c_ip1_aux[:, p0:p1] = (j3c_ip1_batch * dm0).sum(axis=-2)
        # j3c_ip1_bra[:, p0:p1] = np.einsum("tPvu, vj -> tPju", j3c_ip1_batch, mocc_2)
        j3c_ip1_bra[:, p0:p1] = mocc_2.T @ j3c_ip1_batch
        j3c_ip1_j1ao_tmp1 += (FULL3c_ip1[:, p0:p1] * llcd_eri_aux[p0:p1, None, None]).sum(axis=-3)
        for t in range(3):
            j3c_ip1_k1ao_tmp1[t] += j3c_ip1_bra[t, p0:p1].reshape(-1, nao).T @ llcd_eri_bra[p0:p1].reshape(-1, nao)
        for A in range(natm):
            _, _, p0A, p1A = aoslices[A]
            slcA = slice(p0A, p1A)
            tmp1 = fold_j3c_bra[p0:p1, :, slcA].swapaxes(-1, -2).reshape(-1, nocc)
            for t in range(3):
                tmp2 = j3c_ip1_batch[t, :, :, slcA].swapaxes(-1, -2).reshape(-1, nao)
                k1bra_aux0_4[A, t] -= tmp1.T @ tmp2
    k1bra_aux0_4 *= occ_invsqrt[None, None, :, None]
    result["k1bra_aux0_4"] = k1bra_aux0_4

    lcd_j3c_ip1_aux = np.zeros((3, naux, nao))
    for t in range(3):
        lcd_j3c_ip1_aux[t] = solve_by_j2c(j3c_ip1_aux[t], left=True, flip=False)
    del j3c_ip1_aux

    # --- j1ao aux0 --- #

    j1ao_aux0 = np.zeros([natm, 3, nao, nao])
    for A in range(mol.natm):
        sh0, sh1, p0, p1 = aoslices[A]
        slcA = slice(p0, p1)
        # j1ao aux0-1/2
        j1ao_aux0[A, :, slcA, :] -= j3c_ip1_j1ao_tmp1[:, :, slcA].swapaxes(-1, -2)
        j1ao_aux0[A, :, :, slcA] -= j3c_ip1_j1ao_tmp1[:, :, slcA]
        # j1ao aux0-3/4
        tmp1 = lcd_j3c_ip1_aux[:, :, slcA].sum(axis=-1)
        tmp2 = np.einsum("tP, PU -> tU", tmp1, cderi)
        j1ao_aux0[A] -= 2 * lib.unpack_tril(tmp2)
    result["j1ao_aux0"] = j1ao_aux0

    # --- k1ao aux0 --- #

    k1ao_aux0_1 = np.zeros([natm, 3, nao, nao])
    k1ao_aux0_2 = np.zeros([natm, 3, nao, nao])
    for A in range(mol.natm):
        sh0, sh1, p0, p1 = aoslices[A]
        slcA = slice(p0, p1)
        k1ao_aux0_1[A, :, slcA, :] -= j3c_ip1_k1ao_tmp1[:, slcA, :]
        k1ao_aux0_2[A, :, :, slcA] -= j3c_ip1_k1ao_tmp1[:, slcA, :].swapaxes(-1, -2)

    result["k1ao_aux0_1"] = k1ao_aux0_1
    result["k1ao_aux0_2"] = k1ao_aux0_2

    k1bra_aux0_1 = mocc.T @ k1ao_aux0_1
    k1bra_aux0_2 = mocc.T @ k1ao_aux0_2
    result["k1bra_aux0_1"] = k1bra_aux0_1
    result["k1bra_aux0_2"] = k1bra_aux0_2

    k1bra_aux0_3 = np.zeros([natm, 3, nocc, nao])
    # k1bra_aux0_4 = np.zeros([natm, 3, nocc, nao])
    for A in range(mol.natm):
        sh0, sh1, p0, p1 = aoslices[A]
        slcA = slice(p0, p1)
        # k1bra_aux0_3[A] -= np.einsum("tPil, Pju, lj, i -> tiu", j3c_ip1_bra[..., slcA], llcd_eri_bra, mocc_2[slcA], occ_invsqrt)
        for t in range(3):
            tmp1 = mocc_2[slcA].T @ j3c_ip1_bra[t, ..., slcA].swapaxes(-1, -2)  # [P, j, i]
            tmp2 = tmp1.reshape(-1, nocc).T @ llcd_eri_bra.reshape(-1, nao)  # [i, u]
            k1bra_aux0_3[A, t] -= tmp2 * occ_invsqrt[:, None]
        # k1bra_aux0_4[A] -= np.einsum("tPkl, Pil, i -> tik", FULL3c_ip1[..., slcA], fold_j3c_bra[..., slcA], occ_invsqrt)
    result["k1bra_aux0_3"] = k1bra_aux0_3
    # result["k1bra_aux0_4"] = k1bra_aux0_4

    lcd_j3c_ip1_bra = np.zeros((3, naux, nocc, nao))
    for t in range(3):
        lcd_j3c_ip1_bra[t] = solve_by_j2c(j3c_ip1_bra[t], left=True, flip=False)
    del j3c_ip1_bra

    # --- J20-1 --- #
    # dbas_J20_1 = np.einsum("tPu, sPv -> tsuv", lcd_j3c_ip1_aux, lcd_j3c_ip1_aux)
    dbas_J20_1 = np.zeros((3, 3, nao, nao))
    for t in range(3):
        for s in range(3):
            dbas_J20_1[t, s] = lcd_j3c_ip1_aux[s].T @ lcd_j3c_ip1_aux[t]
    dbas_J20_1 *= 2

    # --- J11 preparation --- #

    llcd_j3c_ip1_aux = np.zeros((3, naux, nao))
    for t in range(3):
        llcd_j3c_ip1_aux[t] = solve_by_j2c(lcd_j3c_ip1_aux[t], left=True, flip=True)
    del lcd_j3c_ip1_aux

    # --- J11-2 --- #
    # dbas_J11_2 = np.einsum("tPu, sPR, R -> tsRu", llcd_j3c_ip1_aux, j2c_ip1, llcd_eri_aux)
    dbas_J11_2 = np.zeros([3, 3, naux, nao])
    for t in range(3):
        for s in range(3):
            dbas_J11_2[t, s] = (j2c_ip1[s] * llcd_eri_aux).T @ llcd_j3c_ip1_aux[t]
    dbas_J11_2 *= -2

    # --- J11-3 --- #
    # dbas_J11_3 = np.einsum("tQu, sQR, R -> tsQu", llcd_j3c_ip1_aux, j2c_ip1, llcd_eri_aux)
    tmp1 = (j2c_ip1 * llcd_eri_aux).sum(axis=-1)
    dbas_J11_3 = llcd_j3c_ip1_aux[:, None, :, :] * tmp1[None, :, :, None]
    dbas_J11_3 *= 2

    # --- J11-4 --- #
    # dbas_J11_4 = np.einsum("tQu, sQ -> tsQu", llcd_j3c_ip1_aux, j3c_ip2_aux)
    dbas_J11_4 = llcd_j3c_ip1_aux[:, None, :, :] * j3c_ip2_aux[None, :, :, None]
    dbas_J11_4 *= 2

    # --- K20-1a --- #
    # dbas_K20_1a = np.zeros((3, 3, nao, nao))
    # dbas_K20_1a = np.einsum("tPju, sPjv, ui, vi -> tsvu", lcd_j3c_ip1_bra, lcd_j3c_ip1_bra, mocc_2, mocc_2)
    dbas_K20_1a = np.zeros((3, 3, nao, nao))
    lcd_j3c_ip1_bra_view = lcd_j3c_ip1_bra.reshape(3, naux * nocc, nao)
    for t in range(3):
        for s in range(3):
            dbas_K20_1a[t, s] = (lcd_j3c_ip1_bra_view[s].T @ lcd_j3c_ip1_bra_view[t]) * dm0

    # --- K20-1b --- #
    # dbas_K20_1b = np.einsum("tPju, sPiv, ui, vj -> tsvu", lcd_j3c_ip1_bra, lcd_j3c_ip1_bra, mocc_2, mocc_2)
    dbas_K20_1b = np.zeros((3, 3, nao, nao))
    for p in range(naux):  # PAR-ITER, use buffer for reduction
        tmp1 = mocc_2 @ lcd_j3c_ip1_bra[:, p]  # [t, v, u]
        dbas_K20_1b += tmp1[:, None, :, :] * tmp1[None, :, :, :].swapaxes(-1, -2)

    # --- K11 preparation --- #

    llcd_j3c_ip1_bra = np.zeros((3, naux, nocc, nao))
    for t in range(3):
        llcd_j3c_ip1_bra[t] = solve_by_j2c(lcd_j3c_ip1_bra[t], left=True, flip=True)
    del lcd_j3c_ip1_bra

    # --- K11-2 --- #
    # dbas_K11_2 = np.einsum("tPju, sRP, Rju -> tsRu", llcd_j3c_ip1_bra, j2c_ip1, fold_j3c_bra)
    dbas_K11_2 = np.zeros((3, 3, naux, nao))
    for t in range(3):
        for s in range(3):
            tmp1 = (j2c_ip1[s] @ llcd_j3c_ip1_bra[t].reshape(naux, nocc * nao)).reshape(naux, nocc, nao)
            dbas_K11_2[t, s] = (fold_j3c_bra * tmp1).sum(axis=-2)
    dbas_K11_2 *= 2

    # --- K11-3 --- #
    # dbas_K11_3 = np.einsum("tQju, sQR, Rju -> tsQu", llcd_j3c_ip1_bra, j2c_ip1, fold_j3c_bra)
    dbas_K11_3 = np.zeros((3, 3, naux, nao))
    for s in range(3):
        tmp1 = (j2c_ip1[s] @ fold_j3c_bra.reshape(naux, nocc * nao)).reshape(naux, nocc, nao)
        for t in range(3):
            dbas_K11_3[t, s] = (llcd_j3c_ip1_bra[t] * tmp1).sum(axis=-2)
    dbas_K11_3 *= 2

    # --- K11-4 --- #
    # dbas_K11_4 = np.einsum("tPju, sPji, ui -> tsPu", llcd_j3c_ip1_bra, j3c_ip2_occ, mocc_2)
    dbas_K11_4 = np.zeros((3, 3, naux, nao))
    for s in range(3):
        tmp1 = j3c_ip2_occ[s] @ mocc_2.T
        for t in range(3):
            dbas_K11_4[t, s] = (llcd_j3c_ip1_bra[t] * tmp1).sum(axis=-2)
    dbas_K11_4 *= 2

    de_J11_2 = np.zeros((natm, natm, 3, 3))
    de_J11_3 = np.zeros((natm, natm, 3, 3))
    de_J11_4 = np.zeros((natm, natm, 3, 3))
    de_K11_2 = np.zeros((natm, natm, 3, 3))
    de_K11_3 = np.zeros((natm, natm, 3, 3))
    de_K11_4 = np.zeros((natm, natm, 3, 3))
    for B, (_, _, p0B, p1B) in enumerate(auxslices):
        for A, (_, _, p0A, p1A) in enumerate(aoslices):
            de_J11_2[A, B] = dbas_J11_2[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
            de_J11_3[A, B] = dbas_J11_3[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
            de_J11_4[A, B] = dbas_J11_4[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
            de_K11_2[A, B] = dbas_K11_2[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
            de_K11_3[A, B] = dbas_K11_3[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
            de_K11_4[A, B] = dbas_K11_4[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
    de_J11_2 += de_J11_2.transpose(1, 0, 3, 2)
    de_J11_3 += de_J11_3.transpose(1, 0, 3, 2)
    de_J11_4 += de_J11_4.transpose(1, 0, 3, 2)
    de_K11_2 += de_K11_2.transpose(1, 0, 3, 2)
    de_K11_3 += de_K11_3.transpose(1, 0, 3, 2)
    de_K11_4 += de_K11_4.transpose(1, 0, 3, 2)
    result["de_J11_2"] = de_J11_2
    result["de_J11_3"] = de_J11_3
    result["de_J11_4"] = de_J11_4
    result["de_K11_2"] = de_K11_2
    result["de_K11_3"] = de_K11_3
    result["de_K11_4"] = de_K11_4

    de_J20_1 = np.zeros((natm, natm, 3, 3))
    de_K20_1a = np.zeros((natm, natm, 3, 3))
    de_K20_1b = np.zeros((natm, natm, 3, 3))
    for B, (_, _, p0B, p1B) in enumerate(aoslices):
        for A, (_, _, p0A, p1A) in enumerate(aoslices):
            de_J20_1[A, B] = dbas_J20_1[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
            de_K20_1a[A, B] = dbas_K20_1a[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
            de_K20_1b[A, B] = dbas_K20_1b[..., p0B:p1B, p0A:p1A].sum(axis=(-1, -2))
    de_J20_1 += de_J20_1.transpose(1, 0, 3, 2)
    de_K20_1a += de_K20_1a.transpose(1, 0, 3, 2)
    de_K20_1b += de_K20_1b.transpose(1, 0, 3, 2)
    result["de_J20_1"] = de_J20_1
    result["de_K20_1a"] = de_K20_1a
    result["de_K20_1b"] = de_K20_1b

    # endregion 6

    # all remaining carried inputs are now consumed; free them before returning.
    del (
        cderi,
        dm0,
        j2c_inv,
        j2c_l_inv,
        j2c_ip1,
        llcd_j2c_ip1,
        llcd_eri_aux,
        llcd_eri_occ,
        llcd_eri_bra,
        FULL3c_ip2,
        j3c_ip2_aux,
        j3c_ip2_occ,
        tmp_k1,
    )

    return result
