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
    | 20-1        |   |   |
    | 20-2        | x | x |
    | 20-3        | x | x |
    | 11-1        | x | x |
    | 11-2        |   |   |
    | 11-3        |   |   |
    | 11-4        |   |   |
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
    | f1-aux1-1/2 | x |   |
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

    # endregion 3j2

    # region 3j1. evaluation: non cderi derivative (j1ao part)

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

    # endregion 3k2

    # region 3k1. evaluation: non cderi derivative (k1bra part)

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
        # tmp_k_ao[P,v,u] = einsum("Pij, vj, ui -> Pvu", llcd_eri_occ[P], mocc_2, mocc_2),
        # built per-batch (Pij -> Pvu expansion) to bound memory to batch * nao^2.
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

    # endregion 4

    # region 5. evaluation: ip2-only derivative

    # --- shared ip2 intermediate --- #
    # j3c_ip2_aux[t, P] = einsum("tPvu, vu -> tP", FULL3c_ip2, dm0)            (J; AO-contracted)
    # j3c_ip2_occ[t, P, i, j] = einsum("tPvu, vj, ui -> tPij", FULL3c_ip2, mocc_2, mocc_2)   (K; occ-contracted)
    j3c_ip2_aux = np.zeros((3, naux))
    j3c_ip2_occ = np.zeros((3, naux, nocc, nocc))
    for _sh0, _sh1, p0, p1 in aux_ranges:
        j3c_ip2_batch = FULL3c_ip2[:, p0:p1]  # use generator in real application
        j3c_ip2_aux[:, p0:p1] = (j3c_ip2_batch * dm0).sum(axis=(-1, -2))
        for t in range(3):
            # j3c_ip2_occ[t, p0:p1] = einsum("Pvu, vj, ui -> Pij", FULL3c_ip2[t, p0:p1], mocc_2, mocc_2)
            tmp1 = (j3c_ip2_batch[t] @ mocc_2).swapaxes(-1, -2)  # [P, i, v]
            j3c_ip2_occ[t, p0:p1] = tmp1 @ mocc_2                    # [P, i, j]

    # --- J02-4 --- #
    # tmp1[s, Q] = einsum("sRQ, R -> sQ", j2c_ip1, llcd_eri_aux)
    tmp1 = - (j2c_ip1 * llcd_eri_aux).sum(axis=-1)
    # dbas_J02_4 = einsum("tP, PQ, sQ -> tsPQ", ip2_aux, j2c_inv, tmp1)
    dbas_J02_4 = j3c_ip2_aux[:, None, :, None] * tmp1[None, :, None, :] * j2c_inv
    dbas_J02_4 *= -1

    # --- J02-5 --- #
    # dbas_J02_5 = einsum("tP, PQ, sQ -> tsPQ", j3c_ip2_aux, j2c_inv, j3c_ip2_aux)
    dbas_J02_5 = j3c_ip2_aux[:, None, :, None] * j3c_ip2_aux[None, :, None, :] * j2c_inv
    dbas_J02_5 *= 0.5

    # --- J02-7 --- #
    # dbas_J02_7 = einsum("tP, sPR, R -> tsPR", j3c_ip2_aux, llcd_j2c_ip1, llcd_eri_aux)
    tmp1 = llcd_j2c_ip1 * llcd_eri_aux[None, None, :]
    dbas_J02_7 = j3c_ip2_aux[:, None, :, None] * tmp1[None, :, :, :]
    dbas_J02_7 *= -1

    # --- K02-4 --- #
    # tmp1[s, Q, i, j] = einsum("sRQ, Rij -> Qij", j2c_ip1, llcd_eri_occ)
    tmp1 = (j2c_ip1 @ llcd_eri_occ.reshape(naux, -1)).reshape(3, naux, nocc, nocc)  # [s, Q, i, j]
    # dbas_K02_4 = einsum("tPij, sQij -> tsPQ", j3c_ip2_occ, tmp1) * j2c_inv
    dbas_K02_4 = np.empty((3, 3, naux, naux))
    j3c_ip2_occ_2d = j3c_ip2_occ.reshape(3, naux, nocc * nocc)  # [t, P, ij]
    for s in range(3):
        tmp1_s = tmp1[s].reshape(naux, nocc * nocc).T  # [ij, Q]
        dbas_K02_4[:, s] = (j3c_ip2_occ_2d @ tmp1_s) * j2c_inv  # [t, P, Q]
    dbas_K02_4 *= 1

    # --- K02-5 --- #
    # dbas_K02_5 = einsum("tPij, sQij -> tsPQ", j3c_ip2_occ, j3c_ip2_occ) * j2c_inv
    # contract the shared (i, j) pair of two copies of j3c_ip2_occ; loop over s-component.
    dbas_K02_5 = np.empty((3, 3, naux, naux))
    for s in range(3):
        tmp1_s = j3c_ip2_occ[s].reshape(naux, nocc * nocc).T  # [ij, Q]
        dbas_K02_5[:, s] = (j3c_ip2_occ_2d @ tmp1_s) * j2c_inv  # [t, P, Q]
    dbas_K02_5 *= 0.5

    # --- K02-7 --- #
    # tmp1[t, P, R] = einsum("tPij, Rij -> tPR", j3c_ip2_occ, llcd_eri_occ)
    # dbas_K02_7 = einsum("tPR, sPR -> tsPR", tmp1, llcd_j2c_ip1)
    llcd_eri_occ_2d = llcd_eri_occ.reshape(naux, nocc * nocc).T  # [ij, R]
    tmp1 = np.empty((3, naux, naux))  # [t, P, R]
    for t in range(3):
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

    # --- j1ao aux1-1/2 --- #
    # These are Fock 1st-derivative (not 2nd skeleton) contributions, but their only 3c-2e
    # derivative integral is int3c2e_ip2, so they share the ip2-only region. Both use the
    # aux-atom slice: the derivative is on the auxiliary center A (j1ao_aux1 is wrt aux atoms).
    # j1ao_aux1_1: (00|1)(0|00);  j1ao_aux1_2: (00|0)(1|00)  (solved_itm_j ~ llcd_eri_aux)

    # j1ao_aux1_1[A, t, v, u] = - einsum("tPvu, P -> tvu", j3c_ip2[:, slcA], llcd_eri_aux[slcA])
    j1ao_aux1_1 = np.zeros((natm, 3, nao, nao))
    for A in range(natm):
        _, _, p0, p1 = auxslices[A]
        slcA = slice(p0, p1)
        j1ao_aux1_1[A] = -(FULL3c_ip2[:, slcA] * llcd_eri_aux[slcA, None, None]).sum(axis=1)
    result["j1ao_aux1_1"] = j1ao_aux1_1

    # j1ao_aux1_2: tmp1[A, t, P in A] = j3c_ip2_aux[t, P]; solve_by_j2c (right, flip);
    #              then contract with triangular-packed cderi (same pattern as j1ao_aux1_3/4).
    tmp1 = np.zeros((natm, 3, naux))
    for A in range(natm):
        _, _, p0, p1 = auxslices[A]
        slcA = slice(p0, p1)
        tmp1[A, :, slcA] = j3c_ip2_aux[:, slcA]
    tmp2 = solve_by_j2c(tmp1, left=False, flip=True)
    j1ao_aux1_2_tp = tmp2.reshape(natm * 3, naux) @ cderi
    j1ao_aux1_2 = -lib.unpack_tril(j1ao_aux1_2_tp).reshape(natm, 3, nao, nao)
    result["j1ao_aux1_2"] = j1ao_aux1_2

    # endregion 5

    return result
