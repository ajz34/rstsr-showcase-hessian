from pyscf import gto, dft
import numpy as np
import time


def make_hessian_setup_batch(
    mol: gto.Mole,
    xc: str,
    coords: np.ndarray,
    weights: np.ndarray,
    dm0: np.ndarray,
    mocc: np.ndarray,
    mocc_2: np.ndarray,
    atm_list: list[int] = None,
) -> dict[np.ndarray]:
    # constants
    TX, TY, TZ = 0, 1, 2
    O = 0
    X, Y, Z = 1, 2, 3
    XX, XY, XZ = 4, 5, 6
    YX, YY, YZ = 5, 7, 8
    ZX, ZY, ZZ = 6, 8, 9
    XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
    YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

    # --- setup without much computations --- #

    # basic data
    ngrids = weights.size
    nao = mol.nao
    atm_list = atm_list if atm_list is not None else list(range(mol.natm))
    aoslices = mol.aoslice_by_atom()[atm_list]
    natm = len(atm_list)

    # xc data
    nvar_dict = {"LDA": 1, "GGA": 4, "MGGA": 5}
    xc_type = dft.libxc.xc_type(xc)
    nvar = nvar_dict[xc_type]

    # ao, ao_dm0
    ao_deriv_level_dict = {"LDA": 2, "GGA": 3, "MGGA": 3}
    ncomp_ao_dm0 = {"LDA": 1, "GGA": 4, "MGGA": 4}
    ao_deriv_level = ao_deriv_level_dict[xc_type]

    # --- standard dft: ao, ao_dm0, rho, vxc, fxc --- #

    t0 = time.time()

    ao = dft.numint.eval_ao(mol, coords, deriv=ao_deriv_level)
    ao_dm0 = ao[: ncomp_ao_dm0[xc_type]] @ dm0

    # rho, vxc, fxc
    rho = np.zeros([nvar, ngrids])
    if xc_type in ["LDA", "GGA", "MGGA"]:
        rho[O] = np.einsum("gu, gu -> g", ao[O], ao_dm0[O])
    if xc_type in ["GGA", "MGGA"]:
        rho[X] = 2 * np.einsum("gu, gu -> g", ao[X], ao_dm0[O])
        rho[Y] = 2 * np.einsum("gu, gu -> g", ao[Y], ao_dm0[O])
        rho[Z] = 2 * np.einsum("gu, gu -> g", ao[Z], ao_dm0[O])
    if xc_type in ["MGGA"]:
        rho[4] = 0.5 * (
            +np.einsum("gu, gu -> g", ao[X], ao_dm0[X])
            + np.einsum("gu, gu -> g", ao[Y], ao_dm0[Y])
            + np.einsum("gu, gu -> g", ao[Z], ao_dm0[Z])
        )
    # minor test
    # ref_rho = dft.numint.eval_rho(mol, ao, dm0, xctype="MGGA")[[0, 1, 2, 3, 5]]  # skip LAPL
    # assert np.allclose(rho, ref_rho)

    ni = dft.numint.NumInt()
    _, vxc, fxc, _ = ni.eval_xc_eff(xc, rho, deriv=2, xctype=xc_type)
    wv = weights * vxc

    t1 = time.time()
    print(f"Time for ao, rho, vxc, fxc: {t1 - t0:.3f} s")

    # --- drho --- #

    t0 = time.time()

    drho = np.zeros((natm, 3, 5, ngrids))
    for A in range(natm):
        _, _, p0, p1 = aoslices[A]
        slc = slice(p0, p1)
        ao_slc = ao[:, :, slc]
        ao_dm0_slc = ao_dm0[:, :, slc]
        # components
        DERIV_COMPONENTS = [
            # RHO part
            [(TX, 0), (X, O)],
            [(TY, 0), (Y, O)],
            [(TZ, 0), (Z, O)],
            # SIGMA part (bra deriv 2)
            [(TX, X), (XX, O)],
            [(TX, Y), (XY, O)],
            [(TX, Z), (XZ, O)],
            [(TY, X), (YX, O)],
            [(TY, Y), (YY, O)],
            [(TY, Z), (YZ, O)],
            [(TZ, X), (ZX, O)],
            [(TZ, Y), (ZY, O)],
            [(TZ, Z), (ZZ, O)],
            # SIGMA part (bra deriv 1, ket deriv 1)
            [(TX, X), (X, X)],
            [(TX, Y), (X, Y)],
            [(TX, Z), (X, Z)],
            [(TY, X), (Y, X)],
            [(TY, Y), (Y, Y)],
            [(TY, Z), (Y, Z)],
            [(TZ, X), (Z, X)],
            [(TZ, Y), (Z, Y)],
            [(TZ, Z), (Z, Z)],
            # TAU part
            [(TX, 4), (XX, X)],
            [(TX, 4), (XY, Y)],
            [(TX, 4), (XZ, Z)],
            [(TY, 4), (YX, X)],
            [(TY, 4), (YY, Y)],
            [(TY, 4), (YZ, Z)],
            [(TZ, 4), (ZX, X)],
            [(TZ, 4), (ZY, Y)],
            [(TZ, 4), (ZZ, Z)],
        ]

        for (t, v), (cbra, cket) in DERIV_COMPONENTS:
            drho[A, t, v] -= np.einsum("gu, gu -> g", ao_slc[cbra], ao_dm0_slc[cket])
    # scale symmetric coeff: RHO and SIGMA (0..3) get *2, TAU (4) does not
    drho[:, :, :4] *= 2

    de_fxc = np.einsum("g, Atxg, xyg, Bsyg -> ABts", weights, drho, fxc, drho, optimize=True)

    t1 = time.time()
    print(f"Time for drho: {t1 - t0:.3f} s")

    # --- skeleton deriv 2: de_vxc_diag --- #

    t0 = time.time()

    dao_vxc_diag = np.zeros((6, nao))  # 6 denotes xx, xy, xz, yy, yz, zz

    if xc_type == "LDA":
        raise NotImplementedError("LDA not implemented yet in this function")

    if xc_type in ["GGA", "MGGA"]:
        # contribution 1 (LDA/GGA double-derivative part)
        aowv = (
            np.einsum("gu, g -> gu", ao_dm0[0], wv[0])
            + np.einsum("gu, g -> gu", ao_dm0[1], wv[1])
            + np.einsum("gu, g -> gu", ao_dm0[2], wv[2])
            + np.einsum("gu, g -> gu", ao_dm0[3], wv[3])
        )
        for idx_ts, its in enumerate([XX, XY, XZ, YY, YZ, ZZ]):
            dao_vxc_diag[idx_ts] += 2 * np.einsum("gu, gu -> u", ao[its], aowv)

    if xc_type in ["GGA", "MGGA"]:
        # contribution 2 (GGA triple-derivative part)
        TRIPLE_SIGMA_DIAG = [
            [XXX, XXY, XXZ],  # xx
            [XXY, XYY, XYZ],  # xy
            [XXZ, XYZ, XZZ],  # xz
            [XYY, YYY, YYZ],  # yy
            [XYZ, YYZ, YZZ],  # yz
            [XZZ, YZZ, ZZZ],  # zz
        ]
        for idx_ts, (i3x, i3y, i3z) in enumerate(TRIPLE_SIGMA_DIAG):
            aowv = (
                np.einsum("gu, g -> gu", ao[i3x], wv[1])
                + np.einsum("gu, g -> gu", ao[i3y], wv[2])
                + np.einsum("gu, g -> gu", ao[i3z], wv[3])
            )
            dao_vxc_diag[idx_ts] += 2 * np.einsum("gu, gu -> u", aowv, ao_dm0[0])

    if xc_type == "MGGA":
        # contribution 3 (TAU part)
        TRIPLE_TAU_DIAG = [
            ([XXX, XXY, XXZ, XYY, XYZ, XZZ], 0),  # direction x: ao[triple]^T @ aow_tau_x
            ([XXY, XYY, XYZ, YYY, YYZ, YZZ], 1),  # direction y
            ([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ], 2),  # direction z
        ]
        for trip_idx, r in TRIPLE_TAU_DIAG:
            aowv = np.einsum("gu, g -> gu", ao_dm0[r + 1], wv[4])
            for idx_ts, i3 in enumerate(trip_idx):
                dao_vxc_diag[idx_ts] += np.einsum("gu, gu -> u", ao[i3], aowv)

    de_vxc_diag = np.zeros((natm, natm, 6))
    for A in range(natm):
        _, _, p0A, p1A = aoslices[A]
        slcA = slice(p0A, p1A)
        de_vxc_diag[A, A] += np.einsum("Au -> A", dao_vxc_diag[:, slcA])
    de_vxc_diag = de_vxc_diag[:, :, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(natm, natm, 3, 3)

    t1 = time.time()
    print(f"Time for de_vxc_diag: {t1 - t0:.3f} s")

    # --- skeleton deriv 2: de_vxc (off-diag) --- #

    t0 = time.time()

    dao_vxc = np.zeros((3, 3, nao, nao))

    if xc_type == "LDA":
        raise NotImplementedError("LDA not implemented yet in this function")

    if xc_type in ["GGA", "MGGA"]:
        # GGA part (RHO + SIGMA)
        GGA_CALLS = [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]

        for t in range(3):
            aowv = 0.5 * np.einsum("gu, g -> gu", ao[t + 1], wv[0])
            for r in range(3):
                aowv += np.einsum("gu, g -> gu", ao[GGA_CALLS[t][r]], wv[r + 1])
            for s in range(3):
                dao_vxc[t, s] += 2 * ao[s + 1].T @ aowv

    if xc_type == "MGGA":
        # TAU part
        aowv = [np.einsum("gu, g -> gu", ao[4 + i], wv[4]) for i in range(6)]

        TAU_CALLS = [
            ([0, 1, 2], [XX, XY, XZ]),
            ([1, 3, 4], [YX, YY, YZ]),
            ([2, 4, 5], [ZX, ZY, ZZ]),
        ]

        dao_vxc_tau = np.zeros((3, 3, nao, nao))

        for r_bra, r_ket in TAU_CALLS:
            for t in range(3):
                for s in range(t + 1):
                    dao_vxc_tau[t, s] += 0.5 * aowv[r_bra[s]].T @ ao[r_ket[t]]

        for t in range(3):
            for s in range(t):
                dao_vxc_tau[s, t] = dao_vxc_tau[t, s].T

        dao_vxc += dao_vxc_tau

    dao_vxc += dao_vxc.transpose(1, 0, 3, 2)  # [s,t] with AO indices transposed

    de_vxc = np.zeros((natm, natm, 3, 3))
    for A in range(natm):
        _, _, p0A, p1A = aoslices[A]
        slcA = slice(p0A, p1A)
        for B in range(A + 1):
            _, _, p0B, p1B = aoslices[B]
            slcB = slice(p0B, p1B)
            de_vxc[A, B] += np.einsum("tsuv, uv -> ts", dao_vxc[:, :, slcB, slcA], dm0[slcB, slcA])
            if A != B:
                de_vxc[B, A] = de_vxc[A, B].T

    t1 = time.time()
    print(f"Time for de_vxc: {t1 - t0:.3f} s")

    # --- vmat_ip --- #

    t0 = time.time()

    if xc_type == "LDA":
        raise NotImplementedError("LDA not implemented yet in this function")

    if xc_type in ["GGA", "MGGA"]:
        aowv = 0.5 * np.einsum("g, gu -> gu", wv[0], ao[0])
        for r in range(3):
            aowv += np.einsum("g, gu -> gu", wv[r + 1], ao[r + 1])

        vmat_ip = np.zeros((3, nao, nao))
        for t in range(3):
            vmat_ip[t] += ao[t + 1].T @ aowv

        aowv = np.array([0.5 * wv[0, :, None] * ao[d] for d in [X, Y, Z]])
        aowv[TX] += wv[1, :, None] * ao[XX] + wv[2, :, None] * ao[XY] + wv[3, :, None] * ao[XZ]
        aowv[TY] += wv[1, :, None] * ao[YX] + wv[2, :, None] * ao[YY] + wv[3, :, None] * ao[YZ]
        aowv[TZ] += wv[1, :, None] * ao[ZX] + wv[2, :, None] * ao[ZY] + wv[3, :, None] * ao[ZZ]
        for t in range(3):
            vmat_ip[t] += aowv[t].T @ ao[O]

    if xc_type == "MGGA":
        for r in range(3):
            aow_tau = 0.5 * wv[4, :, None] * ao[r + 1]  # [ngrids, nao]
            for t in range(3):
                vmat_ip[t] += ao[GGA_CALLS[t][r]].T @ aow_tau

    t1 = time.time()
    print(f"Time for vmat_ip: {t1 - t0:.3f} s")

    # --- vmat_deriv1 --- #

    t0 = time.time()

    wf = weights * fxc  # [5, 5, ngrids]

    vmat_deriv1 = np.zeros((natm, 3, nao, nao))

    for A in range(natm):
        if xc_type == "LDA":
            raise NotImplementedError("LDA not implemented yet in this function")

        if xc_type in ["GGA", "MGGA"]:
            wv_f = np.einsum("xyg, txg -> ytg", wf, drho[A])  # [5, 3, ngrids]
            wv_f[0] *= 0.5
            wv_f[4] *= 0.25

            aow_f = np.einsum("ctg, cgm -> tgm", wv_f[:4], ao[:4])  # [3, ngrids, nao]
            for t in range(3):
                vmat_deriv1[A, t] += aow_f[t].T @ ao[O]
        
        if xc_type == "MGGA":
            for j in range(1, 4):
                for t in range(3):
                    aow_tau = wv_f[4, t][:, None] * ao[j]
                    vmat_deriv1[A, t] += aow_tau.T @ ao[j]
        
        _, _, p0, p1 = aoslices[A]
        vmat_deriv1[A, :, p0:p1, :] -= vmat_ip[:, p0:p1, :]
    
    vmat_deriv1 += vmat_deriv1.swapaxes(-1, -2)

    t1 = time.time()
    print(f"Time for vmat_deriv1: {t1 - t0:.3f} s")

    return {
        "de_vxc_diag": de_vxc_diag,
        "de_vxc": de_vxc,
        "de_fxc": de_fxc,
        "vmat_ip": vmat_ip,
        "vmat_deriv1": vmat_deriv1,
    }
