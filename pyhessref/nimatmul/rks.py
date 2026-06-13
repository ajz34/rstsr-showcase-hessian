from pyscf import gto, dft
import numpy as np
import time


# AO derivative component indices (deriv up to 3).
O = 0
X, Y, Z = 1, 2, 3
XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9
XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

IDX_AO_DERIV2 = [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
TRIPLE_SIGMA_DIAG = [
    [XXX, XXY, XXZ],  # xx
    [XXY, XYY, XYZ],  # xy
    [XXZ, XYZ, XZZ],  # xz
    [XYY, YYY, YYZ],  # yy
    [XYZ, YYZ, YZZ],  # yz
    [XZZ, YZZ, ZZZ],  # zz
]
# Triple derivatives organised by direction (X, Y, Z) for the diagonal MGGA tau term.
TRIPLE_TAU_DIAG = [
    ([XXX, XXY, XXZ, XYY, XYZ, XZZ], 0),
    ([XXY, XYY, XYZ, YYY, YYZ, YZZ], 1),
    ([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ], 2),
]

XC_NVAR = {"LDA": 1, "GGA": 4, "MGGA": 5}
XC_AO_DERIV = {"LDA": 2, "GGA": 3, "MGGA": 3}
XC_NCOMP_AO_DM0 = {"LDA": 1, "GGA": 4, "MGGA": 4}


def _eval_rho_vxc_fxc(xc, xc_type, ao, ao_dm0):
    """Compute rho (component-stacked), vxc, fxc for the requested xc_type.

    rho layout follows the eval_xc_eff convention:
        LDA  : (1, ngrid)            [rho]
        GGA  : (4, ngrid)            [rho, drho/dx, drho/dy, drho/dz]
        MGGA : (5, ngrid)            [rho, drho/dx, drho/dy, drho/dz, tau]
    Note: tau here is the LAPL-removed component (index 5 in eval_rho would be tau).
    """
    nvar = XC_NVAR[xc_type]
    ngrids = ao.shape[1]
    rho = np.zeros((nvar, ngrids))

    rho[O] = np.einsum("gu, gu -> g", ao[O], ao_dm0[O])
    if xc_type in ("GGA", "MGGA"):
        rho[X] = 2 * np.einsum("gu, gu -> g", ao[X], ao_dm0[O])
        rho[Y] = 2 * np.einsum("gu, gu -> g", ao[Y], ao_dm0[O])
        rho[Z] = 2 * np.einsum("gu, gu -> g", ao[Z], ao_dm0[O])
    if xc_type == "MGGA":
        rho[4] = 0.5 * (
            np.einsum("gu, gu -> g", ao[X], ao_dm0[X])
            + np.einsum("gu, gu -> g", ao[Y], ao_dm0[Y])
            + np.einsum("gu, gu -> g", ao[Z], ao_dm0[Z])
        )

    ni = dft.numint.NumInt()
    _, vxc, fxc, _ = ni.eval_xc_eff(xc, rho, deriv=2, xctype=xc_type)
    return rho, vxc, fxc


def _make_drho(xc_type, ao, ao_dm0, aoslices):
    """Skeleton derivative of rho components w.r.t. nuclear coordinates.

    Output shape (natm, 3, nvar, ngrids).  The bra-side basis sits on atom A,
    so the derivative acts only on bra basis indices in the slice [p0, p1).
    """
    ngrids = ao.shape[1]
    nvar = XC_NVAR[xc_type]
    natm = len(aoslices)
    drho = np.zeros((natm, 3, nvar, ngrids))

    # (rho_var, t_direction, cbra, cket) tuples that contribute to each rho component.
    # For symmetric components (RHO + grad), result is multiplied by 2 at the end.

    # RHO part — applies for all xc types.
    components = [
        (0, 0, X, O),
        (0, 1, Y, O),
        (0, 2, Z, O),
    ]

    if xc_type in ("GGA", "MGGA"):
        # SIGMA part: bra deriv2 ket val.
        components += [
            (1, 0, XX, O), (2, 0, XY, O), (3, 0, XZ, O),
            (1, 1, YX, O), (2, 1, YY, O), (3, 1, YZ, O),
            (1, 2, ZX, O), (2, 2, ZY, O), (3, 2, ZZ, O),
        ]
        # SIGMA part: bra deriv1 ket deriv1.
        components += [
            (1, 0, X, X), (2, 0, X, Y), (3, 0, X, Z),
            (1, 1, Y, X), (2, 1, Y, Y), (3, 1, Y, Z),
            (1, 2, Z, X), (2, 2, Z, Y), (3, 2, Z, Z),
        ]

    if xc_type == "MGGA":
        # TAU part: bra deriv2 ket deriv1.  tau index = 4.
        components += [
            (4, 0, XX, X), (4, 0, XY, Y), (4, 0, XZ, Z),
            (4, 1, YX, X), (4, 1, YY, Y), (4, 1, YZ, Z),
            (4, 2, ZX, X), (4, 2, ZY, Y), (4, 2, ZZ, Z),
        ]

    for A in range(natm):
        _, _, p0, p1 = aoslices[A]
        slc = slice(p0, p1)
        ao_slc = ao[:, :, slc]
        ao_dm0_slc = ao_dm0[:, :, slc]
        for v, t, cbra, cket in components:
            drho[A, t, v] -= np.einsum("gu, gu -> g", ao_slc[cbra], ao_dm0_slc[cket])

    # Symmetry: RHO + grad components carry a factor 2 (bra↔ket symmetry).
    # TAU (index 4) does NOT — it is built from the asymmetric (∇bra)·(∇ket) form.
    if xc_type in ("GGA", "MGGA"):
        drho[:, :, :4] *= 2
    elif xc_type == "LDA":
        drho[:, :, :1] *= 2
    return drho


def _de_fxc(weights, drho, fxc):
    """fxc contribution: $w_g (\\partial drho)_{A,t,x} f^{xy}_g (\\partial drho)_{B,s,y}$."""
    return np.einsum("g, Atxg, xyg, Bsyg -> ABts", weights, drho, fxc, drho, optimize=True)


def _de_vxc_diag(xc_type, ao, ao_dm0, wv, aoslices, natm, nao):
    """Same-atom block of the XC skeleton 2nd derivative.

    Builds the AO-resolved object dao_vxc_diag[6, nao] and contracts with the
    on-atom slice.  The 6 components are (xx, xy, xz, yy, yz, zz).
    """
    dao_vxc_diag = np.zeros((6, nao))

    # Contribution 1: ao[deriv2] · (sum over rho components of ao_dm0 weighted by wv).
    # LDA only uses the rho channel; GGA/MGGA also bring in the gradient channels.
    aow = np.einsum("gu, g -> gu", ao_dm0[0], wv[0])
    if xc_type in ("GGA", "MGGA"):
        for r in range(3):
            aow += np.einsum("gu, g -> gu", ao_dm0[1 + r], wv[1 + r])
    for idx_ts, its in enumerate([XX, XY, XZ, YY, YZ, ZZ]):
        dao_vxc_diag[idx_ts] += 2 * np.einsum("gu, gu -> u", ao[its], aow)

    if xc_type in ("GGA", "MGGA"):
        # Contribution 2 (GGA triple-derivative against rho-grad weights)
        for idx_ts, (i3x, i3y, i3z) in enumerate(TRIPLE_SIGMA_DIAG):
            aow = (
                np.einsum("gu, g -> gu", ao[i3x], wv[1])
                + np.einsum("gu, g -> gu", ao[i3y], wv[2])
                + np.einsum("gu, g -> gu", ao[i3z], wv[3])
            )
            dao_vxc_diag[idx_ts] += 2 * np.einsum("gu, gu -> u", aow, ao_dm0[0])

    if xc_type == "MGGA":
        # Contribution 3 (TAU triple-derivative)
        for trip_idx, r in TRIPLE_TAU_DIAG:
            aow = np.einsum("gu, g -> gu", ao_dm0[r + 1], wv[4])
            for idx_ts, i3 in enumerate(trip_idx):
                dao_vxc_diag[idx_ts] += np.einsum("gu, gu -> u", ao[i3], aow)

    de_vxc_diag = np.zeros((natm, natm, 6))
    for A in range(natm):
        _, _, p0A, p1A = aoslices[A]
        de_vxc_diag[A, A] = np.einsum("Au -> A", dao_vxc_diag[:, p0A:p1A])
    de_vxc_diag = de_vxc_diag[:, :, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(natm, natm, 3, 3)
    return de_vxc_diag


def _de_vxc_off(xc_type, ao, dm0, wv, aoslices, natm, nao):
    """Two-atom (off-diagonal) block of the XC skeleton 2nd derivative.

    Builds the dense [3, 3, nao, nao] object and contracts it against dm0 over
    the (B, A) AO slices.  We finally symmetrise by adding the [s, t] copy with
    AO indices transposed.
    """
    dao_vxc_off = np.zeros((3, 3, nao, nao))

    if xc_type == "LDA":
        # LDA contribution: ipip[t, s] += (0.5 * wv[0] * ao[s+1])^T @ ao[t+1]
        for t in range(3):
            aowv = 0.5 * np.einsum("gu, g -> gu", ao[t + 1], wv[0])
            for s in range(3):
                dao_vxc_off[t, s] += 2 * ao[s + 1].T @ aowv

    if xc_type in ("GGA", "MGGA"):
        # GGA contribution: ipip[t, s] += sum_r (wv[r+1] * ao[GGA_HESS_AO[t][r]] + 0.5 wv[0] ao[t+1])^T @ ao[s+1]
        for t in range(3):
            aowv = 0.5 * np.einsum("gu, g -> gu", ao[t + 1], wv[0])
            for r in range(3):
                aowv += np.einsum("gu, g -> gu", ao[IDX_AO_DERIV2[t][r]], wv[r + 1])
            for s in range(3):
                dao_vxc_off[t, s] += 2 * ao[s + 1].T @ aowv

    if xc_type == "MGGA":
        # TAU contribution: built lower-triangular then symmetrised.
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
        dao_vxc_off += dao_vxc_tau

    # [t, s] -> [t, s] + [s, t] with AO indices transposed.
    dao_vxc_off += dao_vxc_off.transpose(1, 0, 3, 2)

    de_vxc_off = np.zeros((natm, natm, 3, 3))
    for A in range(natm):
        _, _, p0A, p1A = aoslices[A]
        for B in range(A + 1):
            _, _, p0B, p1B = aoslices[B]
            de_vxc_off[A, B] = np.einsum(
                "tsuv, uv -> ts",
                dao_vxc_off[:, :, p0B:p1B, p0A:p1A],
                dm0[p0B:p1B, p0A:p1A],
            )
            if A != B:
                de_vxc_off[B, A] = de_vxc_off[A, B].T
    return de_vxc_off


def _vmat_ip(xc_type, ao, wv, nao):
    """Gradient-level Vxc matrix (ipip term shared by every atom).

    Returns vmat_ip[3, nao, nao].  This is the AO part that the per-atom
    vmat_deriv1 reuses on its on-atom slice.
    """
    vmat_ip = np.zeros((3, nao, nao))

    if xc_type == "LDA":
        aow = 0.5 * np.einsum("g, gu -> gu", wv[0], ao[O])
        for t in range(3):
            vmat_ip[t] += ao[t + 1].T @ aow
        # Add the bra-derivative-on-A copy:  (0.5 * wv[0] * ao[t+1])^T @ ao[0]
        for t in range(3):
            aow = 0.5 * wv[0, :, None] * ao[t + 1]
            vmat_ip[t] += aow.T @ ao[O]
        return vmat_ip

    # GGA + MGGA share the same SIGMA structure
    aow = 0.5 * np.einsum("g, gu -> gu", wv[0], ao[O])
    for r in range(3):
        aow += np.einsum("g, gu -> gu", wv[1 + r], ao[1 + r])
    for t in range(3):
        vmat_ip[t] += ao[t + 1].T @ aow

    aow_d = np.array([0.5 * wv[0, :, None] * ao[t + 1] for t in range(3)])
    aow_d[0] += wv[1, :, None] * ao[XX] + wv[2, :, None] * ao[XY] + wv[3, :, None] * ao[XZ]
    aow_d[1] += wv[1, :, None] * ao[YX] + wv[2, :, None] * ao[YY] + wv[3, :, None] * ao[YZ]
    aow_d[2] += wv[1, :, None] * ao[ZX] + wv[2, :, None] * ao[ZY] + wv[3, :, None] * ao[ZZ]
    for t in range(3):
        vmat_ip[t] += aow_d[t].T @ ao[O]

    if xc_type == "MGGA":
        for r in range(3):
            aow = 0.5 * wv[4, :, None] * ao[1 + r]
            for t in range(3):
                vmat_ip[t] += ao[IDX_AO_DERIV2[t][r]].T @ aow

    return vmat_ip


def _vmat_deriv1(xc_type, ao, drho, wf, vmat_ip, aoslices, natm, nao):
    """Per-atom Fock skeleton derivative (vxc_deriv1).

    For each atom A, builds the type-dependent fxc part from drho[A], then
    folds in the (atom A only) ipip slice and antisymmetrises in AO indices.
    """
    vmat_deriv1 = np.zeros((natm, 3, nao, nao))

    for A in range(natm):
        _, _, p0, p1 = aoslices[A]

        if xc_type == "LDA":
            # wv_f[t, g] = wf[g] * drho[A, t, 0, g] / 2  (drho already has the *2)
            wv_f = np.einsum("g, tg -> tg", wf[0, 0], drho[A, :, 0]) * 0.5
            for t in range(3):
                aow = wv_f[t][:, None] * ao[O]
                vmat_deriv1[A, t] += aow.T @ ao[O]

        if xc_type in ("GGA", "MGGA"):
            wv_f = np.einsum("xyg, txg -> ytg", wf, drho[A])
            wv_f[0] *= 0.5
            if xc_type == "MGGA":
                wv_f[4] *= 0.25

            aow_f = np.einsum("ctg, cgm -> tgm", wv_f[:4], ao[:4])
            for t in range(3):
                vmat_deriv1[A, t] += aow_f[t].T @ ao[O]

        if xc_type == "MGGA":
            for j in range(1, 4):
                for t in range(3):
                    aow = wv_f[4, t][:, None] * ao[j]
                    vmat_deriv1[A, t] += aow.T @ ao[j]

        # ipip part lives only on atom A's bra rows; sign matches the existing test.
        vmat_deriv1[A, :, p0:p1, :] -= vmat_ip[:, p0:p1, :]

    # Antisymmetrise across AO indices (electron→nuclear coordinate convention).
    vmat_deriv1 += vmat_deriv1.swapaxes(-1, -2)
    return vmat_deriv1


def make_hessian_setup_batch(
    mol: gto.Mole,
    xc: str,
    coords: np.ndarray,
    weights: np.ndarray,
    dm0: np.ndarray,
    atm_list: list[int] = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    nao = mol.nao
    atm_list = atm_list if atm_list is not None else list(range(mol.natm))
    aoslices = mol.aoslice_by_atom()[atm_list]
    natm = len(atm_list)

    xc_type = dft.libxc.xc_type(xc)
    if xc_type not in XC_NVAR:
        raise NotImplementedError(f"xc_type={xc_type} not supported")

    def tic(label, t0):
        if verbose:
            print(f"Time for {label}: {time.time() - t0:.3f} s")

    t0 = time.time()
    ao = dft.numint.eval_ao(mol, coords, deriv=XC_AO_DERIV[xc_type])
    ao_dm0 = ao[: XC_NCOMP_AO_DM0[xc_type]] @ dm0
    _, vxc, fxc = _eval_rho_vxc_fxc(xc, xc_type, ao, ao_dm0)
    wv = weights * vxc
    wf = weights * fxc
    tic("ao, rho, vxc, fxc", t0)

    t0 = time.time()
    drho = _make_drho(xc_type, ao, ao_dm0, aoslices)
    de_fxc = _de_fxc(weights, drho, fxc)
    tic("drho, de_fxc", t0)

    t0 = time.time()
    de_vxc_diag = _de_vxc_diag(xc_type, ao, ao_dm0, wv, aoslices, natm, nao)
    tic("de_vxc_diag", t0)

    t0 = time.time()
    de_vxc_off = _de_vxc_off(xc_type, ao, dm0, wv, aoslices, natm, nao)
    tic("de_vxc_off", t0)

    t0 = time.time()
    vmat_ip = _vmat_ip(xc_type, ao, wv, nao)
    tic("vmat_ip", t0)

    t0 = time.time()
    vmat_deriv1 = _vmat_deriv1(xc_type, ao, drho, wf, vmat_ip, aoslices, natm, nao)
    tic("vmat_deriv1", t0)

    return {
        "de_vxc_diag": de_vxc_diag,
        "de_vxc_off": de_vxc_off,
        "de_fxc": de_fxc,
        "vmat_ip": vmat_ip,
        "vmat_deriv1": vmat_deriv1,
    }
