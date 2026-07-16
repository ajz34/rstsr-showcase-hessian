from pyscf import gto, dft
import numpy as np
import time

from pyhessref.hess_trait_unrestricted import UHessElecInteractAPI
from pyhessref.util import get_dm0_unrestricted


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


def _eval_rho_exc_vxc_fxc_uks(xc, xc_type, ao, ao_dm0a, ao_dm0b):
    """Evaluate the on-grid density together with the first/second functional
    derivatives ``vxc``/``fxc`` for the requested xc functional in the
    spin-polarized (UKS) case.

    Parameters
    ----------
    xc : str
        The xc functional name.
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and derivatives on the grid, shape ``[ncomp, ngrids, nao]``.
    ao_dm0a, ao_dm0b : np.ndarray
        Pre-contracted ``ao @ dm0`` for alpha/beta spins, shape
        ``[ncomp_dm0, ngrids, nao]``.

    Returns
    -------
    rhoa, rhob : np.ndarray
        On-grid density components per spin, shape ``[nvar, ngrids]``.
    exc : np.ndarray
        On-grid XC energy, shape ``[ngrids]``.
    vxc : np.ndarray
        First functional derivative, shape ``[2, nvar, ngrids]``.
    fxc : np.ndarray
        Second functional derivative, shape ``[2, nvar, 2, nvar, ngrids]``.
    """
    nvar = XC_NVAR[xc_type]
    ngrids = ao.shape[1]

    rhoa = np.zeros((nvar, ngrids))
    rhob = np.zeros((nvar, ngrids))

    for rho, ao_dm0 in [(rhoa, ao_dm0a), (rhob, ao_dm0b)]:
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
    exc, vxc, fxc, _ = ni.eval_xc_eff(xc, (rhoa, rhob), deriv=2, xctype=xc_type)
    return rhoa, rhob, exc, vxc, fxc


def _make_drho_uks(xc_type, ao, ao_dm0a, ao_dm0b, aoslices):
    """First-order skeleton derivative of rho components for UKS.

    Returns separate alpha and beta drho arrays.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and derivatives, shape ``[ncomp, ngrids, nao]``.
    ao_dm0a, ao_dm0b : np.ndarray
        Pre-contracted ``ao @ dm0`` for each spin, shape ``[ncomp_dm0, ngrids, nao]``.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]``.

    Returns
    -------
    drhoa, drhob : np.ndarray
        Skeleton derivative of rho per spin, shape ``[natm, 3, nvar, ngrids]``.
    """
    ngrids = ao.shape[1]
    nvar = XC_NVAR[xc_type]
    natm = len(aoslices)
    drhoa = np.zeros((natm, 3, nvar, ngrids))
    drhob = np.zeros((natm, 3, nvar, ngrids))

    # (rho_var, t_direction, cbra, cket) tuples that contribute to each rho component.
    components = [
        (0, 0, X, O),
        (0, 1, Y, O),
        (0, 2, Z, O),
    ]
    if xc_type in ("GGA", "MGGA"):
        components += [
            (1, 0, XX, O), (2, 0, XY, O), (3, 0, XZ, O),
            (1, 1, YX, O), (2, 1, YY, O), (3, 1, YZ, O),
            (1, 2, ZX, O), (2, 2, ZY, O), (3, 2, ZZ, O),
        ]
        components += [
            (1, 0, X, X), (2, 0, X, Y), (3, 0, X, Z),
            (1, 1, Y, X), (2, 1, Y, Y), (3, 1, Y, Z),
            (1, 2, Z, X), (2, 2, Z, Y), (3, 2, Z, Z),
        ]
    if xc_type == "MGGA":
        components += [
            (4, 0, XX, X), (4, 0, XY, Y), (4, 0, XZ, Z),
            (4, 1, YX, X), (4, 1, YY, Y), (4, 1, YZ, Z),
            (4, 2, ZX, X), (4, 2, ZY, Y), (4, 2, ZZ, Z),
        ]

    for A in range(natm):
        _, _, p0, p1 = aoslices[A]
        slc = slice(p0, p1)
        ao_slc = ao[:, :, slc]
        ao_dm0a_slc = ao_dm0a[:, :, slc]
        ao_dm0b_slc = ao_dm0b[:, :, slc]
        for v, t, cbra, cket in components:
            drhoa[A, t, v] -= np.einsum("gu, gu -> g", ao_slc[cbra], ao_dm0a_slc[cket])
            drhob[A, t, v] -= np.einsum("gu, gu -> g", ao_slc[cbra], ao_dm0b_slc[cket])

    # Symmetry: RHO + grad components carry factor 2; TAU does NOT.
    if xc_type in ("GGA", "MGGA"):
        drhoa[:, :, :4] *= 2
        drhob[:, :, :4] *= 2
    elif xc_type == "LDA":
        drhoa[:, :, :1] *= 2
        drhob[:, :, :1] *= 2
    return drhoa, drhob


def _de_fxc_uks(weights, drhoa, drhob, fxc):
    """fxc contribution to the UKS XC skeleton 2nd derivative.

    For UKS, the fxc kernel has spin indices: fxc[s1, x, s2, y, g].
    The contribution sums over all spin pairs:
        de_fxc = w * (drho_a @ fxc_aa @ drho_a + drho_a @ fxc_ab @ drho_b
                     + drho_b @ fxc_ba @ drho_a + drho_b @ fxc_bb @ drho_b)

    Parameters
    ----------
    weights : np.ndarray
        Grid weights, shape ``[ngrids]``.
    drhoa, drhob : np.ndarray
        Skeleton derivative of rho per spin, shape ``[natm, 3, nvar, ngrids]``.
    fxc : np.ndarray
        Second functional derivative, shape ``[2, nvar, 2, nvar, ngrids]``.

    Returns
    -------
    de_fxc : np.ndarray
        fxc contribution, shape ``[natm, natm, 3, 3]``.
    """
    de_fxc = np.zeros_like(_de_fxc_uks_inner(weights, drhoa, fxc[0, :, 0, :, :], drhoa))
    # aa
    de_fxc += _de_fxc_uks_inner(weights, drhoa, fxc[0, :, 0, :, :], drhoa)
    # ab
    de_fxc += _de_fxc_uks_inner(weights, drhoa, fxc[0, :, 1, :, :], drhob)
    # ba
    de_fxc += _de_fxc_uks_inner(weights, drhob, fxc[1, :, 0, :, :], drhoa)
    # bb
    de_fxc += _de_fxc_uks_inner(weights, drhob, fxc[1, :, 1, :, :], drhob)
    return de_fxc


def _de_fxc_uks_inner(weights, drho1, fxc_block, drho2):
    """Single spin-pair fxc contraction."""
    return np.einsum("g, Atxg, xyg, Bsyg -> ABts", weights, drho1, fxc_block, drho2, optimize=True)


def _de_vxc_diag_uks(xc_type, ao, ao_dm0a, ao_dm0b, wva, wvb, aoslices, natm, nao):
    """Same-atom (A == B) block of the UKS XC skeleton 2nd derivative.

    Returns separate alpha and beta contributions.

    Parameters
    ----------
    xc_type : str
    ao : np.ndarray
    ao_dm0a, ao_dm0b : np.ndarray
        Per-spin ao@dm0, shape ``[ncomp_dm0, ngrids, nao]``.
    wva, wvb : np.ndarray
        Per-spin weight*vxc, shape ``[nvar, ngrids]``.
    aoslices, natm, nao : as in RKS

    Returns
    -------
    de_vxc_diag_a, de_vxc_diag_b : np.ndarray
        Each shape ``[natm, natm, 3, 3]``, only diagonal A==B blocks non-zero.
    """
    de_vxc_diag_a = _de_vxc_diag_one_spin(xc_type, ao, ao_dm0a, wva, aoslices, natm, nao)
    de_vxc_diag_b = _de_vxc_diag_one_spin(xc_type, ao, ao_dm0b, wvb, aoslices, natm, nao)
    return de_vxc_diag_a, de_vxc_diag_b


def _de_vxc_diag_one_spin(xc_type, ao, ao_dm0, wv, aoslices, natm, nao):
    """Same-atom vxc diag block for one spin channel (same as RKS)."""
    dao_vxc_diag = np.zeros((6, nao))

    aow = np.einsum("gu, g -> gu", ao_dm0[0], wv[0])
    if xc_type in ("GGA", "MGGA"):
        for r in range(3):
            aow += np.einsum("gu, g -> gu", ao_dm0[1 + r], wv[1 + r])
    for idx_ts, its in enumerate([XX, XY, XZ, YY, YZ, ZZ]):
        dao_vxc_diag[idx_ts] += 2 * np.einsum("gu, gu -> u", ao[its], aow)

    if xc_type in ("GGA", "MGGA"):
        for idx_ts, (i3x, i3y, i3z) in enumerate(TRIPLE_SIGMA_DIAG):
            aow = (
                np.einsum("gu, g -> gu", ao[i3x], wv[1])
                + np.einsum("gu, g -> gu", ao[i3y], wv[2])
                + np.einsum("gu, g -> gu", ao[i3z], wv[3])
            )
            dao_vxc_diag[idx_ts] += 2 * np.einsum("gu, gu -> u", aow, ao_dm0[0])

    if xc_type == "MGGA":
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


def _de_vxc_off_uks(xc_type, ao, dm0a, dm0b, wva, wvb, aoslices, natm, nao):
    """Two-atom (A != B) block of the UKS XC skeleton 2nd derivative.

    Returns separate alpha and beta contributions.

    Parameters
    ----------
    dm0a, dm0b : np.ndarray
        Per-spin density matrices, shape ``[nao, nao]``.
    wva, wvb : np.ndarray
        Per-spin weight*vxc, shape ``[nvar, ngrids]``.

    Returns
    -------
    de_vxc_off_a, de_vxc_off_b : np.ndarray
        Each shape ``[natm, natm, 3, 3]``.
    """
    de_vxc_off_a = _de_vxc_off_one_spin(xc_type, ao, dm0a, wva, aoslices, natm, nao)
    de_vxc_off_b = _de_vxc_off_one_spin(xc_type, ao, dm0b, wvb, aoslices, natm, nao)
    return de_vxc_off_a, de_vxc_off_b


def _de_vxc_off_one_spin(xc_type, ao, dm0, wv, aoslices, natm, nao):
    """Off-diagonal vxc block for one spin channel (same as RKS)."""
    dao_vxc_off = np.zeros((3, 3, nao, nao))

    if xc_type == "LDA":
        for t in range(3):
            aowv = 0.5 * np.einsum("gu, g -> gu", ao[t + 1], wv[0])
            for s in range(3):
                dao_vxc_off[t, s] += 2 * ao[s + 1].T @ aowv

    if xc_type in ("GGA", "MGGA"):
        for t in range(3):
            aowv = 0.5 * np.einsum("gu, g -> gu", ao[t + 1], wv[0])
            for r in range(3):
                aowv += np.einsum("gu, g -> gu", ao[IDX_AO_DERIV2[t][r]], wv[r + 1])
            for s in range(3):
                dao_vxc_off[t, s] += 2 * ao[s + 1].T @ aowv

    if xc_type == "MGGA":
        dao_vxc_tau = np.zeros((3, 3, nao, nao))
        for k in range(3):
            for s in range(3):
                aowv = np.einsum("gu, g -> gu", ao[IDX_AO_DERIV2[k][s]], wv[4])
                for t in range(s, 3):
                    dao_vxc_tau[t, s] += 0.5 * aowv.T @ ao[IDX_AO_DERIV2[k][t]]
        for t in range(3):
            for s in range(t):
                dao_vxc_tau[s, t] = dao_vxc_tau[t, s].T
        dao_vxc_off += dao_vxc_tau

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


def _vmat_ip_uks(xc_type, ao, wva, wvb, nao):
    """Gradient-level Vxc matrix for UKS, per spin.

    Parameters
    ----------
    wva, wvb : np.ndarray
        Per-spin weight*vxc, shape ``[nvar, ngrids]``.

    Returns
    -------
    vmata_ip, vmatb_ip : np.ndarray
        Each shape ``[3, nao, nao]``.
    """
    vmata_ip = _vmat_ip_one_spin(xc_type, ao, wva, nao)
    vmatb_ip = _vmat_ip_one_spin(xc_type, ao, wvb, nao)
    return vmata_ip, vmatb_ip


def _vmat_ip_one_spin(xc_type, ao, wv, nao):
    """Gradient-level Vxc matrix for one spin channel (same as RKS)."""
    vmat_ip = np.zeros((3, nao, nao))

    if xc_type == "LDA":
        aow = np.einsum("g, gu -> gu", wv[0], ao[O])
        for t in range(3):
            vmat_ip[t] += ao[t + 1].T @ aow
        return vmat_ip

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


def _vmat_deriv1_uks(xc_type, ao, drhoa, drhob, wf, vmata_ip, vmatb_ip, aoslices, natm, nao):
    """Per-atom skeleton derivative of the Vxc Fock matrix for UKS.

    Parameters
    ----------
    drhoa, drhob : np.ndarray
        Per-spin drho, shape ``[natm, 3, nvar, ngrids]``.
    wf : np.ndarray
        Weight*fxc, shape ``[2, nvar, 2, nvar, ngrids]``.
    vmata_ip, vmatb_ip : np.ndarray
        Per-spin gradient-level Vxc, shape ``[3, nao, nao]``.

    Returns
    -------
    vmata_deriv1, vmatb_deriv1 : np.ndarray
        Each shape ``[natm, 3, nao, nao]``, antisymmetrised in AO.
    """
    vmata_deriv1 = np.zeros((natm, 3, nao, nao))
    vmatb_deriv1 = np.zeros((natm, 3, nao, nao))

    for A in range(natm):
        _, _, p0, p1 = aoslices[A]

        # For UKS, the fxc contraction couples both spin channels:
        # wva_f = wf_aa @ drho_a + wf_ab @ drho_b
        # wvb_f = wf_ba @ drho_a + wf_bb @ drho_b
        if xc_type == "LDA":
            # LDA: fxc is [2, 1, 2, 1, ngrids]
            wva_f = np.einsum("g, tg -> tg", wf[0, 0, 0, 0], drhoa[A, :, 0]) * 0.5
            wva_f += np.einsum("g, tg -> tg", wf[0, 0, 1, 0], drhob[A, :, 0]) * 0.5
            wvb_f = np.einsum("g, tg -> tg", wf[1, 0, 0, 0], drhoa[A, :, 0]) * 0.5
            wvb_f += np.einsum("g, tg -> tg", wf[1, 0, 1, 0], drhob[A, :, 0]) * 0.5
            for t in range(3):
                aowa = wva_f[t][:, None] * ao[O]
                vmata_deriv1[A, t] += aowa.T @ ao[O]
                aowb = wvb_f[t][:, None] * ao[O]
                vmatb_deriv1[A, t] += aowb.T @ ao[O]

        if xc_type in ("GGA", "MGGA"):
            # wf has shape [2, nvar, 2, nvar, ngrids]
            # wv_sigma_f[y, t, g] = sum_x wf[0, x, sigma, y, g] * drhoa[A, t, x, g]
            #                     + sum_x wf[1, x, sigma, y, g] * drhob[A, t, x, g]
            # Alpha output (sigma=0):
            wva_f = np.einsum("xyg, txg -> ytg", wf[0, :, 0, :, :], drhoa[A])
            wva_f += np.einsum("xyg, txg -> ytg", wf[1, :, 0, :, :], drhob[A])
            # Beta output (sigma=1):
            wvb_f = np.einsum("xyg, txg -> ytg", wf[0, :, 1, :, :], drhoa[A])
            wvb_f += np.einsum("xyg, txg -> ytg", wf[1, :, 1, :, :], drhob[A])

            wva_f[0] *= 0.5
            wvb_f[0] *= 0.5
            if xc_type == "MGGA":
                wva_f[4] *= 0.25
                wvb_f[4] *= 0.25

            aowa_f = np.einsum("ctg, cgm -> tgm", wva_f[:4], ao[:4])
            aowb_f = np.einsum("ctg, cgm -> tgm", wvb_f[:4], ao[:4])
            for t in range(3):
                vmata_deriv1[A, t] += aowa_f[t].T @ ao[O]
                vmatb_deriv1[A, t] += aowb_f[t].T @ ao[O]

        if xc_type == "MGGA":
            for j in range(1, 4):
                for t in range(3):
                    aowa = wva_f[4, t][:, None] * ao[j]
                    vmata_deriv1[A, t] += aowa.T @ ao[j]
                    aowb = wvb_f[4, t][:, None] * ao[j]
                    vmatb_deriv1[A, t] += aowb.T @ ao[j]

        # ipip part: subtract from atom A's bra rows
        vmata_deriv1[A, :, p0:p1, :] -= vmata_ip[:, p0:p1, :]
        vmatb_deriv1[A, :, p0:p1, :] -= vmatb_ip[:, p0:p1, :]

    # Antisymmetrise
    vmata_deriv1 += vmata_deriv1.swapaxes(-1, -2)
    vmatb_deriv1 += vmatb_deriv1.swapaxes(-1, -2)
    return vmata_deriv1, vmatb_deriv1


def make_hessian_setup_batch_uks(
    mol: gto.Mole,
    xc: str,
    coords: np.ndarray,
    weights: np.ndarray,
    dm0a: np.ndarray,
    dm0b: np.ndarray,
    atm_list: list[int] = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute all DFT skeleton ingredients of the UKS Hessian in one pass.

    Parameters
    ----------
    dm0a, dm0b : np.ndarray
        Per-spin density matrices, shape ``[nao, nao]``.

    Returns
    -------
    result : dict[str, np.ndarray]
        Dictionary with entries:
        - ``de_vxc_diag_a``, ``de_vxc_diag_b`` : same-atom XC skeleton, shape ``[natm, natm, 3, 3]``.
        - ``de_vxc_off_a``, ``de_vxc_off_b`` : two-atom XC skeleton, shape ``[natm, natm, 3, 3]``.
        - ``de_fxc`` : fxc-kernel contribution (spin-summed), shape ``[natm, natm, 3, 3]``.
        - ``vmat_ip_a``, ``vmat_ip_b`` : gradient-level Vxc, shape ``[3, nao, nao]``.
        - ``vmat_deriv1_a``, ``vmat_deriv1_b`` : per-atom skeleton derivative, shape ``[natm, 3, nao, nao]``.
    """
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
    ni = dft.numint.NumInt()
    ao = ni.eval_ao(mol, coords, deriv=XC_AO_DERIV[xc_type])
    ncomp_dm0 = XC_NCOMP_AO_DM0[xc_type]
    ao_dm0a = ao[:ncomp_dm0] @ dm0a
    ao_dm0b = ao[:ncomp_dm0] @ dm0b
    rhoa, rhob, exc, vxc, fxc = _eval_rho_exc_vxc_fxc_uks(xc, xc_type, ao, ao_dm0a, ao_dm0b)
    wva = weights * vxc[0]
    wvb = weights * vxc[1]
    wf = weights * fxc
    tic("ao, rho, vxc, fxc", t0)

    t0 = time.time()
    drhoa, drhob = _make_drho_uks(xc_type, ao, ao_dm0a, ao_dm0b, aoslices)
    de_fxc = _de_fxc_uks(weights, drhoa, drhob, fxc)
    tic("drho, de_fxc", t0)

    t0 = time.time()
    de_vxc_diag_a, de_vxc_diag_b = _de_vxc_diag_uks(
        xc_type, ao, ao_dm0a, ao_dm0b, wva, wvb, aoslices, natm, nao
    )
    tic("de_vxc_diag", t0)

    t0 = time.time()
    de_vxc_off_a, de_vxc_off_b = _de_vxc_off_uks(
        xc_type, ao, dm0a, dm0b, wva, wvb, aoslices, natm, nao
    )
    tic("de_vxc_off", t0)

    t0 = time.time()
    vmat_ip_a, vmat_ip_b = _vmat_ip_uks(xc_type, ao, wva, wvb, nao)
    tic("vmat_ip", t0)

    t0 = time.time()
    vmat_deriv1_a, vmat_deriv1_b = _vmat_deriv1_uks(
        xc_type, ao, drhoa, drhob, wf, vmat_ip_a, vmat_ip_b, aoslices, natm, nao
    )
    tic("vmat_deriv1", t0)

    return {
        "de_vxc_diag_a": de_vxc_diag_a,
        "de_vxc_diag_b": de_vxc_diag_b,
        "de_vxc_off_a": de_vxc_off_a,
        "de_vxc_off_b": de_vxc_off_b,
        "de_fxc": de_fxc,
        "vmat_ip_a": vmat_ip_a,
        "vmat_ip_b": vmat_ip_b,
        "vmat_deriv1_a": vmat_deriv1_a,
        "vmat_deriv1_b": vmat_deriv1_b,
    }


def get_uks_response_bra_naive(
    mol: gto.Mole,
    grids,
    xc: str,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    dm0a: np.ndarray,
    dm0b: np.ndarray,
    bra: list[np.ndarray],
    rho_cached=None,
    vxc_cached=None,
    fxc_cached=None,
) -> list[np.ndarray]:
    """Apply the DFT XC fxc kernel to a perturbed bra for UKS.

    Parameters
    ----------
    bra : list[np.ndarray]
        ``[bra_alpha, bra_beta]``, each ``[..., nao, nocc_sigma]``.

    Returns
    -------
    resp_bra : list[np.ndarray]
        ``[resp_alpha, resp_beta]``, same shapes as input.
    """
    nao = mol.nao
    ni = dft.numint.NumInt()

    occidx_a = mo_coeff[0].shape[1] if mo_occ is None else mo_occ[0] > 1e-15
    occidx_b = mo_coeff[1].shape[1] if mo_occ is None else mo_occ[1] > 1e-15
    mocc_a = mo_coeff[0][:, occidx_a]
    mocc_b = mo_coeff[1][:, occidx_b]
    nocc_a = mocc_a.shape[-1]
    nocc_b = mocc_b.shape[-1]

    bra_a_shape = bra[0].shape
    bra_b_shape = bra[1].shape
    assert bra_a_shape[-2] == nao and bra_a_shape[-1] == nocc_a
    assert bra_b_shape[-2] == nao and bra_b_shape[-1] == nocc_b
    bra_a = bra[0].reshape(-1, nao, nocc_a)
    bra_b = bra[1].reshape(-1, nao, nocc_b)

    # Per-spin perturbed density: dm1_s = bra_s @ mocc_s^T + (bra_s @ mocc_s^T)^T
    dm1a = bra_a @ mocc_a.T
    dm1a = dm1a + dm1a.swapaxes(-1, -2)
    dm1b = bra_b @ mocc_b.T
    dm1b = dm1b + dm1b.swapaxes(-1, -2)

    v1a, v1b = ni.nr_uks_fxc(
        mol, grids, xc, (dm0a, dm0b), (dm1a, dm1b), hermi=1,
        rho0=rho_cached, vxc=vxc_cached, fxc=fxc_cached,
    )

    resp_a = v1a @ mocc_a
    resp_b = v1b @ mocc_b
    return [resp_a.reshape(bra_a_shape), resp_b.reshape(bra_b_shape)]


class UHessKSNaive(UHessElecInteractAPI):
    """Naive implementation of the DFT XC contribution to the UKS Hessian.

    Implements ``UHessElecInteractAPI`` for the XC piece. The hybrid J/K piece
    is handled by ``UHessRIJKNaive``; this class handles the grid-integrated XC
    pieces and the fxc-kernel response.

    Parameters
    ----------
    mol : gto.Mole
    xc : str
    grids : pyscf.dft.Grids
    nbatch_grids : int, optional
    """

    def __init__(self, mol: gto.Mole, xc: str, grids, nbatch_grids: int = 16384):
        self.mol = mol
        self.xc = xc
        self.grids = grids
        self.nbatch_grids = nbatch_grids
        self.result = dict()
        self.mo_coeff = None
        self.mo_occ = None
        self.dm0a = None
        self.dm0b = None
        self.rho_cached = None
        self.vxc_cached = None
        self.fxc_cached = None

    def _run_setup_batched(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        """Run batched setup and store results."""
        if "de_xc_skeleton" in self.result and "de_xc_deriv1_ao" in self.result:
            return

        dm0_per_spin = get_dm0_unrestricted(mo_coeff, mo_occ)
        dm0a, dm0b = dm0_per_spin[0], dm0_per_spin[1]
        coords = self.grids.coords
        weights = self.grids.weights
        ngrids = weights.size

        result_sum = None
        for start in range(0, ngrids, self.nbatch_grids):
            stop = min(start + self.nbatch_grids, ngrids)
            partial = make_hessian_setup_batch_uks(
                self.mol, self.xc,
                coords[start:stop], weights[start:stop],
                dm0a, dm0b, verbose=False,
            )
            if result_sum is None:
                result_sum = {k: v.copy() for k, v in partial.items()}
            else:
                for k in result_sum:
                    result_sum[k] += partial[k]

        # Total XC skeleton = (diag_a + off_a + diag_b + off_b + de_fxc)
        self.result["de_xc_skeleton"] = (
            result_sum["de_vxc_diag_a"] + result_sum["de_vxc_off_a"]
            + result_sum["de_vxc_diag_b"] + result_sum["de_vxc_off_b"]
            + result_sum["de_fxc"]
        )
        # Per-spin deriv1_ao
        self.result["de_xc_deriv1_ao"] = np.array([
            result_sum["vmat_deriv1_a"],
            result_sum["vmat_deriv1_b"],
        ])

    def make_skeleton_hess(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        if "de_xc_skeleton" not in self.result:
            self._run_setup_batched(mo_coeff, mo_occ)
        return self.result["de_xc_skeleton"]

    def get_deriv1_ao(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        if "de_xc_deriv1_ao" not in self.result:
            self._run_setup_batched(mo_coeff, mo_occ)
        return self.result["de_xc_deriv1_ao"]

    def make_response_preparation(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        dm0_per_spin = get_dm0_unrestricted(mo_coeff, mo_occ)
        self.dm0a = dm0_per_spin[0]
        self.dm0b = dm0_per_spin[1]

        ni = dft.numint.NumInt()
        self.rho_cached, self.vxc_cached, self.fxc_cached = ni.cache_xc_kernel(
            self.mol, self.grids, self.xc, mo_coeff, mo_occ, spin=1,
        )

    def get_response_bra(self, bra: list[np.ndarray]) -> list[np.ndarray]:
        return get_uks_response_bra_naive(
            self.mol, self.grids, self.xc,
            self.mo_coeff, self.mo_occ, self.dm0a, self.dm0b, bra,
            rho_cached=self.rho_cached,
            vxc_cached=self.vxc_cached,
            fxc_cached=self.fxc_cached,
        )
