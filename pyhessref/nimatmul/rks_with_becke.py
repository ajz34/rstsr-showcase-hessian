"""
Compute Hessian with grid-shift induced contribution (only Becke scheme is supported).
"""

from pyscf import gto, dft
import numpy as np
import time

from pyhessref.hess_trait_restricted import RHessElecInteractAPI
from pyhessref.util import get_dm0_restricted

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


def _eval_rho_exc_vxc_fxc(xc, xc_type, ao, ao_dm0):
    """Evaluate the on-grid density together with the first/second functional
    derivatives ``vxc``/``fxc`` for the requested xc functional.

    The rho layout follows PySCF's ``eval_xc_eff`` convention with channels
    stacked along the leading axis:

    - ``LDA``  : ``[rho]``                                  (1 channel)
    - ``GGA``  : ``[rho, drho/dx, drho/dy, drho/dz]``       (4 channels)
    - ``MGGA`` : ``[rho, drho/dx, drho/dy, drho/dz, tau]``  (5 channels)

    ``tau`` here is the LAPL-removed kinetic energy density.

    Parameters
    ----------
    xc : str
        The xc functional name (e.g. ``"B3LYP"``, ``"TPSS0"``).
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"`` — the functional family.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]``.  Only the value channel ``ao[0]`` (and
        the gradient channels ``ao[1:4]`` for GGA/MGGA) are read here, even
        though additional higher-order channels may be present.
    ao_dm0 : np.ndarray
        Pre-contracted ``ao @ dm0`` on the channel axis, shape
        ``[ncomp_dm0, ngrids, nao]`` — ``ncomp_dm0`` is 1 for LDA and 4
        for GGA/MGGA (rho + 3 gradients).

    Returns
    -------
    rho : np.ndarray
        On-grid density components, shape ``[nvar, ngrids]``.
    exc : np.ndarray
        On-grid XC energy, shape ``[ngrids]``.
    vxc : np.ndarray
        First functional derivative ``f^chi``, shape ``[nvar, ngrids]``.
    fxc : np.ndarray
        Second functional derivative ``f^{chi chi'}``, shape
        ``[nvar, nvar, ngrids]``.
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
    exc, vxc, fxc, _ = ni.eval_xc_eff(xc, rho, deriv=2, xctype=xc_type)
    return rho, exc, vxc, fxc


def _make_drho(xc_type, ao, ao_dm0, aoslices):
    """First-order skeleton derivative of the rho components with respect to
    nuclear coordinates.

    The "skeleton" derivative is the contribution that comes from the basis
    functions following the nucleus they are centred on, holding the density
    matrix fixed.  For each atom A and Cartesian direction t, the derivative
    acts only on bra basis indices in the on-atom slice ``[p0, p1)``, hence
    the per-atom contraction below.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]``.  LDA reads up to 1st-order channels
        (indices 0..3); GGA/MGGA reads up to 2nd-order channels
        (indices 0..9).
    ao_dm0 : np.ndarray
        Pre-contracted ``ao @ dm0``, shape ``[ncomp_dm0, ngrids, nao]``
        with ``ncomp_dm0`` = 1 for LDA and 4 for GGA/MGGA.
    aoslices : np.ndarray
        Per-atom AO slices as returned by ``mol.aoslice_by_atom()`` (or its
        atom-list-restricted view), of shape ``[natm, 4]``.

    Returns
    -------
    drho : np.ndarray
        Skeleton derivative ``d xi^chi / d A_t``, shape
        ``[natm, 3, nvar, ngrids]``.  Symmetric components (rho + grad)
        carry a factor 2 from the bra↔ket symmetry; the tau channel does not.
        The derivative carries the bra-side minus sign convention (so a
        positive nuclear displacement of A drags AO bra functions and the
        contribution to rho falls off in the -bra-deriv direction).
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
            (1, 0, XX, O),
            (2, 0, XY, O),
            (3, 0, XZ, O),
            (1, 1, YX, O),
            (2, 1, YY, O),
            (3, 1, YZ, O),
            (1, 2, ZX, O),
            (2, 2, ZY, O),
            (3, 2, ZZ, O),
        ]
        # SIGMA part: bra deriv1 ket deriv1.
        components += [
            (1, 0, X, X),
            (2, 0, X, Y),
            (3, 0, X, Z),
            (1, 1, Y, X),
            (2, 1, Y, Y),
            (3, 1, Y, Z),
            (1, 2, Z, X),
            (2, 2, Z, Y),
            (3, 2, Z, Z),
        ]

    if xc_type == "MGGA":
        # TAU part: bra deriv2 ket deriv1.  tau index = 4.
        components += [
            (4, 0, XX, X),
            (4, 0, XY, Y),
            (4, 0, XZ, Z),
            (4, 1, YX, X),
            (4, 1, YY, Y),
            (4, 1, YZ, Z),
            (4, 2, ZX, X),
            (4, 2, ZY, Y),
            (4, 2, ZZ, Z),
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
    """fxc contribution to the XC skeleton 2nd derivative.

    This is the part of ``de_vxc`` that comes from contracting the
    first-order rho derivatives on each side with the second functional
    derivative kernel ``fxc``.

    Parameters
    ----------
    weights : np.ndarray
        Grid weights, shape ``[ngrids]``.
    drho : np.ndarray
        Skeleton derivative of rho components, shape
        ``[natm, 3, nvar, ngrids]``.
    fxc : np.ndarray
        Second functional derivative kernel, shape
        ``[nvar, nvar, ngrids]``.

    Returns
    -------
    de_fxc : np.ndarray
        fxc contribution to the Hessian, shape ``[natm, natm, 3, 3]``.
    """
    return np.einsum("g, Atxg, xyg, Bsyg -> ABts", weights, drho, fxc, drho, optimize=True)


def _de_vxc_diag(xc_type, ao, ao_dm0, wv, aoslices, natm, nao):
    """Same-atom (A == B) block of the XC skeleton 2nd derivative.

    Builds the AO-resolved object ``dao_vxc_diag[6, nao]`` whose 6 components
    are the symmetric Cartesian pairs ``(xx, xy, xz, yy, yz, zz)``, then
    contracts with the on-atom slice and re-expands to a dense ``(3, 3)``
    block per atom.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]``.  LDA reads up to 2nd-order channels
        (indices 0..9); GGA/MGGA further reads 3rd-order channels
        (indices 10..19) for the triple-derivative contributions.
    ao_dm0 : np.ndarray
        Pre-contracted ``ao @ dm0``, shape
        ``[ncomp_dm0, ngrids, nao]`` (1 for LDA, 4 for GGA/MGGA).
    wv : np.ndarray
        Weight-times-vxc, shape ``[nvar, ngrids]``.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]`` — only the last two columns
        ``[p0, p1)`` are used for the bra-side AO range of each atom.
    natm : int
        Number of atoms in the (possibly restricted) atom list.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    de_vxc_diag : np.ndarray
        Same-atom block of the XC skeleton 2nd derivative, shape
        ``[natm, natm, 3, 3]`` — only the diagonal ``A == B`` blocks are
        non-zero (off-diagonal blocks are produced by ``_de_vxc_off``).
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
    """Two-atom (A != B) block of the XC skeleton 2nd derivative.

    Builds the dense AO-resolved object ``dao_vxc_off[3, 3, nao, nao]`` (so
    both bra and ket retain their AO indices), symmetrises it under
    ``[t, s, mu, nu] -> [s, t, nu, mu]``, and contracts each ``(A, B)``
    block with the corresponding ``dm0[B, A]`` AO slice.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]``.  Indices 0..3 are always read; GGA/MGGA
        also reads the 2nd-order channels (indices 4..9).
    dm0 : np.ndarray
        Density matrix in AO basis, shape ``[nao, nao]``.
    wv : np.ndarray
        Weight-times-vxc, shape ``[nvar, ngrids]``.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]``.
    natm : int
        Number of atoms in the (possibly restricted) atom list.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    de_vxc_off : np.ndarray
        Two-atom block of the XC skeleton 2nd derivative, shape
        ``[natm, natm, 3, 3]``.  Both ``A == B`` and ``A != B`` entries are
        populated; the natural decomposition into "diagonal vs off-diagonal"
        is by the integral kernel rather than by atom index, so this
        function's diagonal block is *not* zero — it complements
        ``de_vxc_diag``.
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
        # TAU contribution: tau_munu = sum_k 0.5 * (d_k phi_mu)^T (d_k phi_nu).
        # Each AO-derivative block dao_vxc_tau[t, s] picks up
        #   0.5 * sum_k (ao[IDX_AO_DERIV2[k][s]] * wv[4])^T @ ao[IDX_AO_DERIV2[k][t]]
        # for s <= t; the s > t triangle is filled by transposition.  We loop
        # outer-s, inner-t so a single ``aowv`` buffer is alive at a time
        # (matching the LDA/GGA blocks above), instead of caching all 6.
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


def _de_becke_full_parts(dw, ddw, exc, vxc, rho, drho):
    t1 = np.einsum("Atg, xg, Bsxg -> ABts", dw, vxc, drho, optimize=True)
    t2 = np.einsum("AtBsg, g, g -> ABts", ddw, exc, rho[0], optimize=True)
    return {"de_becke_full_1": t1, "de_becke_full_2": t2}


def _de_becke_atom_parts(w, dw, vxc, fxc, drho, prho):
    t1 = -np.einsum("g, txg, xyg, Bsyg -> Bts", w, prho, fxc, drho, optimize=True)
    t2 = -np.einsum("Bsg, xg, txg -> Bts", dw, vxc, prho, optimize=True)
    t3 = np.einsum("g, xyg, syg, txg -> ts", w, fxc, prho, prho, optimize=True)
    return {
        "de_becke_atom_1": t1,
        "de_becke_atom_2": t2,
        "de_becke_atom_3": t3,
    }


def _de_becke_atom_expensive(xc_type, ao, ao_dm0, w, vxc, aoslices, natm, ngrids):
    IDX2 = [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
    IDX3 = [
        [[XXX, XXY, XXZ], [XXY, XYY, XYZ], [XXZ, XYZ, XZZ]],
        [[XXY, XYY, XYZ], [XYY, YYY, YYZ], [XYZ, YYZ, YZZ]],
        [[XXZ, XYZ, XZZ], [XYZ, YYZ, YZZ], [XZZ, YZZ, ZZZ]],
    ]
    if xc_type == "LDA":
        raise NotImplementedError
    pdrho = np.zeros((natm, 3, 3, XC_NVAR[xc_type], ngrids))
    for C in range(natm):
        _, _, p0, p1 = aoslices[C]
        slc = slice(p0, p1)
        ao_slc = ao[:, :, slc]
        ao_dm0_slc = ao_dm0[:, :, slc]
        # x = 0 (rho value)
        for s in range(3):
            for t in range(3):
                term = np.einsum("gu, gu -> g", ao_slc[IDX2[t][s]], ao_dm0_slc[O]) + np.einsum(
                    "gu, gu -> g", ao_slc[s + 1], ao_dm0_slc[t + 1]
                )
                pdrho[C, s, t, 0] += 2 * term
        # x = k+1 (sigma gradient component); needs 3rd-order AO derivatives
        for k in range(3):
            for s in range(3):
                for t in range(3):
                    term = (
                        np.einsum("gu, gu -> g", ao_slc[IDX3[t][s][k]], ao_dm0_slc[O])
                        + np.einsum("gu, gu -> g", ao_slc[IDX2[s][k]], ao_dm0_slc[t + 1])
                        + np.einsum("gu, gu -> g", ao_slc[IDX2[t][s]], ao_dm0_slc[k + 1])
                        + np.einsum("gu, gu -> g", ao_slc[s + 1], ao_dm0_slc[IDX2[t][k]])
                    )
                    pdrho[C, s, t, k + 1] += 2 * term
        # x = 4 (tau component); needs 3rd-order AO derivatives + 2nd-order ao_dm0.
        # tau does NOT carry the bra<->ket symmetry factor 2 (it is built from the
        # asymmetric (nabla bra).(nabla ket) form), so neither drho[..,4] nor d2rho[..,4] do.
        if xc_type == "MGGA":
            for s in range(3):
                for t in range(3):
                    term = 0
                    for k in range(3):
                        term += np.einsum("gu, gu -> g", ao_slc[IDX3[t][s][k]], ao_dm0_slc[k + 1]) + np.einsum(
                            "gu, gu -> g", ao_slc[IDX2[s][k]], ao_dm0_slc[IDX2[t][k]]
                        )
                    pdrho[C, s, t, 4] += term
    pprho = pdrho.sum(axis=0)  # (s, t, x, g) = -d2 rho_x / (d r_s d r_t)

    t4 = np.zeros((natm, 3, 3))
    t5 = np.zeros((3, 3))

    for B in range(natm):
        t4[B] -= np.einsum("g, xg, stxg -> ts", w, vxc, pdrho[B], optimize=True)
    t5 += np.einsum("g, xg, stxg -> ts", w, vxc, pprho, optimize=True)
    return {
        "de_becke_atom_expensive_4": t4,
        "de_becke_atom_expensive_5": t5,
    }


def _vmat_ip(xc_type, ao, wv, nao):
    """Gradient-level Vxc matrix shared across all atoms (the ipip block).

    This is the AO-space object that, when later multiplied on its bra-side
    AO slice, yields the on-atom contribution that ``_vmat_deriv1`` adds to
    the per-atom skeleton Fock derivative.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]``.  LDA uses indices 0..3; GGA/MGGA also
        uses 2nd-order channels at indices 4..9.
    wv : np.ndarray
        Weight-times-vxc, shape ``[nvar, ngrids]``.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    vmat_ip : np.ndarray
        Gradient-level Vxc matrix, shape ``[3, nao, nao]``, indexed by the
        Cartesian direction ``t`` of the bra derivative.  The matrix is not
        symmetrised in AO indices — symmetrisation happens in
        ``_vmat_deriv1`` once the per-atom slicing is done.
    """
    vmat_ip = np.zeros((3, nao, nao))

    if xc_type == "LDA":
        # bra-on-A and ket-on-A halves are identical for LDA
        # (both equal 0.5 * wv[0] * ao[t+1]^T @ ao[0]), so we fold the
        # two 0.5 factors into a single contraction.  This symmetry does
        # NOT extend to GGA/MGGA — see the branch below.
        aow = np.einsum("g, gu -> gu", wv[0], ao[O])
        for t in range(3):
            vmat_ip[t] += ao[t + 1].T @ aow
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


def _vmat_vxc(vmat_ip, aoslices, natm, nao):
    """vxc contribution to the per-atom skeleton derivative of the Vxc Fock
    matrix - the ipip (basis-function derivative) part.

    This is the slice of the gradient-level ``vmat_ip`` matrix that lives on
    each atom ``A``'s bra rows (the on-atom contribution ``_vmat_deriv1``
    previously folded in directly).  It depends only on ``vmat_ip`` and the
    per-atom AO slices - it is spin-diagonal, so UKS reuses it per spin
    channel.

    Parameters
    ----------
    vmat_ip : np.ndarray
        Gradient-level Vxc matrix from ``_vmat_ip``, shape ``[3, nao, nao]``.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]`` - only the last two columns
        ``[p0, p1)`` are used for the bra-side AO range of each atom.
    natm : int
        Number of atoms in the (possibly restricted) atom list.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    vmat_vxc : np.ndarray
        vxc contribution, shape ``[natm, 3, nao, nao]``, assembled across the
        AO axes (``vmat_vxc += vmat_vxc.swapaxes(-1, -2)``).
    """
    vmat_vxc = np.zeros((natm, 3, nao, nao))
    for A in range(natm):
        _, _, p0, p1 = aoslices[A]
        # ipip part lives only on atom A's bra rows; sign matches the existing test.
        vmat_vxc[A, :, p0:p1, :] -= vmat_ip[:, p0:p1, :]
    # Assemble bra + ket (electron->nuclear coordinate convention).
    vmat_vxc += vmat_vxc.swapaxes(-1, -2)
    return vmat_vxc


def _vmat_fxc(xc_type, ao, drho, wf, natm, nao):
    """fxc contribution to the per-atom skeleton derivative of the Vxc Fock
    matrix - the part that comes from the fxc kernel responding to the
    skeleton density derivative ``drho``.

    For each atom ``A`` and Cartesian direction ``t``, this contracts the
    fxc kernel ``wf`` (weight*fxc) folded against ``drho[A]`` with the AO
    value/gradient channels.  It is the genuinely spin-coupled piece for
    UKS (alpha/beta drho mixed through the spin-indexed fxc kernel), so the
    UKS counterpart ``_vmat_fxc_uks`` is kept separate.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]``.  Only indices 0..3 are read here; higher
        channels in ``ao`` are unused but may be present.
    drho : np.ndarray
        Skeleton derivative of rho components (output of ``_make_drho``),
        shape ``[natm, 3, nvar, ngrids]``.
    wf : np.ndarray
        Weight-times-fxc, shape ``[nvar, nvar, ngrids]``.
    natm : int
        Number of atoms in the (possibly restricted) atom list.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    vmat_fxc : np.ndarray
        fxc contribution, shape ``[natm, 3, nao, nao]``, assembled across the
        AO axes (``vmat_fxc += vmat_fxc.swapaxes(-1, -2)``).
    """
    vmat_fxc = np.zeros((natm, 3, nao, nao))

    for A in range(natm):
        if xc_type == "LDA":
            # wv_f[t, g] = wf[g] * drho[A, t, 0, g] / 2  (drho already has the *2)
            wv_f = np.einsum("g, tg -> tg", wf[0, 0], drho[A, :, 0]) * 0.5
            for t in range(3):
                aow = wv_f[t][:, None] * ao[O]
                vmat_fxc[A, t] += aow.T @ ao[O]

        if xc_type in ("GGA", "MGGA"):
            wv_f = np.einsum("xyg, txg -> ytg", wf, drho[A])
            wv_f[0] *= 0.5
            if xc_type == "MGGA":
                wv_f[4] *= 0.25

            aow_f = np.einsum("ctg, cgm -> tgm", wv_f[:4], ao[:4])
            for t in range(3):
                vmat_fxc[A, t] += aow_f[t].T @ ao[O]

        if xc_type == "MGGA":
            for j in range(1, 4):
                for t in range(3):
                    aow = wv_f[4, t][:, None] * ao[j]
                    vmat_fxc[A, t] += aow.T @ ao[j]

    # Assemble bra + ket (electron->nuclear coordinate convention).
    vmat_fxc += vmat_fxc.swapaxes(-1, -2)
    return vmat_fxc


def _vmat_deriv1(xc_type, ao, drho, wf, vmat_ip, aoslices, natm, nao):
    """Per-atom skeleton derivative of the Vxc Fock matrix (``vmat_deriv1``).

    For each atom ``A`` and Cartesian direction ``t``, this is the
    nuclear-coordinate derivative of the Vxc Fock matrix that holds the
    density matrix fixed (i.e. the CP-KS *skeleton* term, not the full
    response).  It is split into two independently assembled contributions:

    - ``_vmat_vxc`` : the ipip basis-derivative part, taken directly from the
      gradient-level ``vmat_ip`` sliced on atom A's bra rows.
    - ``_vmat_fxc`` : the fxc kernel folded against the skeleton density
      derivative ``drho[A]``.

    Each part is assembled across the AO axes (bra + ket) on its own, and the
    two are summed here.  Splitting is exact up to floating-point order: the
    assembled sum ``(F + F.T) + (V + V.T)`` equals the previous
    ``(F + V) + (F + V).T``.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]``.  Only indices 0..3 are read here;
        higher channels in ``ao`` are unused but may be present.
    drho : np.ndarray
        Skeleton derivative of rho components (output of ``_make_drho``),
        shape ``[natm, 3, nvar, ngrids]``.
    wf : np.ndarray
        Weight-times-fxc, shape ``[nvar, nvar, ngrids]``.
    vmat_ip : np.ndarray
        Gradient-level Vxc matrix from ``_vmat_ip``, shape
        ``[3, nao, nao]``.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]``.
    natm : int
        Number of atoms in the (possibly restricted) atom list.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - ``"vmat_fxc"`` : the fxc contribution (assembled across AO axes).
        - ``"vmat_vxc"`` : the vxc (ipip) contribution (assembled across AO axes).
        - ``"vmat_deriv1"`` : the summed skeleton derivative, shape
          ``[natm, 3, nao, nao]``.
    """
    vmat_fxc = _vmat_fxc(xc_type, ao, drho, wf, natm, nao)
    vmat_vxc = _vmat_vxc(vmat_ip, aoslices, natm, nao)
    return {
        "vmat_fxc": vmat_fxc,
        "vmat_vxc": vmat_vxc,
        "vmat_deriv1": vmat_fxc + vmat_vxc,
    }


def make_hessian_setup_batch(
    mol: gto.Mole,
    xc: str,
    coords: np.ndarray,
    weights: np.ndarray,
    dm0: np.ndarray,
    atm_idx: int,
    quadrature_weights: np.ndarray,
    adjustment_factor: np.ndarray,
    hardness: int = 3,
    atm_list: list[int] = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute all DFT skeleton ingredients of the RKS Hessian in one pass.

    Performs the DFT numerical-integration setup once (``ao``, ``rho``,
    ``vxc``, ``fxc``) and feeds it into the helper routines that build the
    XC skeleton 2nd-derivative pieces (``de_vxc_diag``, ``de_vxc_off``,
    ``de_fxc``) and the CP-KS-side ``vmat_ip``/``vmat_deriv1`` matrices.

    The total XC contribution to the skeleton Hessian is
    ``de_vxc_diag + de_vxc_off + de_fxc``.

    Parameters
    ----------
    mol : gto.Mole
        Molecule, used for AO slices and the AO basis dimension.
    xc : str
        XC functional name, e.g. ``"SVWN"`` (LDA), ``"B3LYP"`` (GGA), or
        ``"TPSS0"`` (MGGA).
    coords : np.ndarray
        Grid point coordinates, shape ``[ngrids, 3]``.
    weights : np.ndarray
        Grid weights, shape ``[ngrids]``.
    dm0 : np.ndarray
        Reference density matrix in AO basis, shape ``[nao, nao]``.
    atm_list : list[int], optional
        Subset of atom indices to compute the per-atom outputs for.
        Defaults to all atoms.
    verbose : bool, optional
        When True, print per-stage timings.  Defaults to True.

    Returns
    -------
    result : dict[str, np.ndarray]
        Dictionary with the following entries:

        - ``de_vxc_diag`` : same-atom XC skeleton block, shape
          ``[natm, natm, 3, 3]`` (only ``A == B`` blocks are non-zero).
        - ``de_vxc_off``  : two-atom XC skeleton block, shape
          ``[natm, natm, 3, 3]``.
        - ``de_fxc``      : fxc-kernel contribution, shape
          ``[natm, natm, 3, 3]``.
        - ``vmat_ip``     : gradient-level Vxc, shape ``[3, nao, nao]``.
        - ``vmat_deriv1`` : per-atom skeleton derivative of the Vxc Fock
          matrix, shape ``[natm, 3, nao, nao]``, antisymmetrised in AO.
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
    ao = dft.numint.eval_ao(mol, coords, deriv=XC_AO_DERIV[xc_type])
    ao_dm0 = ao @ dm0
    rho, exc, vxc, fxc = _eval_rho_exc_vxc_fxc(xc, xc_type, ao, ao_dm0)
    wv = weights * vxc
    wf = weights * fxc
    tic("ao, rho, vxc, fxc", t0)

    t0 = time.time()
    drho = _make_drho(xc_type, ao, ao_dm0, aoslices)
    de_fxc = _de_fxc(weights, drho, fxc)
    tic("drho, de_fxc", t0)

    from pyhessref.nimatmul.becke_partition import becke_partition

    ngrids = len(coords)
    becke_result = becke_partition(
        coords,
        atm_coords=mol.atom_coords(),
        atm_indices=np.full(ngrids, atm_idx, dtype=int),
        quadrature_weights=quadrature_weights,
        adjustment_factor=adjustment_factor,
        hardness=hardness,
        nbatch=1024,
        deriv=2,
        deriv_arg={},
    )
    dw = becke_result["dw"]
    ddw = becke_result["ddw"]
    tic("becke_partition", t0)

    t0 = time.time()
    de_becke_full_parts = _de_becke_full_parts(dw, ddw, exc, vxc, rho, drho)
    tic("de_becke_full_parts", t0)

    t0 = time.time()
    prho = drho.sum(axis=0)
    de_becke_atom_parts = _de_becke_atom_parts(weights, dw, vxc, fxc, drho, prho)
    tic("de_becke_atom_parts", t0)

    t0 = time.time()
    de_becke_atom_expensive = _de_becke_atom_expensive(xc_type, ao, ao_dm0, weights, vxc, aoslices, natm, len(weights))
    tic("de_becke_atom_expensive", t0)

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
    vmat = _vmat_deriv1(xc_type, ao, drho, wf, vmat_ip, aoslices, natm, nao)
    tic("vmat_deriv1", t0)

    results = {
        "de_vxc_diag": de_vxc_diag,
        "de_vxc_off": de_vxc_off,
        "de_fxc": de_fxc,
        "vmat_ip": vmat_ip,
        "vmat_deriv1": vmat["vmat_deriv1"],
        "vmat_fxc": vmat["vmat_fxc"],
        "vmat_vxc": vmat["vmat_vxc"],
    }
    results.update(de_becke_full_parts)
    results.update(de_becke_atom_parts)
    results.update(de_becke_atom_expensive)
    return results


def quad_split_by_atom(atm_quad_split: list[int], atm_list: list[int], nbatch_grids: int):
    """Split the grid into batches of size <= nbatch_grids, respecting atom boundaries.

    Parameters
    ----------
    atm_quad_split : list[int]
        Cumulative sum of quadrature points per atom, shape ``[natm + 1]``.
    atm_list : list[int]
        List of atom indices to include in the split.
    nbatch_grids : int
        Maximum number of grids per batch.

    Returns
    -------
    list[tuple[int, int, int]]
        List of (atom_index, start, end) indices for each batch, where each batch contains
        at most ``nbatch_grids`` grids and does not split any atom's grids.
    """
    batches = []
    start = 0
    for A in atm_list:
        end = atm_quad_split[A + 1]
        if end - start > nbatch_grids:
            # Split the current batch if it exceeds the maximum size
            while start < end:
                next_end = min(start + nbatch_grids, end)
                batches.append((start, next_end))
                start = next_end
        else:
            batches.append((A, start, end))
            start = end
    return batches


def get_quad_split(atm_indices: np.ndarray):
    """
    Obtain list of index (natm + 1) from the list of atom indices.
    The list of atom indices is assumed to be sorted and incremental (by one).
    """
    # [0, 0, 1, 1, 1, 2, 2] -> [0, 2, 5, 7]
    atm_quad_split = [0]
    for i in range(1, len(atm_indices)):
        if atm_indices[i] != atm_indices[i - 1]:
            atm_quad_split.append(i)
    atm_quad_split.append(len(atm_indices))
    return atm_quad_split


def make_hessian_setup(
    mol: gto.Mole,
    xc: str,
    coords: np.ndarray,
    weights: np.ndarray,
    dm0: np.ndarray,
    atm_quad_split: list[int],
    quadrature_weights: np.ndarray,
    adjustment_factor: np.ndarray,
    hardness: int = 3,
    atm_list: list[int] = None,
    nbatch_grids: int = 16384,
    verbose: bool = True,
):
    # same to common run_setup_batched, but use `quad_split_by_atom` to split the grid into batches
    atm_list = atm_list if atm_list is not None else list(range(mol.natm))
    batches = quad_split_by_atom(atm_quad_split, atm_list, nbatch_grids)
    natm = len(atm_list)
    nao = mol.nao
    result_sum = {
        "de_becke_full_1": np.zeros((natm, natm, 3, 3)),
        "de_becke_full_2": np.zeros((natm, natm, 3, 3)),
        "de_becke_atom_1": np.zeros((natm, natm, 3, 3)),
        "de_becke_atom_2": np.zeros((natm, natm, 3, 3)),
        "de_becke_atom_3": np.zeros((natm, natm, 3, 3)),
        "de_becke_atom_expensive_4": np.zeros((natm, natm, 3, 3)),
        "de_becke_atom_expensive_5": np.zeros((natm, natm, 3, 3)),
        "de_vxc_diag": np.zeros((natm, natm, 3, 3)),
        "de_vxc_off": np.zeros((natm, natm, 3, 3)),
        "de_fxc": np.zeros((natm, natm, 3, 3)),
        "vmat_ip": np.zeros((3, nao, nao)),
        "vmat_fxc": np.zeros((natm, 3, nao, nao)),
        "vmat_vxc": np.zeros((natm, 3, nao, nao)),
        "vmat_deriv1": np.zeros((natm, 3, nao, nao)),
    }
    for batch_idx, (atm_idx, start, end) in enumerate(batches):
        if verbose:
            print(f"Processing batch {batch_idx + 1}/{len(batches)}: grids {start} to {end}")
        coords_batch = coords[start:end]
        weights_batch = weights[start:end]
        quadrature_weights_batch = quadrature_weights[start:end]
        result_batch = make_hessian_setup_batch(
            mol,
            xc,
            coords_batch,
            weights_batch,
            dm0,
            atm_idx,
            quadrature_weights_batch,
            adjustment_factor,
            hardness=hardness,
            atm_list=atm_list,
            verbose=False,
        )
        for key in ["de_vxc_diag", "de_vxc_off", "de_fxc", "vmat_ip", "vmat_fxc", "vmat_vxc", "vmat_deriv1"]:
            result_sum[key] += result_batch[key]
        for key in ["de_becke_full_1", "de_becke_full_2"]:
            result_sum[key] += result_batch[key]
        for key in ["de_becke_atom_1", "de_becke_atom_2", "de_becke_atom_expensive_4"]:
            result_sum[key][atm_idx] += result_batch[key]
        for key in ["de_becke_atom_3", "de_becke_atom_expensive_5"]:
            result_sum[key][atm_idx, atm_idx] += result_batch[key]

    # symmetrize on the atom indices for the becke parts
    for key in ["de_becke_full_1", "de_becke_atom_1", "de_becke_atom_2", "de_becke_atom_expensive_4"]:
        result_sum[key] += result_sum[key].transpose(1, 0, 3, 2)

    result_sum["de_xc_skeleton_no_becke"] = result_sum["de_vxc_diag"] + result_sum["de_vxc_off"] + result_sum["de_fxc"]
    result_sum["de_xc_skeleton"] = (
        result_sum["de_xc_skeleton_no_becke"]
        + result_sum["de_becke_full_1"]
        + result_sum["de_becke_full_2"]
        + result_sum["de_becke_atom_1"]
        + result_sum["de_becke_atom_2"]
        + result_sum["de_becke_atom_3"]
        + result_sum["de_becke_atom_expensive_4"]
        + result_sum["de_becke_atom_expensive_5"]
    )
    result_sum["de_xc_deriv1_ao"] = result_sum["vmat_deriv1"]
    return result_sum


def get_ks_response_bra_naive(
    mol: gto.Mole,
    grids,
    xc: str,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    dm0: np.ndarray,
    bra: np.ndarray,
    rho_cached=None,
    vxc_cached=None,
    fxc_cached=None,
) -> np.ndarray:
    """Apply the DFT XC fxc kernel to a perturbed bra (the U-coefficient
    half-transformed back to AO), returning the response on the same shape.

    Delegates the actual fxc contraction to ``numint.NumInt.nr_rks_fxc`` —
    the same routine PySCF uses inside ``_gen_rhf_response``.  This wrapper
    handles the bra-side reshape (``[..., nao, nocc]``) and the symmetric
    dm1 build.

    Parameters
    ----------
    mol : gto.Mole
        Molecule.
    grids : pyscf.dft.Grids
        Built grids (with ``coords`` / ``weights``).
    xc : str
        XC functional name.
    mo_coeff : np.ndarray
        MO coefficients, shape ``[nao, nmo]``.
    mo_occ : np.ndarray
        MO occupations, shape ``[nmo]``.
    dm0 : np.ndarray
        Reference density matrix, shape ``[nao, nao]``.
    bra : np.ndarray
        Perturbed bra, shape ``[..., nao, nocc]``.
    rho_cached, vxc_cached, fxc_cached : optional
        Pre-computed kernel returned by ``ni.cache_xc_kernel``; passed
        through to ``nr_rks_fxc`` to avoid re-evaluating the on-grid
        density and functional derivatives on every call.

    Returns
    -------
    resp_bra : np.ndarray
        Response on the same shape as ``bra``.
    """
    nao = mol.nao
    occidx = mo_occ > 1e-15
    mocc = mo_coeff[:, occidx]
    nocc = mocc.shape[-1]

    bra_shape = bra.shape
    assert bra_shape[-2] == nao
    assert bra_shape[-1] == nocc
    bra = bra.reshape(-1, nao, nocc)

    # Symmetric perturbed density matrix.  The factor 2 here matches
    # PySCF's `hessian.rhf.gen_vind` convention (`dm = mo_coeff @ x*2 @ mocc.T`)
    # — it is the closed-shell spin sum, and is also the factor RIJK's
    # `get_rijk_response_bra_naive` absorbs into its 4x J coefficient.
    dm1 = 2 * (bra @ mocc.T)
    dm1 = dm1 + dm1.swapaxes(-1, -2)

    ni = dft.numint.NumInt()
    v1 = ni.nr_rks_fxc(
        mol,
        grids,
        xc,
        dm0,
        dm1,
        hermi=1,
        rho0=rho_cached,
        vxc=vxc_cached,
        fxc=fxc_cached,
    )
    resp_bra = v1 @ mocc
    return resp_bra.reshape(bra_shape)
