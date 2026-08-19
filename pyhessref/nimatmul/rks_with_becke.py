"""
RKS Hessian skeleton ingredients with the Becke grid-shift contribution.

Computes the DFT part of the RKS Hessian decomposition in grid batches,
adding the ``de_becke_*`` grid-shift terms (from the Becke partition
weights moving with the atoms) that restore translational invariance of
the skeleton Hessian, and the analogous f1ao-level ``vmat_becke_*`` terms
for the skeleton Vxc Fock derivative (``vmat_deriv1_grid``).  Only the
Becke partitioning scheme is supported.
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

# Second-order AO derivative components, indexed [t][s] (d^2/dt ds).
IDX_AO_DERIV2 = [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
# Third-order AO derivative triples for the 6 symmetric pairs (xx, xy, xz, yy, yz, zz).
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
# Symmetric Cartesian pair (t, s) -> the 6-component storage index (xx, xy, xz, yy, yz, zz).
IDX_PAIR_TS = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])

XC_NVAR = {"LDA": 1, "GGA": 4, "MGGA": 5}  # number of rho components (channels)
XC_AO_DERIV = {"LDA": 2, "GGA": 3, "MGGA": 3}  # AO derivative order required
XC_NCOMP_AO_DM0 = {"LDA": 1, "GGA": 4, "MGGA": 4}  # ao_dm0 channels (value + 3 gradients)


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


def _make_dao_vxc_diag(xc_type, ao, ao_dm0, wv, nao):
    """Build the AO-resolved diagonal vxc kernel ``dao_vxc_diag[6, nao]``.

    The 6 components are the symmetric Cartesian pairs
    ``(xx, xy, xz, yy, yz, zz)``.  Both the same-atom Hessian block
    ``_de_vxc_diag`` and the grid-shift part ``_de_becke_vxc_parts``
    contract this same kernel, so it is built once per batch and shared.

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
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    dao_vxc_diag : np.ndarray
        Diagonal vxc kernel, shape ``[6, nao]``.
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

    return dao_vxc_diag


def _de_vxc_diag(dao_vxc_diag, aoslices, natm):
    """Same-atom (A == B) block of the XC skeleton 2nd derivative.

    Sums ``dao_vxc_diag`` (from ``_make_dao_vxc_diag``) over each atom's
    on-atom AO slice and re-expands the 6 symmetric Cartesian pairs into a
    dense ``(3, 3)`` block per atom.

    Parameters
    ----------
    dao_vxc_diag : np.ndarray
        Diagonal vxc kernel from ``_make_dao_vxc_diag``, shape ``[6, nao]``.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]`` — only the last two columns
        ``[p0, p1)`` are used for the AO range of each atom.
    natm : int
        Number of atoms in the (possibly restricted) atom list.

    Returns
    -------
    de_vxc_diag : np.ndarray
        Same-atom block of the XC skeleton 2nd derivative, shape
        ``[natm, natm, 3, 3]`` — only the diagonal ``A == B`` blocks are
        non-zero (off-diagonal blocks are produced by ``_de_vxc_off``).
    """
    de_vxc_diag = np.zeros((natm, natm, 6))
    for A in range(natm):
        _, _, p0A, p1A = aoslices[A]
        de_vxc_diag[A, A] = np.einsum("Au -> A", dao_vxc_diag[:, p0A:p1A])
    return de_vxc_diag[:, :, IDX_PAIR_TS]


def _make_dao_vxc_off(xc_type, ao, wv, nao):
    """Build the AO-resolved two-index vxc kernel ``dao_vxc_off[3, 3, nao, nao]``.

    Both bra and ket retain their AO indices; the kernel is symmetrised
    under ``[t, s, mu, nu] -> [s, t, nu, mu]``.  Both the two-atom Hessian
    block ``_de_vxc_off`` and the grid-shift part ``_de_becke_vxc_parts``
    contract this same kernel, so it is built once per batch and shared.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]``.  Indices 0..3 are always read; GGA/MGGA
        also reads the 2nd-order channels (indices 4..9).
    wv : np.ndarray
        Weight-times-vxc, shape ``[nvar, ngrids]``.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    dao_vxc_off : np.ndarray
        Two-index vxc kernel, shape ``[3, 3, nao, nao]``, symmetrised under
        ``[t, s, mu, nu] -> [s, t, nu, mu]``.
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
    return dao_vxc_off


def _de_vxc_off(dao_vxc_off, dm0, aoslices, natm):
    """Two-atom (A != B) block of the XC skeleton 2nd derivative.

    Contracts each ``(A, B)`` block of ``dao_vxc_off`` (from
    ``_make_dao_vxc_off``) with the corresponding ``dm0[B, A]`` AO slice.

    Parameters
    ----------
    dao_vxc_off : np.ndarray
        Two-index vxc kernel from ``_make_dao_vxc_off``, shape
        ``[3, 3, nao, nao]``.
    dm0 : np.ndarray
        Density matrix in AO basis, shape ``[nao, nao]``.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]``.
    natm : int
        Number of atoms in the (possibly restricted) atom list.

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
    """Grid-weight parts of the Becke grid-shift Hessian (notebook t1/t2).

    The Hessian analogue of the grid-shift gradient's ``T1`` term: the
    grid-weight factor ``w_g`` of the XC energy is differentiated instead
    of the integrand.  Both parts are "full" in the sense that every grid
    of the batch contributes to every ``(A, B)`` entry (no grid-atom
    masking), so they accumulate across batches by a plain sum;
    ``de_becke_full_1`` still needs the ``(A, t) <-> (B, s)`` symmetrisation
    applied by ``make_hessian_setup``.

    Parameters
    ----------
    dw : np.ndarray
        First Becke-weight derivative, shape ``[natm, 3, ngrids]``.
    ddw : np.ndarray
        Second Becke-weight derivative, shape
        ``[natm, 3, natm, 3, ngrids]``.
    exc : np.ndarray
        On-grid XC energy per particle, shape ``[ngrids]``.
    vxc : np.ndarray
        First functional derivative, shape ``[nvar, ngrids]``.
    rho : np.ndarray
        On-grid density components, shape ``[nvar, ngrids]`` — only the
        value channel ``rho[0]`` is read (by ``de_becke_full_2``).
    drho : np.ndarray
        Skeleton derivative of rho components (output of ``_make_drho``),
        shape ``[natm, 3, nvar, ngrids]``.

    Returns
    -------
    dict[str, np.ndarray]
        - ``"de_becke_full_1"`` (t1) : ``dw`` contracted with ``vxc`` and
          the skeleton derivative ``drho`` — the weight-gradient term
          differentiated through the AO basis following atom B, shape
          ``[natm, natm, 3, 3]``.
        - ``"de_becke_full_2"`` (t2) : ``ddw`` contracted with the on-grid
          XC energy density ``exc * rho[0]`` — the pure second-order weight
          term, naturally symmetric under ``(A, t) <-> (B, s)`` (by equality
          of mixed partials), shape ``[natm, natm, 3, 3]``.
    """
    t1 = np.einsum("Atg, xg, Bsxg -> ABts", dw, vxc, drho, optimize=True)
    t2 = np.einsum("AtBsg, g, g -> ABts", ddw, exc, rho[0], optimize=True)
    return {"de_becke_full_1": t1, "de_becke_full_2": t2}


def _de_becke_atom_parts(w, dw, vxc, fxc, drho, prho):
    """Grid-atom parts of the Becke grid-shift Hessian (notebook t3/t5/t6).

    The ``dT2`` terms that remain after the t4/t7 -> t8/t9 substitution:
    each contracts the total skeleton derivative ``prho`` (= ``drho.sum(axis=0)``,
    the density response to all atoms moving together) against the
    functional kernel, evaluated on the grids of one atom only (the
    batch's grid atom).  ``make_hessian_setup`` therefore accumulates
    ``de_becke_atom_1/2`` into the ``atm_idx`` row and ``de_becke_atom_3``
    into the ``[atm_idx, atm_idx]`` diagonal block.

    Parameters
    ----------
    w : np.ndarray
        Grid weights of the batch, shape ``[ngrids]``.
    dw : np.ndarray
        First Becke-weight derivative, shape ``[natm, 3, ngrids]``.
    vxc : np.ndarray
        First functional derivative, shape ``[nvar, ngrids]``.
    fxc : np.ndarray
        Second functional derivative, shape ``[nvar, nvar, ngrids]``.
    drho : np.ndarray
        Skeleton derivative of rho components (output of ``_make_drho``),
        shape ``[natm, 3, nvar, ngrids]``.
    prho : np.ndarray
        Total skeleton derivative ``drho.sum(axis=0)``, shape
        ``[3, nvar, ngrids]``.

    Returns
    -------
    dict[str, np.ndarray]
        - ``"de_becke_atom_1"`` (t3) : ``w * fxc`` contracted between
          ``prho`` and the per-atom ``drho[B]``, shape ``[natm, 3, 3]``
          (row for the batch's grid atom).
        - ``"de_becke_atom_2"`` (t5) : ``dw`` contracted with ``vxc`` and
          ``prho``, shape ``[natm, 3, 3]`` (row).
        - ``"de_becke_atom_3"`` (t6) : ``w * fxc`` contracted between
          ``prho`` and itself, shape ``[3, 3]`` (diagonal block).

    Note the in-body variables ``t1/t2/t3`` are numbered by the return-key
    order (1/2/3), which is shifted from the notebook terms above.
    """
    t1 = -np.einsum("g, txg, xyg, Bsyg -> Bts", w, prho, fxc, drho, optimize=True)
    t2 = -np.einsum("Bsg, xg, txg -> Bts", dw, vxc, prho, optimize=True)
    t3 = np.einsum("g, xyg, syg, txg -> ts", w, fxc, prho, prho, optimize=True)
    return {
        "de_becke_atom_1": t1,
        "de_becke_atom_2": t2,
        "de_becke_atom_3": t3,
    }


def _contract_pvxc(pvxc, atm_idx, aoslices, natm):
    """Contract a per-grid-atom skeleton-Vxc kernel into Hessian blocks.

    Per-batch (single grid atom ``atm_idx``) form of the notebook's
    ``contract_pvxc``: the full-AO sum of ``pvxc`` enters the ``A == B``
    block, the per-atom AO-slice sums enter every ``B`` column of row
    ``atm_idx``, and the row is symmetrised under ``(A, t) <-> (B, s)`` —
    the symmetrisation that accounts for the grid moving with the atoms.

    Parameters
    ----------
    pvxc : np.ndarray
        Per-grid-atom skeleton Vxc kernel, shape ``[3, 3, nao]``.
    atm_idx : int
        Grid atom the current batch belongs to.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]``.
    natm : int
        Number of atoms in the (possibly restricted) atom list.

    Returns
    -------
    de_pvxc : np.ndarray
        Hessian blocks, shape ``[natm, natm, 3, 3]`` — only row ``atm_idx``
        and its transpose are non-zero.
    """
    de_pvxc = np.zeros((natm, natm, 3, 3))
    de_pvxc[atm_idx, atm_idx] += np.einsum("tsu -> ts", pvxc)
    for B in range(natm):
        _, _, p0B, p1B = aoslices[B]
        de_pvxc[atm_idx, B] -= 2 * np.einsum("tsu -> ts", pvxc[:, :, p0B:p1B])
    de_pvxc += de_pvxc.transpose(1, 0, 3, 2)
    return de_pvxc


def _de_becke_vxc_parts(xc_type, dao_vxc_diag, dao_vxc_off, dm0, atm_idx, aoslices, natm):
    """vxc-kernel form of the grid-shift terms t8/t9 (basis form of t4/t7).

    Substitutes the previous ``de_becke_atom_expensive_4/5`` (notebook
    t4/t7): contracting the per-grid-atom vxc kernels with the density
    reproduces the same total (``t8 + t9 == t4 + t7``) without building the
    second-order skeleton density derivatives ``pdrho``/``pprho``.  With
    this substitution the grid-shift decomposition needs no ``d2rho`` at
    all — only ``drho``, which enters the other becke parts.

    Since a batch holds the grids of the single atom ``atm_idx``, the
    batch-wide kernels ``dao_vxc_diag``/``dao_vxc_off`` (shared with
    ``de_vxc_diag``/``de_vxc_off``) are exactly the per-grid-atom masked
    kernels of the notebook's t8/t9.

    Parameters
    ----------
    xc_type : str
        One of ``"GGA"``, ``"MGGA"`` — the LDA case is not handled.
    dao_vxc_diag : np.ndarray
        Diagonal vxc kernel from ``_make_dao_vxc_diag``, shape ``[6, nao]``.
    dao_vxc_off : np.ndarray
        Two-index vxc kernel from ``_make_dao_vxc_off``, shape
        ``[3, 3, nao, nao]``.
    dm0 : np.ndarray
        Density matrix in AO basis, shape ``[nao, nao]``.
    atm_idx : int
        Grid atom the current batch belongs to.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]``.
    natm : int
        Number of atoms in the (possibly restricted) atom list.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:

        - ``"de_becke_vxc_diag"`` (t8) : from ``0.5 * dao_vxc_diag`` expanded
          to dense ``(3, 3)`` pairs, shape ``[natm, natm, 3, 3]``.
        - ``"de_becke_vxc_off"`` (t9) : from ``0.5 * dao_vxc_off`` contracted
          with ``dm0`` on the ket AO index, shape ``[natm, natm, 3, 3]``.
    """
    if xc_type == "LDA":
        raise NotImplementedError

    pvxc_diag = 0.5 * dao_vxc_diag[IDX_PAIR_TS]
    pvxc_off = 0.5 * np.einsum("tsuv, uv -> tsu", dao_vxc_off, dm0, optimize=True)
    return {
        "de_becke_vxc_diag": _contract_pvxc(pvxc_diag, atm_idx, aoslices, natm),
        "de_becke_vxc_off": _contract_pvxc(pvxc_off, atm_idx, aoslices, natm),
    }


def _de_becke_atom_expensive(xc_type, ao, ao_dm0, w, vxc, aoslices, natm, ngrids):
    """Reference form of the grid-shift terms t4/t7 (NOT used in evaluation).

    The d2rho-based counterpart of ``_de_becke_vxc_parts``: builds the
    second-order skeleton density derivatives ``pdrho``/``pprho``
    explicitly and contracts them with ``vxc``.  Superseded by the basis
    form t8/t9 in ``make_hessian_setup_batch``; kept only as a reference
    for validating ``t8 + t9 == t4 + t7``.

    Parameters
    ----------
    xc_type : str
        One of ``"GGA"``, ``"MGGA"`` — the LDA case is not handled.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]`` with 3rd-order channels (indices 0..19).
    ao_dm0 : np.ndarray
        Pre-contracted ``ao @ dm0``, shape ``[ncomp_dm0, ngrids, nao]``;
        the MGGA tau part reads the 2nd-order channels (indices up to 9).
    w : np.ndarray
        Grid weights of the batch, shape ``[ngrids]``.
    vxc : np.ndarray
        First functional derivative, shape ``[nvar, ngrids]``.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]``.
    natm : int
        Number of atoms in the (possibly restricted) atom list.
    ngrids : int
        Number of grids in the batch.

    Returns
    -------
    dict[str, np.ndarray]
        - ``"de_becke_atom_expensive_4"`` (t4) : ``w * vxc`` contracted with
          the per-atom ``pdrho[B]``, shape ``[natm, 3, 3]`` (row for the
          batch's grid atom; transpose-symmetrise after accumulation).
        - ``"de_becke_atom_expensive_5"`` (t7) : ``w * vxc`` contracted with
          the total ``pprho``, shape ``[3, 3]`` (diagonal block).
    """
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


def _vxc_fock(xc_type, ao, veff, wg):
    """Symmetric Vxc-style Fock from a generic weight and functional field.

    The standard on-grid Vxc matrix build (``nr_vxc`` convention, 0.5
    factors), with the grid weights ``wg`` and the "vxc-like" functional
    field ``veff`` as free inputs — used by ``_vmat_becke_parts`` for the
    weight part (weights = Becke ``dw``) and the fxc part (veff = fxc-folded
    density derivative) of the f1ao grid-shift.

    Parameters
    ----------
    xc_type : str
        One of ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]`` — only the value/gradient channels
        (indices 0..3) are read.
    veff : np.ndarray
        Functional-derivative field, shape ``[nvar, ngrids]``.
    wg : np.ndarray
        Weight field, shape ``[ngrids]``.

    Returns
    -------
    vxc_fock : np.ndarray
        Symmetric Vxc-style Fock matrix, shape ``[nao, nao]``.
    """
    wv = wg * veff
    wv[O] *= 0.5
    aow = np.einsum("xg, xgu -> gu", wv[:4], ao[:4])
    aow_ao = aow.T @ ao[O]
    vxc_fock = aow_ao + aow_ao.T
    if xc_type == "MGGA":
        wv[4] *= 0.5
        for j in range(1, 4):
            aow = wv[4][:, None] * ao[j]
            vxc_fock += aow.T @ ao[j]
    return vxc_fock


def _vmat_becke_parts(xc_type, ao, vxc, fxc, prho, w, dw, vmat_ip, atm_idx, natm, nao):
    """f1ao-level Becke grid-shift parts of the Vxc Fock derivative (T1/T2).

    First-derivative analogue of the ``de_becke_*`` Hessian terms: the
    grid-shift increment ``Delta = T1 + T2`` that restores translational
    invariance of the skeleton Vxc Fock derivative ``vmat_deriv1`` (the DFT
    part of the CP-KS right-hand side f1ao), i.e. ``sum_A
    vmat_deriv1_grid[A] ~ 0`` for ``vmat_deriv1_grid = vmat_deriv1 +
    Delta`` (assembled by ``make_hessian_setup``).

    - ``vmat_becke_T1`` (weight part) : Vxc-style Fock built with the
      Becke-weight derivative ``dw[A, t]`` as the weight field.  Every grid
      of the batch contributes to every atom's row, so it accumulates
      across batches by a plain sum.
    - ``vmat_becke_T2_ipip`` (functional part, ipip) : the batch's gradient
      Vxc matrix ``vmat_ip`` symmetrised in AO — the batch holds one
      atom's grids, so ``vmat_ip`` already is the per-grid-atom
      ``vmat_ip_A`` of the notebook.
    - ``vmat_becke_T2_fxc`` (functional part, fxc) : the fxc kernel folded
      with the total spatial density derivative ``prho`` (=
      ``drho.sum(axis=0)``; in this module's ``_make_drho`` convention
      ``prho = -d rho / d r``, hence the leading minus), contracted as a
      Vxc-style Fock on the batch weights.

    Parameters
    ----------
    xc_type : str
        One of ``"GGA"``, ``"MGGA"`` — the LDA case is not handled.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]`` — only channels 0..3 are read.
    vxc : np.ndarray
        First functional derivative, shape ``[nvar, ngrids]``.
    fxc : np.ndarray
        Second functional derivative, shape ``[nvar, nvar, ngrids]``.
    prho : np.ndarray
        Total skeleton density derivative ``drho.sum(axis=0)``, shape
        ``[3, nvar, ngrids]``.
    w : np.ndarray
        Grid weights of the batch, shape ``[ngrids]``.
    dw : np.ndarray
        First Becke-weight derivative, shape ``[natm, 3, ngrids]``.
    vmat_ip : np.ndarray
        Gradient-level Vxc matrix of the batch from ``_vmat_ip``, shape
        ``[3, nao, nao]``.
    atm_idx : int
        Atom that generated the batch's grids.
    natm : int
        Number of atoms in the (possibly restricted) atom list.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    dict[str, np.ndarray]
        The three parts above, each of shape ``[natm, 3, nao, nao]``:
        ``vmat_becke_T1`` filled on all rows, the T2 parts only on row
        ``atm_idx`` (accumulated there by ``make_hessian_setup``).
    """
    if xc_type == "LDA":
        raise NotImplementedError

    vmat_becke_T1 = np.zeros((natm, 3, nao, nao))
    for A in range(natm):
        for t in range(3):
            vmat_becke_T1[A, t] = _vxc_fock(xc_type, ao, vxc, dw[A, t])

    vmat_becke_T2_ipip = np.zeros((natm, 3, nao, nao))
    for t in range(3):
        vmat_becke_T2_ipip[atm_idx, t] = vmat_ip[t] + vmat_ip[t].T

    vmat_becke_T2_fxc = np.zeros((natm, 3, nao, nao))
    for t in range(3):
        fxc_prho = np.einsum("xyg, yg -> xg", fxc, prho[t], optimize=True)
        vmat_becke_T2_fxc[atm_idx, t] = _vxc_fock(xc_type, ao, -fxc_prho, w)

    return {
        "vmat_becke_T1": vmat_becke_T1,
        "vmat_becke_T2_ipip": vmat_becke_T2_ipip,
        "vmat_becke_T2_fxc": vmat_becke_T2_fxc,
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
    ``de_vxc_diag + de_vxc_off + de_fxc`` plus the Becke grid-shift parts
    (``de_becke_full_1/2``, ``de_becke_atom_1/2/3``, ``de_becke_vxc_diag``,
    ``de_becke_vxc_off``).

    Parameters
    ----------
    mol : gto.Mole
        Molecule, used for AO slices and the AO basis dimension.
    xc : str
        XC functional name, e.g. ``"SVWN"`` (LDA), ``"B3LYP"`` (GGA), or
        ``"TPSS0"`` (MGGA).
    coords : np.ndarray
        Grid point coordinates, shape ``[ngrids, 3]`` — the grids of one
        call must all belong to the single atom ``atm_idx``.
    weights : np.ndarray
        Grid weights, shape ``[ngrids]``.
    dm0 : np.ndarray
        Reference density matrix in AO basis, shape ``[nao, nao]``.
    atm_idx : int
        Atom that generated the batch's grids (the Becke cell centre); all
        grid-atom-resolved parts are reported in this atom's row/diagonal.
    quadrature_weights : np.ndarray
        Original (pre-partition) quadrature weights, shape ``[ngrids]``.
    adjustment_factor : np.ndarray
        Anti-symmetric Becke radii-adjustment table, shape ``[natm, natm]``.
    hardness : int, optional
        Becke switch-function iteration count.  Defaults to 3.
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
        - ``de_becke_full_1/2``, ``de_becke_atom_1/2``,
          ``de_becke_vxc_diag/off`` : Becke grid-shift contributions
          (notebook terms t1/t2, t3/t5, t8/t9), each of shape
          ``[natm, natm, 3, 3]`` with only the ``atm_idx`` row (and its
          transpose, where applicable) populated.
        - ``de_becke_atom_3`` : diagonal-only grid-shift contribution
          (notebook t6), shape ``[natm, natm, 3, 3]``.
        - ``vmat_ip``     : gradient-level Vxc, shape ``[3, nao, nao]``.
        - ``vmat_vxc``/``vmat_fxc`` : the two pieces of ``vmat_deriv1``.
        - ``vmat_deriv1`` : per-atom skeleton derivative of the Vxc Fock
          matrix, shape ``[natm, 3, nao, nao]``, assembled across the AO
          axes (bra + ket).
        - ``vmat_becke_T1``/``vmat_becke_T2_ipip``/``vmat_becke_T2_fxc`` :
          f1ao-level grid-shift parts (T1 on all rows, T2 parts on the
          ``atm_idx`` row only), each of shape ``[natm, 3, nao, nao]``.
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
    dao_vxc_diag = _make_dao_vxc_diag(xc_type, ao, ao_dm0, wv, nao)
    de_vxc_diag = _de_vxc_diag(dao_vxc_diag, aoslices, natm)
    tic("de_vxc_diag", t0)

    t0 = time.time()
    dao_vxc_off = _make_dao_vxc_off(xc_type, ao, wv, nao)
    de_vxc_off = _de_vxc_off(dao_vxc_off, dm0, aoslices, natm)
    tic("de_vxc_off", t0)

    t0 = time.time()
    de_becke_vxc_parts = _de_becke_vxc_parts(xc_type, dao_vxc_diag, dao_vxc_off, dm0, atm_idx, aoslices, natm)
    tic("de_becke_vxc_parts", t0)

    t0 = time.time()
    vmat_ip = _vmat_ip(xc_type, ao, wv, nao)
    tic("vmat_ip", t0)

    t0 = time.time()
    vmat = _vmat_deriv1(xc_type, ao, drho, wf, vmat_ip, aoslices, natm, nao)
    tic("vmat_deriv1", t0)

    t0 = time.time()
    vmat_becke_parts = _vmat_becke_parts(xc_type, ao, vxc, fxc, prho, weights, dw, vmat_ip, atm_idx, natm, nao)
    tic("vmat_becke_parts", t0)

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
    results.update(de_becke_vxc_parts)
    results.update(vmat_becke_parts)
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
                batches.append((A, start, next_end))
                start = next_end
        else:
            batches.append((A, start, end))
            start = end
    return batches


def get_quad_split(atm_indices: np.ndarray):
    """Cumulative grid boundaries per atom, from the per-grid atom indices.

    Grids built with ``sort_grids=False`` are grouped by atom: each atom
    index appears as exactly one contiguous run.  The boundaries are the
    positions where the atom index changes.

    Parameters
    ----------
    atm_indices : np.ndarray
        Per-grid atom index, shape ``[ngrids]``, grouped by atom (e.g.
        ``grids.atm_idx``).

    Returns
    -------
    atm_quad_split : list[int]
        Cumulative grid counts, shape ``[natm + 1]``; atom A's grids are
        ``[atm_quad_split[A], atm_quad_split[A + 1])``.  Example:
        ``[0, 0, 1, 1, 1, 2, 2] -> [0, 2, 5, 7]``.
    """
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
    """Batched driver for all DFT skeleton ingredients with the grid-shift.

    Splits the grid into atom-respecting batches of at most ``nbatch_grids``
    grids (``quad_split_by_atom``), evaluates ``make_hessian_setup_batch``
    on each, and accumulates the per-batch results into full arrays:

    - full-grid quantities (``de_vxc_diag``, ``de_vxc_off``, ``de_fxc``,
      ``vmat_*``, ``de_becke_full_1/2``, ``de_becke_vxc_diag/off``,
      ``vmat_becke_T1``) are summed over all batches;
    - grid-atom quantities (``de_becke_atom_1/2``,
      ``vmat_becke_T2_ipip/fxc``) accumulate into the batch's ``atm_idx``
      row;
    - ``de_becke_atom_3`` accumulates into the ``[atm_idx, atm_idx]``
      diagonal block.

    ``de_becke_full_1`` and the ``de_becke_atom_1/2`` rows are then
    symmetrised under ``(A, t) <-> (B, s)`` — this also accounts for the
    skeleton gradient moving with the grid (``de_becke_vxc_diag/off`` are
    already symmetrised per batch inside ``_contract_pvxc``).

    Parameters
    ----------
    mol : gto.Mole
        Molecule.
    xc : str
        XC functional name (the grid-shift parts handle GGA/MGGA, not LDA).
    coords : np.ndarray
        Grid point coordinates, shape ``[ngrids, 3]``, grouped by atom.
    weights : np.ndarray
        Grid weights, shape ``[ngrids]``.
    dm0 : np.ndarray
        Reference density matrix in AO basis, shape ``[nao, nao]``.
    atm_quad_split : list[int]
        Cumulative grid counts per atom from ``get_quad_split``, shape
        ``[natm + 1]``.
    quadrature_weights : np.ndarray
        Original (pre-partition) quadrature weights, shape ``[ngrids]``.
    adjustment_factor : np.ndarray
        Anti-symmetric Becke radii-adjustment table, shape ``[natm, natm]``.
    hardness : int, optional
        Becke switch-function iteration count.  Defaults to 3.
    atm_list : list[int], optional
        Subset of atom indices for the per-atom outputs.  Defaults to all
        atoms.
    nbatch_grids : int, optional
        Maximum number of grids per batch; an atom's grids are split into
        several batches when they exceed it.  Defaults to 16384.
    verbose : bool, optional
        When True, print per-batch progress.  Defaults to True.

    Returns
    -------
    result : dict[str, np.ndarray]
        All keys of ``make_hessian_setup_batch`` accumulated over batches,
        plus:

        - ``de_xc_skeleton_no_becke`` : grid-fixed XC skeleton Hessian
          (``de_vxc_diag + de_vxc_off + de_fxc``).
        - ``de_xc_skeleton`` : with all ``de_becke_*`` grid-shift parts
          added; translationally invariant (sum over (A, B) ~1e-13).
        - ``de_xc_deriv1_ao`` : alias of ``vmat_deriv1``.
        - ``vmat_deriv1_grid`` : ``vmat_deriv1`` with the f1ao-level
          grid-shift increment (``vmat_becke_*`` parts) added;
          translationally invariant (sum over A ~1e-12).
    """
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
        "de_becke_vxc_diag": np.zeros((natm, natm, 3, 3)),
        "de_becke_vxc_off": np.zeros((natm, natm, 3, 3)),
        "de_vxc_diag": np.zeros((natm, natm, 3, 3)),
        "de_vxc_off": np.zeros((natm, natm, 3, 3)),
        "de_fxc": np.zeros((natm, natm, 3, 3)),
        "vmat_ip": np.zeros((3, nao, nao)),
        "vmat_fxc": np.zeros((natm, 3, nao, nao)),
        "vmat_vxc": np.zeros((natm, 3, nao, nao)),
        "vmat_deriv1": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T1": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T2_ipip": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T2_fxc": np.zeros((natm, 3, nao, nao)),
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
        for key in [
            "de_vxc_diag",
            "de_vxc_off",
            "de_fxc",
            "vmat_ip",
            "vmat_fxc",
            "vmat_vxc",
            "vmat_deriv1",
            "de_becke_full_1",
            "de_becke_full_2",
            "de_becke_vxc_diag",
            "de_becke_vxc_off",
            "vmat_becke_T1",
        ]:
            result_sum[key] += result_batch[key]
        for key in ["de_becke_atom_1", "de_becke_atom_2"]:
            result_sum[key][atm_idx] += result_batch[key]
        for key in ["vmat_becke_T2_ipip", "vmat_becke_T2_fxc"]:
            result_sum[key][atm_idx] += result_batch[key][atm_idx]
        for key in ["de_becke_atom_3"]:
            result_sum[key][atm_idx, atm_idx] += result_batch[key]

    # symmetrize on the atom indices for the becke parts;
    # de_becke_vxc_diag/off are already symmetrized per batch (in _contract_pvxc)
    for key in ["de_becke_full_1", "de_becke_atom_1", "de_becke_atom_2"]:
        result_sum[key] += result_sum[key].transpose(1, 0, 3, 2)

    result_sum["de_xc_skeleton_no_becke"] = result_sum["de_vxc_diag"] + result_sum["de_vxc_off"] + result_sum["de_fxc"]
    result_sum["de_xc_skeleton"] = (
        result_sum["de_xc_skeleton_no_becke"]
        + result_sum["de_becke_full_1"]
        + result_sum["de_becke_full_2"]
        + result_sum["de_becke_atom_1"]
        + result_sum["de_becke_atom_2"]
        + result_sum["de_becke_atom_3"]
        + result_sum["de_becke_vxc_diag"]
        + result_sum["de_becke_vxc_off"]
    )
    result_sum["de_xc_deriv1_ao"] = result_sum["vmat_deriv1"]
    # f1ao (CP-KS RHS) with the grid-shift increment Delta = T1 + T2:
    # sum_A vmat_deriv1_grid[A] ~ 0 (translational invariance).
    result_sum["vmat_deriv1_grid"] = (
        result_sum["vmat_deriv1"]
        + result_sum["vmat_becke_T1"]
        + result_sum["vmat_becke_T2_ipip"]
        + result_sum["vmat_becke_T2_fxc"]
    )
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
