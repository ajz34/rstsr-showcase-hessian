"""
UKS Hessian skeleton ingredients with the Becke grid-shift contribution.

Unrestricted sibling of ``rks_with_becke``: the single-spin helpers, the
becke partition machinery, and the grid batching driver are shared with the
RKS implementation; only the spin-coupled pieces differ.  The spin
extension of every grid-shift term is the obvious one — terms linear in
``vxc`` become a spin sum (``vxc[0]`` against the alpha quantity plus
``vxc[1]`` against the beta quantity), and terms quadratic in the ``fxc``
kernel become the four spin-pair sum (the same ``aa/ab/ba/bb`` structure
as ``uks._de_fxc_uks``).

Adds the ``de_becke_*`` grid-shift terms (from the Becke partition weights
moving with the atoms) that restore translational invariance of the
skeleton Hessian, and the analogous per-spin f1ao-level ``vmat_becke_*``
terms for the skeleton Vxc Fock derivatives (``vmat_deriv1_grid_a/b``).
Only the Becke partitioning scheme is supported.
"""

from pyscf import gto, dft
import numpy as np
import time

from pyhessref.hess_trait_unrestricted import UHessElecInteractAPI
from pyhessref.util import get_dm0_unrestricted

# Single-spin helpers and the becke machinery (grid batching, pvxc
# contraction, Vxc-style Fock build) come from the RKS implementation, and
# the spin-coupled grid-fixed pieces from the plain UKS implementation.
# This mirrors the ``uks.py`` layout (which imports the grid-fixed
# single-spin helpers from ``rks.py``).
from pyhessref.nimatmul.rks_with_becke import (
    _make_drho,
    _make_dao_vxc_diag,
    _make_dao_vxc_off,
    _de_vxc_diag,
    _de_vxc_off,
    _vmat_ip,
    _vmat_vxc,
    _vxc_fock,
    _contract_pvxc,
    quad_split_by_atom,
    get_quad_split,
    IDX_PAIR_TS,
    XC_NVAR,
    XC_AO_DERIV,
    XC_NCOMP_AO_DM0,
)
from pyhessref.nimatmul.uks import (
    _eval_rho_exc_vxc_fxc_uks,
    _make_drho_uks,
    _de_fxc_uks,
    _vmat_deriv1_uks,
    get_uks_response_bra_naive,
)


def _de_becke_full_parts_uks(dw, ddw, exc, vxc, rhoa, rhob, drhoa, drhob):
    """Grid-weight parts of the Becke grid-shift Hessian (notebook t1/t2),
    UKS spin extension.

    Same structure as ``rks_with_becke._de_becke_full_parts``: the grid
    weight ``w_g`` of the XC energy is differentiated instead of the
    integrand.  The spin extension of the energy density ``exc * rho`` is
    ``exc * (rhoa + rhob)`` (the spin-summed value channel), and the t1
    ``vxc`` contraction becomes a spin sum — ``vxc[0]`` folds against the
    alpha skeleton derivative and ``vxc[1]`` against the beta one.

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
        First functional derivative, shape ``[2, nvar, ngrids]``.
    rhoa, rhob : np.ndarray
        On-grid density components per spin, shape ``[nvar, ngrids]`` —
        only the value channels are read (by ``de_becke_full_2``).
    drhoa, drhob : np.ndarray
        Skeleton derivative of rho per spin (output of
        ``_make_drho_uks``), shape ``[natm, 3, nvar, ngrids]``.

    Returns
    -------
    dict[str, np.ndarray]
        - ``"de_becke_full_1"`` (t1) : ``dw`` contracted with the spin-summed
          ``vxc . drho`` — shape ``[natm, natm, 3, 3]``, still to be
          symmetrised under ``(A, t) <-> (B, s)`` by ``make_hessian_setup_uks``.
        - ``"de_becke_full_2"`` (t2) : ``ddw`` contracted with the spin-summed
          on-grid XC energy density — shape ``[natm, natm, 3, 3]``, naturally
          symmetric.
    """
    t1 = np.einsum("Atg, xg, Bsxg -> ABts", dw, vxc[0], drhoa, optimize=True)
    t1 += np.einsum("Atg, xg, Bsxg -> ABts", dw, vxc[1], drhob, optimize=True)
    t2 = np.einsum("AtBsg, g, g -> ABts", ddw, exc, rhoa[0] + rhob[0], optimize=True)
    return {"de_becke_full_1": t1, "de_becke_full_2": t2}


def _de_becke_atom_parts_uks(w, dw, vxc, fxc, drhoa, drhob, prhoa, prhob):
    """Grid-atom parts of the Becke grid-shift Hessian (notebook t3/t5/t6),
    UKS spin extension.

    Same structure as ``rks_with_becke._de_becke_atom_parts``: each term
    contracts the total skeleton derivative ``prho`` against the
    functional kernel, evaluated on the grids of one atom only (the
    batch's grid atom).  The t5 ``vxc`` contraction becomes a spin sum;
    the t3/t6 ``fxc`` contractions become the four spin-pair sum over
    ``(s1, s2)`` — the same coupling structure as ``uks._de_fxc_uks``.

    Parameters
    ----------
    w : np.ndarray
        Grid weights of the batch, shape ``[ngrids]``.
    dw : np.ndarray
        First Becke-weight derivative, shape ``[natm, 3, ngrids]``.
    vxc : np.ndarray
        First functional derivative, shape ``[2, nvar, ngrids]``.
    fxc : np.ndarray
        Second functional derivative, shape ``[2, nvar, 2, nvar, ngrids]``.
    drhoa, drhob : np.ndarray
        Skeleton derivative of rho per spin, shape
        ``[natm, 3, nvar, ngrids]``.
    prhoa, prhob : np.ndarray
        Total skeleton derivative per spin (``drho.sum(axis=0)``), shape
        ``[3, nvar, ngrids]``.

    Returns
    -------
    dict[str, np.ndarray]
        - ``"de_becke_atom_1"`` (t3) : ``w * fxc`` contracted between
          ``prho`` and the per-atom ``drho[B]`` over all spin pairs, shape
          ``[natm, 3, 3]`` (row for the batch's grid atom).
        - ``"de_becke_atom_2"`` (t5) : ``dw`` contracted with the spin-summed
          ``vxc . prho``, shape ``[natm, 3, 3]`` (row).
        - ``"de_becke_atom_3"`` (t6) : ``w * fxc`` contracted between ``prho``
          and itself over all spin pairs, shape ``[3, 3]`` (diagonal block).

    Note the in-body variables ``t1/t2/t3`` are numbered by the return-key
    order (1/2/3), which is shifted from the notebook terms above (same
    convention as ``rks_with_becke._de_becke_atom_parts``).
    """
    t1 = np.zeros((drhoa.shape[0], 3, 3))
    for prho_l, s1 in [(prhoa, 0), (prhob, 1)]:
        for drho_r, s2 in [(drhoa, 0), (drhob, 1)]:
            t1 -= np.einsum("g, txg, xyg, Bsyg -> Bts", w, prho_l, fxc[s1, :, s2, :], drho_r, optimize=True)

    t2 = -np.einsum("Bsg, xg, txg -> Bts", dw, vxc[0], prhoa, optimize=True)
    t2 -= np.einsum("Bsg, xg, txg -> Bts", dw, vxc[1], prhob, optimize=True)

    t3 = np.zeros((3, 3))
    for prho_l, s1 in [(prhoa, 0), (prhob, 1)]:
        for prho_r, s2 in [(prhoa, 0), (prhob, 1)]:
            t3 += np.einsum("g, xyg, syg, txg -> ts", w, fxc[s1, :, s2, :], prho_r, prho_l, optimize=True)

    return {
        "de_becke_atom_1": t1,
        "de_becke_atom_2": t2,
        "de_becke_atom_3": t3,
    }


def _de_becke_vxc_parts_uks(dao_vxc_diag_a, dao_vxc_diag_b, dao_vxc_off_a, dao_vxc_off_b, dm0a, dm0b, atm_idx, aoslices, natm):
    """vxc-kernel form of the grid-shift terms t8/t9, UKS spin extension.

    Same structure as ``rks_with_becke._de_becke_vxc_parts``, with the
    per-spin kernels (each built from its own ``wv`` weighting and
    ``ao_dm0`` contraction in ``make_hessian_setup_batch_uks``) summed
    before the ``_contract_pvxc`` scatter — the contraction is linear, so
    spin-summing first is equivalent to contracting each spin separately.

    Parameters
    ----------
    dao_vxc_diag_a, dao_vxc_diag_b : np.ndarray
        Diagonal vxc kernel per spin from ``_make_dao_vxc_diag``, shape
        ``[6, nao]``.
    dao_vxc_off_a, dao_vxc_off_b : np.ndarray
        Two-index vxc kernel per spin from ``_make_dao_vxc_off``, shape
        ``[3, 3, nao, nao]``.
    dm0a, dm0b : np.ndarray
        Per-spin density matrices, shape ``[nao, nao]``.
    atm_idx : int
        Grid atom the current batch belongs to.
    aoslices : np.ndarray
        Per-atom AO slices, shape ``[natm, 4]``.
    natm : int
        Number of atoms in the (possibly restricted) atom list.

    Returns
    -------
    dict[str, np.ndarray]
        - ``"de_becke_vxc_diag"`` (t8) : from the spin-summed
          ``0.5 * dao_vxc_diag`` expanded to dense ``(3, 3)`` pairs, shape
          ``[natm, natm, 3, 3]``.
        - ``"de_becke_vxc_off"`` (t9) : from the spin-summed
          ``0.5 * dao_vxc_off`` contracted with the matching per-spin
          ``dm0`` on the ket AO index, shape ``[natm, natm, 3, 3]``.
    """
    pvxc_diag = 0.5 * (dao_vxc_diag_a + dao_vxc_diag_b)[IDX_PAIR_TS]
    pvxc_off = 0.5 * (
        np.einsum("tsuv, uv -> tsu", dao_vxc_off_a, dm0a, optimize=True)
        + np.einsum("tsuv, uv -> tsu", dao_vxc_off_b, dm0b, optimize=True)
    )
    return {
        "de_becke_vxc_diag": _contract_pvxc(pvxc_diag, atm_idx, aoslices, natm),
        "de_becke_vxc_off": _contract_pvxc(pvxc_off, atm_idx, aoslices, natm),
    }


def _vmat_becke_parts_uks(xc_type, ao, vxc, fxc, prhoa, prhob, w, dw, vmat_ip_a, vmat_ip_b, atm_idx, natm, nao):
    """f1ao-level Becke grid-shift parts of the Vxc Fock derivatives (T1/T2),
    UKS spin extension.

    Per-spin analogue of ``rks_with_becke._vmat_becke_parts``: the
    grid-shift increment ``Delta_sigma = T1_sigma + T2_sigma`` that
    restores translational invariance of each spin's skeleton Vxc Fock
    derivative ``vmat_deriv1_sigma`` (the DFT part of the CP-KS
    right-hand side f1ao), i.e. ``sum_A vmat_deriv1_grid_sigma[A] ~ 0``.

    - ``vmat_becke_T1_a/b`` (weight part) : Vxc-style Fock built with the
      Becke-weight derivative ``dw[A, t]`` as the weight field and the
      spin's own ``vxc`` field.  Every grid of the batch contributes to
      every atom's row, so these accumulate across batches by a plain sum.
    - ``vmat_becke_T2_ipip_a/b`` (functional part, ipip) : the batch's
      per-spin gradient Vxc matrix ``vmat_ip`` symmetrised in AO, placed
      on the grid atom's row — the batch holds one atom's grids, so
      ``vmat_ip`` already is the per-grid-atom masked kernel.
    - ``vmat_becke_T2_fxc_a/b`` (functional part, fxc) : the fxc kernel
      spin-coupled and folded with the total spatial density derivative of
      BOTH spins (``fxc[sigma, :, sigma', :]`` against ``prho_sigma'``;
      leading minus from the ``prho = -d rho / d r`` convention of
      ``_make_drho``), contracted as a Vxc-style Fock on the batch
      weights.  This mirrors the ``_vmat_fxc_uks`` spin coupling.

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and its derivatives evaluated on the grid, shape
        ``[ncomp, ngrids, nao]`` — only channels 0..3 are read.
    vxc : np.ndarray
        First functional derivative, shape ``[2, nvar, ngrids]``.
    fxc : np.ndarray
        Second functional derivative, shape ``[2, nvar, 2, nvar, ngrids]``.
    prhoa, prhob : np.ndarray
        Total skeleton density derivative per spin
        (``drho_sigma.sum(axis=0)``), shape ``[3, nvar, ngrids]``.
    w : np.ndarray
        Grid weights of the batch, shape ``[ngrids]``.
    dw : np.ndarray
        First Becke-weight derivative, shape ``[natm, 3, ngrids]``.
    vmat_ip_a, vmat_ip_b : np.ndarray
        Gradient-level Vxc matrices of the batch per spin from
        ``_vmat_ip``, shape ``[3, nao, nao]``.
    atm_idx : int
        Atom that generated the batch's grids.
    natm : int
        Number of atoms in the (possibly restricted) atom list.
    nao : int
        Total number of atomic orbitals.

    Returns
    -------
    dict[str, np.ndarray]
        The six parts above (``_a``/``_b`` suffixed), each of shape
        ``[natm, 3, nao, nao]``: the T1 parts filled on all rows, the T2
        parts only on row ``atm_idx`` (accumulated there by
        ``make_hessian_setup_uks``).
    """
    vmat_becke_T1_a = np.zeros((natm, 3, nao, nao))
    vmat_becke_T1_b = np.zeros((natm, 3, nao, nao))
    for A in range(natm):
        for t in range(3):
            vmat_becke_T1_a[A, t] = _vxc_fock(xc_type, ao, vxc[0], dw[A, t])
            vmat_becke_T1_b[A, t] = _vxc_fock(xc_type, ao, vxc[1], dw[A, t])

    vmat_becke_T2_ipip_a = np.zeros((natm, 3, nao, nao))
    vmat_becke_T2_ipip_b = np.zeros((natm, 3, nao, nao))
    for t in range(3):
        vmat_becke_T2_ipip_a[atm_idx, t] = vmat_ip_a[t] + vmat_ip_a[t].T
        vmat_becke_T2_ipip_b[atm_idx, t] = vmat_ip_b[t] + vmat_ip_b[t].T

    vmat_becke_T2_fxc_a = np.zeros((natm, 3, nao, nao))
    vmat_becke_T2_fxc_b = np.zeros((natm, 3, nao, nao))
    for t in range(3):
        # Spin-coupled folding: vxc of spin sigma responds to the grid-shift
        # density change of both spins,
        # fxc_prho_sigma = sum_sigma' fxc[sigma, :, sigma', :] . prho_sigma'.
        fxc_prho_a = np.einsum("xyg, yg -> xg", fxc[0, :, 0, :], prhoa[t], optimize=True)
        fxc_prho_a += np.einsum("xyg, yg -> xg", fxc[0, :, 1, :], prhob[t], optimize=True)
        fxc_prho_b = np.einsum("xyg, yg -> xg", fxc[1, :, 0, :], prhoa[t], optimize=True)
        fxc_prho_b += np.einsum("xyg, yg -> xg", fxc[1, :, 1, :], prhob[t], optimize=True)
        vmat_becke_T2_fxc_a[atm_idx, t] = _vxc_fock(xc_type, ao, -fxc_prho_a, w)
        vmat_becke_T2_fxc_b[atm_idx, t] = _vxc_fock(xc_type, ao, -fxc_prho_b, w)

    return {
        "vmat_becke_T1_a": vmat_becke_T1_a,
        "vmat_becke_T1_b": vmat_becke_T1_b,
        "vmat_becke_T2_ipip_a": vmat_becke_T2_ipip_a,
        "vmat_becke_T2_ipip_b": vmat_becke_T2_ipip_b,
        "vmat_becke_T2_fxc_a": vmat_becke_T2_fxc_a,
        "vmat_becke_T2_fxc_b": vmat_becke_T2_fxc_b,
    }


def make_hessian_setup_batch_uks(
    mol: gto.Mole,
    xc: str,
    coords: np.ndarray,
    weights: np.ndarray,
    dm0a: np.ndarray,
    dm0b: np.ndarray,
    atm_idx: int,
    quadrature_weights: np.ndarray,
    adjustment_factor: np.ndarray,
    hardness: int = 3,
    atm_list: list[int] = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute all DFT skeleton ingredients of the UKS Hessian (with the
    Becke grid-shift) in one pass over one atom's grids.

    Mirrors ``rks_with_becke.make_hessian_setup_batch``: performs the DFT
    numerical-integration setup once (``ao``, per-spin ``rho``, ``vxc``,
    ``fxc``), then feeds it into the shared single-spin helpers
    (``de_vxc_diag``, ``de_vxc_off``, ``vmat_ip``, per spin), the
    spin-coupled pieces from ``uks.py`` (``de_fxc``, ``vmat_deriv1``), and
    the spin-extended Becke grid-shift parts (``de_becke_*``,
    ``vmat_becke_*``).

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
    dm0a, dm0b : np.ndarray
        Per-spin density matrices in AO basis, shape ``[nao, nao]``.
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

        - ``de_vxc_diag_a/b``, ``de_vxc_off_a/b`` : per-spin XC skeleton
          blocks, shape ``[natm, natm, 3, 3]``.
        - ``de_fxc`` : spin-coupled fxc-kernel contribution, shape
          ``[natm, natm, 3, 3]``.
        - ``de_becke_full_1/2``, ``de_becke_atom_1/2``,
          ``de_becke_vxc_diag/off`` : Becke grid-shift contributions
          (notebook terms t1/t2, t3/t5, t8/t9), spin-extended; each of
          shape ``[natm, natm, 3, 3]`` with only the ``atm_idx`` row (and
          its transpose, where applicable) populated.
        - ``de_becke_atom_3`` : diagonal-only grid-shift contribution
          (notebook t6), shape ``[natm, natm, 3, 3]``.
        - ``vmat_ip_a/b`` : per-spin gradient-level Vxc, shape
          ``[3, nao, nao]``.
        - ``vmat_vxc_a/b``, ``vmat_fxc_a/b`` : the two pieces of
          ``vmat_deriv1`` per spin.
        - ``vmat_deriv1_a/b`` : per-atom skeleton derivative of the
          per-spin Vxc Fock matrix, shape ``[natm, 3, nao, nao]``,
          assembled across the AO axes (bra + ket).
        - ``vmat_becke_T1_a/b``, ``vmat_becke_T2_ipip_a/b``,
          ``vmat_becke_T2_fxc_a/b`` : per-spin f1ao-level grid-shift parts
          (T1 on all rows, T2 parts on the ``atm_idx`` row only), each of
          shape ``[natm, 3, nao, nao]``.
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
    de_becke_full_parts = _de_becke_full_parts_uks(dw, ddw, exc, vxc, rhoa, rhob, drhoa, drhob)
    tic("de_becke_full_parts", t0)

    t0 = time.time()
    prhoa = drhoa.sum(axis=0)
    prhob = drhob.sum(axis=0)
    de_becke_atom_parts = _de_becke_atom_parts_uks(weights, dw, vxc, fxc, drhoa, drhob, prhoa, prhob)
    tic("de_becke_atom_parts", t0)

    t0 = time.time()
    dao_vxc_diag_a = _make_dao_vxc_diag(xc_type, ao, ao_dm0a, wva, nao)
    dao_vxc_diag_b = _make_dao_vxc_diag(xc_type, ao, ao_dm0b, wvb, nao)
    de_vxc_diag_a = _de_vxc_diag(dao_vxc_diag_a, aoslices, natm)
    de_vxc_diag_b = _de_vxc_diag(dao_vxc_diag_b, aoslices, natm)
    tic("de_vxc_diag", t0)

    t0 = time.time()
    dao_vxc_off_a = _make_dao_vxc_off(xc_type, ao, wva, nao)
    dao_vxc_off_b = _make_dao_vxc_off(xc_type, ao, wvb, nao)
    de_vxc_off_a = _de_vxc_off(dao_vxc_off_a, dm0a, aoslices, natm)
    de_vxc_off_b = _de_vxc_off(dao_vxc_off_b, dm0b, aoslices, natm)
    tic("de_vxc_off", t0)

    t0 = time.time()
    de_becke_vxc_parts = _de_becke_vxc_parts_uks(
        dao_vxc_diag_a, dao_vxc_diag_b, dao_vxc_off_a, dao_vxc_off_b, dm0a, dm0b, atm_idx, aoslices, natm
    )
    tic("de_becke_vxc_parts", t0)

    t0 = time.time()
    vmat_ip_a = _vmat_ip(xc_type, ao, wva, nao)
    vmat_ip_b = _vmat_ip(xc_type, ao, wvb, nao)
    tic("vmat_ip", t0)

    t0 = time.time()
    vmat = _vmat_deriv1_uks(
        xc_type, ao, drhoa, drhob, wf, vmat_ip_a, vmat_ip_b, aoslices, natm, nao
    )
    tic("vmat_deriv1", t0)

    t0 = time.time()
    vmat_becke_parts = _vmat_becke_parts_uks(
        xc_type, ao, vxc, fxc, prhoa, prhob, weights, dw, vmat_ip_a, vmat_ip_b, atm_idx, natm, nao
    )
    tic("vmat_becke_parts", t0)

    results = {
        "de_vxc_diag_a": de_vxc_diag_a,
        "de_vxc_diag_b": de_vxc_diag_b,
        "de_vxc_off_a": de_vxc_off_a,
        "de_vxc_off_b": de_vxc_off_b,
        "de_fxc": de_fxc,
        "vmat_ip_a": vmat_ip_a,
        "vmat_ip_b": vmat_ip_b,
        "vmat_deriv1_a": vmat["vmat_deriv1"][0],
        "vmat_deriv1_b": vmat["vmat_deriv1"][1],
        "vmat_fxc_a": vmat["vmat_fxc"][0],
        "vmat_fxc_b": vmat["vmat_fxc"][1],
        "vmat_vxc_a": vmat["vmat_vxc"][0],
        "vmat_vxc_b": vmat["vmat_vxc"][1],
    }
    results.update(de_becke_full_parts)
    results.update(de_becke_atom_parts)
    results.update(de_becke_vxc_parts)
    results.update(vmat_becke_parts)
    return results


def make_hessian_setup_uks(
    mol: gto.Mole,
    xc: str,
    coords: np.ndarray,
    weights: np.ndarray,
    dm0a: np.ndarray,
    dm0b: np.ndarray,
    atm_quad_split: list[int],
    quadrature_weights: np.ndarray,
    adjustment_factor: np.ndarray,
    hardness: int = 3,
    atm_list: list[int] = None,
    nbatch_grids: int = 16384,
    verbose: bool = True,
):
    """Batched driver for all UKS DFT skeleton ingredients with the
    grid-shift.

    Mirrors ``rks_with_becke.make_hessian_setup``: splits the grid into
    atom-respecting batches of at most ``nbatch_grids`` grids
    (``quad_split_by_atom``), evaluates ``make_hessian_setup_batch_uks``
    on each, and accumulates the per-batch results into full arrays:

    - full-grid quantities (``de_vxc_diag_a/b``, ``de_vxc_off_a/b``,
      ``de_fxc``, ``vmat_*``, ``de_becke_full_1/2``,
      ``de_becke_vxc_diag/off``, ``vmat_becke_T1_a/b``) are summed over
      all batches;
    - grid-atom quantities (``de_becke_atom_1/2``,
      ``vmat_becke_T2_ipip_a/b``, ``vmat_becke_T2_fxc_a/b``) accumulate
      into the batch's ``atm_idx`` row;
    - ``de_becke_atom_3`` accumulates into the ``[atm_idx, atm_idx]``
      diagonal block.

    ``de_becke_full_1`` and the ``de_becke_atom_1/2`` rows are then
    symmetrised under ``(A, t) <-> (B, s)`` — ``de_becke_vxc_diag/off``
    are already symmetrised per batch inside ``_contract_pvxc``.

    Parameters
    ----------
    mol : gto.Mole
        Molecule.
    xc : str
        XC functional name.
    coords : np.ndarray
        Grid point coordinates, shape ``[ngrids, 3]``, grouped by atom.
    weights : np.ndarray
        Grid weights, shape ``[ngrids]``.
    dm0a, dm0b : np.ndarray
        Per-spin density matrices in AO basis, shape ``[nao, nao]``.
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
        Maximum number of grids per batch.  Defaults to 16384.
    verbose : bool, optional
        When True, print per-batch progress.  Defaults to True.

    Returns
    -------
    result : dict[str, np.ndarray]
        All keys of ``make_hessian_setup_batch_uks`` accumulated over
        batches, plus:

        - ``de_xc_skeleton_no_becke`` : grid-fixed XC skeleton Hessian
          (``de_vxc_diag_a + de_vxc_off_a + de_vxc_diag_b + de_vxc_off_b
          + de_fxc``).
        - ``de_xc_skeleton`` : with all ``de_becke_*`` grid-shift parts
          added; translationally invariant (sum over (A, B) ~1e-13).
        - ``de_xc_deriv1_ao`` : per-spin stacked alias of
          ``vmat_deriv1_grid_a/b``.
        - ``vmat_deriv1_grid_a/b`` : per-spin ``vmat_deriv1`` with the
          f1ao-level grid-shift increment (``vmat_becke_*`` parts) added;
          translationally invariant per spin (sum over A ~1e-12).
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
        "de_vxc_diag_a": np.zeros((natm, natm, 3, 3)),
        "de_vxc_diag_b": np.zeros((natm, natm, 3, 3)),
        "de_vxc_off_a": np.zeros((natm, natm, 3, 3)),
        "de_vxc_off_b": np.zeros((natm, natm, 3, 3)),
        "de_fxc": np.zeros((natm, natm, 3, 3)),
        "vmat_ip_a": np.zeros((3, nao, nao)),
        "vmat_ip_b": np.zeros((3, nao, nao)),
        "vmat_fxc_a": np.zeros((natm, 3, nao, nao)),
        "vmat_fxc_b": np.zeros((natm, 3, nao, nao)),
        "vmat_vxc_a": np.zeros((natm, 3, nao, nao)),
        "vmat_vxc_b": np.zeros((natm, 3, nao, nao)),
        "vmat_deriv1_a": np.zeros((natm, 3, nao, nao)),
        "vmat_deriv1_b": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T1_a": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T1_b": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T2_ipip_a": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T2_ipip_b": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T2_fxc_a": np.zeros((natm, 3, nao, nao)),
        "vmat_becke_T2_fxc_b": np.zeros((natm, 3, nao, nao)),
    }
    for batch_idx, (atm_idx, start, end) in enumerate(batches):
        if verbose:
            print(f"Processing batch {batch_idx + 1}/{len(batches)}: grids {start} to {end}")
        coords_batch = coords[start:end]
        weights_batch = weights[start:end]
        quadrature_weights_batch = quadrature_weights[start:end]
        result_batch = make_hessian_setup_batch_uks(
            mol,
            xc,
            coords_batch,
            weights_batch,
            dm0a,
            dm0b,
            atm_idx,
            quadrature_weights_batch,
            adjustment_factor,
            hardness=hardness,
            atm_list=atm_list,
            verbose=False,
        )
        for key in [
            "de_vxc_diag_a",
            "de_vxc_diag_b",
            "de_vxc_off_a",
            "de_vxc_off_b",
            "de_fxc",
            "vmat_ip_a",
            "vmat_ip_b",
            "vmat_fxc_a",
            "vmat_fxc_b",
            "vmat_vxc_a",
            "vmat_vxc_b",
            "vmat_deriv1_a",
            "vmat_deriv1_b",
            "de_becke_full_1",
            "de_becke_full_2",
            "de_becke_vxc_diag",
            "de_becke_vxc_off",
            "vmat_becke_T1_a",
            "vmat_becke_T1_b",
        ]:
            result_sum[key] += result_batch[key]
        for key in ["de_becke_atom_1", "de_becke_atom_2"]:
            result_sum[key][atm_idx] += result_batch[key]
        for key in [
            "vmat_becke_T2_ipip_a",
            "vmat_becke_T2_ipip_b",
            "vmat_becke_T2_fxc_a",
            "vmat_becke_T2_fxc_b",
        ]:
            result_sum[key][atm_idx] += result_batch[key][atm_idx]
        for key in ["de_becke_atom_3"]:
            result_sum[key][atm_idx, atm_idx] += result_batch[key]

    # symmetrize on the atom indices for the becke parts;
    # de_becke_vxc_diag/off are already symmetrized per batch (in _contract_pvxc)
    for key in ["de_becke_full_1", "de_becke_atom_1", "de_becke_atom_2"]:
        result_sum[key] += result_sum[key].transpose(1, 0, 3, 2)

    result_sum["de_xc_skeleton_no_becke"] = (
        result_sum["de_vxc_diag_a"]
        + result_sum["de_vxc_off_a"]
        + result_sum["de_vxc_diag_b"]
        + result_sum["de_vxc_off_b"]
        + result_sum["de_fxc"]
    )
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
    # f1ao (CP-KS RHS) with the grid-shift increment Delta_sigma = T1 + T2:
    # sum_A vmat_deriv1_grid_sigma[A] ~ 0 (translational invariance, per spin).
    for s in ("a", "b"):
        result_sum[f"vmat_deriv1_grid_{s}"] = (
            result_sum[f"vmat_deriv1_{s}"]
            + result_sum[f"vmat_becke_T1_{s}"]
            + result_sum[f"vmat_becke_T2_ipip_{s}"]
            + result_sum[f"vmat_becke_T2_fxc_{s}"]
        )
    result_sum["de_xc_deriv1_ao"] = np.array(
        [result_sum["vmat_deriv1_grid_a"], result_sum["vmat_deriv1_grid_b"]]
    )
    return result_sum


class UHessKSNaiveBecke(UHessElecInteractAPI):
    """Naive DFT XC contribution to the UKS Hessian, with Becke grid-shift.

    The grid-shift sibling of ``uks.UHessKSNaive`` (and unrestricted
    counterpart of ``rks_with_becke.RHessKSNaiveBecke``): same interface
    and caching semantics, but ``make_skeleton_hess`` returns the skeleton
    XC Hessian with the ``de_becke_*`` grid-shift parts added
    (``de_xc_skeleton``), and ``get_deriv1_ao`` returns the per-spin
    skeleton Vxc Fock derivatives with the f1ao-level grid-shift increment
    (``vmat_deriv1_grid_a/b``).  Both are translationally invariant (sum
    over atoms ~1e-12), unlike their grid-fixed counterparts.

    The grids must be built with ``sort_grids=False`` so that grids are
    grouped by atom and ``grids.atm_idx`` / ``grids.quadrature_weights``
    are available; the Becke ``adjustment_factor`` and per-atom grid
    boundaries are prepared once in ``__init__``.

    The heavy work is done by ``make_hessian_setup_uks``, called once per
    ``(mo_coeff, mo_occ)`` and cached in ``self.result``.

    Parameters
    ----------
    mol : gto.Mole
        Molecule.
    xc : str
        XC functional, e.g. ``"B3LYP"``.
    grids : pyscf.dft.Grids
        Built grids object, constructed with ``sort_grids=False``.
    nbatch_grids : int, optional
        Batch size for the grid loop.  Defaults to 16384.
    hardness : int, optional
        Becke switch-function iteration count.  Defaults to 3.
    """

    def __init__(self, mol: gto.Mole, xc: str, grids, nbatch_grids: int = 16384, hardness: int = 3):
        self.mol = mol
        self.xc = xc
        self.grids = grids
        self.nbatch_grids = nbatch_grids
        self.hardness = hardness
        natm = mol.natm
        becke_scheme = grids.radii_adjust(mol, grids.atomic_radii)
        self.adjustment_factor = np.array(
            [becke_scheme(i, j, 0) for i in range(natm) for j in range(natm)]
        ).reshape(natm, natm)
        self.atm_quad_split = get_quad_split(grids.atm_idx)
        self.result = dict()
        # filled by make_response_preparation
        self.mo_coeff = None
        self.mo_occ = None
        self.dm0a = None
        self.dm0b = None
        self.rho_cached = None
        self.vxc_cached = None
        self.fxc_cached = None

    def _run_setup_batched(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        """Run `make_hessian_setup_uks` over the full grid and store the
        grid-shift-corrected skeleton / deriv1 quantities in `self.result`.

        No-op if both `de_xc_skeleton` and `de_xc_deriv1_ao` are already cached.
        """
        if "de_xc_skeleton" in self.result and "de_xc_deriv1_ao" in self.result:
            return

        dm0_per_spin = get_dm0_unrestricted(mo_coeff, mo_occ)
        result = make_hessian_setup_uks(
            self.mol,
            self.xc,
            self.grids.coords,
            self.grids.weights,
            dm0_per_spin[0],
            dm0_per_spin[1],
            self.atm_quad_split,
            self.grids.quadrature_weights,
            self.adjustment_factor,
            hardness=self.hardness,
            nbatch_grids=self.nbatch_grids,
            verbose=False,
        )
        self.result = result

    def make_skeleton_hess(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        if "de_xc_skeleton" not in self.result:
            self._run_setup_batched(mo_coeff, mo_occ)
        return self.result["de_xc_skeleton"]

    def get_deriv1_ao(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        if "de_xc_deriv1_ao" not in self.result:
            self._run_setup_batched(mo_coeff, mo_occ)
        return self.result["de_xc_deriv1_ao"]

    def make_response_preparation(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        """Cache `(rho, vxc, fxc)` via PySCF's `cache_xc_kernel` so that every
        subsequent `get_response_bra` call only does the fxc contraction.
        """
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
