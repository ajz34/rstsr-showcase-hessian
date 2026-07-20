from pyscf import gto, dft
import numpy as np
import time

from pyhessref.hess_trait_unrestricted import UHessElecInteractAPI
from pyhessref.util import get_dm0_unrestricted

# The single-spin XC skeleton ingredients (rho/vxc/fxc-free pieces) are shared
# with the RKS implementation; UKS only adds the spin-coupled pieces on top:
# the spin-polarized rho/vxc/fxc evaluation, the four spin-pair fxc contraction,
# and the spin-coupled ``vmat_deriv1``.  This mirrors the Rust layout where
# ``hess_uks.rs`` imports ``get_drho / get_de_vxc_diag / get_de_vxc_off /
# get_vmat_ip`` from ``hess_rks.rs``.
from pyhessref.nimatmul.rks import (
    _make_drho,
    _de_vxc_diag,
    _de_vxc_off,
    _vmat_ip,
    _vmat_vxc,
    # AO derivative component indices (only the value/gradient channels are
    # referenced directly in the UKS-specific routines below; the higher-order
    # indices live with the shared single-spin helpers in ``rks.py``).
    O,
    X,
    Y,
    Z,
    XC_NVAR,
    XC_AO_DERIV,
    XC_NCOMP_AO_DM0,
)


def _eval_rho_exc_vxc_fxc_uks(xc, xc_type, ao, ao_dm0a, ao_dm0b):
    """Evaluate the on-grid density together with the first/second functional
    derivatives ``vxc``/``fxc`` for the requested xc functional in the
    spin-polarized (UKS) case.

    The per-spin rho assembly is identical to the RKS routine in
    ``rks._eval_rho_exc_vxc_fxc``; only the spin-polarized ``eval_xc_eff``
    call (taking the ``(rhoa, rhob)`` tuple) differs, so it is kept here.

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

    Thin per-spin wrapper around the RKS routine ``_make_drho`` - the
    skeleton derivative is spin-diagonal, so alpha and beta are computed
    independently.  Mirrors ``get_drho_uks`` in the Rust implementation.

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
    drhoa = _make_drho(xc_type, ao, ao_dm0a, aoslices)
    drhob = _make_drho(xc_type, ao, ao_dm0b, aoslices)
    return drhoa, drhob


def _de_fxc_uks_inner(weights, drho1, fxc_block, drho2):
    """Single spin-pair fxc contraction."""
    return np.einsum("g, Atxg, xyg, Bsyg -> ABts", weights, drho1, fxc_block, drho2, optimize=True)


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


def _vmat_fxc_uks(xc_type, ao, drhoa, drhob, wf, natm, nao):
    """fxc contribution to the per-atom skeleton derivative of the Vxc Fock
    matrix for UKS - the spin-coupled part.

    Unlike the spin-diagonal ``_vmat_vxc`` (reused from RKS per spin), the fxc
    contraction here couples the two spin channels:
        wva_f = wf_aa @ drho_a + wf_ab @ drho_b
        wvb_f = wf_ba @ drho_a + wf_bb @ drho_b
    Returned per spin, each assembled across the AO axes (bra + ket).

    Parameters
    ----------
    xc_type : str
        One of ``"LDA"``, ``"GGA"``, ``"MGGA"``.
    ao : np.ndarray
        AO and derivatives, shape ``[ncomp, ngrids, nao]``.  Only indices
        0..3 are read here.
    drhoa, drhob : np.ndarray
        Per-spin drho, shape ``[natm, 3, nvar, ngrids]``.
    wf : np.ndarray
        Weight*fxc, shape ``[2, nvar, 2, nvar, ngrids]``.
    natm : int
    nao : int

    Returns
    -------
    vmata_fxc, vmatb_fxc : np.ndarray
        Each shape ``[natm, 3, nao, nao]``, assembled across the AO axes.
    """
    vmata_fxc = np.zeros((natm, 3, nao, nao))
    vmatb_fxc = np.zeros((natm, 3, nao, nao))

    for A in range(natm):
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
                vmata_fxc[A, t] += aowa.T @ ao[O]
                aowb = wvb_f[t][:, None] * ao[O]
                vmatb_fxc[A, t] += aowb.T @ ao[O]

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
                vmata_fxc[A, t] += aowa_f[t].T @ ao[O]
                vmatb_fxc[A, t] += aowb_f[t].T @ ao[O]

        if xc_type == "MGGA":
            for j in range(1, 4):
                for t in range(3):
                    aowa = wva_f[4, t][:, None] * ao[j]
                    vmata_fxc[A, t] += aowa.T @ ao[j]
                    aowb = wvb_f[4, t][:, None] * ao[j]
                    vmatb_fxc[A, t] += aowb.T @ ao[j]

    # Assemble bra + ket per spin.
    vmata_fxc += vmata_fxc.swapaxes(-1, -2)
    vmatb_fxc += vmatb_fxc.swapaxes(-1, -2)
    return vmata_fxc, vmatb_fxc


def _vmat_deriv1_uks(xc_type, ao, drhoa, drhob, wf, vmata_ip, vmatb_ip, aoslices, natm, nao):
    """Per-atom skeleton derivative of the Vxc Fock matrix for UKS.

    Split into a per-spin vxc contribution (the ipip basis-derivative part,
    reused from the RKS ``_vmat_vxc``) and a spin-coupled fxc contribution
    (``_vmat_fxc_uks``).  Each is assembled independently and summed per spin;
    the split is exact up to floating-point order (same as the RKS split).

    Parameters
    ----------
    drhoa, drhob : np.ndarray
        Per-spin drho, shape ``[natm, 3, nvar, ngrids]``.
    wf : np.ndarray
        Weight*fxc, shape ``[2, nvar, 2, nvar, ngrids]``.
    vmata_ip, vmatb_ip : np.ndarray
        Per-spin gradient-level Vxc (from the shared RKS ``_vmat_ip``),
        shape ``[3, nao, nao]``.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Dictionary with keys ``"vmat_fxc"``, ``"vmat_vxc"``, ``"vmat_deriv1"``,
        each mapping to a per-spin ``[alpha, beta]`` list of arrays, each of
        shape ``[natm, 3, nao, nao]`` and assembled across the AO axes.
    """
    vmata_fxc, vmatb_fxc = _vmat_fxc_uks(xc_type, ao, drhoa, drhob, wf, natm, nao)
    vmata_vxc = _vmat_vxc(vmata_ip, aoslices, natm, nao)
    vmatb_vxc = _vmat_vxc(vmatb_ip, aoslices, natm, nao)
    return {
        "vmat_fxc": [vmata_fxc, vmatb_fxc],
        "vmat_vxc": [vmata_vxc, vmatb_vxc],
        "vmat_deriv1": [vmata_fxc + vmata_vxc, vmatb_fxc + vmatb_vxc],
    }


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

    The spin-diagonal pieces (``de_vxc_diag``, ``de_vxc_off``, ``vmat_ip``,
    ``drho``) reuse the RKS single-spin helpers from ``rks.py``, called once
    per spin channel.  Only the spin-coupled pieces (``_eval_rho_exc_vxc_fxc_uks``,
    ``_de_fxc_uks``, ``_vmat_deriv1_uks``) are UKS-specific.

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

    # Spin-diagonal pieces: delegate to the RKS single-spin helpers.
    t0 = time.time()
    de_vxc_diag_a = _de_vxc_diag(xc_type, ao, ao_dm0a, wva, aoslices, natm, nao)
    de_vxc_diag_b = _de_vxc_diag(xc_type, ao, ao_dm0b, wvb, aoslices, natm, nao)
    tic("de_vxc_diag", t0)

    t0 = time.time()
    de_vxc_off_a = _de_vxc_off(xc_type, ao, dm0a, wva, aoslices, natm, nao)
    de_vxc_off_b = _de_vxc_off(xc_type, ao, dm0b, wvb, aoslices, natm, nao)
    tic("de_vxc_off", t0)

    t0 = time.time()
    vmat_ip_a = _vmat_ip(xc_type, ao, wva, nao)
    vmat_ip_b = _vmat_ip(xc_type, ao, wvb, nao)
    tic("vmat_ip", t0)

    t0 = time.time()
    vmat = _vmat_deriv1_uks(
        xc_type, ao, drhoa, drhob, wf, vmat_ip_a, vmat_ip_b, aoslices, natm, nao
    )
    vmat_deriv1_a, vmat_deriv1_b = vmat["vmat_deriv1"]
    vmat_fxc_a, vmat_fxc_b = vmat["vmat_fxc"]
    vmat_vxc_a, vmat_vxc_b = vmat["vmat_vxc"]
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
        "vmat_fxc_a": vmat_fxc_a,
        "vmat_fxc_b": vmat_fxc_b,
        "vmat_vxc_a": vmat_vxc_a,
        "vmat_vxc_b": vmat_vxc_b,
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
        mol,
        grids,
        xc,
        (dm0a, dm0b),
        (dm1a, dm1b),
        hermi=1,
        rho0=rho_cached,
        vxc=vxc_cached,
        fxc=fxc_cached,
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
                self.mol,
                self.xc,
                coords[start:stop],
                weights[start:stop],
                dm0a,
                dm0b,
                verbose=False,
            )
            if result_sum is None:
                result_sum = {k: v.copy() for k, v in partial.items()}
            else:
                for k in result_sum:
                    result_sum[k] += partial[k]

        # Total XC skeleton = (diag_a + off_a + diag_b + off_b + de_fxc)
        self.result["de_xc_skeleton"] = (
            result_sum["de_vxc_diag_a"]
            + result_sum["de_vxc_off_a"]
            + result_sum["de_vxc_diag_b"]
            + result_sum["de_vxc_off_b"]
            + result_sum["de_fxc"]
        )
        # Per-spin deriv1_ao
        self.result["de_xc_deriv1_ao"] = np.array(
            [
                result_sum["vmat_deriv1_a"],
                result_sum["vmat_deriv1_b"],
            ]
        )

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
            self.mol,
            self.grids,
            self.xc,
            mo_coeff,
            mo_occ,
            spin=1,
        )

    def get_response_bra(self, bra: list[np.ndarray]) -> list[np.ndarray]:
        return get_uks_response_bra_naive(
            self.mol,
            self.grids,
            self.xc,
            self.mo_coeff,
            self.mo_occ,
            self.dm0a,
            self.dm0b,
            bra,
            rho_cached=self.rho_cached,
            vxc_cached=self.vxc_cached,
            fxc_cached=self.fxc_cached,
        )
