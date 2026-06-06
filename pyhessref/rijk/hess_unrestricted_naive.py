import numpy as np
from pyscf import gto
from functools import partial
from pyscf.df.grad.rhf import _int3c_wrapper

from pyhessref.hess_trait_unrestricted import UHessElecInteractAPI
from pyhessref.rijk.hess_restricted_naive import (
    get_decomposed_rij_skeleton_deriv2_naive,
    get_decomposed_rik_skeleton_deriv2_naive,
    get_rij_deriv1_ao_naive,
    get_rik_deriv1_ao_naive,
)

# override einsum for some efficiency
einsum = partial(np.einsum, optimize=True)


def _fake_mo_for_total_density(mo_coeff: np.ndarray, mo_occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a fake (mo_coeff, mo_occ) pair such that the restricted-style total density
    matrix constructed from it equals the UHF total density ``D^alpha + D^beta``.

    Concretely, we stack the occupied orbitals of both spins and assign an occupation of 1
    to each. Then ``mocc * occ @ mocc.T`` equals ``D^alpha + D^beta``.

    This lets us reuse the restricted J function (which only depends on total density)
    without modifying it.
    """
    mocca = mo_coeff[0][:, mo_occ[0] > 1e-15]
    moccb = mo_coeff[1][:, mo_occ[1] > 1e-15]
    fake_coeff = np.hstack([mocca, moccb])
    fake_occ = np.ones(mocca.shape[1] + moccb.shape[1])
    return fake_coeff, fake_occ


def get_decomposed_uij_skeleton_deriv2_naive(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray
) -> dict[str, np.ndarray]:
    """UHF Coulomb skeleton second derivative.

    The Coulomb part depends only on the total density, so we reuse the restricted
    implementation by constructing a fake ``mo_coeff/mo_occ`` whose induced density
    is the UHF total density.
    """
    fake_coeff, fake_occ = _fake_mo_for_total_density(mo_coeff, mo_occ)
    return get_decomposed_rij_skeleton_deriv2_naive(mol, aux, fake_coeff, fake_occ)


def get_decomposed_uik_skeleton_deriv2_naive(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray
) -> dict[str, np.ndarray]:
    """UHF exchange skeleton second derivative.

    UHF K = K^alpha + K^beta, with each spin channel built from its own occupied
    orbitals (no occupation-number scaling, since UHF occ = 1). The restricted
    implementation uses ``mocc_2 = mocc * sqrt(occ)``; passing UHF (mo_coeff, mo_occ)
    per spin gives ``mocc_2 = mocc`` and the cited spin-channel formulas with the
    same numeric coefficients (see 05-3 prototype notebook).
    """
    keys = None
    out = None
    for s in range(2):
        per_spin = get_decomposed_rik_skeleton_deriv2_naive(mol, aux, mo_coeff[s], mo_occ[s])
        if out is None:
            keys = list(per_spin.keys())
            out = {k: per_spin[k].copy() for k in keys}
        else:
            for k in keys:
                out[k] += per_spin[k]
    return out


def get_uij_deriv1_ao_naive(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray
) -> dict[str, np.ndarray]:
    """UHF first-order skeleton derivative of Coulomb in AO basis.

    Spin-independent: depends only on total density. Returns a dict with the same
    keys as the restricted version (``j1ao_aux0`` and ``j1ao_aux1``), each of shape
    ``[natm, 3, nao, nao]``.
    """
    fake_coeff, fake_occ = _fake_mo_for_total_density(mo_coeff, mo_occ)
    return get_rij_deriv1_ao_naive(mol, aux, fake_coeff, fake_occ)


def get_uik_deriv1_ao_naive(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray
) -> dict[str, np.ndarray]:
    """UHF first-order skeleton derivative of exchange in AO basis, spin-resolved.

    Returns a dict whose values have shape ``[2, natm, 3, nao, nao]`` (the leading
    dimension indexes spin).
    """
    out = {}
    for s in range(2):
        per_spin = get_rik_deriv1_ao_naive(mol, aux, mo_coeff[s], mo_occ[s])
        for k, v in per_spin.items():
            if k not in out:
                out[k] = np.zeros((2,) + v.shape)
            out[k][s] = v
    return out


def get_uijk_response_bra_naive(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray, bra: list[np.ndarray]
) -> list[np.ndarray]:
    r"""UHF response of RI-JK given bra (per-spin perturbed coefficients).

    The Coulomb response couples the two spin channels (since J sees the total density),
    while the exchange response is local to each spin (same-spin only).

    Per-spin response:

    .. math::
        R^\sigma_{\mu i} = 2 J[D_1^\alpha + D_1^\beta]_{\mu\nu} C^\sigma_{\nu i}
                        - K^\sigma[D_1^\sigma]_{\mu\nu} C^\sigma_{\nu i}

    where :math:`D_1^\sigma = U^{\sigma,\text{bra}} C^{\sigma,T}_{\text{occ}}` is the
    density built from the perturbed coefficients of spin :math:`\sigma`. (The factor of 2
    on J reflects the symmetrization of the density-matrix derivative; the factor of 1
    on K matches the UHF per-spin exchange amplitude.)

    Parameters
    ----------
    bra : list[np.ndarray]
        ``[bra_alpha, bra_beta]``. Each entry has shape ``[..., nao, nocc_sigma]``
        and the leading dimensions must agree.

    Returns
    -------
    resp : list[np.ndarray]
        ``[resp_alpha, resp_beta]`` with shapes matching the corresponding ``bra`` entries.
    """
    nao = mol.nao
    mocca = mo_coeff[0][:, mo_occ[0] > 1e-15]
    moccb = mo_coeff[1][:, mo_occ[1] > 1e-15]
    mocc = [mocca, moccb]
    nocc = [mocca.shape[1], moccb.shape[1]]

    # sanity / reshape
    in_shapes = [bra[s].shape for s in range(2)]
    bra = [bra[s].reshape(-1, nao, nocc[s]) for s in range(2)]
    nset = bra[0].shape[0]
    assert bra[1].shape[0] == nset

    int2c2e = aux.intor("int2c2e")
    int2c2e_inv = np.linalg.inv(int2c2e)
    int3c2e = _int3c_wrapper(mol, aux, "int3c2e", "s1")()

    # contract once with the integrals; then assemble per spin

    # J part: depends on sum of (bra @ mocc.T) over spins; gives a single response operator
    # that hits both spins. We use the half-transformed einsum form (matching prototype 05-5).
    resp = [None, None]
    for s in range(2):
        r = np.zeros_like(bra[s])
        # J contribution (sees total density): sum over spin channel tau
        for tau in range(2):
            r += 2 * einsum(
                "uvP, PQ, klQ, Akj, lj, vi -> Aui",
                int3c2e,
                int2c2e_inv,
                int3c2e,
                bra[tau],
                mocc[tau],
                mocc[s],
            )
        # K contribution (same-spin only). Two contributions from symmetrization of D_1^sigma.
        r -= einsum(
            "uvP, PQ, klQ, Avj, lj, ki -> Aui",
            int3c2e,
            int2c2e_inv,
            int3c2e,
            bra[s],
            mocc[s],
            mocc[s],
        )
        r -= einsum(
            "uvP, PQ, klQ, Akj, vj, li -> Aui",
            int3c2e,
            int2c2e_inv,
            int3c2e,
            bra[s],
            mocc[s],
            mocc[s],
        )
        resp[s] = r.reshape(in_shapes[s])
    return resp


class UHessRIJKNaive(UHessElecInteractAPI):
    """A naive implementation of the RI-JK Hessian for unrestricted Hartree-Fock.

    Mirrors `RHessRIJKNaive`. The auxiliary derivative is always taken to full order.

    The skeleton hessian is assembled as::

        de_JK = scale_j * de_J - scale_k * de_K

    where the K coefficient is -1 (not -0.5 as in RHF), because UHF ``de_K = K^alpha + K^beta``
    already absorbs the spin sum (see prototype 04 and 05-3 notebooks).
    """

    def __init__(self, mol: gto.Mole, aux: gto.Mole, scale_j: float = 1.0, scale_k: float = 1.0):
        self.mol = mol
        self.aux = aux
        self.scale_j = scale_j
        self.scale_k = scale_k
        self.mo_coeff = None
        self.mo_occ = None
        self.result = dict()

    def make_skeleton_hess(self, mo_coeff, mo_occ):
        de_J_skeleton = get_decomposed_uij_skeleton_deriv2_naive(self.mol, self.aux, mo_coeff, mo_occ)
        de_K_skeleton = get_decomposed_uik_skeleton_deriv2_naive(self.mol, self.aux, mo_coeff, mo_occ)

        self.result.update(de_J_skeleton)
        self.result.update(de_K_skeleton)

        de_J = de_J_skeleton["de_J20"] + de_J_skeleton["de_J11"] + de_J_skeleton["de_J02"]
        de_K = de_K_skeleton["de_K20"] + de_K_skeleton["de_K11"] + de_K_skeleton["de_K02"]
        # UHF: K coefficient is -1 (not -0.5 as in RHF) because de_K already includes spin sum.
        de_JK = self.scale_j * de_J - self.scale_k * de_K
        return de_JK

    def get_deriv1_ao(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        j1ao_dict = get_uij_deriv1_ao_naive(self.mol, self.aux, mo_coeff, mo_occ)
        k1ao_dict = get_uik_deriv1_ao_naive(self.mol, self.aux, mo_coeff, mo_occ)

        self.result.update(j1ao_dict)
        self.result.update(k1ao_dict)

        # J is spin-independent (shape [natm, 3, nao, nao]); K is spin-resolved [2, natm, 3, nao, nao]
        j1ao = j1ao_dict["j1ao_aux0"] + j1ao_dict["j1ao_aux1"]
        k1ao = k1ao_dict["k1ao_aux0"] + k1ao_dict["k1ao_aux1"]
        # Broadcast J to both spins, then subtract per-spin K (no 0.5 factor for UHF).
        deriv_ao = np.broadcast_to(self.scale_j * j1ao, (2,) + j1ao.shape).copy()
        deriv_ao -= self.scale_k * k1ao
        return deriv_ao

    def make_response_preparation(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

    def get_response_bra(self, bra: list[np.ndarray]) -> list[np.ndarray]:
        resp = get_uijk_response_bra_naive(self.mol, self.aux, self.mo_coeff, self.mo_occ, bra)
        # Apply scale_j to J part and scale_k to K part. Since the response routine combines
        # J and K with default factors (scale 1), we need to redo if scales differ from defaults.
        # For now, only support scale_j == scale_k == 1.0; raise if not.
        if not (self.scale_j == 1.0 and self.scale_k == 1.0):
            raise NotImplementedError(
                "Non-trivial scale_j / scale_k not supported in UHessRIJKNaive.get_response_bra. "
                "Decompose into J and K calls separately if needed."
            )
        return resp
