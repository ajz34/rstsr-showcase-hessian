"""Optimized-prototype UHF RI-JK Hessian, mirroring the RHF optimized prototype.

Like `hess_unrestricted_naive`, this reuses the restricted optimized skeleton
(`get_decomposed_skeleton_separated`) for the UHF J and K terms:

- J (Coulomb) is spin-independent (depends only on the total density ``D^alpha + D^beta``),
  so we feed the restricted optimizer a *fake* ``(mo_coeff, mo_occ)`` whose induced density is
  the UHF total density (see `_fake_mo_for_total_density`).
- K (exchange) is ``K^alpha + K^beta``; each spin channel is built from its own occupied
  orbitals (UHF occ = 1, so ``mocc_2 = mocc``), so we run the restricted optimizer once per
  spin and sum the K keys.

The restricted optimizer computes J and K together (they share the 3c integrals), so each call
also produces the "other" half as a by-product that is simply discarded -- acceptable for a
prototype. The first-derivative is produced directly in the half-transformed bra form
(``get_deriv1_bra``); the full AO form is not materialized, mirroring `RHessRIJKOptPrototype`.
"""

import numpy as np
from pyscf import gto

from pyhessref.hess_trait_unrestricted import UHessElecInteractAPI
from pyhessref.rijk.hess_restricted_opt_prototype import get_decomposed_skeleton_separated
from pyhessref.rijk.hess_unrestricted_naive import (
    _fake_mo_for_total_density,
    get_uijk_response_bra_naive,
)


# --- skeleton / first-derivative key groups (shared with the RHF optimizer output) --- #
_J20_KEYS = ("de_J20_1", "de_J20_2", "de_J20_3")
_J11_KEYS = ("de_J11_1", "de_J11_2", "de_J11_3", "de_J11_4")
_J02_KEYS = ("de_J02_1", "de_J02_2", "de_J02_3a", "de_J02_3b", "de_J02_4",
             "de_J02_5", "de_J02_6", "de_J02_7", "de_J02_8")
_K20_KEYS = ("de_K20_1a", "de_K20_1b", "de_K20_2", "de_K20_3")
_K11_KEYS = ("de_K11_1", "de_K11_2", "de_K11_3", "de_K11_4")
_K02_KEYS = ("de_K02_1", "de_K02_2", "de_K02_3a", "de_K02_3b", "de_K02_4",
             "de_K02_5", "de_K02_6", "de_K02_7", "de_K02_8")
_J1AO_KEYS = ("j1ao_aux0", "j1ao_aux1_1", "j1ao_aux1_2", "j1ao_aux1_3", "j1ao_aux1_4")
_K1BRA_KEYS = ("k1bra_aux0_1", "k1bra_aux0_2", "k1bra_aux0_3", "k1bra_aux0_4",
               "k1bra_aux1_1", "k1bra_aux1_2", "k1bra_aux1_3", "k1bra_aux1_4")


def get_decomposed_uij_skeleton_deriv2_opt(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray, cderi: np.ndarray,
    nbatch_aux: int = 72,
) -> dict[str, np.ndarray]:
    """UHF Coulomb skeleton second derivative (optimized).

    J depends only on the total density, so the restricted optimizer is run on a fake
    ``(mo_coeff, mo_occ)`` inducing ``D^alpha + D^beta``. Only the J keys are returned; the K
    keys the optimizer also produces (built from the fake occupied set) are discarded.
    """
    fake_coeff, fake_occ = _fake_mo_for_total_density(mo_coeff, mo_occ)
    res = get_decomposed_skeleton_separated(mol, aux, fake_coeff, fake_occ, cderi, nbatch_aux)
    out = {k: res[k] for k in _J20_KEYS + _J11_KEYS + _J02_KEYS + _J1AO_KEYS}
    # summed totals, matching the naive return contract
    out["de_J20"] = sum(out[k] for k in _J20_KEYS)
    out["de_J11"] = sum(out[k] for k in _J11_KEYS)
    out["de_J02"] = sum(out[k] for k in _J02_KEYS)
    return out


def get_decomposed_uik_skeleton_deriv2_opt(
    mol: gto.Mole, aux: gto.Mole, mo_coeff: np.ndarray, mo_occ: np.ndarray, cderi: np.ndarray,
    nbatch_aux: int = 72,
) -> dict[str, np.ndarray]:
    """UHF exchange skeleton second derivative (optimized): ``K^alpha + K^beta``.

    The restricted optimizer is run once per spin (UHF occ = 1 gives ``mocc_2 = mocc``, matching
    the per-spin exchange formulas). The K skeleton keys are summed over spins; the per-spin
    ``k1bra_*`` first-derivative keys are kept spin-resolved (returned as ``k1bra_aux*_<s>``)
    since their ``nocc`` axis differs between spins and cannot be stacked.
    """
    out = {}
    for s in range(2):
        res = get_decomposed_skeleton_separated(
            mol, aux, mo_coeff[s], mo_occ[s], cderi, nbatch_aux
        )
        # K skeleton keys: summed over spins.
        for k in _K20_KEYS + _K11_KEYS + _K02_KEYS:
            out[k] = res[k] + out.get(k, 0.0)
        # K first-derivative bra keys: spin-resolved (nocc differs), tagged with spin suffix.
        for k in _K1BRA_KEYS:
            out[f"{k}_{s}"] = res[k]
    # summed totals, matching the naive return contract
    out["de_K20"] = sum(out[k] for k in _K20_KEYS)
    out["de_K11"] = sum(out[k] for k in _K11_KEYS)
    out["de_K02"] = sum(out[k] for k in _K02_KEYS)
    return out


class UHessRIJKOptPrototype(UHessElecInteractAPI):
    """Optimized-prototype RI-JK Hessian for unrestricted HF, implementing `UHessElecInteractAPI`.

    Mirrors `UHessRIJKNaive` but backs the skeleton / first-derivative terms with the optimized
    `get_decomposed_skeleton_separated` (reused for UHF J via fake-mo total density, and for UHF K
    per-spin). See `RHessRIJKOptPrototype` for the RHF counterpart.

    Notes on `get_deriv1_bra` vs `get_deriv1_ao`:
    - The optimized prototype evaluates the exchange K1 directly in the left half-transformed bra
      form ``k1bra^sigma = mocc[sigma].T @ k1ao^sigma`` (shape ``[natm, 3, nocc_sigma, nao]``);
      the full AO form ``[nao, nao]`` is never produced for K. So `get_deriv1_bra` is the real
      implementation and `get_deriv1_ao` is a dummy.
    - The bra form returned by the API is the *right* half-transform ``deriv_ao @ mocc[sigma]``
      (``[natm, 3, nao, nocc_sigma]``). For J (held in AO) this is ``j1ao @ mocc[sigma]``. For K,
      the per-spin total ``k1ao^sigma`` is symmetric, so
      ``k1ao^sigma @ mocc[sigma] = (mocc[sigma].T @ k1ao^sigma).swapaxes(-1, -2)``, i.e. the right
      transform is obtained from the stored left transform by a swapaxes -- no full ``k1ao`` is built.

    Skeleton scaling is ``scale_j * de_J - scale_k * de_K`` (K coefficient -1, not -0.5 as in RHF),
    because UHF ``de_K = K^alpha + K^beta`` already absorbs the spin sum (matches `UHessRIJKNaive`).

    The response (`get_response_bra`) reuses the naive UHF RI-JK response; the optimization work
    here covers skeleton + first derivative, not the CPHF response.
    """

    def __init__(self, mol: gto.Mole, aux: gto.Mole, cderi: np.ndarray, nbatch_aux: int = 72,
                 scale_j: float = 1.0, scale_k: float = 1.0):
        self.mol = mol
        self.aux = aux
        self.cderi = cderi
        self.nbatch_aux = nbatch_aux
        self.scale_j = scale_j
        self.scale_k = scale_k
        self.mo_coeff = None
        self.mo_occ = None
        self.result = dict()
        # cache for the (expensive) skeleton computation; both make_skeleton_hess and
        # get_deriv1_bra consume it, so compute at most once per (mo_coeff[0], mo_coeff[1]).
        self._skel = None
        self._skel_key = None

    def _ensure_skeleton(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> dict[str, np.ndarray]:
        key = (id(mo_coeff[0]), id(mo_coeff[1]))
        if self._skel is not None and self._skel_key == key:
            return self._skel

        # J: spin-independent, run optimizer on fake (total-density) mo.
        fake_coeff, fake_occ = _fake_mo_for_total_density(mo_coeff, mo_occ)
        res_J = get_decomposed_skeleton_separated(
            self.mol, self.aux, fake_coeff, fake_occ, self.cderi, self.nbatch_aux
        )
        # K^sigma: run optimizer once per spin.
        res_K = [
            get_decomposed_skeleton_separated(
                self.mol, self.aux, mo_coeff[s], mo_occ[s], self.cderi, self.nbatch_aux
            )
            for s in range(2)
        ]

        skel = {}
        # J skeleton + J first-derivative (AO) keys -- from the J call.
        for k in _J20_KEYS + _J11_KEYS + _J02_KEYS + _J1AO_KEYS:
            skel[k] = res_J[k]
        # K skeleton keys -- summed over spins.
        for k in _K20_KEYS + _K11_KEYS + _K02_KEYS:
            skel[k] = res_K[0][k] + res_K[1][k]
        # K first-derivative bra keys -- spin-resolved (nocc differs), tagged with spin suffix.
        for s in range(2):
            for k in _K1BRA_KEYS:
                skel[f"{k}_{s}"] = res_K[s][k]

        self._skel = skel
        self._skel_key = key
        self.result.update(skel)
        return skel

    def make_skeleton_hess(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        res = self._ensure_skeleton(mo_coeff, mo_occ)
        de_J = sum(res[k] for k in _J20_KEYS) + sum(res[k] for k in _J11_KEYS) + sum(res[k] for k in _J02_KEYS)
        de_K = sum(res[k] for k in _K20_KEYS) + sum(res[k] for k in _K11_KEYS) + sum(res[k] for k in _K02_KEYS)
        # UHF: K coefficient is -1 (not -0.5 as in RHF) because de_K already includes the spin sum.
        de_JK = self.scale_j * de_J - self.scale_k * de_K
        return de_JK

    def get_deriv1_ao(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        """Dummy -- not implemented.

        The optimized prototype evaluates K1 only in the half-transformed bra form
        (per-spin ``[nocc_sigma, nao]``); the full spin-resolved AO form
        ``[2, natm, 3, nao, nao]`` is intentionally never produced. Use `get_deriv1_bra` instead
        (which `UHessSCF` consumes directly).
        """
        raise NotImplementedError(
            "UHessRIJKOptPrototype does not produce the full AO first-derivative; "
            "use get_deriv1_bra instead."
        )

    def get_deriv1_bra(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> list[np.ndarray]:
        res = self._ensure_skeleton(mo_coeff, mo_occ)
        # J: held in AO form ([nao, nao]), right half-transform per spin.
        j1ao = sum(res[k] for k in _J1AO_KEYS)
        out = []
        for s in range(2):
            occidx = mo_occ[s] > 1e-15
            mocc_s = mo_coeff[s][:, occidx]
            # K: stored as left half-transform k1bra^s = mocc_s.T @ k1ao^s ([nocc_s, nao]); the
            # per-spin total k1ao^s is symmetric, so the right transform k1ao^s @ mocc_s is
            # k1bra^s.swapaxes(-1, -2).
            k1bra_s = sum(res[f"{k}_{s}"] for k in _K1BRA_KEYS)
            deriv_bra = self.scale_j * (j1ao @ mocc_s) - self.scale_k * k1bra_s.swapaxes(-1, -2)
            out.append(deriv_bra)
        return out

    def make_response_preparation(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

    def get_response_bra(self, bra: list[np.ndarray]) -> list[np.ndarray]:
        return get_uijk_response_bra_naive(
            self.mol, self.aux, self.mo_coeff, self.mo_occ, bra,
            scale_j=self.scale_j, scale_k=self.scale_k,
        )
