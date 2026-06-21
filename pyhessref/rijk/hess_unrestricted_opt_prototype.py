"""Optimized-prototype UHF RI-JK Hessian, mirroring the RHF optimized prototype.

Backed by the J/K-separated `get_decomposed_skeleton_separated`, which natively handles UHF:
pass the 3D ``[2,nao,nmo]`` ``mo_coeff`` / 2D ``[2,nmo]`` ``mo_occ`` and it runs **one** pass
that shares every 3c-integral batch across J (total density) and both K spins (K^alpha, K^beta) --
vs the previous design's 3 separate optimizer calls.

- J (Coulomb) is spin-independent: built from the total density ``D^alpha + D^beta`` (the
  separated function builds ``dm0`` internally, or accepts it via the ``dm0`` kwarg).
- K (exchange) is ``K^alpha + K^beta``; each spin channel is built from its own occupied
  orbitals (UHF occ = 1, so ``mocc_2 = mocc``).

The first-derivative is produced directly in the half-transformed bra form (``get_deriv1_bra``);
the full AO form is not materialized, mirroring `RHessRIJKOptPrototype`.
"""

import numpy as np
from pyscf import gto

from pyhessref.hess_trait_unrestricted import UHessElecInteractAPI
from pyhessref.rijk.hess_restricted_opt_prototype import (
    get_decomposed_skeleton_separated,
    _J20_KEYS, _J11_KEYS, _J02_KEYS,
    _K20_KEYS, _K11_KEYS, _K02_KEYS,
    _J1AO_KEYS, _K1BRA_KEYS,
)
from pyhessref.rijk.hess_unrestricted_naive import get_uijk_response_bra_naive


class UHessRIJKOptPrototype(UHessElecInteractAPI):
    """Optimized-prototype RI-JK Hessian for unrestricted HF, implementing `UHessElecInteractAPI`.

    Backed by the J/K-separated `get_decomposed_skeleton_separated`. A **single** call with the
    UHF ``mo_coeff``/``mo_occ`` (``do_j=True, do_k=True``) produces J (from total density) and
    both K spins, sharing every 3c-integral batch -- the integral-minimization win over the
    previous 3-call design. See `RHessRIJKOptPrototype` for the RHF counterpart and the
    ``get_deriv1_bra`` / ``get_deriv1_ao`` (dummy) rationale.

    Skeleton scaling is ``scale_j * de_J - scale_k * de_K`` (K coefficient -1, not -0.5 as in
    RHF), because UHF ``de_K = K^alpha + K^beta`` already absorbs the spin sum (matches
    `UHessRIJKNaive`). The response (`get_response_bra`) reuses the naive UHF RI-JK response.
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
        self._j_res = None
        self._k_res = None
        self._skel_key = None

    def _ensure_skeleton(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        key = (id(mo_coeff[0]), id(mo_coeff[1]))
        if self._j_res is not None and self._skel_key == key:
            return
        j_res, k_res = get_decomposed_skeleton_separated(
            self.mol, self.aux, mo_coeff, mo_occ, self.cderi, self.nbatch_aux,
            do_j=True, do_k=True,
        )
        self._j_res = j_res
        self._k_res = k_res  # list of 2 dicts (UHF): [k_alpha, k_beta]
        self._skel_key = key
        self.result.update(j_res)
        for kr in k_res:
            self.result.update(kr)

    def make_skeleton_hess(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        self._ensure_skeleton(mo_coeff, mo_occ)
        j_res, k_res = self._j_res, self._k_res
        de_J = sum(j_res[k] for k in _J20_KEYS) + sum(j_res[k] for k in _J11_KEYS) + sum(j_res[k] for k in _J02_KEYS)
        de_K = sum(kr[k] for kr in k_res for k in _K20_KEYS) \
            + sum(kr[k] for kr in k_res for k in _K11_KEYS) \
            + sum(kr[k] for kr in k_res for k in _K02_KEYS)
        # UHF: K coefficient is -1 (not -0.5 as in RHF) because de_K already includes the spin sum.
        return self.scale_j * de_J - self.scale_k * de_K

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
        self._ensure_skeleton(mo_coeff, mo_occ)
        j_res, k_res = self._j_res, self._k_res
        # J: held in AO form ([nao, nao]), shared across spins; right half-transform per spin.
        j1ao = sum(j_res[k] for k in _J1AO_KEYS)
        out = []
        for s in range(2):
            occidx = mo_occ[s] > 1e-15
            mocc_s = mo_coeff[s][:, occidx]
            # K: stored as left half-transform k1bra^s = mocc_s.T @ k1ao^s ([nocc_s, nao]); the
            # per-spin total k1ao^s is symmetric, so the right transform k1ao^s @ mocc_s is
            # k1bra^s.swapaxes(-1, -2).
            k1bra_s = sum(k_res[s][k] for k in _K1BRA_KEYS)
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
