from pyscf import gto
import numpy as np

from pyhessref.hess_trait_unrestricted import UHessCoreAPI, UHessElecInteractAPI
from pyhessref.ovlp import RHessOvlp
from pyhessref.krylov_block import krylov_block
from pyhessref.util import get_dme0_unrestricted, pack_uhf_mo_pair, unpack_uhf_mo_pair


class UHessSCF:
    """Working solver and maintainer of all hessian components for unrestricted SCF method.

    Mirrors `RHessSCF`. The main differences:

    - ``mo_coeff`` / ``mo_occ`` / ``mo_energy`` are stored per spin (leading dimension 2).
    - CP-HF unknowns ``mo1``, the Fock skeleton derivative ``f1mo`` and the overlap
      derivative ``s1mo`` are kept as ``list[np.ndarray]`` of length 2, because the
      number of occupied orbitals may differ between alpha and beta channels and
      cannot be combined into a single regular ndarray.
    - The Krylov solver receives a flattened 2D array of shape
      ``[nset, nmo*nocc_alpha + nmo*nocc_beta]``; flattening and unflattening are
      handled by `pack_uhf_mo_pair` / `unpack_uhf_mo_pair`.
    - CP-HF Hessian factors are ``2, 2, 1`` (RHF uses ``4, 4, 2``), reflecting that
      UHF uses occupation 1 per spin and sums over both spins.
    """

    def __init__(
        self,
        mol: gto.Mole,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        mo_energy: np.ndarray,
        ovlp_obj: RHessOvlp,
        core_list: list[UHessCoreAPI],
        el_list: list[UHessElecInteractAPI],
        level_shift: float = 0,
    ):
        """
        Parameters
        ----------
        mol : gto.Mole
            Molecule object.
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape ``[2, nao, nmo]``.
        mo_occ : np.ndarray
            Occupation numbers, shape ``[2, nmo]``.
        mo_energy : np.ndarray
            Orbital energies, shape ``[2, nmo]``.
        ovlp_obj : RHessOvlp
            Overlap matrix derivative provider (spin-independent; we reuse the restricted
            class). Pass the **total** energy-weighted density when calling ``make_hess``.
        core_list : list[UHessCoreAPI]
            Core derivative providers (nuclear repulsion, hcore, ...).
        el_list : list[UHessElecInteractAPI]
            Electron-interaction derivative providers (RI-JK, ...).
        level_shift : float, optional
            Level shift in CPHF denominator.
        """
        self.mol = mol
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.mo_energy = mo_energy

        self.ovlp_obj = ovlp_obj
        self.core_list = core_list
        self.el_list = el_list

        self.level_shift = level_shift

        # cached per-spin descriptors (populated lazily by _spin_descriptors)
        self.result = dict()

    # ------------------------------------------------------------------ helpers

    def _spin_descriptors(self):
        """Return per-spin (mocc, eocc, evir, nocc, nmo) tuples and shape metadata."""
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mo_energy = self.mo_energy
        nmo = mo_coeff.shape[-1]
        mocc, eocc, evir, nocc = [], [], [], []
        for s in range(2):
            occidx = mo_occ[s] > 1e-15
            mocc.append(mo_coeff[s][:, occidx])
            eocc.append(mo_energy[s][occidx])
            evir.append(mo_energy[s][~occidx])
            nocc.append(int(occidx.sum()))
        return mocc, eocc, evir, nocc, nmo

    def _e_ai(self):
        _, eocc, evir, _, _ = self._spin_descriptors()
        return [evir[s][:, None] - eocc[s][None, :] for s in range(2)]

    def _shape_per_spin(self):
        """Trailing shape ``(nmo, nocc_sigma)`` for each spin."""
        _, _, _, nocc, nmo = self._spin_descriptors()
        return [(nmo, nocc[0]), (nmo, nocc[1])]

    # ------------------------------------------------------------------ CP-HF

    def compute_dimensionless_cphf_rhs(self) -> dict[str, list]:
        """Compute the dimensionless CPHF right-hand side for UHF.

        See `RHessSCF.compute_dimensionless_cphf_rhs` for the meaning of *dimensionless*.

        Returns
        -------
        dict
            - ``rhs``: list ``[rhs_alpha, rhs_beta]``, each of shape ``[natm, 3, nmo, nocc_sigma]``.
            - ``f1mo``: list of per-spin Fock skeleton derivative in MO basis.
            - ``s1mo``: list of per-spin overlap derivative in MO basis (note: ``s1mo`` here
              uses the same spin-specific MO transformation as f1mo, even though the AO
              integral is spin-independent).
        """
        mo_coeff = self.mo_coeff
        mocc, eocc, _, nocc, nmo = self._spin_descriptors()
        e_ai_list = self._e_ai()
        level_shift = self.level_shift
        natm = self.mol.natm
        nao = mo_coeff.shape[1]

        # --- core (spin-independent) skeleton derivative in AO basis --- #
        f1ao_core = np.zeros([natm, 3, nao, nao])
        for core_obj in self.core_list:
            gen = core_obj.generator_deriv1()
            if gen is None:
                continue
            for A in range(natm):
                f1ao_core[A] += gen(A)

        # --- electron-interaction (spin-resolved) skeleton derivative, half-transformed --- #
        f1bra_el = [np.zeros([natm, 3, nao, nocc[s]]) for s in range(2)]
        for el_obj in self.el_list:
            bra = el_obj.get_deriv1_bra(self.mo_coeff, self.mo_occ)
            for s in range(2):
                f1bra_el[s] += bra[s]

        # --- assemble per-spin f1mo --- #
        f1mo = [None, None]
        for s in range(2):
            f1bra = f1bra_el[s] + f1ao_core @ mocc[s]
            f1mo[s] = mo_coeff[s].T @ f1bra

        # --- s1mo (per spin; AO part is shared) --- #
        gen_ovlp = self.ovlp_obj.generator_deriv1()
        s1ao = np.zeros([natm, 3, nao, nao])
        for A in range(natm):
            s1ao[A] += gen_ovlp(A)
        s1mo = [mo_coeff[s].T @ s1ao @ mocc[s] for s in range(2)]

        # --- dimensionless rhs --- #
        rhs = [None, None]
        for s in range(2):
            b1mo = f1mo[s] - s1mo[s] * eocc[s]
            e_ai_shift = e_ai_list[s] + level_shift
            rhs_s = np.zeros([natm, 3, nmo, nocc[s]])
            rhs_s[:, :, nocc[s] :, :] = -b1mo[:, :, nocc[s] :, :] / e_ai_shift[None, None, :, :]
            rhs_s[:, :, : nocc[s], :] = -0.5 * s1mo[s][:, :, : nocc[s], :]
            rhs[s] = rhs_s

        return {"rhs": rhs, "f1mo": f1mo, "s1mo": s1mo}

    def make_response_preparation(self, mo_coeff: np.ndarray = None, mo_occ: np.ndarray = None):
        mo_coeff = mo_coeff if mo_coeff is not None else self.mo_coeff
        mo_occ = mo_occ if mo_occ is not None else self.mo_occ
        for el_obj in self.el_list:
            el_obj.make_response_preparation(mo_coeff, mo_occ)

    def response_mo(self, mo1: list[np.ndarray]) -> list[np.ndarray]:
        """Compute the response in MO space.

        Parameters
        ----------
        mo1 : list[np.ndarray]
            ``[mo1_alpha, mo1_beta]``. Each entry has shape ``[..., nmo, nocc_sigma]``.

        Returns
        -------
        resp : list[np.ndarray]
            Per-spin response of the same shapes.
        """
        mo_coeff = self.mo_coeff
        # bra-transform per spin
        ubra = [mo_coeff[s] @ mo1[s] for s in range(2)]
        resp = [np.zeros_like(mo1[s]) for s in range(2)]
        for el_obj in self.el_list:
            r = el_obj.get_response_bra(ubra)
            for s in range(2):
                resp[s] += mo_coeff[s].T @ r[s]
        return resp

    def response_dimless_cphf(self, mo1: list[np.ndarray]) -> list[np.ndarray]:
        """Dimensionless response for the CP-HF Krylov iteration."""
        _, eocc, evir, nocc, _ = self._spin_descriptors()
        level_shift = self.level_shift
        e_ai_list = self._e_ai()

        resp = self.response_mo(mo1)
        for s in range(2):
            if level_shift != 0.0:
                resp[s] -= mo1[s] * level_shift
            e_ai_shift = e_ai_list[s] + level_shift
            resp[s][..., nocc[s] :, :] /= e_ai_shift
            resp[s][..., : nocc[s], :] = 0
        return resp

    def solve_dimless_cphf(self, rhs: list[np.ndarray]) -> list[np.ndarray]:
        """Solve the dimensionless CP-HF equation using a Krylov solver.

        The two spin channels are flattened and concatenated into a single
        ``[nset, n_alpha + n_beta]`` array before being fed to the solver.
        """
        shapes = [rhs[s].shape for s in range(2)]
        nset = shapes[0][0]
        assert shapes[1][0] == nset
        per_spin_shape = [s[1:] for s in shapes]  # trailing (3, nmo, nocc_sigma) or (..., nmo, nocc)

        # Flatten the leading "atom × component" dimensions into a single nset dim
        nset_flat = int(np.prod([s for s in shapes[0][:-2]]))  # natm * 3
        trailing_a = shapes[0][-2:]  # (nmo, nocc_a)
        trailing_b = shapes[1][-2:]
        rhs_flat = [rhs[s].reshape(nset_flat, *shapes[s][-2:]) for s in range(2)]
        rhs_packed = pack_uhf_mo_pair(rhs_flat)  # [nset_flat, size_a + size_b]

        def response_cphf_packed(x_packed: np.ndarray) -> np.ndarray:
            x_list = unpack_uhf_mo_pair(x_packed, trailing_a, trailing_b)
            y_list = self.response_dimless_cphf(x_list)
            return pack_uhf_mo_pair(y_list)

        mo1_packed = krylov_block(response_cphf_packed, rhs_packed)
        mo1_flat = unpack_uhf_mo_pair(mo1_packed, trailing_a, trailing_b)
        mo1 = [mo1_flat[s].reshape(shapes[s]) for s in range(2)]
        # Pin mo1[oo] = rhs[oo] = -0.5*s1mo[oo] for each spin.  The
        # response operator zeros the oo block, so the equation degenerates
        # to mo1[oo] = rhs[oo] there; without this overwrite Krylov leaves
        # ~1e-5 noise that propagates to mo_e1 and corrupts de_cphf for
        # hybrid-DFT.  See restricted analogue in hess_scf_restricted.py.
        _, _, _, nocc, _ = self._spin_descriptors()
        for s in range(2):
            mo1[s][..., : nocc[s], :] = rhs[s][..., : nocc[s], :]
        return mo1

    def finalize_cphf(self, mo1: list[np.ndarray], pre_cphf_dict: dict) -> dict:
        """Finalize the CP-HF calculation: re-impose the exact CPHF for vir-occ block
        (also removes level shift), and compute the occ-occ Fock derivative ``mo_e1``.
        """
        _, eocc, evir, nocc, _ = self._spin_descriptors()
        e_ai_list = self._e_ai()
        e_ij_list = [eocc[s][:, None] - eocc[s][None, :] for s in range(2)]

        f1mo = pre_cphf_dict["f1mo"]
        s1mo = pre_cphf_dict["s1mo"]
        resp = self.response_mo(mo1)

        mo_e1 = [None, None]
        for s in range(2):
            b1mo = f1mo[s] - s1mo[s] * eocc[s] + resp[s]
            mo1[s][:, :, nocc[s] :, :] = -b1mo[:, :, nocc[s] :, :] / e_ai_list[s]
            mo_e1[s] = b1mo[:, :, : nocc[s], :] + mo1[s][:, :, : nocc[s], :] * e_ij_list[s]

        return {"mo1": mo1, "mo_e1": mo_e1}

    def get_cphf_hess(
        self,
        f1mo: list[np.ndarray],
        s1mo: list[np.ndarray],
        mo1: list[np.ndarray],
        mo_e1: list[np.ndarray],
    ) -> np.ndarray:
        """Assemble the CP-HF contribution to the Hessian for UHF.

        UHF coefficients are ``2, 2, 1`` (versus ``4, 4, 2`` for RHF), reflecting the
        per-spin occupation of 1 and the spin sum.
        """
        natm = self.mol.natm
        _, eocc, _, nocc, _ = self._spin_descriptors()

        de_cphf = np.zeros([natm, natm, 3, 3])
        for s in range(2):
            s1oo = s1mo[s][:, :, : nocc[s], :]
            for A in range(natm):
                for B in range(A + 1):
                    de_cphf[A, B] += 2 * (f1mo[s][A][:, None] * mo1[s][B][None, :]).sum(axis=(-1, -2))
                    de_cphf[A, B] -= 2 * (s1mo[s][A][:, None] * mo1[s][B][None, :] * eocc[s]).sum(axis=(-1, -2))
                    de_cphf[A, B] -= 1 * (s1oo[A][:, None] * mo_e1[s][B][None, :]).sum(axis=(-1, -2))

        # symmetrize the upper triangle (lower triangle was filled in the spin loop)
        for A in range(natm):
            for B in range(A):
                de_cphf[B, A] = de_cphf[A, B].T
        return de_cphf

    def make_cphf_hess(self) -> np.ndarray:
        pre_cphf_dict = self.compute_dimensionless_cphf_rhs()
        self.make_response_preparation(self.mo_coeff, self.mo_occ)
        mo1 = self.solve_dimless_cphf(pre_cphf_dict["rhs"])
        result_cphf = self.finalize_cphf(mo1, pre_cphf_dict)
        return self.get_cphf_hess(
            pre_cphf_dict["f1mo"], pre_cphf_dict["s1mo"], result_cphf["mo1"], result_cphf["mo_e1"]
        )

    def make_skeleton_hess(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        natm = self.mol.natm
        de_skeleton = np.zeros([natm, natm, 3, 3])
        for core_obj in self.core_list:
            de_skeleton += core_obj.make_skeleton_hess(mo_coeff, mo_occ)
        for el_obj in self.el_list:
            de_skeleton += el_obj.make_skeleton_hess(mo_coeff, mo_occ)
        return de_skeleton

    def make_hess(self) -> np.ndarray:
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mo_energy = self.mo_energy
        dme0_per_spin = get_dme0_unrestricted(mo_coeff, mo_occ, mo_energy)
        dme0_total = dme0_per_spin.sum(axis=0)

        de_skeleton = self.make_skeleton_hess(mo_coeff, mo_occ)
        de_ovlp = self.ovlp_obj.make_hess(dme0_total)
        de_cphf = self.make_cphf_hess()
        return de_skeleton + de_ovlp + de_cphf
