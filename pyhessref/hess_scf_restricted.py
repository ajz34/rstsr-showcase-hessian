from pyscf import gto
import numpy as np

from pyhessref.hess_trait_restricted import RHessCoreAPI, RHessElecInteractAPI
from pyhessref.ovlp import RHessOvlp
from pyhessref.krylov_block import krylov_block
from pyhessref.util import get_dme0_restricted


class RHessSCF:
    def __init__(
        self,
        mol: gto.Mole,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        mo_energy: np.ndarray,
        ovlp_obj: RHessOvlp,
        core_list: list[RHessCoreAPI],
        el_list: list[RHessElecInteractAPI],
        level_shift: float = 0,
    ):
        """Working solver and maintainer of all hessian components for restricted SCF method.

        Parameters
        ----------
        mol : gto.Mole
            Molecule object.
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape [nao, nmo].
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape [nmo].
        mo_energy : np.ndarray
            Molecular orbital energies, shape [nmo].
        ovlp_obj : RHessOvlp
            Overlap matrix derivative provider.
        core_list : list[RHessCoreAPI]
            List of core derivative providers. Usually includes
            - Nuclear-repulsion contribution (`HessNucRepl`)
            - One-electron Hamiltonian contribution (`RHessHcore`)
        el_list : list[RHessElecInteractAPI]
            List of electron-interaction derivative providers. For example,
            - RI-JK contribution (`RHessRIJKNaive` or optimized variants)
        level_shift : float, optional
            Level shift added to the denominator in CPHF to improve convergence. Default is 0.
        """
        self.mol = mol
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.mo_energy = mo_energy

        self.ovlp_obj = ovlp_obj
        self.core_list = core_list
        self.el_list = el_list

        self.level_shift = level_shift

        self.result = dict()

    def compute_dimensionless_cphf_rhs(self) -> dict[str, np.ndarray]:
        """Compute the dimensionless CPHF right-hand side, along with necessary intermediates for later steps.

        Note there are some differences compared to usual CP-HF:
        - Usual CP-HF is `(ea - ei) U - AU = B`, where now we handle something like
          `U + (A / (ea - ei)) U = - B / (ea - ei)`
        - We now handle the U in all-occ block, instead of standard vir-occ block;
          this will omit the response evaluation during rhs (B), making the rhs evaluation cheap,
          but we also need to carefully handle the CP-HF equation.
          this behavior should be similar to PySCF's `solve_withs1`.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing:
            - "rhs": The dimensionless CPHF right-hand side, shape [natm, 3, nmo, nocc].
            - "f1mo": The first-order derivative of the Fock matrix in MO basis, shape [natm, 3, nmo, nocc].
            - "s1mo": The first-order derivative of the overlap matrix in MO basis, shape [natm, 3, nmo, nocc].
        """
        # dimensionality setting
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mo_energy = self.mo_energy
        level_shift = self.level_shift

        nao, nmo = mo_coeff.shape
        occidx = mo_occ > 1e-15
        nocc = occidx.sum()
        mocc = mo_coeff[:, occidx]
        natm = self.mol.natm

        eocc = mo_energy[occidx]
        evir = mo_energy[~occidx]
        e_ai = evir[:, None] - eocc[None, :]
        e_ai_shift = e_ai + level_shift

        # --- f1mo --- #

        # fock skeleton derivative (core contribution)
        f1ao_core = np.zeros([natm, 3, nao, nao])
        for core_obj in self.core_list:
            gen_core_deriv1 = core_obj.generator_deriv1()
            if gen_core_deriv1 is None:
                continue
            for A in range(natm):
                f1ao_core[A] += gen_core_deriv1(A)

        # fock skeleton derivative (electron interaction contribution, half-transformed to bra)
        f1bra_el = np.zeros([natm, 3, nao, nocc])
        for el_obj in self.el_list:
            f1bra_el += el_obj.get_deriv1_bra(mo_coeff, mo_occ)

        # construct whole f1mo
        f1bra = f1bra_el + f1ao_core @ mocc
        f1mo = mo_coeff.T @ f1bra

        # --- s1mo --- #

        gen_ovlp_deriv1 = self.ovlp_obj.generator_deriv1()
        s1ao = np.zeros([natm, 3, nao, nao])
        for A in range(natm):
            s1ao[A] += gen_ovlp_deriv1(A)
        s1mo = mo_coeff.T @ s1ao @ mocc

        # --- dimensionless cphf rhs --- #

        b1mo = f1mo - s1mo * eocc
        rhs = np.zeros([natm, 3, nmo, nocc])
        rhs[:, :, nocc:, :] = -b1mo[:, :, nocc:, :] / e_ai_shift[None, None, :, :]
        rhs[:, :, :nocc, :] = -0.5 * s1mo[:, :, :nocc, :]

        return {
            "rhs": rhs,
            "f1mo": f1mo,
            "s1mo": s1mo,
        }

    def make_response_preparation(self, mo_coeff: np.ndarray = None, mo_occ: np.ndarray = None):
        """Prepare the response for CPHF calculation.

        This involves all electron-interaction objects.

        Parameters
        ----------
        mo_coeff : np.ndarray, optional
            Molecular orbital coefficients.
        mo_occ : np.ndarray, optional
            Molecular orbital occupancies.
        """
        mo_coeff = mo_coeff if mo_coeff is not None else self.mo_coeff
        mo_occ = mo_occ if mo_occ is not None else self.mo_occ
        for el_obj in self.el_list:
            el_obj.make_response_preparation(mo_coeff, mo_occ)

    def response_mo(self, mo1: np.ndarray) -> np.ndarray:
        """Compute the response of the system to a given perturbation in MO space (mo1), which is needed for CPHF.

        Parameters
        ----------
        mo1 : np.ndarray
            Perturbation in MO space, shape [..., nmo, nocc].

        Returns
        -------
        resp : np.ndarray
            Response in MO space, shape [..., nmo, nocc].
        """
        mo_coeff = self.mo_coeff

        ubra = mo_coeff @ mo1
        resp = np.zeros_like(mo1)
        for el_obj in self.el_list:
            resp += mo_coeff.T @ el_obj.get_response_bra(ubra)
        return resp

    def response_dimless_cphf(self, mo1: np.ndarray) -> np.ndarray:
        """Compute the dimensionless response for CP-HF calculation.

        Compared to usual CP-HF response, this additionally handles
        - the level shift in denominator
        - the zeroing of occupied-part response (we use mo1[occ, occ] part for evaluating resp[vir, occ],
          but we actually only want to solve the mo1[vir, occ] part and freeze mo1[occ, occ] part to always
          be 0.5 times of ovlp_deriv1).

        Parameters
        ----------
        mo1 : np.ndarray
            Perturbation in MO space, shape [..., nmo, nocc].

        Returns
        -------
        resp : np.ndarray
            Dimensionless response in MO space, shape [..., nmo, nocc].
        """
        mo_occ = self.mo_occ
        mo_energy = self.mo_energy
        occidx = mo_occ > 1e-15
        nocc = occidx.sum()
        level_shift = self.level_shift

        eocc = mo_energy[occidx]
        evir = mo_energy[~occidx]
        e_ai = evir[:, None] - eocc[None, :]
        e_ai_shift = e_ai + level_shift

        resp = self.response_mo(mo1)

        # handle dimensionless denominator and force handle virtual-part only
        if level_shift != 0.0:
            resp -= mo1 * level_shift
        resp[..., nocc:, :] /= e_ai_shift
        resp[..., :nocc, :] = 0
        return resp

    def solve_dimless_cphf(self, rhs: np.ndarray) -> np.ndarray:
        """Solve the dimensionless CP-HF equation using a Krylov solver.

        This should solves `U + resp(U) = rhs`. Note difference of standard CP-HF equation as mentioned in functions above.

        Parameters
        ----------
        rhs : np.ndarray
            Dimensionless right-hand side, shape [natm, 3, nmo, nocc].

        Returns
        -------
        mo1 : np.ndarray
            Perturbation in MO space that solves the dimensionless CP-HF equation, shape [natm, 3, nmo, nocc].
        """
        rhs_shape = rhs.shape
        nmo, nocc = rhs.shape[-2], rhs.shape[-1]
        rhs = rhs.reshape(-1, nmo * nocc)

        def response_cphf_flattened(x: np.ndarray):
            x = x.reshape(-1, nmo, nocc)
            y = self.response_dimless_cphf(x)
            return y.reshape(-1, nmo * nocc)

        mo1 = krylov_block(response_cphf_flattened, rhs)
        return mo1.reshape(rhs_shape)

    def finalize_cphf(self, mo1: np.ndarray, pre_cphf_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Finalize the CP-HF calculation by computing necessary intermediates for Hessian assembly.

        This includes:
        - Re-computing the mo1 (as post-iteration computation), as well as removing the level shift.
        - Computing the derivative of occupied orbital energy with respect to perturbation (mo_e1).
          Note occupied orbital energy (shape [nocc]) is diagonal of Fock, and Fock matrix is diagonal.
          However, with the definition that `U[occ, occ] = -0.5 S1[occ, occ]`, the off-diagonal part of
          derivative of Fock in occupied-occupied block is not zero. That's why this term is actually matrix.

        Parameters
        ----------
        mo1 : np.ndarray
            Perturbation in MO space obtained from Krylov solver, shape [natm, 3, nmo, nocc].
        pre_cphf_dict : dict[str, np.ndarray]
            The dictionary returned by `compute_dimensionless_cphf_rhs`, containing necessary intermediates for finalizing CP-HF results.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary containing:
            - "mo1": The finalized perturbation in MO space, shape [natm, 3, nmo, nocc].
            - "mo_e1": The derivative of occupied orbital energies (Fock matrix) with respect to perturbation, shape [natm, 3, nocc, nocc].
        """
        mo_occ = self.mo_occ
        mo_energy = self.mo_energy
        occidx = mo_occ > 1e-15
        nocc = occidx.sum()
        eocc = mo_energy[occidx]
        evir = mo_energy[~occidx]
        e_ai = evir[:, None] - eocc[None, :]
        e_ij = eocc[:, None] - eocc[None, :]

        # last-iter the cp-hf equation, and remove the level-shift
        f1mo = pre_cphf_dict["f1mo"]
        s1mo = pre_cphf_dict["s1mo"]
        b1mo = f1mo - s1mo * eocc + self.response_mo(mo1)
        mo1[:, :, nocc:, :] = -b1mo[:, :, nocc:, :] / e_ai

        # get the derivative of fock matrix in occ-occ block (derivative of orbital energy with rotation)
        mo_e1 = b1mo[:, :, :nocc, :] + mo1[:, :, :nocc, :] * e_ij
        return {
            "mo1": mo1,
            "mo_e1": mo_e1,
        }

    def get_cphf_hess(self, f1mo: np.ndarray, s1mo: np.ndarray, mo1: np.ndarray, mo_e1: np.ndarray) -> np.ndarray:
        """Compute the CP-HF contribution to the Hessian using the finalized CP-HF results.

        Parameters
        ----------
        f1mo : np.ndarray
            The first-order skeleton derivative of the Fock matrix in MO basis, shape [natm, 3, nmo, nocc].
            This term should be already computed in `compute_dimensionless_cphf_rhs` and passed through `pre_cphf_dict`.
        s1mo : np.ndarray
            The first-order skeleton derivative of the overlap matrix in MO basis, shape [natm, 3, nmo, nocc].
            This term should be already computed in `compute_dimensionless_cphf_rhs` and passed through `pre_cphf_dict`.
        mo1 : np.ndarray
            The finalized perturbation in MO space obtained from `finalize_cphf`, shape [natm, 3, nmo, nocc].
        mo_e1 : np.ndarray
            The derivative of occupied orbital energies (Fock matrix) with respect to perturbation, obtained from `finalize_cphf`, shape [natm, 3, nocc, nocc].

        Returns
        -------
        de_cphf : np.ndarray
            The CP-HF contribution to the Hessian, shape [natm, natm, 3, 3].
        """
        natm = self.mol.natm
        occidx = self.mo_occ > 1e-15
        nocc = occidx.sum()
        eocc = self.mo_energy[occidx]

        s1oo = s1mo[:, :, :nocc, :]

        de_cphf = np.zeros([natm, natm, 3, 3])
        for A in range(natm):
            for B in range(A + 1):
                de_cphf[A, B] += 4 * (f1mo[A][:, None] * mo1[B][None, :]).sum(axis=(-1, -2))
                de_cphf[A, B] -= 4 * (s1mo[A][:, None] * mo1[B][None, :] * eocc).sum(axis=(-1, -2))
                de_cphf[A, B] -= 2 * (s1oo[A][:, None] * mo_e1[B][None, :]).sum(axis=(-1, -2))
            for B in range(A):
                de_cphf[B, A] = de_cphf[A, B].T
        return de_cphf

    def make_cphf_hess(self) -> np.ndarray:
        """Compute the CP-HF contribution to the Hessian by running through the entire CP-HF workflow:

        - Compute the dimensionless CPHF right-hand side and necessary intermediates.
        - Prepare the response for CPHF calculation.
        - Solve the dimensionless CP-HF equation using a Krylov solver.
        - Finalize the CP-HF results by computing necessary intermediates for Hessian assembly.
        - Compute the CP-HF contribution to the Hessian using the finalized CP-HF results.

        Returns
        -------
        de_cphf : np.ndarray
            The CP-HF contribution to the Hessian, shape [natm, natm, 3, 3].
        """
        pre_cphf_dict = self.compute_dimensionless_cphf_rhs()
        self.make_response_preparation(self.mo_coeff, self.mo_occ)
        mo1 = self.solve_dimless_cphf(pre_cphf_dict["rhs"])
        result_cphf = self.finalize_cphf(mo1, pre_cphf_dict)
        mo1 = result_cphf["mo1"]
        mo_e1 = result_cphf["mo_e1"]
        f1mo = pre_cphf_dict["f1mo"]
        s1mo = pre_cphf_dict["s1mo"]
        de_cphf = self.get_cphf_hess(f1mo, s1mo, mo1, mo_e1)
        return de_cphf

    def make_skeleton_hess(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        """Compute the total skeleton contribution to the Hessian.

        **Total** means that we sum over all skeleton contributions from both core and electron-interaction objects.

        Parameters
        ----------
        mo_coeff : np.ndarray
            The molecular orbital coefficients, shape [norb, nmo].
        mo_occ : np.ndarray
            The orbital occupations, shape [nmo].

        Returns
        -------
        de_skeleton : np.ndarray
            The total skeleton contribution to the Hessian, shape [natm, natm, 3, 3].
        """
        natm = self.mol.natm
        de_skeleton = np.zeros([natm, natm, 3, 3])
        for core_obj in self.core_list:
            de_skeleton += core_obj.make_skeleton_hess(mo_coeff, mo_occ)
        for el_obj in self.el_list:
            de_skeleton += el_obj.make_skeleton_hess(mo_coeff, mo_occ)
        return de_skeleton

    def make_hess(self) -> np.ndarray:
        """Compute the total Hessian by summing over skeleton, overlap, and CP-HF contributions.

        Returns
        -------
        de_hess : np.ndarray
            The total Hessian, shape [natm, natm, 3, 3].
        """
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mo_energy = self.mo_energy
        dme0 = get_dme0_restricted(mo_coeff, mo_occ, mo_energy)

        de_skeleton = self.make_skeleton_hess(mo_coeff, mo_occ)
        de_ovlp = self.ovlp_obj.make_hess(dme0)
        de_cphf = self.make_cphf_hess()
        de_hess = de_skeleton + de_ovlp + de_cphf
        return de_hess
