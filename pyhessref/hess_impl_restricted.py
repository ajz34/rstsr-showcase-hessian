from pyscf import gto
import numpy as np

from pyhessref.hess_trait_restricted import RHessCoreAPI, RHessElecInteractAPI
from pyhessref.ovlp import RHessOvlp
from pyhessref.krylov_block import krylov_block
from pyhessref.util import get_dme0_restricted


class RHessImpl:

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
        """Working solver and maintainer of all hessian components for restricted SCF method."""
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

    def make_response_preparation(self, mo_coeff: np.ndarray = None, mo_occ: np.ndarray = None) -> np.ndarray:
        mo_coeff = mo_coeff if mo_coeff is not None else self.mo_coeff
        mo_occ = mo_occ if mo_occ is not None else self.mo_occ
        for el_obj in self.el_list:
            el_obj.make_response_preparation(mo_coeff, mo_occ)

    def response_mo(self, mo1: np.ndarray) -> np.ndarray:
        mo_coeff = self.mo_coeff

        ubra = mo_coeff @ mo1
        resp = np.zeros_like(mo1)
        for el_obj in self.el_list:
            resp += mo_coeff.T @ el_obj.get_response_bra(ubra)
        return resp

    def response_dimless_cphf(self, mo1: np.ndarray) -> np.ndarray:
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

    def solve_dimless_cphf(self, rhs: np.ndarray) -> dict[str, np.ndarray]:
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
        dme0 = get_dme0_restricted(mo_coeff, mo_occ, mo_energy)
        
        de_skeleton = self.make_skeleton_hess(mo_coeff, mo_occ)
        de_ovlp = self.ovlp_obj.make_hess(dme0)
        de_cphf = self.make_cphf_hess()
        de_hess = de_skeleton + de_ovlp + de_cphf
        return de_hess
