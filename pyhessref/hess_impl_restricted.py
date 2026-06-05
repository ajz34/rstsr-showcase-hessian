from pyscf import gto
import numpy as np

from pyhessref.hess_trait_restricted import RHessCoreAPI, RHessElecInteractAPI
from pyhessref.ovlp import RHessOvlp


class RHessImpl:

    def __init__(
        self,
        mol: gto.Mole,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        mo_energy: np.ndarray,
        ovlp_obj: RHessOvlp,
        core_list: list[RHessCoreAPI],
        interact_list: list[RHessElecInteractAPI],
        level_shift: float = 0,
    ):
        """Working solver and maintainer of all hessian components for restricted SCF method."""
        self.mol = mol
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.mo_energy = mo_energy

        self.ovlp_obj = ovlp_obj
        self.core_list = core_list
        self.interact_list = interact_list

        self.level_shift = level_shift

        self.result = dict()

    def compute_dimensionless_cphf_rhs(self):
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
        for interact_obj in self.interact_list:
            f1bra_el += interact_obj.get_deriv1_bra(mo_coeff, mo_occ)

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
