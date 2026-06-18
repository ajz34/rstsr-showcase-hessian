import unittest

import numpy as np
from pyscf import gto, scf, lib, df, hessian


from pyhessref.util import get_dme0_restricted
from pyhessref.nuc_repl import HessNucRepl, get_nuc_repl_hess
from pyhessref.hcore import RHessHcore, generator_hcore_deriv2, get_hess_hcore, generator_hcore_deriv1
from pyhessref.ovlp import RHessOvlp
from pyhessref.rijk.hess_restricted_naive import (
    RHessRIJKNaive,
    get_decomposed_rij_skeleton_deriv2_naive,
    get_decomposed_rik_skeleton_deriv2_naive,
    get_rij_deriv1_ao_naive,
    get_rik_deriv1_ao_naive,
)


def setUpModule():
    global mol, aux, mf, mf_hess, ref_value
    lib.num_threads(4)

    xyz = """
    N  0   0   0
    H  1.0 0.1 0.2
    H  0.3 1.1 0.2
    H  0.1 0.1 1.2
    """
    PATH_PROTOTYPE = "prototype/"  # assuming run at project root
    PATH_REF = PATH_PROTOTYPE + "nh3_r_hf_decomp.npz"

    mol = gto.Mole(atom=xyz, basis="def2-TZVP", max_memory=8000).build()
    mf = scf.RHF(mol).density_fit()
    ref_value = np.load(PATH_REF)
    mf.mo_coeff = ref_value["mo_coeff"]
    mf.mo_occ = ref_value["mo_occ"]
    mf.mo_energy = ref_value["mo_energy"]
    mf.with_df.build()
    mf.converged = True
    mf_hess = mf.Hessian()
    aux = mf.with_df.auxmol


class TestHessianRHFOptPrototype(unittest.TestCase):
    def test_get_decomposed_skeleton(self):
        from pyhessref.rijk.hess_restricted_opt_prototype import get_decomposed_skeleton

        cderi = mf.with_df._cderi
        result = get_decomposed_skeleton(mol, aux, mf.mo_coeff, mf.mo_occ, cderi, nbatch_aux=72, atm_list=None)
        ref_j2 = get_decomposed_rij_skeleton_deriv2_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        ref_k2 = get_decomposed_rik_skeleton_deriv2_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        ref_j1ao = get_rij_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        ref_k1ao = get_rik_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        # concate the reference values into a single dict
        ref_dict = {}
        ref_dict.update(ref_j2)
        ref_dict.update(ref_k2)
        ref_dict.update(ref_j1ao)
        ref_dict.update(ref_k1ao)
        for key in ["de_J02_2"]:
            print(f"{key:>20}, val {lib.fp(result[key]):>20.12f}, ref {lib.fp(ref_dict[key]):>20.12f}")
            self.assertTrue(np.allclose(result[key], ref_dict[key], rtol=1e-4, atol=1e-6))


