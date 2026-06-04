import unittest

import numpy as np
from pyscf import gto, scf


def setUpModule():
    global mol, mf, mf_hess, ref_value

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
    mf_hess = mf.Hessian().run()


class TestHessianRHF(unittest.TestCase):
    def test_nuc_repl_hess(self):
        from pyhessref.rijk.pure_hessian_rhf import get_nuc_repl_hess

        hess_nuc_repl = get_nuc_repl_hess(mol)
        np.testing.assert_allclose(hess_nuc_repl, ref_value["de_nuc"], atol=1e-8)
