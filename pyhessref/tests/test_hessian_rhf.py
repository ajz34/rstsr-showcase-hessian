import unittest

import numpy as np
from pyscf import gto, scf, lib


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
    def test_hess_nuc_repl(self):
        from pyhessref.nuc_repl import get_nuc_repl_hess

        hess_nuc_repl = get_nuc_repl_hess(mol)
        # numerical check
        self.assertTrue(np.allclose(hess_nuc_repl, ref_value["de_nuc"]))
        self.assertAlmostEqual(lib.fp(hess_nuc_repl), 10.942151503672441)
        # class check
        from pyhessref.nuc_repl import HessNucRepl

        hess_nuc_repl_obj = HessNucRepl(mol)
        de_nuc_repl = hess_nuc_repl_obj.make_hess(mf.mo_coeff, mf.mo_occ, mf.mo_energy)
        self.assertTrue(np.allclose(de_nuc_repl, ref_value["de_nuc"]))

    def test_generator_hcore_deriv2(self):
        from pyhessref.hcore import generator_hcore_deriv2

        gen_hcore_deriv2 = generator_hcore_deriv2(mol)
        # functionality check
        for A in range(mol.natm):
            for B in range(mol.natm):
                hcore_deriv2_AB = gen_hcore_deriv2(A, B)
                self.assertTrue(
                    np.allclose(hcore_deriv2_AB, mf_hess.hcore_generator()(A, B))
                )
        # numerical check
        self.assertAlmostEqual(lib.fp(gen_hcore_deriv2(0, 0)), -72.29474171640412)
        self.assertAlmostEqual(lib.fp(gen_hcore_deriv2(0, 1)), 12.858221292861833)

    def test_hess_hcore(self):
        from pyhessref.hcore import get_hess_hcore

        hess_hcore = get_hess_hcore(mol, mf.make_rdm1())
        # numerical check
        self.assertTrue(np.allclose(hess_hcore, ref_value["de_hcore"]))
        self.assertAlmostEqual(lib.fp(hess_hcore), -16.993496707453197)
        # class check
        from pyhessref.hcore import HessHcore

        hess_hcore_obj = HessHcore(mol)
        de_hcore = hess_hcore_obj.make_hess(mf.mo_coeff, mf.mo_occ, mf.mo_energy)
        self.assertTrue(np.allclose(de_hcore, ref_value["de_hcore"]))

    def test_hess_ovlp(self):
        # class check
        from pyhessref.ovlp import HessOvlp

        hess_ovlp_obj = HessOvlp(mol)
        de_ovlp = hess_ovlp_obj.make_hess(mf.mo_coeff, mf.mo_occ, mf.mo_energy)
        self.assertTrue(np.allclose(de_ovlp, ref_value["de_ovlp"]))
        self.assertAlmostEqual(lib.fp(de_ovlp), 0.7050335726988588)
