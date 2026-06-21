import unittest

import numpy as np
from pyscf import gto, scf, lib, df, hessian

from pyhessref.ovlp import RHessOvlp
from pyhessref.nuc_repl import UHessNucRepl
from pyhessref.hcore import UHessHcore
from pyhessref.rijk.hess_unrestricted_naive import (
    UHessRIJKNaive,
    get_decomposed_uij_skeleton_deriv2_naive,
    get_decomposed_uik_skeleton_deriv2_naive,
    get_uij_deriv1_ao_naive,
    get_uik_deriv1_ao_naive,
)
from pyhessref.rijk.hess_unrestricted_opt_prototype import UHessRIJKOptPrototype
from pyhessref.hess_scf_unrestricted import UHessSCF


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
    PATH_REF = PATH_PROTOTYPE + "nh3_u_hf_decomp.npz"

    mol = gto.Mole(atom=xyz, basis="def2-TZVP", charge=2, spin=2, max_memory=32000).build()
    mf = scf.UHF(mol).density_fit()
    ref_value = np.load(PATH_REF)
    mf.mo_coeff = ref_value["mo_coeff"]
    mf.mo_occ = ref_value["mo_occ"]
    mf.mo_energy = ref_value["mo_energy"]
    mf.with_df.build()
    mf.converged = True
    mf_hess = mf.Hessian().run()
    aux = mf.with_df.auxmol


class TestHessianUHFOptPrototype(unittest.TestCase):
    def test_uhess_rijk_opt_api(self):
        """The optimized-prototype class matches UHessRIJKNaive on skeleton + deriv1_bra."""
        cderi = mf.with_df._cderi
        obj_naive = UHessRIJKNaive(mol, aux)
        obj_opt = UHessRIJKOptPrototype(mol, aux, cderi, nbatch_aux=72)

        # skeleton hessian (matmul vs einsum accumulation order -> ~1e-6 wobble; relax to 1e-5)
        sk_naive = obj_naive.make_skeleton_hess(mf.mo_coeff, mf.mo_occ)
        sk_opt = obj_opt.make_skeleton_hess(mf.mo_coeff, mf.mo_occ)
        self.assertTrue(np.allclose(sk_opt, sk_naive, atol=1e-5, rtol=1e-4))

        # first-derivative bra form (the API method UHessSCF consumes), per spin
        bra_naive = obj_naive.get_deriv1_bra(mf.mo_coeff, mf.mo_occ)
        bra_opt = obj_opt.get_deriv1_bra(mf.mo_coeff, mf.mo_occ)
        self.assertEqual(len(bra_opt), 2)
        for s in range(2):
            self.assertEqual(bra_opt[s].shape, bra_naive[s].shape)
            self.assertTrue(np.allclose(bra_opt[s], bra_naive[s], atol=1e-6, rtol=1e-5), msg=f"spin {s}")

        # the optimized class deliberately does not produce the full AO first-derivative
        with self.assertRaises(NotImplementedError):
            obj_opt.get_deriv1_ao(mf.mo_coeff, mf.mo_occ)

    def test_make_hess(self):
        """End-to-end Hessian through UHessSCF using the optimized-prototype RI-JK object.

        Mirrors test_hessian_uhf_naive.test_make_hess, but swaps the electronic-interaction
        provider from UHessRIJKNaive to the optimized UHessRIJKOptPrototype (which evaluates K1
        in the half-transformed bra form via get_deriv1_bra). The assembled Hessian must
        reproduce the stored reference.
        """
        cderi = mf.with_df._cderi
        hess_impl = UHessSCF(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[UHessNucRepl(mol), UHessHcore(mol)],
            el_list=[UHessRIJKOptPrototype(mol, aux, cderi, nbatch_aux=72)],
        )
        de_hess = hess_impl.make_hess()
        self.assertTrue(np.allclose(de_hess, ref_value["de_ref"], atol=1e-5, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(de_hess), 0.6241806384454698, places=4)
