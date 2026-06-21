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
from pyhessref.hess_scf_restricted import RHessSCF


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
        ref_j1ao = get_rij_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        ref_k1ao = get_rik_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        ref_dict = dict(ref_value).copy()
        ref_dict.update(ref_j1ao)
        ref_dict.update(ref_k1ao)
        for key in sorted(result.keys()):
            print(f"{key:<20}, val {lib.fp(result[key]):>20.12f}, ref {lib.fp(ref_dict[key]):>20.12f}")
            self.assertTrue(np.allclose(result[key], ref_dict[key], rtol=1e-4, atol=1e-6))

    def test_get_decomposed_skeleton_separated(self):
        from pyhessref.rijk.hess_restricted_opt_prototype import (
            get_decomposed_skeleton,
            get_decomposed_skeleton_separated,
        )

        cderi = mf.with_df._cderi
        j_res, k_res = get_decomposed_skeleton_separated(
            mol, aux, mf.mo_coeff, mf.mo_occ, cderi, nbatch_aux=72, atm_list=None
        )
        # structured return: J dict + list of 1 K-spin dict (RHF)
        self.assertIsInstance(j_res, dict)
        self.assertEqual(len(k_res), 1)
        result = {**j_res, **k_res[0]}

        ref_j1ao = get_rij_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        ref_k1ao = get_rik_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        ref_dict = dict(ref_value).copy()
        ref_dict.update(ref_j1ao)
        ref_dict.update(ref_k1ao)
        # must produce the same key set as the baseline
        baseline = get_decomposed_skeleton(mol, aux, mf.mo_coeff, mf.mo_occ, cderi, nbatch_aux=72, atm_list=None)
        self.assertEqual(set(result.keys()), set(baseline.keys()))
        for key in sorted(result.keys()):
            print(f"{key:<20}, val {lib.fp(result[key]):>20.12f}, ref {lib.fp(ref_dict[key]):>20.12f}")
            self.assertTrue(np.allclose(result[key], ref_dict[key], rtol=1e-4, atol=1e-6))

    def test_get_decomposed_skeleton_separated_dojk(self):
        """do_j / do_k flags skip the unused half's tensors and work entirely."""
        from pyhessref.rijk.hess_restricted_opt_prototype import (
            get_decomposed_skeleton_separated,
        )

        cderi = mf.with_df._cderi
        # J-only
        j_only, k_only = get_decomposed_skeleton_separated(
            mol, aux, mf.mo_coeff, mf.mo_occ, cderi, nbatch_aux=72, do_j=True, do_k=False
        )
        self.assertIsInstance(j_only, dict)
        self.assertEqual(k_only, [])
        self.assertTrue(all(k.startswith("de_J") or k.startswith("j1ao") for k in j_only))
        # K-only
        j_none, k_only2 = get_decomposed_skeleton_separated(
            mol, aux, mf.mo_coeff, mf.mo_occ, cderi, nbatch_aux=72, do_j=False, do_k=True
        )
        self.assertIsNone(j_none)
        self.assertEqual(len(k_only2), 1)
        self.assertTrue(all(k.startswith("de_K") or k.startswith("k1") for k in k_only2[0]))
        # J-only with user-supplied dm0 must match J of the full call
        from pyhessref.util import get_dm0_restricted
        dm0 = get_dm0_restricted(mf.mo_coeff, mf.mo_occ)
        j_dm0, _ = get_decomposed_skeleton_separated(
            mol, aux, mf.mo_coeff, mf.mo_occ, cderi, nbatch_aux=72, do_j=True, do_k=False, dm0=dm0
        )
        for k in j_only:
            self.assertTrue(np.allclose(j_only[k], j_dm0[k], atol=1e-12), msg=k)

    def test_rhess_rijk_opt_api(self):
        """The optimized-prototype class matches RHessRIJKNaive on skeleton + deriv1_bra."""
        from pyhessref.rijk.hess_restricted_opt_prototype import RHessRIJKOptPrototype

        cderi = mf.with_df._cderi
        obj_naive = RHessRIJKNaive(mol, aux)
        obj_opt = RHessRIJKOptPrototype(mol, aux, cderi, nbatch_aux=72)

        # skeleton hessian
        sk_naive = obj_naive.make_skeleton_hess(mf.mo_coeff, mf.mo_occ)
        sk_opt = obj_opt.make_skeleton_hess(mf.mo_coeff, mf.mo_occ)
        self.assertTrue(np.allclose(sk_opt, sk_naive, atol=1e-6, rtol=1e-5))

        # first-derivative bra form (the API method RHessSCF consumes)
        bra_naive = obj_naive.get_deriv1_bra(mf.mo_coeff, mf.mo_occ)
        bra_opt = obj_opt.get_deriv1_bra(mf.mo_coeff, mf.mo_occ)
        self.assertTrue(np.allclose(bra_opt, bra_naive, atol=1e-6, rtol=1e-5))

        # the optimized class deliberately does not produce the full AO first-derivative
        with self.assertRaises(NotImplementedError):
            obj_opt.get_deriv1_ao(mf.mo_coeff, mf.mo_occ)

    def test_make_hess(self):
        """End-to-end Hessian through RHessSCF using the optimized-prototype RI-JK object.

        Mirrors test_hessian_rhf_naive.test_make_hess, but swaps the electronic-interaction
        provider from RHessRIJKNaive to the optimized RHessRIJKOptPrototype (which evaluates
        K1 in the half-transformed bra form via get_deriv1_bra). The assembled Hessian must
        reproduce the stored reference.
        """
        from pyhessref.rijk.hess_restricted_opt_prototype import RHessRIJKOptPrototype

        cderi = mf.with_df._cderi
        hess_impl = RHessSCF(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[HessNucRepl(mol), RHessHcore(mol)],
            el_list=[RHessRIJKOptPrototype(mol, aux, cderi, nbatch_aux=72)],
        )
        de_hess = hess_impl.make_hess()
        self.assertTrue(np.allclose(de_hess, ref_value["de_ref"], atol=1e-6, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(de_hess), 1.4704252379360374, places=5)
