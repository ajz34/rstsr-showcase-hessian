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
    get_rij_deriv1_ao_naive,
    get_rik_deriv1_ao_naive,
)
from pyhessref.hess_impl_restricted import RHessImpl


def setUpModule():
    global mol, aux, mf, mf_hess, ref_value

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
    aux = mf.with_df.auxmol


class TestHessianRHF(unittest.TestCase):
    def test_hess_nuc_repl(self):
        hess_nuc_repl = get_nuc_repl_hess(mol)
        # numerical check
        self.assertTrue(np.allclose(hess_nuc_repl, ref_value["de_nuc"]))
        self.assertAlmostEqual(lib.fp(hess_nuc_repl), 10.942151503672441)
        # class check
        hess_nuc_repl_obj = HessNucRepl(mol)
        de_nuc_repl = hess_nuc_repl_obj.make_skeleton_hess(mf.mo_coeff, mf.mo_occ)
        self.assertTrue(np.allclose(de_nuc_repl, ref_value["de_nuc"]))

    def test_generator_hcore_deriv2(self):
        gen_hcore_deriv2 = generator_hcore_deriv2(mol)
        # functionality check
        for A in range(mol.natm):
            for B in range(mol.natm):
                hcore_deriv2_AB = gen_hcore_deriv2(A, B)
                self.assertTrue(np.allclose(hcore_deriv2_AB, mf_hess.hcore_generator()(A, B)))
        # numerical check
        self.assertAlmostEqual(lib.fp(gen_hcore_deriv2(0, 0)), -72.29474171640412)
        self.assertAlmostEqual(lib.fp(gen_hcore_deriv2(0, 1)), 12.858221292861833)

    def test_hess_hcore(self):
        hess_hcore = get_hess_hcore(mol, mf.make_rdm1())
        # numerical check
        self.assertTrue(np.allclose(hess_hcore, ref_value["de_hcore"]))
        self.assertAlmostEqual(lib.fp(hess_hcore), -16.993496707453197)
        # class check
        hess_hcore_obj = RHessHcore(mol)
        de_hcore = hess_hcore_obj.make_skeleton_hess(mf.mo_coeff, mf.mo_occ)
        self.assertTrue(np.allclose(de_hcore, ref_value["de_hcore"]))

    def test_hess_ovlp(self):
        # class check
        hess_ovlp_obj = RHessOvlp(mol)
        dme0 = get_dme0_restricted(mf.mo_coeff, mf.mo_occ, mf.mo_energy)
        de_ovlp = hess_ovlp_obj.make_hess(dme0)
        self.assertTrue(np.allclose(de_ovlp, ref_value["de_ovlp"]))
        self.assertAlmostEqual(lib.fp(de_ovlp), 0.7050335726988588)

    def test_hess_JK_skeleton_naive(self):
        # function check
        de_J_skeleton = get_decomposed_rij_skeleton_deriv2_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        for key, val in de_J_skeleton.items():
            self.assertTrue(np.allclose(val, ref_value[key], atol=1e-5, rtol=1e-4))

        de_K_skeleton = get_decomposed_rij_skeleton_deriv2_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        for key, val in de_K_skeleton.items():
            self.assertTrue(np.allclose(val, ref_value[key], atol=1e-5, rtol=1e-4))

    def test_generator_hcore_deriv1(self):
        gen_hcore_deriv1 = generator_hcore_deriv1(mol)
        # functionality check
        for A in range(mol.natm):
            hcore_deriv1_A = gen_hcore_deriv1(A)
            hcore_deriv1_A_ref = mf.nuc_grad_method().hcore_generator(mol)(A)
            self.assertTrue(np.allclose(hcore_deriv1_A, hcore_deriv1_A_ref))
        # numerical check
        self.assertAlmostEqual(lib.fp(gen_hcore_deriv1(0)), -19.44142929546185)
        self.assertAlmostEqual(lib.fp(gen_hcore_deriv1(3)), 23.88285913576012)

    def test_rij_deriv1(self):
        j1ao_dict = get_rij_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        j1ao = j1ao_dict["j1ao_aux0"] + j1ao_dict["j1ao_aux1"]

        # functionality check
        j1ao_ref = np.array([r[2] for r in df.hessian.rhf._gen_jk(mf_hess, mf.mo_coeff, mf.mo_occ)])
        self.assertTrue(np.allclose(j1ao, j1ao_ref))
        # numerical check
        self.assertAlmostEqual(lib.fp(j1ao_dict["j1ao_aux0"]), 35.38555993698421)
        self.assertAlmostEqual(lib.fp(j1ao_dict["j1ao_aux1"]), 0.11465211252634573)

    def test_kij_deriv1(self):
        k1ao_dict = get_rik_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        k1ao = k1ao_dict["k1ao_aux0"] + k1ao_dict["k1ao_aux1"]

        # functionality check
        k1ao_ref = np.array([r[3] for r in df.hessian.rhf._gen_jk(mf_hess, mf.mo_coeff, mf.mo_occ)])
        self.assertTrue(np.allclose(k1ao, k1ao_ref))
        # numerical check
        self.assertAlmostEqual(lib.fp(k1ao_dict["k1ao_aux0"]), 1.5425060495529097)
        self.assertAlmostEqual(lib.fp(k1ao_dict["k1ao_aux1"]), 0.20670656219034203)

    def test_f1ao(self):
        # functionality check
        hess_hcore_obj = RHessHcore(mol)
        hess_rijk_obj = RHessRIJKNaive(mol, aux)

        h1ao = np.array([hess_hcore_obj.generator_deriv1()(A) for A in range(mol.natm)])
        jk1ao = hess_rijk_obj.get_deriv1_ao(mf.mo_coeff, mf.mo_occ)
        f1ao = h1ao + jk1ao

        f1ao_ref = mf_hess.make_h1(mf.mo_coeff, mf.mo_occ)
        self.assertTrue(np.allclose(f1ao, f1ao_ref))

        # numerical check
        self.assertAlmostEqual(lib.fp(f1ao), 0.03306328818631421)

    def test_resp_bra(self):

        # double check of mo1 (U_{pi}^A) sanity
        mo1 = ref_value["mo1"]
        self.assertAlmostEqual(lib.fp(mo1), -0.02385155247256418)

        nmo = mo1.shape[-2]
        nocc = mo1.shape[-1]

        # functionality check
        hess_rijk_obj = RHessRIJKNaive(mol, aux)
        hess_rijk_obj.make_response_preparation(mf.mo_coeff, mf.mo_occ)
        mo1_bra = mf.mo_coeff @ mo1
        resp_bra = hess_rijk_obj.get_response_bra(mo1_bra)
        resp = mf.mo_coeff.T @ resp_bra

        resp_ref = hessian.rhf.gen_vind(mf, mf.mo_coeff, mf.mo_occ)(mo1.reshape(-1, nmo, nocc)).reshape(mo1.shape)
        self.assertTrue(np.allclose(resp, resp_ref))

        # numerical check
        self.assertAlmostEqual(lib.fp(resp), -0.07694258336883628)

    def test_dimensionless_cphf_rhs(self):
        hess_impl = RHessImpl(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[HessNucRepl(mol), RHessHcore(mol)],
            el_list=[RHessRIJKNaive(mol, aux)],
        )

        # before krylov, first obtain dimensionless rhs part
        pre_cphf_dict = hess_impl.compute_dimensionless_cphf_rhs()
        self.assertAlmostEqual(lib.fp(pre_cphf_dict["rhs"]), -0.027755691019085788)

        # solve cphf
        rhs = pre_cphf_dict["rhs"]
        hess_impl.make_response_preparation()
        mo1 = hess_impl.solve_dimless_cphf(rhs)
        self.assertTrue(np.allclose(mo1, ref_value["mo1"], atol=1e-6, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(mo1), -0.02385155247256418, places=6)

        # finalize cphf
        result_cphf = hess_impl.finalize_cphf(mo1, pre_cphf_dict)
        self.assertTrue(np.allclose(result_cphf["mo1"], ref_value["mo1"], atol=1e-6, rtol=1e-4))
        self.assertTrue(np.allclose(result_cphf["mo_e1"], ref_value["mo_e1"], atol=1e-6, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(result_cphf["mo1"]), -0.02385155247256418, places=6)
        self.assertAlmostEqual(lib.fp(result_cphf["mo_e1"]), 0.2961618130386303, places=6)

        # compute de_cphf
        mo1 = result_cphf["mo1"]
        mo_e1 = result_cphf["mo_e1"]
        f1mo = pre_cphf_dict["f1mo"]
        s1mo = pre_cphf_dict["s1mo"]
        de_cphf = hess_impl.get_cphf_hess(f1mo, s1mo, mo1, mo_e1)
        self.assertTrue(np.allclose(de_cphf, ref_value["de_cphf"], atol=1e-6, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(de_cphf), 1.0888788930763051, places=6)
    
    def test_make_hess(self):
        hess_impl = RHessImpl(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[HessNucRepl(mol), RHessHcore(mol)],
            el_list=[RHessRIJKNaive(mol, aux)],
        )
        de_hess = hess_impl.make_hess()
        self.assertTrue(np.allclose(de_hess, ref_value["de_ref"], atol=1e-6, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(de_hess), 1.4704252379360374, places=5)
