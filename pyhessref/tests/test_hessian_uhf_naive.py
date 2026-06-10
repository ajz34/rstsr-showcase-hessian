import unittest

import numpy as np
from pyscf import gto, scf, lib, df, hessian

from pyhessref.util import get_dme0_unrestricted, get_dm0_unrestricted
from pyhessref.nuc_repl import UHessNucRepl, get_nuc_repl_hess
from pyhessref.hcore import UHessHcore, generator_hcore_deriv2, get_hess_hcore, generator_hcore_deriv1
from pyhessref.ovlp import RHessOvlp
from pyhessref.rijk.hess_unrestricted_naive import (
    UHessRIJKNaive,
    get_decomposed_uij_skeleton_deriv2_naive,
    get_decomposed_uik_skeleton_deriv2_naive,
    get_uij_deriv1_ao_naive,
    get_uik_deriv1_ao_naive,
)
from pyhessref.hess_scf_unrestricted import UHessSCF


def setUpModule():
    global mol, aux, mf, mf_hess, ref_value

    xyz = """
    N  0   0   0
    H  1.0 0.1 0.2
    H  0.3 1.1 0.2
    H  0.1 0.1 1.2
    """
    PATH_PROTOTYPE = "prototype/"
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


class TestHessianUHF(unittest.TestCase):
    def test_hess_nuc_repl(self):
        hess_nuc_repl = get_nuc_repl_hess(mol)
        self.assertTrue(np.allclose(hess_nuc_repl, ref_value["de_nuc"]))
        self.assertAlmostEqual(lib.fp(hess_nuc_repl), 10.942151503672441, places=10)

        hess_nuc_repl_obj = UHessNucRepl(mol)
        de_nuc_repl = hess_nuc_repl_obj.make_skeleton_hess(mf.mo_coeff, mf.mo_occ)
        self.assertTrue(np.allclose(de_nuc_repl, ref_value["de_nuc"]))

    def test_hess_hcore(self):
        dm0 = get_dm0_unrestricted(mf.mo_coeff, mf.mo_occ).sum(axis=0)
        hess_hcore = get_hess_hcore(mol, dm0)
        self.assertTrue(np.allclose(hess_hcore, ref_value["de_hcore"]))
        self.assertAlmostEqual(lib.fp(hess_hcore), -19.367829669982456, places=10)

        hess_hcore_obj = UHessHcore(mol)
        de_hcore = hess_hcore_obj.make_skeleton_hess(mf.mo_coeff, mf.mo_occ)
        self.assertTrue(np.allclose(de_hcore, ref_value["de_hcore"]))

    def test_hess_ovlp(self):
        dme0 = get_dme0_unrestricted(mf.mo_coeff, mf.mo_occ, mf.mo_energy).sum(axis=0)
        hess_ovlp_obj = RHessOvlp(mol)
        de_ovlp = hess_ovlp_obj.make_hess(dme0)
        self.assertTrue(np.allclose(de_ovlp, ref_value["de_ovlp"]))
        self.assertAlmostEqual(lib.fp(de_ovlp), 1.7951443986220534, places=10)

    def test_hess_J_skeleton_naive(self):
        de_J_skeleton = get_decomposed_uij_skeleton_deriv2_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        for key in ["de_J20", "de_J11", "de_J02"]:
            self.assertTrue(np.allclose(de_J_skeleton[key], ref_value[key], atol=1e-5, rtol=1e-4), msg=key)
        self.assertAlmostEqual(lib.fp(de_J_skeleton["de_J20"]), 4.902587371193881, places=8)
        self.assertAlmostEqual(lib.fp(de_J_skeleton["de_J11"]), 8.88765043727085, places=8)
        self.assertAlmostEqual(lib.fp(de_J_skeleton["de_J02"]), -4.445673838381621, places=8)

    def test_hess_K_skeleton_naive(self):
        de_K_skeleton = get_decomposed_uik_skeleton_deriv2_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        for key in ["de_K20", "de_K11", "de_K02"]:
            self.assertTrue(np.allclose(de_K_skeleton[key], ref_value[key], atol=1e-5, rtol=1e-4), msg=key)
        self.assertAlmostEqual(lib.fp(de_K_skeleton["de_K20"]), -0.5883149652263711, places=8)
        self.assertAlmostEqual(lib.fp(de_K_skeleton["de_K11"]), 4.536955856266865, places=8)
        self.assertAlmostEqual(lib.fp(de_K_skeleton["de_K02"]), -2.2682891819149544, places=8)

    def test_rij_deriv1(self):
        j1ao_dict = get_uij_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        j1ao = j1ao_dict["j1ao_aux0"] + j1ao_dict["j1ao_aux1"]
        # Check against PySCF (J is spin-independent, reference is [natm, 3, nao, nao])
        j1ao_ref = np.array([r[2] for r in df.hessian.uhf._gen_jk(mf_hess, mf.mo_coeff, mf.mo_occ)])
        self.assertTrue(np.allclose(j1ao, j1ao_ref))
        self.assertAlmostEqual(lib.fp(j1ao_dict["j1ao_aux0"]), 27.320873266136108, places=8)
        self.assertAlmostEqual(lib.fp(j1ao_dict["j1ao_aux1"]), 0.12413515517879808, places=8)

    def test_rik_deriv1(self):
        k1ao_dict = get_uik_deriv1_ao_naive(mol, aux, mf.mo_coeff, mf.mo_occ)
        # k1ao_dict values have shape [2, natm, 3, nao, nao]
        # PySCF returns per-spin k1ao: (vk1a, vk1b)
        k1ao_ref = np.array([r[3] for r in df.hessian.uhf._gen_jk(mf_hess, mf.mo_coeff, mf.mo_occ)])
        # k1ao_ref shape is [natm, (vk1a, vk1b)] — need to check actual structure
        for s in range(2):
            k1ao_s = k1ao_dict["k1ao_aux0"][s] + k1ao_dict["k1ao_aux1"][s]
            k1ao_ref_s = np.array([r[3][s] for r in df.hessian.uhf._gen_jk(mf_hess, mf.mo_coeff, mf.mo_occ)])
            self.assertTrue(np.allclose(k1ao_s, k1ao_ref_s), msg=f"spin {s}")
        self.assertAlmostEqual(lib.fp(k1ao_dict["k1ao_aux0"]), -6.127504869346246, places=8)
        self.assertAlmostEqual(lib.fp(k1ao_dict["k1ao_aux1"]), 0.05442798516090062, places=8)

    def test_f1ao(self):
        hess_hcore_obj = UHessHcore(mol)
        hess_rijk_obj = UHessRIJKNaive(mol, aux)

        h1ao = np.array([hess_hcore_obj.generator_deriv1()(A) for A in range(mol.natm)])
        jk1ao = hess_rijk_obj.get_deriv1_ao(mf.mo_coeff, mf.mo_occ)
        # jk1ao shape: [2, natm, 3, nao, nao]
        # f1ao_σ = h1ao + jk1ao[σ] (for UHF, jk1ao already has j1ao - k1ao_σ)
        f1ao = np.array([h1ao + jk1ao[s] for s in range(2)])

        f1ao_ref = mf_hess.make_h1(mf.mo_coeff, mf.mo_occ)
        f1ao_ref = [np.asarray(f1ao_ref[s]) for s in range(2)]
        for s in range(2):
            self.assertTrue(np.allclose(f1ao[s], f1ao_ref[s]), msg=f"spin {s}")
        self.assertAlmostEqual(lib.fp(f1ao), 8.191588278238157, places=8)

    def test_resp_bra(self):
        mo1_a = ref_value["mo1_a"]
        mo1_b = ref_value["mo1_b"]

        nmo = mo1_a.shape[-2]
        nocc_a = mo1_a.shape[-1]
        nocc_b = mo1_b.shape[-1]

        hess_rijk_obj = UHessRIJKNaive(mol, aux)
        hess_rijk_obj.make_response_preparation(mf.mo_coeff, mf.mo_occ)
        mo1_bra_a = mf.mo_coeff[0] @ mo1_a
        mo1_bra_b = mf.mo_coeff[1] @ mo1_b
        resp_bra = hess_rijk_obj.get_response_bra([mo1_bra_a, mo1_bra_b])
        resp = [mf.mo_coeff[s].T @ resp_bra[s] for s in range(2)]

        # Reference from PySCF gen_vind
        from pyhessref.util import pack_uhf_mo_pair, unpack_uhf_mo_pair

        fvind = hessian.uhf.gen_vind(mf, mf.mo_coeff, mf.mo_occ)
        mo1_flat = pack_uhf_mo_pair([mo1_a.reshape(-1, nmo, nocc_a), mo1_b.reshape(-1, nmo, nocc_b)])
        resp_flat = fvind(mo1_flat)
        resp_ref = unpack_uhf_mo_pair(resp_flat, (nmo, nocc_a), (nmo, nocc_b))
        resp_ref = [resp_ref[s].reshape(mo1_a.shape[0], 3, nmo, [nocc_a, nocc_b][s]) for s in range(2)]

        for s in range(2):
            self.assertTrue(np.allclose(resp[s], resp_ref[s], atol=1e-10, rtol=1e-8), msg=f"spin {s}")
        self.assertAlmostEqual(lib.fp(resp[0]), -0.6682133336210753, places=8)
        self.assertAlmostEqual(lib.fp(resp[1]), 1.0034934136306894, places=8)

    def test_dimensionless_cphf_rhs(self):
        hess_impl = UHessSCF(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[UHessNucRepl(mol), UHessHcore(mol)],
            el_list=[UHessRIJKNaive(mol, aux)],
        )

        pre_cphf_dict = hess_impl.compute_dimensionless_cphf_rhs()
        rhs = pre_cphf_dict["rhs"]
        self.assertAlmostEqual(lib.fp(rhs[0]), -0.01785256539468953, places=8)
        self.assertAlmostEqual(lib.fp(rhs[1]), 0.14550989432158085, places=8)

        hess_impl.make_response_preparation()
        mo1 = hess_impl.solve_dimless_cphf(rhs)

        mo1_a_ref = ref_value["mo1_a"]
        mo1_b_ref = ref_value["mo1_b"]
        # Pre-finalize Krylov solution is accurate to ~1e-5 (matches RHF behavior).
        self.assertTrue(np.allclose(mo1[0], mo1_a_ref, atol=1e-4, rtol=1e-3))
        self.assertTrue(np.allclose(mo1[1], mo1_b_ref, atol=1e-4, rtol=1e-3))
        self.assertAlmostEqual(lib.fp(mo1[0]), 0.04797427280601669, places=4)
        self.assertAlmostEqual(lib.fp(mo1[1]), -1.1346573239117455, places=4)

        result_cphf = hess_impl.finalize_cphf(mo1, pre_cphf_dict)
        mo1_fin = result_cphf["mo1"]
        mo_e1 = result_cphf["mo_e1"]
        self.assertTrue(np.allclose(mo1_fin[0], mo1_a_ref, atol=1e-5, rtol=1e-4))
        self.assertTrue(np.allclose(mo1_fin[1], mo1_b_ref, atol=1e-5, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(mo1_fin[0]), 0.04797427280601669, places=4)
        self.assertAlmostEqual(lib.fp(mo1_fin[1]), -1.1346573239117455, places=4)
        mo_e1_a_ref = ref_value["mo_e1_a"]
        mo_e1_b_ref = ref_value["mo_e1_b"]
        # mo_e1 depends on response of CPHF solution, so accuracy is one order lower than mo1.
        self.assertTrue(np.allclose(mo_e1[0], mo_e1_a_ref, atol=1e-4, rtol=1e-3))
        self.assertTrue(np.allclose(mo_e1[1], mo_e1_b_ref, atol=1e-4, rtol=1e-3))
        self.assertAlmostEqual(lib.fp(mo_e1[0]), -1.1979763394388616, places=4)
        self.assertAlmostEqual(lib.fp(mo_e1[1]), -0.20920766550023265, places=4)

        de_cphf = hess_impl.get_cphf_hess(pre_cphf_dict["f1mo"], pre_cphf_dict["s1mo"], mo1_fin, mo_e1)
        self.assertTrue(np.allclose(de_cphf, ref_value["de_cphf"], atol=1e-6, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(de_cphf), -0.40949468934990596, places=6)

    def test_make_hess(self):
        hess_impl = UHessSCF(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[UHessNucRepl(mol), UHessHcore(mol)],
            el_list=[UHessRIJKNaive(mol, aux)],
        )
        de_hess = hess_impl.make_hess()
        self.assertTrue(np.allclose(de_hess, ref_value["de_ref"], atol=1e-5, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(de_hess), 0.6241806384454698, places=5)
