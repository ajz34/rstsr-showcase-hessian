import unittest

import numpy as np
from pyscf import gto, scf, lib, dft, df
from pyhessref.nimatmul.uks_with_becke import get_quad_split, make_hessian_setup_uks, UHessKSNaiveBecke
from pyhessref.rijk.hess_unrestricted_naive import UHessRIJKNaive
from pyhessref.hess_scf_unrestricted import UHessSCF
from pyhessref.hcore import UHessHcore
from pyhessref.nuc_repl import UHessNucRepl
from pyhessref.ovlp import RHessOvlp
from pyhessref.util import get_dm0_unrestricted


def setUpModule():
    global mol, aux, mf, mf_hess, grids, ref_value
    lib.num_threads(4)

    xyz = """
    N  0   0   0
    H  1.0 0.1 0.2
    H  0.3 1.1 0.2
    H  0.1 0.1 1.2
    """
    PATH_PROTOTYPE = "prototype/"  # assuming run at project root
    PATH_REF = PATH_PROTOTYPE + "nh3_u_b3lyp_decomp.npz"

    mol = gto.Mole(atom=xyz, basis="def2-TZVP", spin=2, max_memory=8000).build()
    mf = dft.UKS(mol, xc="B3LYP").density_fit()
    ref_value = np.load(PATH_REF)
    mf.mo_coeff = ref_value["mo_coeff"]
    mf.mo_occ = ref_value["mo_occ"]
    mf.mo_energy = ref_value["mo_energy"]
    mf.with_df.build()
    mf.converged = True
    mf_hess = mf.Hessian()
    aux = mf.with_df.auxmol
    grids = dft.Grids(mol).build(sort_grids=False)


class TestHessianUKS(unittest.TestCase):
    def test_setup(self):
        print(list(ref_value.keys()))

    def test_make_hessian_setup_whole(self):
        natm = mol.natm
        atm_quad_split = get_quad_split(grids.atm_idx)
        quadrature_weights = grids.quadrature_weights
        becke_scheme = grids.radii_adjust(mol, grids.atomic_radii)
        adjustment_factor = np.array([becke_scheme(i, j, 0) for i in range(natm) for j in range(natm)]).reshape(
            natm, natm
        )
        dm0a, dm0b = get_dm0_unrestricted(mf.mo_coeff, mf.mo_occ)
        result = make_hessian_setup_uks(
            mol, mf.xc, grids.coords, grids.weights, dm0a, dm0b, atm_quad_split, quadrature_weights, adjustment_factor
        )
        # grid-fixed skeleton parts match the (grid-fixed) reference
        self.assertTrue(np.allclose(result["de_fxc"], ref_value["de_fxc"]))
        self.assertTrue(np.allclose(result["de_vxc_diag_a"], ref_value["de_vxc_diag_a"]))
        self.assertTrue(np.allclose(result["de_vxc_diag_b"], ref_value["de_vxc_diag_b"]))
        self.assertTrue(np.allclose(result["de_vxc_off_a"], ref_value["de_vxc_off_a"]))
        self.assertTrue(np.allclose(result["de_vxc_off_b"], ref_value["de_vxc_off_b"]))
        self.assertTrue(np.allclose(result["vmat_ip_a"], ref_value["vmat_ip_a"]))
        self.assertTrue(np.allclose(result["vmat_ip_b"], ref_value["vmat_ip_b"]))
        self.assertTrue(np.allclose(result["vmat_deriv1_a"], ref_value["vmat_deriv1_a"]))
        self.assertTrue(np.allclose(result["vmat_deriv1_b"], ref_value["vmat_deriv1_b"]))
        self.assertAlmostEqual(lib.fp(result["de_fxc"]), -20.608327622840, places=5)

        de_xc_skeleton = result["de_xc_skeleton"]
        print("de_xc_skeleton reduced\n", de_xc_skeleton.sum(axis=(0, 1)))
        assert (
            np.abs(de_xc_skeleton.sum(axis=(0, 1))).max() < 1e-9
        ), "translational invariance check failed for de_xc_skeleton"

        # f1ao (CP-KS RHS) grid-shift: per-spin vmat_deriv1_grid sums to ~0 over atoms
        for s in "ab":
            print(f"vmat_deriv1_{s} (skeleton) sum(A) max:", np.abs(result[f"vmat_deriv1_{s}"].sum(axis=0)).max())
            print(f"vmat_deriv1_grid_{s} sum(A) max:     ", np.abs(result[f"vmat_deriv1_grid_{s}"].sum(axis=0)).max())
            assert (
                np.abs(result[f"vmat_deriv1_grid_{s}"].sum(axis=0)).max() < 1e-9
            ), f"translational invariance check failed for vmat_deriv1_grid_{s} (f1ao)"

    def test_make_hess(self):
        """End-to-end UKS Hessian via UHessSCF with the Becke grid-shift KS object.

        Mirrors `test_make_hess` in `test_hessian_uks_b3lyp.py`, but the XC
        skeleton Hessian and the CP-KS right-hand side carry the grid-shift
        increment, so the result is translationally invariant while differing
        from the (grid-fixed) PySCF reference by the grid-shift magnitude
        (~2e-3 for the spin-polarized GGA case).
        """
        natm = mol.natm
        ni = mf._numint
        _, _, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        hess_obj_rijk = UHessRIJKNaive(mol, aux, scale_j=1.0, scale_k=hyb)
        hess_obj_ks = UHessKSNaiveBecke(mol, mf.xc, grids)

        hess_impl = UHessSCF(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[UHessNucRepl(mol), UHessHcore(mol)],
            el_list=[hess_obj_rijk, hess_obj_ks],
        )
        de_hess = hess_impl.make_hess()

        diff = np.abs(de_hess - ref_value["de_ref"])
        print("max|de_hess - de_ref| (grid-shift corrected vs PySCF ref):", diff.max())
        self.assertTrue(
            np.allclose(de_hess, ref_value["de_ref"], atol=4e-3, rtol=0),
            msg=f"max abs diff = {diff.max()}",
        )

        # translational invariance: sum over both atom indices -> ~0
        de_hess_4d = de_hess.reshape(natm, 3, natm, 3)
        invariance = np.abs(de_hess_4d.sum(axis=(0, 2))).max()
        print("full Hessian translational invariance max:", invariance)
        self.assertLess(invariance, 1e-7, msg=f"translational invariance = {invariance}")
