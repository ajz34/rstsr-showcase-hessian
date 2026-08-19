import unittest

import numpy as np
from pyscf import gto, scf, lib, dft
from pyhessref.nimatmul.rks_with_becke import get_quad_split, make_hessian_setup, RHessKSNaiveBecke
from pyhessref.rijk.hess_restricted_naive import RHessRIJKNaive
from pyhessref.hess_scf_restricted import RHessSCF
from pyhessref.hcore import RHessHcore
from pyhessref.nuc_repl import HessNucRepl
from pyhessref.ovlp import RHessOvlp
from pyhessref.util import get_dm0_restricted


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
    PATH_REF = PATH_PROTOTYPE + "nh3_r_tpss0_decomp.npz"

    mol = gto.Mole(atom=xyz, basis="def2-TZVP", max_memory=8000).build()
    mf = scf.RKS(mol, xc="TPSS0").density_fit()
    ref_value = np.load(PATH_REF)
    mf.mo_coeff = ref_value["mo_coeff"]
    mf.mo_occ = ref_value["mo_occ"]
    mf.mo_energy = ref_value["mo_energy"]
    mf.with_df.build()
    mf.converged = True
    mf_hess = mf.Hessian()
    aux = mf.with_df.auxmol
    grids = dft.Grids(mol).build(sort_grids=False)


class TestHessianRKS(unittest.TestCase):
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
        dm0 = get_dm0_restricted(mf.mo_coeff, mf.mo_occ)
        result = make_hessian_setup(
            mol, mf.xc, grids.coords, grids.weights, dm0, atm_quad_split, quadrature_weights, adjustment_factor
        )
        self.assertTrue(np.allclose(result["de_vxc_diag"], ref_value["de_vxc_diag"]))
        self.assertTrue(np.allclose(result["de_vxc_off"], ref_value["de_vxc_off"]))
        self.assertTrue(np.allclose(result["de_fxc"], ref_value["de_fxc"]))
        self.assertTrue(np.allclose(result["vmat_ip"], ref_value["vmat_ip"]))
        self.assertTrue(np.allclose(result["vmat_deriv1"], ref_value["vmat_deriv1"]))
        self.assertAlmostEqual(lib.fp(result["de_vxc_diag"]), 44.683863589574251, places=5)
        self.assertAlmostEqual(lib.fp(result["de_vxc_off"]), -16.124876249597346, places=5)
        self.assertAlmostEqual(lib.fp(result["de_fxc"]), -29.390069496787994, places=5)
        self.assertAlmostEqual(lib.fp(result["vmat_deriv1"]), -3.418468953177161, places=5)

        de_xc_skeleton = result["de_xc_skeleton"]
        print("de_xc_skeleton reduced\n", de_xc_skeleton.sum(axis=(0, 1)))
        assert (
            np.abs(de_xc_skeleton.sum(axis=(0, 1))).max() < 1e-9
        ), "translational invariance check failed for de_xc_skeleton"

        # f1ao (CP-KS RHS) grid-shift: vmat_deriv1_grid sums to ~0 over atoms
        print("vmat_deriv1 (skeleton) sum(A) max:", np.abs(result["vmat_deriv1"].sum(axis=0)).max())
        print("vmat_deriv1_grid sum(A) max:     ", np.abs(result["vmat_deriv1_grid"].sum(axis=0)).max())
        assert (
            np.abs(result["vmat_deriv1_grid"].sum(axis=0)).max() < 1e-9
        ), "translational invariance check failed for vmat_deriv1_grid (f1ao)"

    def test_make_hess(self):
        """End-to-end RKS Hessian via RHessSCF with the Becke grid-shift KS object.

        Mirrors `test_make_hess` in `test_hessian_rks_tpss0.py`, but the XC
        skeleton Hessian and the CP-KS right-hand side carry the grid-shift
        increment, so the result is translationally invariant while differing
        from the (grid-fixed) PySCF reference by the grid-shift magnitude
        (~4e-3 for MGGA, ~10x the GGA case, dominated by the tau channel).
        """
        natm = mol.natm
        hyb = float(ref_value["hyb"])
        hess_obj_rijk = RHessRIJKNaive(mol, aux, scale_j=1.0, scale_k=hyb)
        hess_obj_ks = RHessKSNaiveBecke(mol, mf.xc, grids)

        hess_impl = RHessSCF(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[HessNucRepl(mol), RHessHcore(mol)],
            el_list=[hess_obj_rijk, hess_obj_ks],
        )
        de_hess = hess_impl.make_hess()

        diff = np.abs(de_hess - ref_value["de_ref"])
        print("max|de_hess - de_ref| (grid-shift corrected vs PySCF ref):", diff.max())
        self.assertTrue(
            np.allclose(de_hess, ref_value["de_ref"], atol=1e-2, rtol=0),
            msg=f"max abs diff = {diff.max()}",
        )

        # translational invariance: sum over both atom indices -> ~0
        de_hess_4d = de_hess.reshape(natm, 3, natm, 3)
        invariance = np.abs(de_hess_4d.sum(axis=(0, 2))).max()
        print("full Hessian translational invariance max:", invariance)
        self.assertLess(invariance, 1e-7, msg=f"translational invariance = {invariance}")
