import unittest
from unittest import result

import numpy as np
from pyscf import gto, scf, lib, dft, df, hessian
from pyhessref.nimatmul.rks_with_becke import get_quad_split, make_hessian_setup
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
    PATH_REF = PATH_PROTOTYPE + "nh3_r_b3lyp_decomp.npz"

    mol = gto.Mole(atom=xyz, basis="def2-TZVP", max_memory=8000).build()
    mf = scf.RKS(mol, xc="B3LYP").density_fit()
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
        self.assertAlmostEqual(lib.fp(result["de_vxc_diag"]), 49.688766385730304, places=5)
        self.assertAlmostEqual(lib.fp(result["de_vxc_off"]), -29.337474734527515, places=5)
        self.assertAlmostEqual(lib.fp(result["de_fxc"]), -21.249874465163057, places=5)
        self.assertAlmostEqual(lib.fp(result["vmat_deriv1"]), -3.8658927361526123, places=5)

        de_xc_skeleton = result["de_xc_skeleton"]
        print("de_xc_skeleton reduced\n", de_xc_skeleton.sum(axis=(0, 1)))
        assert (
            np.abs(de_xc_skeleton.sum(axis=(0, 1))).max() < 1e-9
        ), "translational invariance check failed for de_xc_skeleton"

    def test_make_hessian_setup_f1ao_grid(self):
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
        # skeleton (grid-fixed) DFT part of f1ao: non-invariant at ~1e-5 level
        print("vmat_deriv1 (skeleton) sum(A) max:", np.abs(result["vmat_deriv1"].sum(axis=0)).max())
        print("vmat_deriv1_grid sum(A) max:     ", np.abs(result["vmat_deriv1_grid"].sum(axis=0)).max())
        assert (
            np.abs(result["vmat_deriv1_grid"].sum(axis=0)).max() < 1e-9
        ), "translational invariance check failed for vmat_deriv1_grid (f1ao)"
