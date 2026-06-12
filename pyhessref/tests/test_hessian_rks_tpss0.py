import unittest

import numpy as np
from pyscf import gto, scf, lib, dft, df, hessian
from pyhessref.nimatmul.rks import make_hessian_setup_batch
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
    grids = dft.Grids(mol)
    grids.coords = ref_value["grid_coords"]
    grids.weights = ref_value["grid_weights"]


class TestHessianRKS(unittest.TestCase):
    def test_setup(self):
        print(list(ref_value.keys()))

    def test_make_hessian_setup_whole(self):
        dm0 = get_dm0_restricted(mf.mo_coeff, mf.mo_occ)
        mocc = mf.mo_coeff[:, mf.mo_occ > 0]
        mocc_2 = mocc @ mocc.T
        result = make_hessian_setup_batch(mol, mf.xc, grids.coords, grids.weights, dm0, mocc, mocc_2)
        self.assertTrue(np.allclose(result["de_vxc_diag"], ref_value["de_vxc_diag"]))
        self.assertTrue(np.allclose(result["de_vxc"], ref_value["de_vxc"]))
        self.assertTrue(np.allclose(result["de_fxc"], ref_value["de_fxc"]))
        self.assertTrue(np.allclose(result["vmat_ip"], ref_value["vmat_ip"]))
        self.assertTrue(np.allclose(result["vmat_deriv1"], ref_value["vmat_deriv1"]))
        
