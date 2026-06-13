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
    grids = dft.Grids(mol)
    grids.coords = ref_value["grid_coords"]
    grids.weights = ref_value["grid_weights"]


class TestHessianRKS(unittest.TestCase):
    def test_setup(self):
        print(list(ref_value.keys()))

    def test_make_hessian_setup_whole(self):
        dm0 = get_dm0_restricted(mf.mo_coeff, mf.mo_occ)
        result = make_hessian_setup_batch(mol, mf.xc, grids.coords, grids.weights, dm0)
        self.assertTrue(np.allclose(result["de_vxc_diag"], ref_value["de_vxc_diag"]))
        self.assertTrue(np.allclose(result["de_vxc_off"], ref_value["de_vxc_off"]))
        self.assertTrue(np.allclose(result["de_fxc"], ref_value["de_fxc"]))
        self.assertTrue(np.allclose(result["vmat_ip"], ref_value["vmat_ip"]))
        self.assertTrue(np.allclose(result["vmat_deriv1"], ref_value["vmat_deriv1"]))
        self.assertAlmostEqual(lib.fp(result["de_vxc_diag"]), 49.688766385730304, places=5)
        self.assertAlmostEqual(lib.fp(result["de_vxc_off"]), -29.337474734527515, places=5)
        self.assertAlmostEqual(lib.fp(result["de_fxc"]), -21.249874465163057, places=5)
        self.assertAlmostEqual(lib.fp(result["vmat_deriv1"]), -3.8658927361526123, places=5)

    def test_make_hessian_setup_batched(self):
        """Run the setup in roughly equal grid batches (4 batches total)
        and check the summed result matches the whole-grid reference.
        Every output of make_hessian_setup_batch is linear in the grid
        weights, so a sum over disjoint grid batches is exact (up to
        floating-point error).
        """
        dm0 = get_dm0_restricted(mf.mo_coeff, mf.mo_occ)
        ngrids = grids.weights.size
        batch_size = ngrids // 4 + 1

        result_sum = None
        for start in range(0, ngrids, batch_size):
            stop = min(start + batch_size, ngrids)
            coords_batch = grids.coords[start:stop]
            weights_batch = grids.weights[start:stop]
            partial = make_hessian_setup_batch(
                mol, mf.xc, coords_batch, weights_batch, dm0, verbose=False,
            )
            if result_sum is None:
                result_sum = {k: v.copy() for k, v in partial.items()}
            else:
                for k in result_sum:
                    result_sum[k] += partial[k]

        for k in ("de_vxc_diag", "de_vxc_off", "de_fxc", "vmat_ip", "vmat_deriv1"):
            self.assertTrue(
                np.allclose(result_sum[k], ref_value[k]),
                msg=f"batched {k} mismatch",
            )
