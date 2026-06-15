import unittest

import numpy as np
from pyscf import gto, scf, lib, dft, df
from pyhessref.nimatmul.uks import (
    get_uks_response_bra_naive,
    make_hessian_setup_batch_uks,
    UHessKSNaive,
)
from pyhessref.rijk.hess_unrestricted_naive import UHessRIJKNaive
from pyhessref.hess_scf_unrestricted import UHessSCF
from pyhessref.hcore import UHessHcore
from pyhessref.nuc_repl import UHessNucRepl
from pyhessref.ovlp import RHessOvlp
from pyhessref.util import get_dm0_unrestricted


def setUpModule():
    global mol, aux, mf, grids, ref_value
    lib.num_threads(4)

    xyz = """
    N  0   0   0
    H  1.0 0.1 0.2
    H  0.3 1.1 0.2
    H  0.1 0.1 1.2
    """
    PATH_PROTOTYPE = "prototype/"
    PATH_REF = PATH_PROTOTYPE + "nh3_u_b3lyp_decomp.npz"

    mol = gto.Mole(atom=xyz, basis="def2-TZVP", spin=2, max_memory=8000).build()
    mf = dft.UKS(mol, xc="B3LYP").density_fit()
    ref_value = np.load(PATH_REF)
    mf.mo_coeff = ref_value["mo_coeff"]
    mf.mo_occ = ref_value["mo_occ"]
    mf.mo_energy = ref_value["mo_energy"]
    mf.with_df.build()
    mf.converged = True
    aux = mf.with_df.auxmol
    grids = dft.Grids(mol)
    grids.coords = ref_value["grid_coords"]
    grids.weights = ref_value["grid_weights"]


class TestHessianUKS(unittest.TestCase):
    def test_setup(self):
        print(list(ref_value.keys()))

    def test_make_hessian_setup_whole(self):
        dm0_per_spin = get_dm0_unrestricted(mf.mo_coeff, mf.mo_occ)
        dm0a, dm0b = dm0_per_spin[0], dm0_per_spin[1]
        result = make_hessian_setup_batch_uks(
            mol, mf.xc, grids.coords, grids.weights, dm0a, dm0b
        )
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

    def test_make_hessian_setup_batched(self):
        dm0_per_spin = get_dm0_unrestricted(mf.mo_coeff, mf.mo_occ)
        dm0a, dm0b = dm0_per_spin[0], dm0_per_spin[1]
        coords = grids.coords
        weights = grids.weights
        ngrids = weights.size

        # Run in 4 batches
        result_sum = None
        for start in range(0, ngrids, ngrids // 4):
            stop = min(start + ngrids // 4, ngrids)
            partial = make_hessian_setup_batch_uks(
                mol, mf.xc, coords[start:stop], weights[start:stop], dm0a, dm0b, verbose=False
            )
            if result_sum is None:
                result_sum = {k: v.copy() for k, v in partial.items()}
            else:
                for k in result_sum:
                    result_sum[k] += partial[k]

        self.assertTrue(np.allclose(result_sum["de_fxc"], ref_value["de_fxc"]))

    def test_make_hess(self):
        ni = mf._numint
        _, _, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

        ks_obj = UHessKSNaive(mol, mf.xc, grids)
        hess_scf = UHessSCF(
            mol,
            mf.mo_coeff,
            mf.mo_occ,
            mf.mo_energy,
            ovlp_obj=RHessOvlp(mol),
            core_list=[UHessNucRepl(mol), UHessHcore(mol)],
            el_list=[
                UHessRIJKNaive(mol, aux, scale_j=1.0, scale_k=hyb),
                ks_obj,
            ],
        )
        de_hess = hess_scf.make_hess()
        self.assertTrue(np.allclose(de_hess, ref_value["de_ref"], atol=5e-4, rtol=1e-4))
        self.assertAlmostEqual(lib.fp(de_hess), 0.661032172085, places=4)

    def test_response(self):
        dm0_per_spin = get_dm0_unrestricted(mf.mo_coeff, mf.mo_occ)
        dm0a, dm0b = dm0_per_spin[0], dm0_per_spin[1]

        vmat_deriv1_mo_a = ref_value["vmat_deriv1_mo_a"]
        vmat_deriv1_mo_b = ref_value["vmat_deriv1_mo_b"]

        ks_obj = UHessKSNaive(mol, mf.xc, grids)
        ks_obj.make_response_preparation(mf.mo_coeff, mf.mo_occ)

        bra_a = vmat_deriv1_mo_a.reshape(-1, mol.nao, vmat_deriv1_mo_a.shape[-1])
        bra_b = vmat_deriv1_mo_b.reshape(-1, mol.nao, vmat_deriv1_mo_b.shape[-1])

        resp = get_uks_response_bra_naive(
            mol,
            grids,
            mf.xc,
            mf.mo_coeff,
            mf.mo_occ,
            dm0a,
            dm0b,
            [bra_a, bra_b],
            rho_cached=ks_obj.rho_cached,
            vxc_cached=ks_obj.vxc_cached,
            fxc_cached=ks_obj.fxc_cached,
        )
        total = np.concatenate([resp[0].ravel(), resp[1].ravel()])
        self.assertAlmostEqual(lib.fp(total), -0.023459554665, places=6)


if __name__ == "__main__":
    unittest.main()
