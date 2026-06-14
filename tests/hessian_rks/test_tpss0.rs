use crate::hessian_rks::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use libxc::prelude::*;
use rstest::rstest;
use rstsr_showcase_hessian::numint_matmul::hess_rks::make_hessian_setup_batch;
use rstsr_showcase_hessian::prelude::*;
use rstsr_showcase_hessian::util::density_matrices::get_dm0_restricted;

#[rstest]
fn test_whole(hess_case_tpss0: &CaseAmoniaRKS) {
    let CaseAmoniaRKS { mol, mo_coeff, mo_occ, grid_coords, grid_weights, ref_dict, .. } = hess_case_tpss0.clone();
    let mut ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let xc_func = LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSS0", LibXCSpin::Unpolarized);
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let result = make_hessian_setup_batch(&mol, &xc_func, &mut ni, dm0.view(), None, true);

    assert!(rt::allclose(result["de_fxc"].view(), ref_dict["de_fxc"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_diag"].view(), ref_dict["de_vxc_diag"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_off"].view(), ref_dict["de_vxc_off"].t(), (1e-4, 1e-6)));

    assert_abs_diff_eq!(fp(result["de_fxc"].view()), -29.390069496788165, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_diag"].view()), 44.68386358957363, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_off"].view()), -16.124876249597378, epsilon = 1e-5);
}
