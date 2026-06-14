use crate::hessian_rks::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use libxc::prelude::*;
use rstest::rstest;
use rstsr_showcase_hessian::numint_matmul::hess_rks::make_hessian_setup_batch;
use rstsr_showcase_hessian::numint_matmul::hess_rks::make_hessian_setup_with_parallel;
use rstsr_showcase_hessian::prelude::*;
use rstsr_showcase_hessian::util::density_matrices::get_dm0_restricted;

#[rstest]
fn test_whole(hess_case_b3lyp: &CaseAmoniaRKS) {
    let CaseAmoniaRKS { mol, mo_coeff, mo_occ, grid_coords, grid_weights, ref_dict, .. } = hess_case_b3lyp.clone();
    let mut ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let xc_func_list = [(1.0, LibXCFunctional::from_identifier("HYB_GGA_XC_B3LYP", LibXCSpin::Unpolarized))];
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let result = make_hessian_setup_batch(&mol, &xc_func_list, &mut ni, dm0.view(), None, true);

    println!("fp de_fxc: {}", fp(result["de_fxc"].view()));
    println!("fp de_vxc_diag: {}", fp(result["de_vxc_diag"].view()));
    println!("fp de_vxc_off: {}", fp(result["de_vxc_off"].view()));
    println!("fp vmat_deriv1: {}", fp(result["vmat_deriv1"].view()));

    assert!(rt::allclose(result["de_fxc"].view(), ref_dict["de_fxc"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_diag"].view(), ref_dict["de_vxc_diag"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_off"].view(), ref_dict["de_vxc_off"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_ip"].view(), ref_dict["vmat_ip"].transpose([1, 2, 0]), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_deriv1"].view(), ref_dict["vmat_deriv1"].t(), (1e-4, 1e-6)));

    assert_abs_diff_eq!(fp(result["de_fxc"].view()), -21.249874465163057, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_diag"].view()), 49.688766385730304, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_off"].view()), -29.337474734527515, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["vmat_deriv1"].view()), -3.8658927361526123, epsilon = 1e-6);
}

#[rstest]
fn test_batched(hess_case_b3lyp: &CaseAmoniaRKS) {
    let CaseAmoniaRKS { mol, mo_coeff, mo_occ, grid_coords, grid_weights, ref_dict, .. } = hess_case_b3lyp.clone();
    let mut ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let xc_func_list = [(1.0, LibXCFunctional::from_identifier("HYB_GGA_XC_B3LYP", LibXCSpin::Unpolarized))];
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let result = make_hessian_setup_with_parallel(&mol, &xc_func_list, &mut ni, dm0.view(), None, true);

    println!("fp de_fxc: {}", fp(result["de_fxc"].view()));
    println!("fp de_vxc_diag: {}", fp(result["de_vxc_diag"].view()));
    println!("fp de_vxc_off: {}", fp(result["de_vxc_off"].view()));
    println!("fp vmat_deriv1: {}", fp(result["vmat_deriv1"].view()));

    assert!(rt::allclose(result["de_fxc"].view(), ref_dict["de_fxc"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_diag"].view(), ref_dict["de_vxc_diag"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_off"].view(), ref_dict["de_vxc_off"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_ip"].view(), ref_dict["vmat_ip"].transpose([1, 2, 0]), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_deriv1"].view(), ref_dict["vmat_deriv1"].t(), (1e-4, 1e-6)));

    assert_abs_diff_eq!(fp(result["de_fxc"].view()), -21.249874465163057, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_diag"].view()), 49.688766385730304, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_off"].view()), -29.337474734527515, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["vmat_deriv1"].view()), -3.8658927361526123, epsilon = 1e-6);
}
