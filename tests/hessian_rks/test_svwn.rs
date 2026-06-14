use crate::hessian_rks::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use libxc::prelude::*;
use rstest::rstest;
use rstsr_showcase_hessian::numint_matmul::hess_rks::make_hessian_setup_batch;
use rstsr_showcase_hessian::prelude::*;
use rstsr_showcase_hessian::util::density_matrices::get_dm0_restricted;

#[rstest]
fn test_whole(hess_case_svwn: &CaseAmoniaRKS) {
    let CaseAmoniaRKS { mol, mo_coeff, mo_occ, grid_coords, grid_weights, ref_dict, .. } = hess_case_svwn.clone();
    let mut ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    // SVWN = SLATER + VWN5 in PySCF (libxc: LDA_X + LDA_C_VWN, both with unit weight).
    let xc_func_list = [
        (1.0, LibXCFunctional::from_identifier("LDA_X", LibXCSpin::Unpolarized)),
        (1.0, LibXCFunctional::from_identifier("LDA_C_VWN", LibXCSpin::Unpolarized)),
    ];
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

    assert_abs_diff_eq!(fp(result["de_fxc"].view()), -20.132874762943892, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_diag"].view()), 52.68061529362255, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_off"].view()), -33.60278999768945, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["vmat_deriv1"].view()), -4.579019717395777, epsilon = 1e-6);
}
