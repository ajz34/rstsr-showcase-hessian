use crate::hessian_rks::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use libxc::prelude::*;
use rstest::rstest;
use rstsr_showcase_hessian::numint_matmul::hess_rks::make_hessian_setup;
use rstsr_showcase_hessian::numint_matmul::hess_rks::make_hessian_setup_batched;
use rstsr_showcase_hessian::numint_matmul::hess_rks::get_rks_response_bra;
use rstsr_showcase_hessian::numint_matmul::hess_rks::RHessKSNIMatmul;
use rstsr_showcase_hessian::prelude::*;
use rstsr_showcase_hessian::util::density_matrices::get_dm0_restricted;

#[rstest]
fn test_make_hess(hess_case_tpss0: &CaseAmoniaRKS) {
    let CaseAmoniaRKS { mol, aux, mo_coeff, mo_occ, mo_energy, grid_coords, grid_weights, ref_dict, .. } =
        hess_case_tpss0;

    let mo_coeff = mo_coeff.view().into_contig(ColMajor);
    let mo_occ = mo_occ.view().into_contig(ColMajor);
    let mo_energy = mo_energy.view().into_contig(ColMajor);
    let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
    let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
    let mut hcore_obj = RHessHcore::new(mol, &DeviceTsr::default());
    // TPSS0 scales the exchange contribution by 0.25
    let mut rijk_obj = RHessRIJKNaive::new(mol, aux, 1.0, 0.25);

    let ni = NIMatmul::new(mol, grid_coords, grid_weights);
    let xc_func_list = [(1.0, LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSS0", LibXCSpin::Unpolarized))];
    let mut nimatmul_obj = RHessKSNIMatmul::new(mol, &xc_func_list, ni, true);

    let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
    let hcore_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
    let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj, &mut nimatmul_obj];
    let config = HessSCFConfig::default();
    let mut hess_scf =
        RHessSCF::new(mo_coeff, mo_occ, mo_energy, ovlp_obj, nuc_list, hcore_list, el_list, config, None);

    let de_hess = hess_scf.make_hess();
    let de_hess_ref = ref_dict["de_ref"].transpose([2, 3, 0, 1]);

    assert!(rt::allclose(de_hess.view(), de_hess_ref.view(), (1e-4, 5e-5)));
    assert_abs_diff_eq!(fp(de_hess.view()), 1.477331064221, epsilon = 1e-4);

    println!("Result keys of hessian object: {:?}", hess_scf.result.keys());
    println!("Timing of hessian");
    for (key, value) in hess_scf.timing.iter() {
        println!("    {:60}: {:10.6} seconds", key, value);
    }
}

#[rstest]
fn test_response(hess_case_tpss0: &CaseAmoniaRKS) {
    let CaseAmoniaRKS { mol, mo_coeff, mo_occ, grid_coords, grid_weights, ref_dict, .. } = hess_case_tpss0;
    let mut ni = NIMatmul::new(mol, grid_coords, grid_weights);
    let xc_func_list = [(1.0, LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSS0", LibXCSpin::Unpolarized))];
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let (result, _) = make_hessian_setup(mol, &xc_func_list, &mut ni, dm0.view(), None);

    let mut ni = ni.duplicate();
    let vmat_deriv1_mo = ref_dict["vmat_deriv1_mo"].transpose([2, 3, 1, 0]).into_contig(ColMajor);
    let den_type = XCDenType::TAU;
    let fxc_eff = result["fxc"].view();
    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, occidx);
    let (resp, timing) = get_rks_response_bra(&mut ni, den_type, fxc_eff, vmat_deriv1_mo.view(), mocc.view());
    println!("fp resp, {:?}", fp(resp.swapaxes(0, 1)));
    println!("timing: {:?}", timing);
    assert_abs_diff_eq!(fp(resp.swapaxes(0, 1)), -0.036689489483, epsilon = 1e-4);
}

#[rstest]
fn test_whole(hess_case_tpss0: &CaseAmoniaRKS) {
    let CaseAmoniaRKS { mol, mo_coeff, mo_occ, grid_coords, grid_weights, ref_dict, .. } = hess_case_tpss0.clone();
    let mut ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let xc_func_list = [(1.0, LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSS0", LibXCSpin::Unpolarized))];
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let (result, timing) = make_hessian_setup(&mol, &xc_func_list, &mut ni, dm0.view(), None);

    assert!(rt::allclose(result["de_fxc"].view(), ref_dict["de_fxc"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_diag"].view(), ref_dict["de_vxc_diag"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_off"].view(), ref_dict["de_vxc_off"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_ip"].view(), ref_dict["vmat_ip"].transpose([1, 2, 0]), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_deriv1"].view(), ref_dict["vmat_deriv1"].t(), (1e-4, 1e-6)));

    assert_abs_diff_eq!(fp(result["de_fxc"].view()), -29.390069496788165, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_diag"].view()), 44.68386358957363, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_off"].view()), -16.124876249597378, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["vmat_deriv1"].view()), -3.4184689531771597, epsilon = 1e-6);

    let mut total_time = 0.0;
    for (key, value) in timing.iter() {
        println!("Timing for {key:>20}: {value:.4} sec");
        total_time += value;
    }
    println!("Total time: {total_time:.4} sec");
}

#[rstest]
fn test_batched(hess_case_tpss0: &CaseAmoniaRKS) {
    let CaseAmoniaRKS { mol, mo_coeff, mo_occ, grid_coords, grid_weights, ref_dict, .. } = hess_case_tpss0.clone();
    let mut ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let xc_func_list = [(1.0, LibXCFunctional::from_identifier("HYB_MGGA_XC_TPSS0", LibXCSpin::Unpolarized))];
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let (result, _) = make_hessian_setup_batched(&mol, &xc_func_list, &mut ni, dm0.view(), None, true);

    assert!(rt::allclose(result["de_fxc"].view(), ref_dict["de_fxc"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_diag"].view(), ref_dict["de_vxc_diag"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_off"].view(), ref_dict["de_vxc_off"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_ip"].view(), ref_dict["vmat_ip"].transpose([1, 2, 0]), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_deriv1"].view(), ref_dict["vmat_deriv1"].t(), (1e-4, 1e-6)));

    assert_abs_diff_eq!(fp(result["de_fxc"].view()), -29.390069496788165, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_diag"].view()), 44.68386358957363, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["de_vxc_off"].view()), -16.124876249597378, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(result["vmat_deriv1"].view()), -3.4184689531771597, epsilon = 1e-6);
}
