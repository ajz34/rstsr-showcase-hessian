use crate::hessian_uks::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use libxc::prelude::*;
use rstest::rstest;
use rstsr_showcase_hessian::numint_matmul::hess_uks::*;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_whole(hess_case_b3lyp: &CaseAmoniaUKS) {
    let CaseAmoniaUKS { mol, mo_coeff, mo_occ, grid_coords, grid_weights, ref_dict, .. } = hess_case_b3lyp.clone();
    let mut ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let xc_func_list = [(1.0, LibXCFunctional::from_identifier("HYB_GGA_XC_B3LYP", LibXCSpin::Polarized))];

    let occidx_a = mo_occ[0].view().greater(0).into_vec();
    let occidx_b = mo_occ[1].view().greater(0).into_vec();
    let mocc_a = mo_coeff[0].bool_select(-1, &occidx_a);
    let mocc_b = mo_coeff[1].bool_select(-1, &occidx_b);
    let dm0a = &mocc_a % mocc_a.t();
    let dm0b = &mocc_b % mocc_b.t();

    let (result, _) = make_hessian_setup_uks(&mol, &xc_func_list, &mut ni, dm0a.view(), dm0b.view(), None);

    println!("fp de_fxc: {}", fp(result["de_fxc"].view()));
    println!("fp de_vxc_diag_a: {}", fp(result["de_vxc_diag_a"].view()));
    println!("fp de_vxc_off_a: {}", fp(result["de_vxc_off_a"].view()));
    println!("fp vmat_deriv1_a: {}", fp(result["vmat_deriv1_a"].view()));

    assert!(rt::allclose(result["de_fxc"].view(), ref_dict["de_fxc"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_diag_a"].view(), ref_dict["de_vxc_diag_a"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["de_vxc_off_a"].view(), ref_dict["de_vxc_off_a"].t(), (1e-4, 1e-6)));
    assert!(rt::allclose(result["vmat_deriv1_a"].view(), ref_dict["vmat_deriv1_a"].t(), (1e-4, 1e-6)));

    assert_abs_diff_eq!(fp(result["de_fxc"].view()), -20.608327622840, epsilon = 1e-4);
}

#[rstest]
fn test_make_hess(hess_case_b3lyp: &CaseAmoniaUKS) {
    let CaseAmoniaUKS { mol, aux, mo_coeff, mo_occ, mo_energy, grid_coords, grid_weights, ref_dict, .. } =
        hess_case_b3lyp;

    let mo_coeff = [mo_coeff[0].view().into_contig(ColMajor), mo_coeff[1].view().into_contig(ColMajor)];
    let mo_occ = [mo_occ[0].view().into_contig(ColMajor), mo_occ[1].view().into_contig(ColMajor)];
    let mo_energy = [mo_energy[0].view().into_contig(ColMajor), mo_energy[1].view().into_contig(ColMajor)];

    let ovlp_obj = UHessOvlp::new(mol, &DeviceTsr::default());
    let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
    let mut hcore_obj = UHessHcore::new(mol, &DeviceTsr::default());
    // B3LYP hybrid coefficient = 0.2
    let mut rijk_obj = UHessRIJK::new_without_cderi(mol, aux, 1.0, 0.2);

    let ni = NIMatmul::new(mol, grid_coords, grid_weights);
    let xc_func_list = [(1.0, LibXCFunctional::from_identifier("HYB_GGA_XC_B3LYP", LibXCSpin::Polarized))];
    let mut nimatmul_obj = UHessKSNIMatmul::new(mol, &xc_func_list, ni, true);

    let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
    let core_list: Vec<&mut dyn UHessCoreAPI> = vec![&mut hcore_obj];
    let el_list: Vec<&mut dyn UHessElecInteractAPI> = vec![&mut rijk_obj, &mut nimatmul_obj];
    let config = HessSCFConfig::default();
    let mut hess_scf = UHessSCF::new(
        mo_coeff.map(|c| c.to_owned()),
        mo_occ.map(|c| c.to_owned()),
        mo_energy.map(|c| c.to_owned()),
        ovlp_obj,
        nuc_list,
        core_list,
        el_list,
        config,
        None,
    );

    let de_hess = hess_scf.make_hess();
    let de_hess_ref = ref_dict["de_ref"].transpose([2, 3, 0, 1]);

    assert!(rt::allclose(de_hess.view(), de_hess_ref.view(), (1e-3, 5e-4)));
    assert_abs_diff_eq!(fp(de_hess.view()), 0.661022070857, epsilon = 1e-4);

    println!("Result keys of hessian object: {:?}", hess_scf.result.keys());
    println!("Timing of hessian");
    for (key, value) in hess_scf.timing.iter() {
        println!("    {:60}: {:10.6} seconds", key, value);
    }
}
