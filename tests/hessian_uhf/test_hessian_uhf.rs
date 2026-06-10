use crate::hessian_uhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_dimensionless_cphf_rhs(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, mo_energy, .. } = hess_case;

    let ovlp_obj = UHessOvlp::new(mol, &DeviceTsr::default());
    let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
    let mut hcore_obj = UHessHcore::new(mol, &DeviceTsr::default());
    let mut rijk_obj = UHessRIJKNaive::new(mol, aux, 1.0, 1.0);
    let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
    let hcore_list: Vec<&mut dyn UHessCoreAPI> = vec![&mut hcore_obj];
    let el_list: Vec<&mut dyn UHessElecInteractAPI> = vec![&mut rijk_obj];
    let config = HessSCFConfig::default();
    let mut hess_scf = UHessSCF::new(
        mo_coeff.clone(),
        mo_occ.clone(),
        mo_energy.clone(),
        ovlp_obj,
        nuc_list,
        hcore_list,
        el_list,
        config,
        None,
    );

    // before krylov, first obtain dimensionless rhs part
    let pre_cphf_dict = hess_scf.compute_dimless_cphf_rhs();
    assert_abs_diff_eq!(fp(pre_cphf_dict["rhs_0"].swapaxes(0, 1)), -0.01785256539468953, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(pre_cphf_dict["rhs_1"].swapaxes(0, 1)), 0.14550989432158085, epsilon = 1e-5);

    // solve cphf
    let rhs = [pre_cphf_dict["rhs_0"].view(), pre_cphf_dict["rhs_1"].view()];
    hess_scf.make_response_preparation();
    let mo1 = hess_scf.solve_dimless_cphf(&rhs);

    let ref_mo1_0 = hess_case.ref_dict["mo1_a"].transpose((2, 3, 1, 0));
    let ref_mo1_1 = hess_case.ref_dict["mo1_b"].transpose((2, 3, 1, 0));
    assert!(rt::allclose(mo1[0].view(), ref_mo1_0.view(), (1e-3, 1e-4)));
    assert!(rt::allclose(mo1[1].view(), ref_mo1_1.view(), (1e-3, 1e-4)));
    assert_abs_diff_eq!(fp(mo1[0].swapaxes(0, 1)), 0.04797427280601669, epsilon = 1e-4);
    assert_abs_diff_eq!(fp(mo1[1].swapaxes(0, 1)), -1.1346573239117455, epsilon = 1e-4);
}
