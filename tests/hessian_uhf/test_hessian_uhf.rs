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
}
