use crate::hessian_rhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_f1ao(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;

    let hess_hcore_obj = HessHcore::new(mol, &DeviceTsr::default());
    let mut hess_rijk_obj = RHessRIJKNaive::new(mol, aux, 1.0, 1.0);

    let natm = mol.natm();
    let gen_h1ao = hess_hcore_obj.generator_deriv1().unwrap();
    let h1ao_list = (0..natm).map(gen_h1ao).collect_vec();
    let h1ao = rt::stack((h1ao_list, -1));
    let jk1ao = hess_rijk_obj.get_deriv1_ao(mo_coeff.view(), mo_occ.view());
    let f1ao = &h1ao + &jk1ao;
    assert_abs_diff_eq!(fp(f1ao.view().swapaxes(0, 1)), 0.03306328817997084, epsilon = 1e-6);
}

#[rstest]
fn test_dimensionless_cphf_rhs(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, mo_energy, ref_dict } = hess_case;

    let mo_coeff = mo_coeff.view().into_contig(ColMajor);
    let mo_occ = mo_occ.view().into_contig(ColMajor);
    let mo_energy = mo_energy.view().into_contig(ColMajor);
    let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
    let mut hcore_obj = HessHcore::new(mol, &DeviceTsr::default());
    let mut rijk_obj = RHessRIJKNaive::new(mol, aux, 1.0, 1.0);
    let hcore_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
    let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
    let config = RHessSCFConfig::default();
    let mut hess_scf = RHessSCF::new(mo_coeff, mo_occ, mo_energy, ovlp_obj, hcore_list, el_list, config);

    // before krylov, first obtain dimensionless rhs part
    let pre_cphf_dict = hess_scf.compute_dimless_cphf_rhs();
    assert_abs_diff_eq!(fp(pre_cphf_dict["rhs"].swapaxes(0, 1)), -0.027755691019085788, epsilon = 1e-6);
    assert_abs_diff_eq!(fp(pre_cphf_dict["f1mo"].swapaxes(0, 1)), 9.624352641672411, epsilon = 1e-6);
    assert_abs_diff_eq!(fp(pre_cphf_dict["s1mo"].swapaxes(0, 1)), -3.0146480401818847, epsilon = 1e-6);

    // solve cphf
    let rhs = pre_cphf_dict["rhs"].view();
    hess_scf.make_response_preparation();
    let mo1 = hess_scf.solve_dimless_cphf(rhs);
    let ref_mo1 = ref_dict["mo1"].transpose((2, 3, 1, 0));
    assert!(rt::allclose(mo1.view(), ref_mo1.view(), (1e-4, 1e-6)));
    assert_abs_diff_eq!(fp(mo1.swapaxes(0, 1)), -0.02385155247256418, epsilon = 1e-6);
}
