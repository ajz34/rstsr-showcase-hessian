use crate::hessian_rhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_hess_nuc_repl(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, ref_dict, .. } = hess_case;

    // compute results
    let mut hess_nuc_repl = HessNucRepl::new(mol, &DeviceTsr::default());
    let de_nuc_repl = hess_nuc_repl.make_skeleton_hess(None);

    // compare to reference
    let de_nuc_repl_ref = ref_dict["de_nuc"].to_owned().into_transpose((2, 3, 0, 1));
    assert!(rt::allclose(de_nuc_repl.view(), de_nuc_repl_ref.view(), (1e-4, 1e-6)));
    assert_abs_diff_eq!(fp(de_nuc_repl.view()), 10.942151503672441, epsilon = 1e-6);
}

#[rstest]
fn test_generator_hcore_deriv2(hess_case: &CaseAmoniaRHF) {
    use rstsr_showcase_hessian::hessian::hcore::generator_hcore_deriv2;

    let CaseAmoniaRHF { mol, .. } = hess_case;
    let device = DeviceTsr::default();
    let mut gen_hcore_deriv2 = generator_hcore_deriv2(mol, &device);
    assert_abs_diff_eq!(fp(gen_hcore_deriv2(0, 0).view()), -72.29474171640412, epsilon = 1e-6);
    assert_abs_diff_eq!(fp(gen_hcore_deriv2(0, 1).view()), 12.858221292861833, epsilon = 1e-6);
}

#[rstest]
fn test_hess_hcore(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, mo_coeff, mo_occ, ref_dict, .. } = hess_case;

    // compute results
    let mut hess_hcore = RHessHcore::new(mol, &DeviceTsr::default());
    let de_hcore = hess_hcore.make_skeleton_hess(mo_coeff.view(), mo_occ.view(), None);

    // compare to reference
    let de_hcore_ref = ref_dict["de_hcore"].to_owned().into_reverse_axes();
    assert!(rt::allclose(de_hcore.view(), de_hcore_ref.view(), (1e-4, 1e-6)));
    assert_abs_diff_eq!(fp(de_hcore.view()), -16.993496707453197, epsilon = 1e-6);
}

#[rstest]
fn test_generator_hcore_deriv1(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, .. } = hess_case;
    let hess_core = RHessHcore::new(mol, &DeviceTsr::default());
    let mut gen_hcore_deriv1 = hess_core.generator_deriv1();
    assert_abs_diff_eq!(fp(gen_hcore_deriv1(0).view()), -19.44142929546185, epsilon = 1e-6);
    assert_abs_diff_eq!(fp(gen_hcore_deriv1(3).view()), 23.88285913576012, epsilon = 1e-6);
}

#[rstest]
fn test_hess_ovlp(hess_case: &CaseAmoniaRHF) {
    use rstsr_showcase_hessian::util::density_matrices::get_dme0_restricted;
    let CaseAmoniaRHF { mol, mo_coeff, mo_occ, mo_energy, ref_dict, .. } = hess_case;

    // compute results
    let dme0 = get_dme0_restricted(mo_coeff.view(), mo_occ.view(), mo_energy.view());
    let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
    let de_ovlp = ovlp_obj.make_hess(dme0.view(), None);

    // compare to reference
    let de_ovlp_ref = ref_dict["de_ovlp"].to_owned().into_transpose((2, 3, 0, 1));
    assert!(rt::allclose(de_ovlp.view(), de_ovlp_ref.view(), (1e-4, 1e-6)));
    assert_abs_diff_eq!(fp(de_ovlp.view()), 0.7050335726988588, epsilon = 1e-6);
}
