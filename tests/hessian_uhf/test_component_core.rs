use crate::hessian_uhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_hess_nuc_repl(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, ref_dict, .. } = hess_case;

    // compute results
    let mut hess_nuc_repl = HessNucRepl::new(mol, &DeviceTsr::default());
    let de_nuc_repl = hess_nuc_repl.make_skeleton_hess(None);

    // compare to reference
    let de_nuc_repl_ref = ref_dict["de_nuc"].to_owned().into_transpose((2, 3, 0, 1));
    assert!(rt::allclose(de_nuc_repl.view(), de_nuc_repl_ref.view(), (1e-4, 1e-6)));
    assert_abs_diff_eq!(fp(de_nuc_repl.view()), 10.942151503672441, epsilon = 1e-6);
}

#[rstest]
fn test_hess_hcore(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, mo_coeff, mo_occ, ref_dict, .. } = hess_case;

    let mo_coeff = mo_coeff.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_occ = mo_occ.iter().map(|x| x.view()).collect_array().unwrap();

    // compute results
    let mut hess_hcore = UHessHcore::new(mol, &DeviceTsr::default());
    let de_hcore = hess_hcore.make_skeleton_hess(mo_coeff, mo_occ, None);

    // compare to reference
    let de_hcore_ref = ref_dict["de_hcore"].to_owned().into_reverse_axes();
    assert!(rt::allclose(de_hcore.view(), de_hcore_ref.view(), (1e-4, 1e-6)));
    assert_abs_diff_eq!(fp(de_hcore.view()), -19.367829669982456, epsilon = 1e-6);
}
