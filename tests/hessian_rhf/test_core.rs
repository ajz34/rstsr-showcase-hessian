use crate::hessian_rhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;

#[rstest]
fn test_hess_nuc_repl(hess_case: CaseAmoniaRHF) {
    let de_nuc_repl_ref = hess_case.ref_dict["de_nuc"].to_owned().into_reverse_axes();
    assert_abs_diff_eq!(fp(de_nuc_repl_ref.view()), 10.942151503672441, epsilon = 1e-6);
}
