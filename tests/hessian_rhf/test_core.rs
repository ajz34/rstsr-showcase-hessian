use crate::hessian_rhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::hessian::hess_trait_restricted::RHessCoreAPI;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_hess_nuc_repl(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, mo_coeff, mo_occ, ref_dict, .. } = hess_case;

    // compute results
    let mut hess_nuc_repl = HessNucRepl::new(mol);
    let de_nuc_repl = hess_nuc_repl.make_skeleton_hess(mo_coeff.view(), mo_occ.view());

    // compare to reference
    let de_nuc_repl_ref = ref_dict["de_nuc"].to_owned().into_reverse_axes();
    assert!(rt::allclose(de_nuc_repl.view(), de_nuc_repl_ref.view(), (1e-4, 1e-6)));
    assert_abs_diff_eq!(fp(de_nuc_repl.view()), 10.942151503672441, epsilon = 1e-6);
}
