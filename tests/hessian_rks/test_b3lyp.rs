use crate::hessian_rks::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_anyway(hess_case_b3lyp: &CaseAmoniaRKS) {
    println!("{:?}", hess_case_b3lyp);
}
