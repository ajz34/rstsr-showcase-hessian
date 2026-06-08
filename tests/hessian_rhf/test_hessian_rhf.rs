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
