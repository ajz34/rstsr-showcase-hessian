use crate::hessian_rhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;

use rstsr_showcase_hessian::hessian::ri_jk_restricted_naive::*;

#[rstest]
fn test_hess_ri_jk_skeleton_naive(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, ref_dict, .. } = hess_case;

    // test ri-j
    let de_dict = get_decomposed_rij_skeleton_deriv2_naive(mol, aux, mo_coeff.view(), mo_occ.view());
    for (key, de) in de_dict {
        let de_ref = ref_dict[key].to_owned().into_reverse_axes();
        assert!(rt::allclose(de.view(), de_ref.view(), (1e-4, 1e-6)));
    }

    // test ri-k
    let de_dict = get_decomposed_rik_skeleton_deriv2_naive(mol, aux, mo_coeff.view(), mo_occ.view());
    for (key, de) in de_dict {
        let de_ref = ref_dict[key].to_owned().into_reverse_axes();
        assert!(rt::allclose(de.view(), de_ref.view(), (1e-4, 1e-6)));
    }
}

#[rstest]
fn test_hess_ri_jk_deriv1_ao_naive(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;

    let de_dict = get_rij_deriv1_ao_naive(mol, aux, mo_coeff.view(), mo_occ.view());
    // the deriv1 ao here is [u, v, t, A], while for python it is [A, t, u, v]; swap first two axes
    assert_abs_diff_eq!(fp(de_dict["j1ao_aux0"].swapaxes(0, 1)), 35.38555993698421, epsilon = 1e-6);
    assert_abs_diff_eq!(fp(de_dict["j1ao_aux1"].swapaxes(0, 1)), 0.11465211252634573, epsilon = 1e-6);

    let de_dict = get_rik_deriv1_ao_naive(mol, aux, mo_coeff.view(), mo_occ.view());
    // the deriv1 ao here is [u, v, t, A], while for python it is [A, t, u, v]; swap first two axes
    assert_abs_diff_eq!(fp(de_dict["k1ao_aux0"].swapaxes(0, 1)), 1.5425060495529097, epsilon = 1e-6);
    assert_abs_diff_eq!(fp(de_dict["k1ao_aux1"].swapaxes(0, 1)), 0.20670656219034203, epsilon = 1e-6);
}

#[rstest]
fn test_hess_ri_jk_response_bra(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, ref_dict, .. } = hess_case;

    // double check of mo1 sanity
    assert_abs_diff_eq!(fp(ref_dict["mo1"].t()), -0.02385155247256418, epsilon = 1e-6);

    // row-major [A, t, u, v] -> col-major [u, v, t, A]
    let mo1 = ref_dict["mo1"].transpose((2, 3, 1, 0)).into_contig(ColMajor);

    let mo1_bra = mo_coeff % mo1;
    let resp_bra = get_rijk_response_bra_naive(mol, aux, mo_coeff.view(), mo_occ.view(), mo1_bra.view());
    let resp = mo_coeff.t() % resp_bra;
    assert_abs_diff_eq!(fp(resp.swapaxes(0, 1)), -0.2656880294075937, epsilon = 1e-6);
}
