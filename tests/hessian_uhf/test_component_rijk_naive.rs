use crate::hessian_uhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::hessian::ri_jk_unrestricted_naive::*;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_hess_rij_skeleton_naive(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, ref_dict, .. } = hess_case;

    let mo_coeff = mo_coeff.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_occ = mo_occ.iter().map(|x| x.view()).collect_array().unwrap();
    let de_j_skeleton = get_decomposed_rij_skeleton_deriv2_unrestricted_naive(mol, aux, &mo_coeff, &mo_occ, None);

    for key in ["de_J20", "de_J11", "de_J02"] {
        let de_j_skeleton_ref = ref_dict[key].to_owned().into_reverse_axes();
        assert!(rt::allclose(de_j_skeleton[key].view(), de_j_skeleton_ref.view(), (1e-4, 1e-6)));
    }
    assert_abs_diff_eq!(fp(de_j_skeleton["de_J20"].view()), 4.902587371193881, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(de_j_skeleton["de_J11"].view()), 8.88765043727085, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(de_j_skeleton["de_J02"].view()), -4.445673838381621, epsilon = 1e-5);
}

#[rstest]
fn test_hess_rik_skeleton_naive(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, ref_dict, .. } = hess_case;

    let mo_coeff = mo_coeff.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_occ = mo_occ.iter().map(|x| x.view()).collect_array().unwrap();
    let de_j_skeleton = get_decomposed_rik_skeleton_deriv2_unrestricted_naive(mol, aux, &mo_coeff, &mo_occ, None);

    for key in ["de_K20", "de_K11", "de_K02"] {
        let de_j_skeleton_ref = ref_dict[key].to_owned().into_reverse_axes();
        assert!(rt::allclose(de_j_skeleton[key].view(), de_j_skeleton_ref.view(), (1e-4, 1e-6)));
    }
    assert_abs_diff_eq!(fp(de_j_skeleton["de_K20"].view()), -0.5883149652263711, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(de_j_skeleton["de_K11"].view()), 4.536955856266865, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(de_j_skeleton["de_K02"].view()), -2.2682891819149544, epsilon = 1e-5);
}

#[rstest]
fn test_rij_deriv1(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;

    let mo_coeff = mo_coeff.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_occ = mo_occ.iter().map(|x| x.view()).collect_array().unwrap();
    let j1ao_dict = get_rij_deriv1_ao_unrestricted_naive(mol, aux, &mo_coeff, &mo_occ, None);

    assert_abs_diff_eq!(fp(j1ao_dict["j1ao_aux0"].view()), 27.320873266136108, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(j1ao_dict["j1ao_aux1"].view()), 0.12413515517879808, epsilon = 1e-5);
}

#[rstest]
fn test_rik_deriv1(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;

    let mo_coeff = mo_coeff.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_occ = mo_occ.iter().map(|x| x.view()).collect_array().unwrap();
    let j1ao_dict = get_rik_deriv1_ao_unrestricted_naive(mol, aux, &mo_coeff, &mo_occ, None);

    assert_abs_diff_eq!(fp(j1ao_dict["k1ao_aux0"].view()), -6.127504869346246, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(j1ao_dict["k1ao_aux1"].view()), 0.05442798516090062, epsilon = 1e-5);
}

#[rstest]
fn test_ri_jk_resp_bra(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, ref_dict, .. } = hess_case;

    let mut hess_rijk_obj = UHessRIJKNaive::new(mol, aux, 1.0, 1.0);
    let mo_coeff = mo_coeff.iter().map(|x| x.view()).collect_array::<2>().unwrap();
    let mo_occ = mo_occ.iter().map(|x| x.view()).collect_array::<2>().unwrap();
    hess_rijk_obj.make_response_preparation(&mo_coeff, &mo_occ);

    let mo1_a = ref_dict["mo1_a"].to_owned().into_transpose([2, 3, 1, 0]);
    let mo1_b = ref_dict["mo1_b"].to_owned().into_transpose([2, 3, 1, 0]);
    let mo1_bra_a = &mo_coeff[0] % &mo1_a;
    let mo1_bra_b = &mo_coeff[1] % &mo1_b;

    let resp_bra = hess_rijk_obj.get_response_bra(&[mo1_bra_a.view(), mo1_bra_b.view()]);
    let resp = [mo_coeff[0].t() % &resp_bra[0], mo_coeff[1].t() % &resp_bra[1]];

    assert_abs_diff_eq!(fp(resp[0].swapaxes(0, 1)), -0.6682133336210753, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(resp[1].swapaxes(0, 1)), 1.0034934136306894, epsilon = 1e-5);
}
