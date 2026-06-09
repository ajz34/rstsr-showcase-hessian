use crate::hessian::ri_jk_restricted_naive;
use crate::prelude::*;

pub fn get_decomposed_rij_skeleton_deriv2_unrestricted_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: [TsrView; 2],
    mo_occ: [TsrView; 2],
    atm_list: Option<&[usize]>,
) -> HashMap<&'static str, Tsr> {
    // concate mo_coeff and mo_occ
    let mo_coeff_stack: Tsr = rt::concatenate((mo_coeff, -1));
    let mo_occ_stack: Tsr = rt::concatenate((mo_occ, -1));

    ri_jk_restricted_naive::get_decomposed_rij_skeleton_deriv2_naive(
        mol,
        aux,
        mo_coeff_stack.view(),
        mo_occ_stack.view(),
        atm_list,
    )
}

pub fn get_decomposed_rik_skeleton_deriv2_unrestricted_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: [TsrView; 2],
    mo_occ: [TsrView; 2],
    atm_list: Option<&[usize]>,
) -> HashMap<&'static str, Tsr> {
    // compute alpha and beta separately
    const 上: usize = 0;
    const 下: usize = 1;

    let de_rik_alpha = ri_jk_restricted_naive::get_decomposed_rik_skeleton_deriv2_naive(
        mol,
        aux,
        mo_coeff[上].view(),
        mo_occ[上].view(),
        atm_list,
    );
    let de_rik_beta = ri_jk_restricted_naive::get_decomposed_rik_skeleton_deriv2_naive(
        mol,
        aux,
        mo_coeff[下].view(),
        mo_occ[下].view(),
        atm_list,
    );

    let mut result = HashMap::new();
    for &key in de_rik_alpha.keys() {
        let de_alpha = &de_rik_alpha[key];
        let de_beta = &de_rik_beta[key];
        result.insert(key, de_alpha + de_beta);
    }
    result
}
