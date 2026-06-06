use crate::prelude::*;

const TOL_OCC: f64 = 1e-15;

/// Generate the density matrix for current SCF component.
///
/// # Parameters
///
/// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
/// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
///
/// # Returns
///
/// - `dm0` : shape `[nao, nao]`. The density matrix for current SCF component.
pub fn get_dm0_restricted(mo_coeff: TsrView, mo_occ: TsrView) -> Tsr {
    let [_nao, nmo] = mo_coeff.shape().to_vec().try_into().unwrap();
    check_shape!(mo_occ.shape(), [nmo], "mo_occ shape not correct.");

    let occidx = mo_occ.view().greater(TOL_OCC).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    &mocc * occ.i((None, ..)) % &mocc.t()
}
