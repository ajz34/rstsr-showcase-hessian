use crate::prelude::*;

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

    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    &mocc * occ.i((None, ..)) % &mocc.t()
}

/// Generate the orbital-energy weighted density matrix for current SCF component.
///
/// # Parameters
///
/// - `mo_coeff` : shape `[nao, nmo]`. Molecular orbital coefficients.
/// - `mo_occ` : shape `[nmo]`. Molecular orbital occupation numbers.
/// - `mo_energy` : shape `[nmo]`. Molecular orbital energies.
///
/// # Returns
///
/// - `dme0` : shape `[nao, nao]`. The orbital-energy weighted density matrix for current SCF
///   component.
pub fn get_dme0_restricted(mo_coeff: TsrView, mo_occ: TsrView, mo_energy: TsrView) -> Tsr {
    let [_nao, nmo] = mo_coeff.shape().to_vec().try_into().unwrap();
    check_shape!(mo_occ.shape(), [nmo], "mo_occ shape not correct.");
    check_shape!(mo_energy.shape(), [nmo], "mo_energy shape not correct.");

    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    let eocc = mo_energy.bool_select(-1, &occidx);
    &mocc * (occ * eocc).i((None, ..)) % &mocc.t()
}

/// Pack the lower triangular part of a matrix into a 1D array, multiply non-diagonal values by 2.
pub fn pack_triu_tilde_2d(dm: TsrView) -> Tsr {
    assert_eq!(dm.ndim(), 2);
    assert_eq!(dm.shape()[0], dm.shape()[1]);
    let nao = dm.shape()[0];
    let mut dm_triu: Tsr = 2.0 * dm.pack_triu();
    for i in 0..nao {
        dm_triu[[(i + 2) * (i + 1) / 2 - 1]] *= 0.5;
    }
    dm_triu
}

/// Pack the lower triangular part of a multi-dimensional array into a smaller-one dimension array,
/// multiply non-diagonal values by 2.
pub fn pack_triu_tilde(dm: TsrView) -> Tsr {
    if dm.ndim() == 2 {
        return pack_triu_tilde_2d(dm);
    }
    assert!(dm.ndim() > 2);
    assert_eq!(dm.shape()[0], dm.shape()[1]);
    let shape_remaining = &dm.shape()[2..];
    let nao = dm.shape()[0];
    let nao_tp = nao * (nao + 1) / 2;
    let dm = dm.reshape((nao, nao, -1));
    let mut out = rt::zeros(([nao_tp, dm.shape()[2]], dm.device()));
    for (i, dm_i) in dm.axes_iter(-1).enumerate() {
        out.i_mut((.., i)).assign(&pack_triu_tilde_2d(dm_i));
    }
    let shape_recap = [nao_tp].iter().chain(shape_remaining.iter()).copied().collect_vec();
    out.into_shape(shape_recap)
}
