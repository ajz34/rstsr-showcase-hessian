// see also pyhessref/nimatmul/rks_with_becke.py
//
// Becke grid-shift contribution to the RKS Hessian: the grid is glued to the
// atoms that generated it, so the weight factor of the XC energy carries
// nuclear-coordinate derivatives.  This module duplicates the term-level code
// of `hess_rks.rs` (mirroring how `rks_with_becke.py` carries its own copies of
// `rks.py`) and adds the seven skeleton terms `de_becke_*` and the three f1ao
// corrections `vmat_becke_*`, restoring translational invariance of
// `de_xc_skeleton` and `vmat_deriv1_grid`.
//
// The 2nd Becke derivative `ddw` is never materialized: the only consumer
// (`de_becke_full_2`) contracts `ddw` with `exc * rho[0]` over the grid, which
// is exactly the `contract_ddw` channel of `becke_partition` with `nset = 1`.

use super::becke_partition::{becke_partition, AtmIndices, BeckeDerivArg};
use super::hess_rks::get_rks_response_bra_batched;
use super::prelude::*;

/* #region const dimensions/indices definition */

const O: usize = 0;
const X: usize = 1;
const Y: usize = 2;
const Z: usize = 3;
const XX: usize = 4;
const XY: usize = 5;
const XZ: usize = 6;
const YX: usize = 5;
const YY: usize = 7;
const YZ: usize = 8;
const ZX: usize = 6;
const ZY: usize = 8;
const ZZ: usize = 9;
const XXX: usize = 10;
const XXY: usize = 11;
const XXZ: usize = 12;
const XYY: usize = 13;
const XYZ: usize = 14;
const XZZ: usize = 15;
const YYY: usize = 16;
const YYZ: usize = 17;
const YZZ: usize = 18;
const ZZZ: usize = 19;

const IDX_AO_DERIV2: [[usize; 3]; 3] = [[XX, XY, XZ], [XY, YY, YZ], [XZ, YZ, ZZ]];

pub const fn get_hess_ao_deriv(xc_type: XCDenType) -> usize {
    match xc_type {
        RHO => 2,
        SIGMA => 3,
        TAU => 3,
        LAPL => unimplemented!(),
    }
}

pub const fn get_hess_ncomp_ao_dm0(xc_type: XCDenType) -> usize {
    match xc_type {
        RHO => 1,
        SIGMA => 4,
        TAU => 4,
        LAPL => unimplemented!(),
    }
}

/* #endregion */

/* #region macro for indexing last dimension */

macro_rules! index {
    ($tsr: ident, $($idx:expr),*) => {
        $tsr.i((Ellipsis, $($idx),*))
    };
}

macro_rules! index_mut {
    ($tsr: ident, $($idx:expr),*) => {
        (*&mut $tsr.i_mut((Ellipsis, $($idx),*)))
    };
}

/* #endregion */

/* #region basic pure functions of skeleton hessian evaluation */

/// Becke grid-attribution boundaries for a batch that holds only atom `atm_idx`'s
/// grids: atoms before `atm_idx` get the empty interval `[0, 0)`, atom `atm_idx`
/// owns `[0, n)`, and atoms after own the empty `[n, n)`.
pub fn by_atom_batch(natm: usize, atm_idx: usize, n: usize) -> Vec<usize> {
    let mut v = vec![0; natm + 1];
    for x in v.iter_mut().skip(atm_idx + 1) {
        *x = n;
    }
    v
}

/// Split the grid into batches of at most `nbatch_grids` grids, respecting atom
/// boundaries (`quad_split_by_atom` in the pyhessref reference).
pub fn quad_split_by_atom(atm_quad_split: &[usize], nbatch_grids: usize, natm: usize) -> Vec<(usize, usize, usize)> {
    let mut batches = Vec::new();
    for A in 0..natm {
        let mut start = atm_quad_split[A];
        let end = atm_quad_split[A + 1];
        while start < end {
            let next_end = (start + nbatch_grids).min(end);
            batches.push((A, start, next_end));
            start = next_end;
        }
    }
    batches
}

/// Same as [`super::hess_rks::get_rho_vxc_fxc`], but also returns the per-particle
/// XC energy density `exc [ngrids]` (order-0 output of the libxc evaluation),
/// needed by the `cddw` contraction of `de_becke_full_2`.
pub fn get_rho_exc_vxc_fxc(
    xc_func_list: &[(f64, LibXCFunctional)],
    ao: TsrView,
    ao_dm0: TsrView,
) -> (Tsr, Tsr, Tsr, Tsr) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let xc_type = xc_func_list
        .iter()
        .map(|(_, f)| determine_den_type(f))
        .max_by_key(|t| t.num_nvar())
        .expect("xc_func_list must not be empty");
    let nvar = xc_type.num_nvar();
    let ngrids = ao.shape()[0];
    let device = ao.device().clone();

    let mut rho = rt::zeros(([ngrids, nvar], &device));
    index_mut!(rho, 0) += rt::vecdot(index!(ao, 0), index!(ao_dm0, O), 1);
    if matches!(xc_type, SIGMA | TAU) {
        index_mut!(rho, X) += 2 * rt::vecdot(index!(ao, X), index!(ao_dm0, O), 1);
        index_mut!(rho, Y) += 2 * rt::vecdot(index!(ao, Y), index!(ao_dm0, O), 1);
        index_mut!(rho, Z) += 2 * rt::vecdot(index!(ao, Z), index!(ao_dm0, O), 1);
    }
    if matches!(xc_type, TAU) {
        index_mut!(rho, 4) += 0.5
            * (rt::vecdot(index!(ao, X), index!(ao_dm0, X), 1)
                + rt::vecdot(index!(ao, Y), index!(ao_dm0, Y), 1)
                + rt::vecdot(index!(ao, Z), index!(ao_dm0, Z), 1))
    }

    let mut exc = rt::zeros(([ngrids], &device));
    let mut vxc = rt::zeros(([ngrids, nvar], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, nvar], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        let rho_i = rho.i((.., ..nvar_i));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [e_i, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        exc += *scale * e_i.into_shape([ngrids]);
        *&mut vxc.i_mut((.., ..nvar_i)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, ..nvar_i)) += *scale * fxc_i;
    }

    (rho, exc, vxc, fxc)
}

pub fn get_drho(xc_type: XCDenType, ao: TsrView, ao_dm0: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    // direct transformation of `pyhessref/nimatmul/rks_with_becke.py`, function `_make_drho`

    let ngrids = ao.shape()[0];
    let nvar = xc_type.num_nvar();
    let natm = aoslices.len();
    let device = ao.device().clone();

    let mut drho = rt::zeros(([ngrids, nvar, 3, natm], &device));

    // components: [rho_var, t_direction, cbra, cket]
    let mut components = vec![(0, 0, X, O), (0, 1, Y, O), (0, 2, Z, O)];
    if matches!(xc_type, SIGMA | TAU) {
        let sigma_bra2_ket0 = [
            [(1, 0, XX, O), (2, 0, XY, O), (3, 0, XZ, O)],
            [(1, 1, YX, O), (2, 1, YY, O), (3, 1, YZ, O)],
            [(1, 2, ZX, O), (2, 2, ZY, O), (3, 2, ZZ, O)],
        ];
        components.extend(sigma_bra2_ket0.concat());
        let sigma_bra1_ket1 = [
            [(1, 0, X, X), (2, 0, X, Y), (3, 0, X, Z)],
            [(1, 1, Y, X), (2, 1, Y, Y), (3, 1, Y, Z)],
            [(1, 2, Z, X), (2, 2, Z, Y), (3, 2, Z, Z)],
        ];
        components.extend(sigma_bra1_ket1.concat());
    }
    if matches!(xc_type, TAU) {
        let tau_bra2_ket1 = [
            [(4, 0, XX, X), (4, 0, XY, Y), (4, 0, XZ, Z)],
            [(4, 1, YX, X), (4, 1, YY, Y), (4, 1, YZ, Z)],
            [(4, 2, ZX, X), (4, 2, ZY, Y), (4, 2, ZZ, Z)],
        ];
        components.extend(tau_bra2_ket1.concat());
    }

    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        for &(v, t, cbra, cket) in &components {
            *&mut drho.i_mut((.., v, t, A)) -= rt::vecdot(ao.i((.., slc, cbra)), ao_dm0.i((.., slc, cket)), 1);
        }
    }

    match xc_type {
        RHO => *&mut drho.i_mut((.., 0..1)) *= 2.0,
        SIGMA | TAU => *&mut drho.i_mut((.., 0..4)) *= 2.0,
        LAPL => unimplemented!(),
    }
    drho
}

pub fn get_de_fxc(wf: TsrView, drho: TsrView) -> Tsr {
    // gxy, gxtA, gysB -> tsAB

    let [ngrids, nvar, _, natm] = drho.shape().iter().cloned().collect_array().unwrap();

    let tmp1 = rt::vecdot(wf.i((.., .., .., None, None)), drho.i((.., .., None, .., ..)), 1);
    let tmp1 = tmp1.reshape([ngrids * nvar, natm * 3]);
    let drho = drho.reshape([ngrids * nvar, natm * 3]);
    let tmp2 = tmp1.t() % drho;

    tmp2.reshape([3, natm, 3, natm]).transpose([0, 2, 1, 3]).into_contig(ColMajor)
}

/// Builder part of [`get_de_vxc_diag`]: the AO-resolved diagonal vxc kernel
/// `[nao, 6]` (pairs xx, xy, xz, yy, yz, zz), shared with `de_becke_vxc_diag`.
pub fn make_dao_vxc_diag(xc_type: XCDenType, ao: TsrView, ao_dm0: TsrView, wv: TsrView) -> Tsr {
    const TRIPLE_SIGMA_DIAG: [[usize; 3]; 6] =
        [[XXX, XXY, XXZ], [XXY, XYY, XYZ], [XXZ, XYZ, XZZ], [XYY, YYY, YYZ], [XYZ, YYZ, YZZ], [XZZ, YZZ, ZZZ]];
    const TRIPLE_TAU_DIAG: [[usize; 6]; 3] =
        [[XXX, XXY, XXZ, XYY, XYZ, XZZ], [XXY, XYY, XYZ, YYY, YYZ, YZZ], [XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]];

    let nao = ao.shape()[1];
    let device = ao.device().clone();

    let mut dao_vxc_diag: Tsr = rt::zeros(([nao, 6], &device));

    // contribution 1: lda/gga ao deriv 2
    let mut aow = index!(ao_dm0, O) * index!(wv, 0);
    if matches!(xc_type, SIGMA | TAU) {
        aow += index!(ao_dm0, X) * index!(wv, X);
        aow += index!(ao_dm0, Y) * index!(wv, Y);
        aow += index!(ao_dm0, Z) * index!(wv, Z);
    }
    for (idx_ts, its) in [XX, XY, XZ, YY, YZ, ZZ].into_iter().enumerate() {
        index_mut!(dao_vxc_diag, idx_ts) += 2 * rt::vecdot(index!(ao, its), &aow, 0);
    }

    // contribution 2: gga ao deriv 3
    if matches!(xc_type, SIGMA | TAU) {
        for (idx_ts, &[i3x, i3y, i3z]) in TRIPLE_SIGMA_DIAG.iter().enumerate() {
            let aow =
                index!(ao, i3x) * index!(wv, X) + index!(ao, i3y) * index!(wv, Y) + index!(ao, i3z) * index!(wv, Z);
            index_mut!(dao_vxc_diag, idx_ts) += 2 * rt::vecdot(&aow, index!(ao_dm0, O), 0);
        }
    }

    // contribution 3: tau ao deriv 3
    if matches!(xc_type, TAU) {
        for (r, &idx_tri) in TRIPLE_TAU_DIAG.iter().enumerate() {
            let aow = index!(ao_dm0, r + 1) * index!(wv, 4);
            for (idx_ts, &i3) in idx_tri.iter().enumerate() {
                index_mut!(dao_vxc_diag, idx_ts) += rt::vecdot(index!(ao, i3), &aow, 0);
            }
        }
    }

    dao_vxc_diag
}

/// Reduction part of [`get_de_vxc_diag`]: AO-wise to atom-wise blocks, then the
/// symmetric `[6] -> [3, 3]` pair expansion.
pub fn get_de_vxc_diag(dao_vxc_diag: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
    let device = dao_vxc_diag.device().clone();

    let mut de_vxc_diag = rt::zeros(([6, natm, natm], &device));
    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        de_vxc_diag.i_mut((.., A, A)).assign(dao_vxc_diag.i(slc).sum_axes(0));
    }
    de_vxc_diag.index_select(0, [0, 1, 2, 1, 3, 4, 2, 4, 5]).into_shape([3, 3, natm, natm])
}

/// Builder part of [`get_de_vxc_off`]: the two-index vxc kernel `[nao, nao, 3, 3]`,
/// symmetrised under `[t, s, mu, nu] -> [s, t, nu, mu]` (cf `_make_dao_vxc_off`),
/// shared with `de_becke_vxc_off`.
pub fn make_dao_vxc_off(xc_type: XCDenType, ao: TsrView, wv: TsrView) -> Tsr {
    let nao = ao.shape()[1];
    let device = ao.device().clone();

    let mut dao_vxc_off: Tsr = rt::zeros(([nao, nao, 3, 3], &device));

    if matches!(xc_type, RHO) {
        for t in 0..3 {
            let aowv = index!(wv, 0) * index!(ao, t + 1);
            for s in 0..3 {
                index_mut!(dao_vxc_off, t, s).matmul_from(aowv.t(), index!(ao, s + 1), 1.0, 1.0);
            }
        }
    }

    if matches!(xc_type, SIGMA | TAU) {
        for t in 0..3 {
            let mut aowv: Tsr = 0.5 * index!(wv, 0) * index!(ao, t + 1);
            for r in 0..3 {
                aowv += index!(wv, r + 1) * index!(ao, IDX_AO_DERIV2[t][r]);
            }
            for s in 0..3 {
                index_mut!(dao_vxc_off, t, s).matmul_from(aowv.t(), index!(ao, s + 1), 2.0, 1.0);
            }
        }
    }

    if matches!(xc_type, TAU) {
        let mut dao_vxc_tau: Tsr = rt::zeros(([nao, nao, 3, 3], &device));
        for r in 0..3 {
            for t in 0..3 {
                let aowv: Tsr = 0.5 * index!(wv, 4) * index!(ao, IDX_AO_DERIV2[t][r]);
                for s in 0..t + 1 {
                    index_mut!(dao_vxc_tau, t, s).matmul_from(aowv.t(), index!(ao, IDX_AO_DERIV2[s][r]), 1.0, 1.0);
                }
            }
        }

        for t in 0..3 {
            for s in 0..t + 1 {
                index_mut!(dao_vxc_off, t, s) += &index!(dao_vxc_tau, t, s);
            }
            for s in 0..t {
                index_mut!(dao_vxc_off, s, t) += &index!(dao_vxc_tau, t, s).t();
            }
        }
    }

    // symmetrised under [t, s, mu, nu] -> [s, t, nu, mu]
    &dao_vxc_off + dao_vxc_off.transpose([1, 0, 3, 2])
}

/// Reduction part of [`get_de_vxc_off`]: contract the kernel with `dm0` AO slices.
pub fn get_de_vxc_off(dao_vxc_off: TsrView, dm0: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
    let device = dao_vxc_off.device().clone();

    let mut de_vxc_off = rt::zeros(([3, 3, natm, natm], &device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        let slcA = rt::slice!(p0A, p1A);
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            let slcB = rt::slice!(p0B, p1B);
            let contrib = rt::vecdot(dao_vxc_off.i((slcA, slcB)), dm0.i((slcA, slcB)), ([0, 1], [0, 1]));
            de_vxc_off.i_mut((.., .., A, B)).assign(&contrib);
            de_vxc_off.i_mut((.., .., B, A)).assign(contrib.t());
        }
    }

    de_vxc_off
}

pub fn get_vmat_ip(xc_type: XCDenType, ao: TsrView, wv: TsrView) -> Tsr {
    // direct transformation of `pyhessref/nimatmul/rks_with_becke.py`, function `_vmat_ip`

    let nao = ao.shape()[1];
    let device = ao.device().clone();

    let mut vmat_ip = rt::zeros(([nao, nao, 3], &device));

    if matches!(xc_type, RHO) {
        let aow: Tsr = index!(wv, 0) * index!(ao, O);
        for t in 0..3 {
            index_mut!(vmat_ip, t).matmul_from(&index!(ao, t + 1).t(), &aow, 1.0, 1.0);
        }
        return vmat_ip;
    }

    if matches!(xc_type, SIGMA | TAU) {
        let mut aow: Tsr = 0.5 * index!(wv, 0) * index!(ao, O);
        for r in 0..3 {
            aow += index!(wv, r + 1) * index!(ao, r + 1);
        }
        for t in 0..3 {
            index_mut!(vmat_ip, t).matmul_from(&index!(ao, t + 1).t(), &aow, 1.0, 1.0);
        }

        for t in 0..3 {
            let mut aow_d: Tsr = 0.5 * index!(wv, 0) * index!(ao, t + 1);
            for r in 0..3 {
                aow_d += index!(wv, r + 1) * index!(ao, IDX_AO_DERIV2[t][r]);
            }
            index_mut!(vmat_ip, t).matmul_from(&aow_d.t(), &index!(ao, O), 1.0, 1.0);
        }
    }

    if matches!(xc_type, TAU) {
        for r in 0..3 {
            let aow: Tsr = 0.5 * index!(wv, 4) * index!(ao, r + 1);
            for t in 0..3 {
                index_mut!(vmat_ip, t).matmul_from(&index!(ao, IDX_AO_DERIV2[t][r]).t(), &aow, 1.0, 1.0);
            }
        }
    }

    vmat_ip
}

pub fn get_vmat_vxc(vmat_ip: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    // direct transformation of `pyhessref/nimatmul/rks_with_becke.py`, function `_vmat_vxc`

    let natm = aoslices.len();
    let nao = vmat_ip.shape()[0];

    let mut vmat_vxc: Tsr = rt::zeros(([nao, nao, 3, natm], vmat_ip.device()));

    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        *&mut vmat_vxc.i_mut((slc, .., .., A)) -= vmat_ip.i((slc, .., ..));
    }

    &vmat_vxc + vmat_vxc.swapaxes(0, 1)
}

pub fn get_vmat_fxc(xc_type: XCDenType, ao: TsrView, drho: TsrView, wf: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    // direct transformation of `pyhessref/nimatmul/rks_with_becke.py`, function `_vmat_fxc`

    let natm = aoslices.len();
    let nao = ao.shape()[1];

    let mut vmat_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], ao.device()));

    for A in 0..natm {
        if matches!(xc_type, RHO) {
            let wf_rho: Tsr = 0.5 * index!(wf, O, O) * drho.i((.., O, .., A));
            for t in 0..3 {
                let aow = index!(wf_rho, t) * index!(ao, O);
                index_mut!(vmat_fxc, t, A).matmul_from(aow.t(), index!(ao, O), 1.0, 1.0);
            }
        }

        if matches!(xc_type, SIGMA | TAU) {
            let mut wf_rho = rt::vecdot(&wf, drho.i((.., .., None, .., A)), 1);
            *&mut wf_rho.i_mut((.., 0)) *= 0.5;
            if matches!(xc_type, TAU) {
                *&mut wf_rho.i_mut((.., 4)) *= 0.25;
            }
            for t in 0..3 {
                let aow = rt::vecdot(wf_rho.i((.., None, ..4, t)), ao.i((.., .., ..4)), 2);
                index_mut!(vmat_fxc, t, A).matmul_from(aow.t(), index!(ao, O), 1.0, 1.0);
            }

            if matches!(xc_type, TAU) {
                for r in [X, Y, Z] {
                    for t in 0..3 {
                        let aow = wf_rho.i((.., 4, t)) * index!(ao, r);
                        index_mut!(vmat_fxc, t, A).matmul_from(aow.t(), index!(ao, r), 1.0, 1.0);
                    }
                }
            }
        }
    }

    &vmat_fxc + vmat_fxc.swapaxes(0, 1)
}

pub fn get_vmat_deriv1(
    xc_type: XCDenType,
    ao: TsrView,
    drho: TsrView,
    wf: TsrView,
    vmat_ip: TsrView,
    aoslices: &[[usize; 4]],
) -> Tsr {
    let vmat_fxc = get_vmat_fxc(xc_type, ao, drho, wf, aoslices);
    let vmat_vxc = get_vmat_vxc(vmat_ip, aoslices);
    &vmat_fxc + &vmat_vxc
}

/* #endregion */

/* #region becke grid-shift parts: skeleton hessian */

/// `de_becke_atom_1` (notebook t3): `-einsum("g, txg, xyg, Bsyg -> Bts", w, prho, fxc, drho)`,
/// evaluated on the batch's grids only — the result fills the batch atom's row.
///
/// - `w`: `[ngrids]` grid weights of the batch.
/// - `prho`: `[ngrids, nvar, 3]` total skeleton derivative.
///
/// Returns `[3, 3, natm]` (tsB) for the row of the batch's grid atom.
pub fn get_de_becke_atom_1(w: TsrView, prho: TsrView, fxc: TsrView, drho: TsrView) -> Tsr {
    // fxc_drho [g, x, s, B] = sum_y fxc[g, x, y] drho[g, y, s, B]
    let fxc_drho = rt::vecdot(fxc.i((.., .., .., None, None)), drho.i((.., None, .., .., ..)), 2);
    // fold in the batch grid weights
    let fxc_drho = fxc_drho * w.i((.., None, None, None));
    // t3 [s, B, t] = -sum_{g, x} prho[g, x, t] fxc_drho[g, x, s, B]
    -rt::vecdot(fxc_drho.i((.., .., None, .., ..)), prho.i((.., .., .., None, None)), ([0, 1], [0, 1]))
}

/// `de_becke_atom_2` (notebook t5): `-einsum("Bsg, xg, txg -> Bts", dw, vxc, prho)`.
///
/// - `dw`: grid-first becke `dw`, `[ngrids, 3, natm]` (Fortran-order wrap of the C-order `[A, t,
///   g]` becke output buffer; see `make_hessian_setup_batch_becke`).
///
/// Returns `[3, 3, natm]` (tsB) for the row of the batch's grid atom.
pub fn get_de_becke_atom_2(dw: TsrView, vxc: TsrView, prho: TsrView) -> Tsr {
    // vdw2 [g, x, s, B] = vxc[g, x] dw[g, s, B]
    let vdw2 = dw.i((.., None, .., ..)) * vxc.i((.., .., None, None));
    // t5 [s, B, t] = -sum_{g, x} vdw2[g, x, s, B] prho[g, x, t]
    -rt::vecdot(vdw2.i((.., .., None, .., ..)), prho.i((.., .., .., None, None)), ([0, 1], [0, 1]))
}

/// `de_becke_atom_3` (notebook t6): `einsum("g, xyg, syg, txg -> ts", w, fxc, prho, prho)`,
/// evaluated on the batch's grids only — fills the `[atm_idx, atm_idx]` diagonal block.
///
/// Returns `[3, 3]` (ts).
pub fn get_de_becke_atom_3(w: TsrView, prho: TsrView, fxc: TsrView) -> Tsr {
    // fp [g, x, t] = sum_y fxc[g, x, y] prho[g, y, t]
    let fp = rt::vecdot(fxc.i((.., .., .., None)), prho.i((.., None, .., ..)), 2);
    // wprho [g, x, s]
    let wprho = &prho * w.i((.., None, None));
    // t6 [s, t] = sum_{g, x} wprho[g, x, s] fp[g, x, t]
    rt::vecdot(wprho.i((.., .., None, ..)), fp.i((.., .., .., None)), ([0, 1], [0, 1]))
}

/// Contract a per-grid-atom skeleton-Vxc kernel into the batch atom's
/// Hessian row (`_contract_pvxc` in the pyhessref reference).
///
/// - `pvxc`: `[3, 3, nao]`.
///
/// Returns `[3, 3, natm]` (tsB): the row of atom `atm_idx`, before the
/// `(A, t) <-> (B, s)` symmetrisation — the batched driver scatters the row
/// into the last (A) axis of the `[3, 3, natm, natm]` accumulator and applies
/// the symmetrisation once after the accumulation.
pub fn contract_pvxc(pvxc: TsrView, atm_idx: usize, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
    let mut row: Tsr = rt::zeros(([3, 3, natm], pvxc.device()));

    *&mut row.i_mut((.., .., atm_idx)) += pvxc.sum_axes(2);
    for (B, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        *&mut row.i_mut((.., .., B)) -= 2.0 * &pvxc.i((.., .., slc)).sum_axes(2);
    }

    row
}

/// `de_becke_vxc_diag` (t8) / `de_becke_vxc_off` (t9): the basis form of t4/t7,
/// contracting the per-batch `dao_vxc_*` kernels (shared with `de_vxc_*`).
///
/// Returns both parts, each the batch atom's `[3, 3, natm]` row.
pub fn get_de_becke_vxc_parts(
    dao_vxc_diag: TsrView,
    dao_vxc_off: TsrView,
    dm0: TsrView,
    atm_idx: usize,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let nao = dao_vxc_diag.shape()[0];

    // pvxc_diag [3, 3, nao] = 0.5 * dao_vxc_diag[IDX_PAIR_TS]
    const IDX_PAIR_TS: [usize; 9] = [0, 1, 2, 1, 3, 4, 2, 4, 5];
    let pvxc_diag: Tsr = 0.5 * dao_vxc_diag.index_select(1, IDX_PAIR_TS).t().into_shape([3, 3, nao]);

    // pvxc_off[t, s, u] = 0.5 * sum_v dao_vxc_off[v, u, t, s] dm0[u, v]  (einsum
    // "tsuv, uv -> tsu" on the python [t, s, u, v] kernel; this module's kernel
    // stores the AO indices transposed relative to python, so the free index is
    // the FIRST AO axis of this module's [u, v, t, s] storage)
    let pvxc_off: Tsr = 0.5 * rt::vecdot(dao_vxc_off, dm0, 0).transpose([1, 2, 0]).into_contig(ColMajor);
    (contract_pvxc(pvxc_diag.view(), atm_idx, aoslices), contract_pvxc(pvxc_off.view(), atm_idx, aoslices))
}

/* #endregion */

/* #region becke grid-shift parts: f1ao (CP-KS RHS) */

/// Symmetric Vxc-style Fock from a generic weight and functional field
/// (`_vxc_fock` in the pyhessref reference).
///
/// - `veff`: `[ngrids, nvar]` functional-derivative field.
/// - `wg`: `[ngrids]` weight field (becke `dw[A, t]` row for T1, batch weights for T2_fxc).
pub fn vxc_fock(xc_type: XCDenType, ao: TsrView, veff: TsrView, wg: TsrView) -> Tsr {
    let mut wv = &veff * &wg;
    *&mut wv.i_mut((.., O)) *= 0.5;

    if matches!(xc_type, RHO) {
        let aow = index!(wv, O) * index!(ao, O);
        let aow_ao = aow.t() % index!(ao, O);
        return &aow_ao + aow_ao.t();
    }

    let aow = rt::vecdot(ao.i((.., .., ..4)), wv.i((.., None, ..4)), -1);
    let aow_ao = aow.t() % index!(ao, O);
    let mut vxc_fock = &aow_ao + aow_ao.t();

    if matches!(xc_type, TAU) {
        *&mut wv.i_mut((.., 4)) *= 0.5;
        for j in 1..4 {
            let aow = index!(wv, 4) * index!(ao, j);
            vxc_fock += &(aow.t() % index!(ao, j));
        }
    }

    vxc_fock
}

/// f1ao-level Becke grid-shift parts (T1/T2_ipip/T2_fxc of
/// `_vmat_becke_parts` in the pyhessref reference).
///
/// - `dw`: grid-first becke `dw`, `[ngrids, 3, natm]` (Fortran-order wrap of the C-order `[A, t,
///   g]` becke output buffer; see `make_hessian_setup_batch_becke`).
/// - `prho`: `[ngrids, nvar, 3]`, `w`: `[ngrids]`, `vmat_ip`: `[nao, nao, 3]`.
///
/// Returns `vmat_becke_T1` `[nao, nao, 3, natm]` (filled on all rows) and the
/// T2 parts `[nao, nao, 3]` — the batch atom's row, scattered into the
/// `[nao, nao, 3, natm]` accumulators by the batched driver.
#[allow(clippy::too_many_arguments)]
pub fn get_vmat_becke_parts(
    xc_type: XCDenType,
    ao: TsrView,
    vxc: TsrView,
    fxc: TsrView,
    prho: TsrView,
    w: TsrView,
    dw: TsrView,
    vmat_ip: TsrView,
) -> (Tsr, Tsr, Tsr) {
    let nao = ao.shape()[1];
    let natm = dw.shape()[2];
    let device = ao.device().clone();

    // T1: Vxc-style Fock with the becke dw[A, t] rows as weights (all rows)
    let mut vmat_becke_t1 = rt::zeros(([nao, nao, 3, natm], &device));
    for A in 0..natm {
        for t in 0..3 {
            let fock = vxc_fock(xc_type, ao.view(), vxc.view(), index!(dw, t, A));
            *&mut vmat_becke_t1.i_mut((.., .., t, A)) += &fock;
        }
    }

    // T2_ipip: batch's vmat_ip symmetrised in AO
    let vmat_becke_t2_ipip = &vmat_ip + vmat_ip.swapaxes(0, 1);

    // T2_fxc: fxc folded with prho[t], contracted on the batch weights
    let mut vmat_becke_t2_fxc = rt::zeros(([nao, nao, 3], &device));
    let ngrids = fxc.shape()[0];
    let nvar = fxc.shape()[1];
    for t in 0..3 {
        // fxc_prho [g, x] = sum_y fxc[g, x, y] prho[g, y, t]
        let prho_t = prho.i((.., .., t));
        let neg_fxc_prho: Tsr =
            -1.0 * rt::vecdot(fxc.i((.., .., .., None)), prho_t.i((.., None, .., None)), 2).into_shape([ngrids, nvar]);
        let fock = vxc_fock(xc_type, ao.view(), neg_fxc_prho.view(), w.view());
        *&mut vmat_becke_t2_fxc.i_mut((.., .., t)) += &fock;
    }

    (vmat_becke_t1, vmat_becke_t2_ipip, vmat_becke_t2_fxc)
}

/* #endregion */

/* #region per-batch evaluation */

/// Per-batch evaluation of all skeleton ingredients with the grid-shift
/// (`make_hessian_setup_batch` in the pyhessref reference).  The batch must hold
/// grids of the single atom `atm_idx` (ByAtom attribution).
///
/// Full-grid outputs (`[3, 3, natm, natm]` for the skeleton parts, `[nao, nao,
/// 3, natm]` for `vmat_becke_T1`) accumulate across batches by a plain sum.
/// The grid-atom outputs carry only the batch atom's contribution and are
/// scattered into the full accumulators by [`make_hessian_setup_becke`]:
/// `de_becke_atom_1/2` and `de_becke_vxc_diag/off` as `[3, 3, natm]` rows,
/// `de_becke_atom_3` as the `[3, 3]` diagonal block, and `vmat_becke_T2_*` as
/// `[nao, nao, 3]`.
#[allow(clippy::too_many_arguments)]
pub fn make_hessian_setup_batch_becke(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    atm_idx: usize,
    quadrature_weights: &[f64],
    adjustment_factor: &[f64],
    hardness: usize,
) -> HashMap<&'static str, Tsr> {
    let natm = mol.natm();
    let device = dm0.device().clone();
    let aoslices = mol.aoslice_by_atom();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    // clone grid data before the mutable AO cache borrow of `ni`
    let grid_coords = ni.coords.clone();
    let weights_data = ni.weights.clone();
    let ngrids = weights_data.len();

    // --- ao, rho, exc, vxc, fxc --- //

    let ao = ni.get_cached_ao(get_hess_ao_deriv(xc_type));
    let ncomp_ao_dm0 = get_hess_ncomp_ao_dm0(xc_type);
    let ao_dm0 = index!(ao, ..ncomp_ao_dm0) % &dm0;
    let (rho, exc, vxc, fxc) = get_rho_exc_vxc_fxc(xc_func_list, ao.view(), ao_dm0.view());

    let weights = rt::asarray((weights_data, &device));
    let wv = &weights * &vxc;
    let wf = &weights * &fxc;

    // --- drho, prho --- //

    let drho = get_drho(xc_type, ao.view(), ao_dm0.view(), &aoslices);
    // prho [ngrids, nvar, 3] = drho summed over atoms
    let prho = drho.sum_axes(3);

    // --- without-becke parts --- //

    let de_fxc = get_de_fxc(wf.view(), drho.view());
    let dao_vxc_diag = make_dao_vxc_diag(xc_type, ao.view(), ao_dm0.view(), wv.view());
    let de_vxc_diag = get_de_vxc_diag(dao_vxc_diag.view(), &aoslices);
    let dao_vxc_off = make_dao_vxc_off(xc_type, ao.view(), wv.view());
    let de_vxc_off = get_de_vxc_off(dao_vxc_off.view(), dm0.view(), &aoslices);

    let vmat_ip = get_vmat_ip(xc_type, ao.view(), wv.view());
    let vmat_fxc = get_vmat_fxc(xc_type, ao.view(), drho.view(), wf.view(), &aoslices);
    let vmat_vxc = get_vmat_vxc(vmat_ip.view(), &aoslices);
    let vmat_deriv1 = get_vmat_deriv1(xc_type, ao.view(), drho.view(), wf.view(), vmat_ip.view(), &aoslices);

    // --- becke partition: dw in full, ddw only through the cddw contraction --- //

    // cddw (nset = 1): the only ddw consumer is de_becke_full_2 = ddw . (exc * rho0)
    let cddw = (&exc * rho.i((.., 0))).into_vec();

    let atm_coords = mol.atom_coords();
    let boundaries = by_atom_batch(natm, atm_idx, ngrids);
    let deriv_arg = BeckeDerivArg {
        output_w: false,
        output_dw: true,
        output_ddw: false,
        contract_w: None,
        contract_dw: None,
        contract_ddw: Some(&cddw),
    };
    let becke_result = becke_partition(
        &grid_coords,
        &atm_coords,
        AtmIndices::ByAtom(&boundaries),
        quadrature_weights,
        adjustment_factor,
        hardness,
        1024,
        2,
        Some(deriv_arg),
    );
    // dw flat is C-order [A, t, g] == Fortran-order [g, t, A];
    // ddc flat is C-order [A, t, B, s, iset] == Fortran-order [iset, s, B, t, A]
    let dw = rt::asarray((becke_result.dw.unwrap(), [ngrids, 3, natm].f(), &device));

    // --- becke grid-shift parts --- //

    // de_becke_full_1 (notebook t1): einsum("Atg, xg, Bsxg -> ABts", dw, vxc, drho);
    let de_becke_full_1 = {
        // vxc_drho [g, s, B] = sum_x vxc[g, x] drho[g, x, s, B]
        let vxc_drho = rt::vecdot(drho.view(), vxc.view(), 1);
        // t1 [s, t, B, A] = sum_g dw[g, t, A] vxc_drho[g, s, B]
        rt::vecdot(dw.i((.., None, .., None, ..)), vxc_drho.i((.., .., None, .., None)), 0)
    };

    // de_becke_full_2 (notebook t2): einsum("AtBsg, g, g -> ABts", ddw, exc, rho[0]) via
    // the cddw contraction above (nset = 1), naturally symmetric;
    // ddc flat is C-order [A, t, B, s, iset] == Fortran-order [iset, s, B, t, A]
    let de_becke_full_2 = rt::asarray((becke_result.ddc.unwrap(), [3, natm, 3, natm].f(), &device))
        .transpose([2, 0, 3, 1])
        .into_contig(ColMajor);

    // grid-atom parts: compact tensors for the batch atom's row (resp.
    // diagonal block); the scatter into `[3, 3, natm, natm]` is done by
    // `make_hessian_setup_becke`
    let de_becke_atom_1 = get_de_becke_atom_1(weights.view(), prho.view(), fxc.view(), drho.view());
    let de_becke_atom_2 = get_de_becke_atom_2(dw.view(), vxc.view(), prho.view());
    let de_becke_atom_3 = get_de_becke_atom_3(weights.view(), prho.view(), fxc.view());

    let (de_becke_vxc_diag, de_becke_vxc_off) =
        get_de_becke_vxc_parts(dao_vxc_diag.view(), dao_vxc_off.view(), dm0.view(), atm_idx, &aoslices);

    let (vmat_becke_t1, vmat_becke_t2_ipip, vmat_becke_t2_fxc) = get_vmat_becke_parts(
        xc_type,
        ao.view(),
        vxc.view(),
        fxc.view(),
        prho.view(),
        weights.view(),
        dw.view(),
        vmat_ip.view(),
    );

    HashMap::from([
        ("fxc", fxc),
        ("de_vxc_diag", de_vxc_diag),
        ("de_vxc_off", de_vxc_off),
        ("de_fxc", de_fxc),
        ("vmat_ip", vmat_ip),
        ("vmat_fxc", vmat_fxc),
        ("vmat_vxc", vmat_vxc),
        ("vmat_deriv1", vmat_deriv1),
        ("de_becke_full_1", de_becke_full_1),
        ("de_becke_full_2", de_becke_full_2),
        ("de_becke_atom_1", de_becke_atom_1),
        ("de_becke_atom_2", de_becke_atom_2),
        ("de_becke_atom_3", de_becke_atom_3),
        ("de_becke_vxc_diag", de_becke_vxc_diag),
        ("de_becke_vxc_off", de_becke_vxc_off),
        ("vmat_becke_T1", vmat_becke_t1),
        ("vmat_becke_T2_ipip", vmat_becke_t2_ipip),
        ("vmat_becke_T2_fxc", vmat_becke_t2_fxc),
    ])
}

/* #endregion */

/* #region batched driver */

/// `x + x.transpose(1, 0, 3, 2)` on a `[3, 3, natm, natm]` (tsAB) tensor: the
/// `(A, t) <-> (B, s)` symmetrisation of the pyhessref reference.
fn symmetrize_ts_ab(x: Tsr) -> Tsr {
    &x + x.transpose([1, 0, 3, 2])
}

/// Batched driver for all DFT skeleton ingredients with the grid-shift
/// (`make_hessian_setup` in the pyhessref reference).
///
/// Splits the (atom-grouped) grid into batches of at most `nbatch_grids` grids
/// that never cross an atom boundary, evaluates
/// [`make_hessian_setup_batch_becke`] on each, and accumulates: the full-grid
/// keys by a plain sum, the grid-atom keys (`de_becke_atom_1/2`,
/// `de_becke_vxc_diag/off`, `vmat_becke_T2_ipip/fxc` as rows,
/// `de_becke_atom_3` as the diagonal block) by scattering into the batch
/// atom's slice of the last (A) axis.  The keys
/// `de_becke_full_1/atom_1/atom_2/vxc_diag/vxc_off` are then symmetrised under
/// `(A, t) <-> (B, s)`; `de_becke_full_2` is naturally symmetric.
///
/// Returns all keys of the per-batch function accumulated over batches, plus
/// `de_xc_skeleton_no_becke`, `de_xc_skeleton`, and `vmat_deriv1_grid` (=
/// `vmat_deriv1` + T1 + T2_ipip + T2_fxc).
#[allow(clippy::too_many_arguments)]
pub fn make_hessian_setup_becke(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    atm_quad_split: &[usize],
    quadrature_weights: &[f64],
    adjustment_factor: &[f64],
    hardness: usize,
    nbatch_grids: usize,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    assert!(
        atm_list.is_none() || atm_list.unwrap().len() == mol.natm(),
        "the becke grid-shift currently requires the full atom list"
    );
    let natm = mol.natm();
    let nao = mol.nao();
    let ngrids = ni.weights.len();
    let nchunk = ni.nchunk;
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
    let nvar = xc_type.num_nvar();
    let deriv_level = get_hess_ao_deriv(xc_type);
    let device = dm0.device().clone();

    let batches = quad_split_by_atom(atm_quad_split, nbatch_grids, natm);
    let nbatches = batches.len();

    let fxc_full: Tsr = rt::zeros(([ngrids, nvar, nvar], &device));
    let de_fxc: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_ip: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_vxc: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_deriv1: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let de_becke_full_1: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_full_2: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_1: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_2: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_3: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_vxc_diag: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_vxc_off: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_becke_t1: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_t2_ipip: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_t2_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], &device));

    let timing = Arc::new(Mutex::new(IndexMap::from([("total", 0.0)])));
    let time_total = std::time::Instant::now();

    // atomic guard to avoid racing write
    let guard = Mutex::new(());

    for (ibatch, (atm_idx, start_batch, end_batch)) in batches.into_iter().enumerate() {
        // handle AO integral at batch level
        let mut ni_batch = ni.split_batch(start_batch, end_batch);
        ni_batch.get_cached_ao(deriv_level);

        // other parts can be parallelized by chunk, with atomic guard for reduction
        (start_batch..end_batch).into_par_iter().step_by(nchunk).for_each(|start| {
            let end = (start + nchunk).min(end_batch);
            let mut ni_chunk = ni_batch.split_batch(start - start_batch, end - start_batch);
            let result_chunk = make_hessian_setup_batch_becke(
                mol,
                xc_func_list,
                &mut ni_chunk,
                dm0.view(),
                atm_idx,
                &quadrature_weights[start..end],
                adjustment_factor,
                hardness,
            );
            // fill fxc (disjoint grid ranges, no guard needed)
            unsafe {
                let fxc_slc = fxc_full.i(start..end);
                let mut fxc_slc = fxc_slc.force_mut();
                fxc_slc.assign(&result_chunk["fxc"]);
            }
            // add up other tensors
            unsafe {
                let _lock = guard.lock().unwrap();
                *&mut de_fxc.force_mut() += &result_chunk["de_fxc"];
                *&mut de_vxc_diag.force_mut() += &result_chunk["de_vxc_diag"];
                *&mut de_vxc_off.force_mut() += &result_chunk["de_vxc_off"];
                *&mut vmat_ip.force_mut() += &result_chunk["vmat_ip"];
                *&mut vmat_fxc.force_mut() += &result_chunk["vmat_fxc"];
                *&mut vmat_vxc.force_mut() += &result_chunk["vmat_vxc"];
                *&mut vmat_deriv1.force_mut() += &result_chunk["vmat_deriv1"];
                *&mut de_becke_full_1.force_mut() += &result_chunk["de_becke_full_1"];
                *&mut de_becke_full_2.force_mut() += &result_chunk["de_becke_full_2"];
                *&mut vmat_becke_t1.force_mut() += &result_chunk["vmat_becke_T1"];
                // grid-atom parts: scatter the batch atom's compact contribution
                // into its row of the last (A) axis; the (A, t) <-> (B, s)
                // symmetrisation is applied once after the accumulation
                *&mut de_becke_atom_1.i((Ellipsis, atm_idx)).force_mut() +=
                    &result_chunk["de_becke_atom_1"].transpose([1, 0, 2]);
                *&mut de_becke_atom_2.i((Ellipsis, atm_idx)).force_mut() +=
                    &result_chunk["de_becke_atom_2"].transpose([1, 0, 2]);
                *&mut de_becke_atom_3.i((Ellipsis, atm_idx, atm_idx)).force_mut() +=
                    &result_chunk["de_becke_atom_3"].t();
                *&mut de_becke_vxc_diag.i((Ellipsis, atm_idx)).force_mut() +=
                    &result_chunk["de_becke_vxc_diag"].transpose([1, 0, 2]);
                *&mut de_becke_vxc_off.i((Ellipsis, atm_idx)).force_mut() +=
                    &result_chunk["de_becke_vxc_off"].transpose([1, 0, 2]);
                *&mut vmat_becke_t2_ipip.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_T2_ipip"];
                *&mut vmat_becke_t2_fxc.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_T2_fxc"];
            }
        });

        timing.lock().unwrap().insert("total", time_total.elapsed().as_secs_f64());
        if verbose {
            println!(
                "In make_hessian_setup_becke, batch {}/{} (atom {atm_idx}): grids {start_batch}..{end_batch}",
                ibatch + 1,
                nbatches,
            );
            println!("  Elapsed time from start (Wall time): {:.4} sec", timing.lock().unwrap()["total"]);
        }
    }

    // symmetrize on the atom indices for the becke parts (the grid-atom keys
    // were accumulated as unsymmetrized rows; de_becke_full_2 is naturally
    // symmetric)
    let de_becke_full_1 = symmetrize_ts_ab(de_becke_full_1);
    let de_becke_atom_1 = symmetrize_ts_ab(de_becke_atom_1);
    let de_becke_atom_2 = symmetrize_ts_ab(de_becke_atom_2);
    let de_becke_vxc_diag = symmetrize_ts_ab(de_becke_vxc_diag);
    let de_becke_vxc_off = symmetrize_ts_ab(de_becke_vxc_off);

    // final assemblies
    let de_xc_skeleton_no_becke = &de_vxc_diag + &de_vxc_off + &de_fxc;
    let de_xc_skeleton = &de_xc_skeleton_no_becke
        + &de_becke_full_1
        + &de_becke_full_2
        + &de_becke_atom_1
        + &de_becke_atom_2
        + &de_becke_atom_3
        + &de_becke_vxc_diag
        + &de_becke_vxc_off;
    let vmat_deriv1_grid = &vmat_deriv1 + &vmat_becke_t1 + &vmat_becke_t2_ipip + &vmat_becke_t2_fxc;

    let result = HashMap::from([
        ("fxc", fxc_full),
        ("de_fxc", de_fxc),
        ("de_vxc_diag", de_vxc_diag),
        ("de_vxc_off", de_vxc_off),
        ("vmat_ip", vmat_ip),
        ("vmat_fxc", vmat_fxc),
        ("vmat_vxc", vmat_vxc),
        ("vmat_deriv1", vmat_deriv1),
        ("de_becke_full_1", de_becke_full_1),
        ("de_becke_full_2", de_becke_full_2),
        ("de_becke_atom_1", de_becke_atom_1),
        ("de_becke_atom_2", de_becke_atom_2),
        ("de_becke_atom_3", de_becke_atom_3),
        ("de_becke_vxc_diag", de_becke_vxc_diag),
        ("de_becke_vxc_off", de_becke_vxc_off),
        ("vmat_becke_T1", vmat_becke_t1),
        ("vmat_becke_T2_ipip", vmat_becke_t2_ipip),
        ("vmat_becke_T2_fxc", vmat_becke_t2_fxc),
        ("de_xc_skeleton_no_becke", de_xc_skeleton_no_becke),
        ("de_xc_skeleton", de_xc_skeleton),
        ("vmat_deriv1_grid", vmat_deriv1_grid),
    ]);

    let timing = timing.lock().unwrap().clone();
    (result, timing)
}

/* #endregion */

/* #region final implementation of RKS Hessian with becke grid-shift */

/// RKS Hessian XC component with the Becke grid-shift
/// (`RHessKSNaiveBecke` in the pyhessref reference).
///
/// Grids must be atom-grouped (`sort_grids=False` in pyscf; the ByAtom
/// attribution scheme of `becke_partition`).  `adjustment_factor` is the
/// antisymmetric radii table flattened in column-major `(natm, natm)` order
/// (the convention of [`super::becke_partition::becke_partition`]).
pub struct RHessKSNIMatmulBecke<'a> {
    pub mol: CInt,
    pub xc_func_list: &'a [(f64, LibXCFunctional)],
    pub ni: NIMatmul<'a>,
    pub ni_cpks: Option<NIMatmul<'a>>,
    pub quadrature_weights: Vec<f64>,
    pub atm_quad_split: Vec<usize>,
    pub adjustment_factor: Vec<f64>,
    pub hardness: usize,
    pub nbatch_grids: usize,
    pub verbose: bool,
    pub intmd: HashMap<String, Tsr>,
    pub result: HashMap<String, Tsr>,
}

impl<'a> RHessKSNIMatmulBecke<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mol: &CInt,
        xc_func_list: &'a [(f64, LibXCFunctional)],
        ni: NIMatmul<'a>,
        quadrature_weights: Vec<f64>,
        atm_quad_split: Vec<usize>,
        adjustment_factor: Vec<f64>,
        hardness: usize,
        nbatch_grids: usize,
        verbose: bool,
    ) -> Self {
        Self {
            mol: mol.clone(),
            xc_func_list,
            ni,
            ni_cpks: None,
            quadrature_weights,
            atm_quad_split,
            adjustment_factor,
            hardness,
            nbatch_grids,
            verbose,
            intmd: HashMap::new(),
            result: HashMap::new(),
        }
    }

    /// Perform the Hessian setup for RKS calculations, with the grid-shift.
    ///
    /// Mirrors [`super::hess_rks::RHessKSNIMatmul::make_hessian_setup`]: `fxc` is
    /// stored as `cpks_fxc` (unless a CP-KS-specific grid is given) for the
    /// response; `de_xc_skeleton` and `vmat_deriv1_grid` are the main results.
    pub fn make_hessian_setup(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) {
        let dm0 = get_dm0_restricted(mo_coeff, mo_occ);
        let (result, _timing) = make_hessian_setup_becke(
            &self.mol,
            self.xc_func_list,
            &mut self.ni,
            dm0.view(),
            &self.atm_quad_split,
            &self.quadrature_weights,
            &self.adjustment_factor,
            self.hardness,
            self.nbatch_grids,
            atm_list,
            self.verbose,
        );

        for (key, val) in result.into_iter() {
            if key == "fxc" {
                if self.ni_cpks.is_none() {
                    self.intmd.insert("cpks_fxc".to_string(), val);
                }
            } else {
                self.intmd.insert(key.to_string(), val);
            }
        }
    }

    /// Check if the Hessian setup is done by verifying the presence of the
    /// "de_xc_skeleton" key in the intermediate results.
    pub fn is_hessian_setup_done(&self) -> bool {
        self.intmd.contains_key("de_xc_skeleton")
    }
}

impl<'a> HessUtilAPI for RHessKSNIMatmulBecke<'a> {}

impl<'a> RHessElecInteractAPI for RHessKSNIMatmulBecke<'a> {
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        self.intmd["de_xc_skeleton"].to_owned()
    }

    fn get_deriv1_ao(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        self.intmd["vmat_deriv1_grid"].to_owned()
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        self.intmd.insert("mo_coeff".to_string(), mo_coeff.into_contig(ColMajor));
        self.intmd.insert("mo_occ".to_string(), mo_occ.into_contig(ColMajor));
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        let ni_cpks = self.ni_cpks.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff = self.intmd.get("mo_coeff").unwrap();
        let mo_occ = self.intmd.get("mo_occ").unwrap();
        let fxc_eff = self.intmd.get("cpks_fxc").unwrap();
        let occidx = mo_occ.view().greater(0).into_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);

        let (resp, _timing) = get_rks_response_bra_batched(
            ni_cpks,
            determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec()),
            fxc_eff.view(),
            bra,
            mocc.view(),
            self.verbose,
        );
        resp
    }
}

/* #endregion */
