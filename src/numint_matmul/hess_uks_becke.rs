// see also pyhessref/nimatmul/uks_with_becke.py
//
// Unrestricted sibling of `hess_rks_becke`: the single-spin helpers, the becke
// partition machinery, and the chunk-level batching driver are shared with the
// RKS implementation; only the spin-coupled pieces differ.  The spin extension
// of every grid-shift term is the obvious one — terms linear in `vxc` become a
// spin sum (vxc[alpha] against the alpha quantity plus vxc[beta] against the
// beta quantity), and terms quadratic in the fxc kernel become the four
// spin-pair sum (the same aa/ab/ba/bb structure as `get_de_fxc_uks`).
//
// Like `hess_rks_becke` (which carries its own copies of the term-level
// functions of `hess_rks`), this module carries its own copies of the
// spin-coupled pieces of `hess_uks` and is meant as a future replacement of
// `hess_uks`; the single-spin helpers are imported from `hess_rks_becke`, and
// only the response reuses `hess_uks::get_uks_response_bra_batched`.
//
// # Index and shape conventions
//
// - Index letters: `g` grid point; `u`/`v` AO basis; `x`/`y` rho component; `t` Cartesian direction
//   of the Hessian row atom `A`; `s` direction of the column atom `B`; sigma in {alpha, beta} spin.
// - On-grid tensors are grid-leading: `rho [ngrids, nvar, 2]`, `vxc [ngrids, nvar, 2]`, `fxc
//   [ngrids, nvar, 2, nvar, 2]`, `ao [ngrids, nao, ncomp]`.
// - Hessian-like tensors `[3, 3, natm, natm]` (t, s, A, B); skeleton-Fock-like tensors `[nao, nao,
//   3, natm]` (u, v, t, A), per spin.
// - `nvar` (number of rho components per spin): RHO 1, SIGMA 4, TAU 5 (see
//   [`XCDenType::num_nvar`]).
// - AO derivative components (constants below): value `O` 0; gradient `X..Z` 1..3; 2nd derivatives
//   `XX..ZZ` 4..9 (symmetric pairs, see `IDX_AO_DERIV2` in `hess_rks_becke`); 3rd derivatives
//   `XXX..ZZZ` 10..19.

use super::becke_partition::{becke_partition_with_tables, AtmIndices, BeckeDerivArg, BeckeMolTables};
use super::hess_rks_becke::{
    by_atom_chunk, contract_pvxc, get_de_becke_atom_1, get_de_becke_atom_2, get_de_vxc_diag, get_de_vxc_off, get_drho,
    get_hess_ao_deriv, get_hess_ncomp_ao_dm0, get_vmat_ip, get_vmat_vxc, make_dao_vxc_diag, make_dao_vxc_off,
    quad_split_by_atom, vxc_fock,
};
use super::hess_uks::get_uks_response_bra_batched;
use super::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/* #region const dimensions/indices definition */

const O: usize = 0;
const X: usize = 1;
const Y: usize = 2;
const Z: usize = 3;

#[allow(non_upper_case_globals)]
const α: usize = 0;
#[allow(non_upper_case_globals)]
const β: usize = 1;

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

/// On-grid spin-polarized density with 1st/2nd functional derivatives and the
/// per-particle XC energy density (pyhessref `_eval_rho_exc_vxc_fxc_uks`).
///
/// Same as [`super::hess_uks::get_rho_vxc_fxc_uks`], but also returns the
/// order-0 output `exc`, needed by the `cddw` contraction of `de_becke_full_2`
/// (there contracted with the spin-summed value channel `rhoa[0] + rhob[0]`).
///
/// # Parameters
///
/// - `xc_func_list` : list of `(scale, functional)` pairs.  The overall family is the strictest one
///   across the list; contributions of looser families are added into their leading `nvar_i` slice.
/// - `ao` : shape `[ngrids, nao, ncomp]` (g, u, component).  AO values and derivatives; only each
///   family's leading channels are read.
/// - `ao_dm0α`, `ao_dm0β` : shape `[ngrids, nao, ncomp_ao_dm0]`.  Leading AO channels contracted
///   with the per-spin density matrices.
///
/// # Returns
///
/// - `rho` : shape `[ngrids, nvar, 2]` (g, x, sigma).  On-grid density components per spin.
/// - `exc` : shape `[ngrids]`.  Per-particle XC energy density (spin-summed).
/// - `vxc` : shape `[ngrids, nvar, 2]`.  1st functional derivative.
/// - `fxc` : shape `[ngrids, nvar, 2, nvar, 2]`.  2nd functional derivative.
pub fn get_rho_exc_vxc_fxc_uks(
    xc_func_list: &[(f64, LibXCFunctional)],
    ao: TsrView,
    ao_dm0α: TsrView,
    ao_dm0β: TsrView,
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

    let mut rho = rt::zeros(([ngrids, nvar, 2], &device));
    for (σ, ao_dm0σ) in [(α, &ao_dm0α), (β, &ao_dm0β)] {
        index_mut!(rho, 0, σ) += rt::vecdot(index!(ao, 0), index!(ao_dm0σ, O), 1);
        if matches!(xc_type, SIGMA | TAU) {
            index_mut!(rho, X, σ) += 2 * rt::vecdot(index!(ao, X), index!(ao_dm0σ, O), 1);
            index_mut!(rho, Y, σ) += 2 * rt::vecdot(index!(ao, Y), index!(ao_dm0σ, O), 1);
            index_mut!(rho, Z, σ) += 2 * rt::vecdot(index!(ao, Z), index!(ao_dm0σ, O), 1);
        }
        if matches!(xc_type, TAU) {
            index_mut!(rho, 4, σ) += 0.5
                * (rt::vecdot(index!(ao, X), index!(ao_dm0σ, X), 1)
                    + rt::vecdot(index!(ao, Y), index!(ao_dm0σ, Y), 1)
                    + rt::vecdot(index!(ao, Z), index!(ao_dm0σ, Z), 1))
        }
    }

    let mut exc = rt::zeros(([ngrids], &device));
    let mut vxc = rt::zeros(([ngrids, nvar, 2], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, 2, nvar, 2], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        let rho_i = rho.i((.., ..nvar_i, ..));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [e_i, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        exc += *scale * e_i.into_shape([ngrids]);
        *&mut vxc.i_mut((.., ..nvar_i, ..)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, .., ..nvar_i, ..)) += *scale * fxc_i;
    }

    (rho, exc, vxc, fxc)
}

/// Single spin-pair fxc contraction `einsum("g, Atxg, xyg, Bsyg -> ABts",
/// weights, drho1, fxc_block, drho2)`, the inner loop of [`get_de_fxc_uks`].
///
/// # Parameters
///
/// - `wf_block` : shape `[ngrids, nvar, nvar]` (g, x, y); a single spin block of `wf`.
/// - `drho1`, `drho2` : shape `[ngrids, nvar, 3, natm]` (output of [`get_drho`]).
///
/// # Returns
///
/// - `de_fxc` : shape `[3, 3, natm, natm]` (t, s, A, B).
fn get_de_fxc_uks_inner(wf_block: TsrView, drho1: TsrView, drho2: TsrView) -> Tsr {
    let [ngrids, nvar, _, natm] = drho1.shape().iter().cloned().collect_array().unwrap();

    let tmp1 = rt::vecdot(wf_block.i((.., .., .., None, None)), drho1.i((.., .., None, .., ..)), 1);
    let tmp1 = tmp1.reshape([ngrids * nvar, natm * 3]);
    let drho2 = drho2.reshape([ngrids * nvar, natm * 3]);
    let tmp2 = tmp1.t() % drho2;

    tmp2.reshape([3, natm, 3, natm]).transpose([0, 2, 1, 3]).into_contig(ColMajor)
}

/// fxc contribution to the UKS XC skeleton Hessian (pyhessref `_de_fxc_uks`):
/// the four spin-pair sum of [`super::hess_rks_becke::get_de_fxc`] blocks,
/// `einsum("gxy, gxtA, gysB -> tsAB", wf, drho, drho)` per (s1, s2) pair.
///
/// # Parameters
///
/// - `wf` : shape `[ngrids, nvar, 2, nvar, 2]` (g, x, s1, y, s2).  Grid-weighted spin-polarized fxc
///   kernel.
/// - `drhoα`, `drhoβ` : shape `[ngrids, nvar, 3, natm]`, per-spin outputs of
///   [`super::hess_rks_becke::get_drho`].
///
/// # Returns
///
/// - `de_fxc` : shape `[3, 3, natm, natm]` (t, s, A, B), spin-pair summed.
pub fn get_de_fxc_uks(wf: TsrView, drhoα: TsrView, drhoβ: TsrView) -> Tsr {
    let de_αα = get_de_fxc_uks_inner(wf.i((.., .., α, .., α)), drhoα.view(), drhoα.view());
    let de_αβ = get_de_fxc_uks_inner(wf.i((.., .., α, .., β)), drhoα.view(), drhoβ.view());
    let de_βα = get_de_fxc_uks_inner(wf.i((.., .., β, .., α)), drhoβ.view(), drhoα.view());
    let de_ββ = get_de_fxc_uks_inner(wf.i((.., .., β, .., β)), drhoβ.view(), drhoβ.view());

    &de_αα + &de_αβ + &de_βα + &de_ββ
}

/// fxc contribution to the per-atom skeleton derivative of the Vxc Fock
/// matrices for UKS (pyhessref `_vmat_fxc_uks`) — the spin-coupled piece.
/// Unlike the spin-diagonal [`get_vmat_vxc`] (reused from RKS per spin), the
/// fxc contraction here couples the two spin channels:
/// `wvα_f = wf_αα @ drho_α + wf_βα @ drho_β` and symmetrically for beta.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads channels 0..3.
/// - `drhoα`, `drhoβ` : shape `[ngrids, nvar, 3, natm]`, per-spin outputs of
///   [`super::hess_rks_becke::get_drho`].
/// - `wf` : shape `[ngrids, nvar, 2, nvar, 2]`.  Grid-weighted spin-polarized fxc kernel.
/// - `aoslices` : shape `[natm, 4]`; per-atom `[shl0, shl1, p0, p1]` AO slices.
///
/// # Returns
///
/// - `vmatα_fxc`, `vmatβ_fxc` : shape `[nao, nao, 3, natm]` each, assembled across the AO axes (bra
///   + ket).
#[allow(clippy::too_many_arguments)]
pub fn get_vmat_fxc_uks(
    xc_type: XCDenType,
    ao: TsrView,
    drhoα: TsrView,
    drhoβ: TsrView,
    wf: TsrView,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let natm = aoslices.len();
    let nao = ao.shape()[1];
    let device = ao.device();

    let mut vmatα_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], device));
    let mut vmatβ_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], device));

    for A in 0..natm {
        for t in 0..3 {
            if matches!(xc_type, RHO) {
                let wf_αα_00: Tsr = 0.5 * wf.i((.., O, α, O, α));
                let wf_βα_00: Tsr = 0.5 * wf.i((.., O, β, O, α));
                let wvα_f: Tsr = wf_αα_00 * drhoα.i((.., O, t, A)) + wf_βα_00 * drhoβ.i((.., O, t, A));

                let wf_αβ_00: Tsr = 0.5 * wf.i((.., O, α, O, β));
                let wf_ββ_00: Tsr = 0.5 * wf.i((.., O, β, O, β));
                let wvβ_f: Tsr = wf_αβ_00 * drhoα.i((.., O, t, A)) + wf_ββ_00 * drhoβ.i((.., O, t, A));

                let aowα = wvα_f * index!(ao, O);
                index_mut!(vmatα_fxc, t, A).matmul_from(aowα.t(), index!(ao, O), 1.0, 1.0);
                let aowβ = wvβ_f * index!(ao, O);
                index_mut!(vmatβ_fxc, t, A).matmul_from(aowβ.t(), index!(ao, O), 1.0, 1.0);
                continue;
            }

            let wf_αα = wf.i((.., .., α, .., α));
            let wf_αβ = wf.i((.., .., α, .., β));
            let wf_βα = wf.i((.., .., β, .., α));
            let wf_ββ = wf.i((.., .., β, .., β));

            let drhoα_tA = drhoα.i((.., .., t, A));
            let drhoβ_tA = drhoβ.i((.., .., t, A));

            let wf_rho_αα = rt::vecdot(&wf_αα, &drhoα_tA, 1);
            let wf_rho_βα = rt::vecdot(&wf_βα, &drhoβ_tA, 1);
            let wf_rho_αβ = rt::vecdot(&wf_αβ, &drhoα_tA, 1);
            let wf_rho_ββ = rt::vecdot(&wf_ββ, &drhoβ_tA, 1);
            let mut wf_rho_α = wf_rho_αα + wf_rho_βα;
            let mut wf_rho_β = wf_rho_αβ + wf_rho_ββ;

            if matches!(xc_type, SIGMA | TAU) {
                *&mut wf_rho_α.i_mut((.., 0)) *= 0.5;
                *&mut wf_rho_β.i_mut((.., 0)) *= 0.5;
                if matches!(xc_type, TAU) {
                    *&mut wf_rho_α.i_mut((.., 4)) *= 0.25;
                    *&mut wf_rho_β.i_mut((.., 4)) *= 0.25;
                }
                for c in 0..4 {
                    let aowα = wf_rho_α.i((.., c)) * index!(ao, c);
                    index_mut!(vmatα_fxc, t, A).matmul_from(aowα.t(), index!(ao, O), 1.0, 1.0);
                    let aowβ = wf_rho_β.i((.., c)) * index!(ao, c);
                    index_mut!(vmatβ_fxc, t, A).matmul_from(aowβ.t(), index!(ao, O), 1.0, 1.0);
                }
            }

            if matches!(xc_type, TAU) {
                for r in [X, Y, Z] {
                    let aowα = wf_rho_α.i((.., 4)) * index!(ao, r);
                    index_mut!(vmatα_fxc, t, A).matmul_from(aowα.t(), index!(ao, r), 1.0, 1.0);
                    let aowβ = wf_rho_β.i((.., 4)) * index!(ao, r);
                    index_mut!(vmatβ_fxc, t, A).matmul_from(aowβ.t(), index!(ao, r), 1.0, 1.0);
                }
            }
        }
    }

    let vmatα_fxc = &vmatα_fxc + vmatα_fxc.swapaxes(0, 1);
    let vmatβ_fxc = &vmatβ_fxc + vmatβ_fxc.swapaxes(0, 1);

    (vmatα_fxc, vmatβ_fxc)
}

/* #endregion */

/* #region becke grid-shift parts: skeleton hessian */

/// `de_becke_atom_1` (notebook t3), UKS spin extension:
/// `-einsum("g, txg, xyg, Bsyg -> Bts", w, prho_l, fxc[s1, :, s2, :], drho_r)`
/// summed over the four spin pairs (s1, s2) — the same coupling structure as
/// [`get_de_fxc_uks`].  Each pair reuses the single-spin
/// [`get_de_becke_atom_1`] on the `fxc[s1, :, s2, :]` spin block; the free
/// direction axes `s`/`t` interchange is inherited per pair and is equivalent
/// under the driver's symmetrisation.
///
/// # Parameters
///
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `prhoα`, `prhoβ` : shape `[ngrids, nvar, 3]` (g, x, t).  Total skeleton derivative `drho`
///   summed over atoms, per spin.
/// - `fxc` : shape `[ngrids, nvar, 2, nvar, 2]`.
/// - `drhoα`, `drhoβ` : shape `[ngrids, nvar, 3, natm]`.
///
/// # Returns
///
/// - `de_becke_atom_1` : shape `[3, 3, natm]` (t, s, A), for the last (B) axis scatter of the `[3,
///   3, natm, natm]` accumulator (see [`get_de_becke_atom_1`]).
pub fn get_de_becke_atom_1_uks(
    w: TsrView,
    prhoα: TsrView,
    prhoβ: TsrView,
    fxc: TsrView,
    drhoα: TsrView,
    drhoβ: TsrView,
) -> Tsr {
    let [_, _, _, natm] = drhoα.shape().iter().cloned().collect_array().unwrap();
    let device = drhoα.device().clone();

    let mut de_becke_atom_1: Tsr = rt::zeros(([3, 3, natm], &device));
    for (prho_l, s1) in [(&prhoα, α), (&prhoβ, β)] {
        for (drho_r, s2) in [(&drhoα, α), (&drhoβ, β)] {
            let term = get_de_becke_atom_1(w.view(), prho_l.view(), fxc.i((.., .., s1, .., s2)), drho_r.view());
            de_becke_atom_1 = &de_becke_atom_1 + &term;
        }
    }
    de_becke_atom_1
}

/// `de_becke_atom_2` (notebook t5), UKS spin extension:
/// `-einsum("Bsg, xg, txg -> Bts", dw, vxc[σ], prho_σ)` summed over the two
/// spins.  Each spin reuses the single-spin [`get_de_becke_atom_2`].
///
/// # Parameters
///
/// - `dw` : shape `[ngrids, 3, natm]` (g, t, A).  Grid-first Becke `dw` (see
///   [`make_hessian_setup_chunk_becke_uks`]).
/// - `vxc` : shape `[ngrids, nvar, 2]`.
/// - `prhoα`, `prhoβ` : shape `[ngrids, nvar, 3]` (g, x, t).
///
/// # Returns
///
/// - `de_becke_atom_2` : shape `[3, 3, natm]` (t, s, A), for the last (B) axis scatter.
pub fn get_de_becke_atom_2_uks(dw: TsrView, vxc: TsrView, prhoα: TsrView, prhoβ: TsrView) -> Tsr {
    let term_α = get_de_becke_atom_2(dw.view(), vxc.i((.., .., α)), prhoα);
    let term_β = get_de_becke_atom_2(dw.view(), vxc.i((.., .., β)), prhoβ);
    &term_α + &term_β
}

/// `de_becke_atom_3` (notebook t6), UKS spin extension:
/// `einsum("g, xyg, syg, txg -> ts", w, fxc[s1, :, s2, :], prho_r, prho_l)`
/// summed over the four spin pairs — same structure as
/// [`super::hess_rks_becke::get_de_becke_atom_3`], but with the two prho slots
/// filled by different spins, so the body is spelled out instead of reusing
/// the single-spin function.
///
/// # Parameters
///
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `prhoα`, `prhoβ` : shape `[ngrids, nvar, 3]` (g, x, t).
/// - `fxc` : shape `[ngrids, nvar, 2, nvar, 2]`.
///
/// # Returns
///
/// - `de_becke_atom_3` : shape `[3, 3]` (t, s), for the `[atm_idx, atm_idx]` diagonal block
///   scatter.
pub fn get_de_becke_atom_3_uks(w: TsrView, prhoα: TsrView, prhoβ: TsrView, fxc: TsrView) -> Tsr {
    let device = prhoα.device().clone();

    let mut de_becke_atom_3: Tsr = rt::zeros(([3, 3], &device));
    for (prho_l, s1) in [(&prhoα, α), (&prhoβ, β)] {
        for (prho_r, s2) in [(&prhoα, α), (&prhoβ, β)] {
            // fp [g, x, t] = sum_y fxc[s1, x, s2, y] prho_r[g, y, t]
            let fp = rt::vecdot(fxc.i((.., .., s1, .., s2, None)), prho_r.i((.., None, .., ..)), 2);
            // wprho_l [g, x, s]
            let wprho_l = prho_l * w.i((.., None, None));
            // t6 [t, s] = sum_{g, x} fp[g, x, t] wprho_l[g, x, s]
            let term = rt::vecdot(fp.i((.., .., None, ..)), wprho_l.i((.., .., .., None)), ([0, 1], [0, 1]));
            de_becke_atom_3 = &de_becke_atom_3 + &term;
        }
    }
    de_becke_atom_3
}

/// `de_becke_vxc_diag` (notebook t8) / `de_becke_vxc_off` (t9), UKS spin
/// extension: same structure as
/// [`super::hess_rks_becke::get_de_becke_vxc_parts`], with the per-spin kernels
/// (each built from its own `wv` weighting and `ao_dm0` contraction in
/// [`make_hessian_setup_chunk_becke_uks`]) summed before the
/// [`contract_pvxc`] scatter — the contraction is linear, so spin-summing
/// first is equivalent to contracting each spin separately.
///
/// # Parameters
///
/// - `dao_vxc_diag_α`, `dao_vxc_diag_β` : shape `[nao, 6]`, per-spin outputs of
///   [`super::hess_rks_becke::make_dao_vxc_diag`].
/// - `dao_vxc_off_α`, `dao_vxc_off_β` : shape `[nao, nao, 3, 3]` (u, v, t, s), per-spin outputs of
///   [`super::hess_rks_becke::make_dao_vxc_off`].
/// - `dm0α`, `dm0β` : shape `[nao, nao]`.  Per-spin density matrices in AO basis.
/// - `atm_idx` : atom that generated the chunk's grids.
/// - `aoslices` : shape `[natm, 4]`.
///
/// # Returns
///
/// - `de_becke_vxc_diag` : shape `[3, 3, natm]`, from the spin-summed `0.5 * dao_vxc_diag` expanded
///   to dense (3, 3) pairs.
/// - `de_becke_vxc_off` : shape `[3, 3, natm]`, from the spin-summed `0.5 * dao_vxc_off` contracted
///   with the matching per-spin `dm0` on the leading AO axis.
///
/// Both for the last (B) axis scatter (see [`contract_pvxc`]).
#[allow(clippy::too_many_arguments)]
pub fn get_de_becke_vxc_parts_uks(
    dao_vxc_diag_α: TsrView,
    dao_vxc_diag_β: TsrView,
    dao_vxc_off_α: TsrView,
    dao_vxc_off_β: TsrView,
    dm0α: TsrView,
    dm0β: TsrView,
    atm_idx: usize,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let nao = dao_vxc_diag_α.shape()[0];

    // pvxc_diag [nao, 3, 3] = 0.5 * (dao_vxc_diag_α + dao_vxc_diag_β)[IDX_PAIR_TS]; the symmetric
    // pairs make it invariant under (t, s), so it doubles as the interchanged
    // kernel
    const IDX_PAIR_TS: [usize; 9] = [0, 1, 2, 1, 3, 4, 2, 4, 5];
    let pvxc_diag: Tsr = 0.5 * (&dao_vxc_diag_α + &dao_vxc_diag_β).index_select(1, IDX_PAIR_TS).into_shape([nao, 3, 3]);

    // pvxc_off[u, t, s] = 0.5 * sum_v (dao_vxc_off_α[u, v, t, s] dm0α[u, v] + dao_vxc_off_β[u, v,
    // t, s] dm0β[u, v])  (einsum "tsuv, uv -> tsu" on the python [t, s, u, v] kernels per spin,
    // with the free direction axes interchanged; by the kernels' [u, v, t, s] -> [v, u, s, t]
    // symmetry, built into `make_dao_vxc_off`, contracting the SECOND AO axis yields the
    // interchanged orientation directly, with no transpose)
    let pvxc_off: Tsr = 0.5 * (rt::vecdot(dao_vxc_off_α, dm0α, 1) + rt::vecdot(dao_vxc_off_β, dm0β, 1));
    (contract_pvxc(pvxc_diag.view(), atm_idx, aoslices), contract_pvxc(pvxc_off.view(), atm_idx, aoslices))
}

/* #endregion */

/* #region becke grid-shift parts: f1ao (CP-KS RHS) */

/// f1ao-level Becke grid-shift parts of the per-spin skeleton Vxc Fock
/// derivatives (pyhessref `_vmat_becke_parts_uks`): the per-spin increments
/// `T1_σ + T2_σ` that restore translational invariance of each spin's
/// `vmat_deriv1_σ` (the DFT part of the CP-KS right-hand side f1ao), i.e.
/// `sum_A vmat_deriv1_grid_σ[A] ~ 0`.
///
/// - T1 (weight part): per-spin [`vxc_fock`] built with the Becke `dw[g, t, A]` slices as the
///   weight field and the spin's own `vxc` field; every grid of the chunk contributes to every
///   atom's row.
/// - T2_ipip: the chunk's per-spin `vmat_ip` symmetrised in AO — the chunk holds one atom's grids,
///   so `vmat_ip` already is the per-grid-atom kernel.
/// - T2_fxc: the fxc kernel spin-coupled and folded with the total spatial density derivative of
///   BOTH spins (`fxc[σ, :, σ', :]` against `prho_σ'`; leading minus from the `prho = -d rho / dr`
///   convention of [`get_drho`]), contracted as a [`vxc_fock`] on the chunk weights.  This mirrors
///   the [`get_vmat_fxc_uks`] spin coupling.
///
/// # Parameters
///
/// - `xc_type` : density family.
/// - `ao` : shape `[ngrids, nao, ncomp]`; reads channels 0..3.
/// - `vxc` : shape `[ngrids, nvar, 2]`.
/// - `fxc` : shape `[ngrids, nvar, 2, nvar, 2]`.
/// - `prhoα`, `prhoβ` : shape `[ngrids, nvar, 3]` (g, x, t), per spin.
/// - `w` : shape `[ngrids]`.  Grid weights of the chunk.
/// - `dw` : shape `[ngrids, 3, natm]` (g, t, A).  Grid-first Becke `dw` (see
///   [`get_de_becke_atom_2_uks`]).
/// - `vmat_ip_α`, `vmat_ip_β` : shape `[nao, nao, 3]`, per-spin outputs of
///   [`super::hess_rks_becke::get_vmat_ip`].
///
/// # Returns
///
/// - `vmat_becke_T1_α`, `vmat_becke_T1_β` : shape `[nao, nao, 3, natm]`, filled on all rows.
/// - `vmat_becke_T2_ipip_α/β`, `vmat_becke_T2_fxc_α/β` : shape `[nao, nao, 3]` — the chunk atom's
///   row, scattered into the `[nao, nao, 3, natm]` accumulators by the driver.
#[allow(clippy::too_many_arguments)]
pub fn get_vmat_becke_parts_uks(
    xc_type: XCDenType,
    ao: TsrView,
    vxc: TsrView,
    fxc: TsrView,
    prhoα: TsrView,
    prhoβ: TsrView,
    w: TsrView,
    dw: TsrView,
    vmat_ip_α: TsrView,
    vmat_ip_β: TsrView,
) -> (Tsr, Tsr, Tsr, Tsr, Tsr, Tsr) {
    let nao = ao.shape()[1];
    let natm = dw.shape()[2];
    let device = ao.device().clone();
    let ngrids = fxc.shape()[0];
    let nvar = fxc.shape()[1];

    // T1: per-spin Vxc-style Fock with the becke dw[A, t] rows as weights (all rows)
    let mut vmat_becke_t1_α = rt::zeros(([nao, nao, 3, natm], &device));
    let mut vmat_becke_t1_β = rt::zeros(([nao, nao, 3, natm], &device));
    for A in 0..natm {
        for t in 0..3 {
            let fock_α = vxc_fock(xc_type, ao.view(), vxc.i((.., .., α)), index!(dw, t, A));
            *&mut vmat_becke_t1_α.i_mut((.., .., t, A)) += &fock_α;
            let fock_β = vxc_fock(xc_type, ao.view(), vxc.i((.., .., β)), index!(dw, t, A));
            *&mut vmat_becke_t1_β.i_mut((.., .., t, A)) += &fock_β;
        }
    }

    // T2_ipip: chunk's per-spin vmat_ip symmetrised in AO
    let vmat_becke_t2_ipip_α = &vmat_ip_α + vmat_ip_α.swapaxes(0, 1);
    let vmat_becke_t2_ipip_β = &vmat_ip_β + vmat_ip_β.swapaxes(0, 1);

    // T2_fxc: fxc spin-coupled and folded with prho[t] of both spins, contracted on the chunk
    // weights; fxc_prho_σ [g, x] = sum_{σ', y} fxc[σ, x, σ', y] prho_σ'[g, y, t]
    let mut vmat_becke_t2_fxc_α = rt::zeros(([nao, nao, 3], &device));
    let mut vmat_becke_t2_fxc_β = rt::zeros(([nao, nao, 3], &device));
    for t in 0..3 {
        let prhoα_t = prhoα.i((.., .., t));
        let prhoβ_t = prhoβ.i((.., .., t));

        let fxc_prho_α: Tsr = (rt::vecdot(fxc.i((.., .., α, .., α, None)), prhoα_t.i((.., None, .., None)), 2)
            + rt::vecdot(fxc.i((.., .., α, .., β, None)), prhoβ_t.i((.., None, .., None)), 2))
        .into_shape([ngrids, nvar]);
        let neg_fxc_prho_α: Tsr = -1.0 * fxc_prho_α;
        let fock_α = vxc_fock(xc_type, ao.view(), neg_fxc_prho_α.view(), w.view());
        *&mut vmat_becke_t2_fxc_α.i_mut((.., .., t)) += &fock_α;

        let fxc_prho_β: Tsr = (rt::vecdot(fxc.i((.., .., β, .., α, None)), prhoα_t.i((.., None, .., None)), 2)
            + rt::vecdot(fxc.i((.., .., β, .., β, None)), prhoβ_t.i((.., None, .., None)), 2))
        .into_shape([ngrids, nvar]);
        let neg_fxc_prho_β: Tsr = -1.0 * fxc_prho_β;
        let fock_β = vxc_fock(xc_type, ao.view(), neg_fxc_prho_β.view(), w.view());
        *&mut vmat_becke_t2_fxc_β.i_mut((.., .., t)) += &fock_β;
    }

    (
        vmat_becke_t1_α,
        vmat_becke_t1_β,
        vmat_becke_t2_ipip_α,
        vmat_becke_t2_ipip_β,
        vmat_becke_t2_fxc_α,
        vmat_becke_t2_fxc_β,
    )
}

/* #endregion */

/* #region per-chunk evaluation */

/// Per-chunk evaluation of all UKS skeleton ingredients with the grid-shift
/// (pyhessref `make_hessian_setup_batch_uks`).  Unrestricted sibling of
/// [`super::hess_rks_becke::make_hessian_setup_chunk_becke`]: the chunk must
/// hold grids of the single atom `atm_idx` (ByAtom attribution) and computes
/// its own AO integrals through `ni.get_cached_ao`.
///
/// # Parameters
///
/// - `mol` : molecule (AO slices, dimensions).
/// - `xc_func_list` : list of `(scale, functional)` pairs.
/// - `ni` : numerical-integration driver restricted to the chunk's grids.
/// - `dm0α`, `dm0β` : shape `[nao, nao]`.  Per-spin reference density matrices in AO basis.
/// - `atm_idx` : atom that generated the chunk's grids.
/// - `quadrature_weights` : shape `[nchunk_grids]`.  Pre-partition quadrature weights of the chunk.
/// - `tables` : precomputed molecular tables of the Becke partition (shared across chunks).
/// - `hardness` : Becke switch-function hardness.
///
/// # Returns
///
/// Map from key to the chunk's contribution.  Full-grid keys accumulate
/// across chunks by a plain sum; grid-atom keys carry only the chunk atom's
/// contribution and are scattered by [`make_hessian_setup_becke_uks`]:
///
/// - Sum: `fxc [ngrids, nvar, 2, nvar, 2]` (disjoint grid ranges); `de_fxc`, `de_vxc_diag_a/b`,
///   `de_vxc_off_a/b`, `de_becke_full_1/2` `[3, 3, natm, natm]`; `vmat_ip_a/b [nao, nao, 3]`;
///   `vmat_fxc_a/b`, `vmat_vxc_a/b`, `vmat_deriv1_a/b`, `vmat_becke_T1_a/b` `[nao, nao, 3, natm]`.
/// - Scatter into column `B = atm_idx` (direction axes interchanged): `de_becke_atom_1/2`,
///   `de_becke_vxc_diag/off` `[3, 3, natm]`; `vmat_becke_T2_ipip_a/b`, `vmat_becke_T2_fxc_a/b`
///   `[nao, nao, 3]`.
/// - Scatter into the `[atm_idx, atm_idx]` diagonal block: `de_becke_atom_3` `[3, 3]`.
#[allow(clippy::too_many_arguments)]
pub fn make_hessian_setup_chunk_becke_uks(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0α: TsrView,
    dm0β: TsrView,
    atm_idx: usize,
    quadrature_weights: &[f64],
    tables: &BeckeMolTables,
    hardness: usize,
) -> HashMap<&'static str, Tsr> {
    let natm = mol.natm();
    let device = dm0α.device().clone();
    let aoslices = mol.aoslice_by_atom();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    // owned copies of the chunk's grid data; `ni` stays borrowed by the AO cache
    let grid_coords = ni.coords.clone();
    let weights_data = ni.weights.clone();
    let ngrids = weights_data.len();

    // --- ao, rho, exc, vxc, fxc --- //

    let ao = ni.get_cached_ao(get_hess_ao_deriv(xc_type));
    let ncomp_ao_dm0 = get_hess_ncomp_ao_dm0(xc_type);
    let ao_dm0α = index!(ao, ..ncomp_ao_dm0) % &dm0α;
    let ao_dm0β = index!(ao, ..ncomp_ao_dm0) % &dm0β;
    let (rho, exc, vxc, fxc) = get_rho_exc_vxc_fxc_uks(xc_func_list, ao.view(), ao_dm0α.view(), ao_dm0β.view());

    let weights = rt::asarray((weights_data, &device));
    let wvα = &weights * vxc.i((.., .., α));
    let wvβ = &weights * vxc.i((.., .., β));
    let wf = &weights * &fxc;

    // --- drho, prho (per spin; the skeleton derivative is spin-diagonal) --- //

    let drhoα = get_drho(xc_type, ao.view(), ao_dm0α.view(), &aoslices);
    let drhoβ = get_drho(xc_type, ao.view(), ao_dm0β.view(), &aoslices);
    // prho_σ [ngrids, nvar, 3] = drho_σ summed over atoms
    let prhoα = drhoα.sum_axes(3);
    let prhoβ = drhoβ.sum_axes(3);

    // --- without-becke parts --- //

    let de_fxc = get_de_fxc_uks(wf.view(), drhoα.view(), drhoβ.view());

    let dao_vxc_diag_α = make_dao_vxc_diag(xc_type, ao.view(), ao_dm0α.view(), wvα.view());
    let dao_vxc_diag_β = make_dao_vxc_diag(xc_type, ao.view(), ao_dm0β.view(), wvβ.view());
    let de_vxc_diag_α = get_de_vxc_diag(dao_vxc_diag_α.view(), &aoslices);
    let de_vxc_diag_β = get_de_vxc_diag(dao_vxc_diag_β.view(), &aoslices);

    let dao_vxc_off_α = make_dao_vxc_off(xc_type, ao.view(), wvα.view());
    let dao_vxc_off_β = make_dao_vxc_off(xc_type, ao.view(), wvβ.view());
    let de_vxc_off_α = get_de_vxc_off(dao_vxc_off_α.view(), dm0α.view(), &aoslices);
    let de_vxc_off_β = get_de_vxc_off(dao_vxc_off_β.view(), dm0β.view(), &aoslices);

    let vmat_ip_α = get_vmat_ip(xc_type, ao.view(), wvα.view());
    let vmat_ip_β = get_vmat_ip(xc_type, ao.view(), wvβ.view());

    // per-atom skeleton Vxc Fock derivative per spin: spin-coupled fxc part plus per-spin
    // basis-derivative (ipip) part; both are already assembled across the AO axes
    let (vmat_fxc_α, vmat_fxc_β) =
        get_vmat_fxc_uks(xc_type, ao.view(), drhoα.view(), drhoβ.view(), wf.view(), &aoslices);
    let vmat_vxc_α = get_vmat_vxc(vmat_ip_α.view(), &aoslices);
    let vmat_vxc_β = get_vmat_vxc(vmat_ip_β.view(), &aoslices);
    let vmat_deriv1_α = &vmat_fxc_α + &vmat_vxc_α;
    let vmat_deriv1_β = &vmat_fxc_β + &vmat_vxc_β;

    // --- becke partition: dw in full, ddw only through the cddw contraction --- //

    // cddw (nset = 1): the only ddw consumer is de_becke_full_2 = ddw . (exc * (rhoa[0] + rhob[0]))
    let cddw = (&exc * (rho.i((.., 0, α)) + rho.i((.., 0, β)))).into_vec();

    let boundaries = by_atom_chunk(natm, atm_idx, ngrids);
    let deriv_arg = BeckeDerivArg {
        output_w: false,
        output_dw: true,
        output_ddw: false,
        contract_w: None,
        contract_dw: None,
        contract_ddw: Some(&cddw),
    };
    let becke_result = becke_partition_with_tables(
        tables,
        &grid_coords,
        AtmIndices::ByAtom(&boundaries),
        quadrature_weights,
        hardness,
        1024,
        2,
        Some(deriv_arg),
    );
    // dw flat is C-order [A, t, g] == Fortran-order [g, t, A];
    // ddc flat is C-order [A, t, B, s, iset] == Fortran-order [iset, s, B, t, A]
    let dw = rt::asarray((becke_result.dw.unwrap(), [ngrids, 3, natm].f(), &device));

    // --- becke grid-shift parts --- //

    // de_becke_full_1 (notebook t1): einsum("Atg, xg, Bsxg -> ABts", dw, vxc[σ], drho_σ), summed
    // over σ; t1 [t, s, A, B] = sum_g dw[g, t, A] vxc_drho[g, s, B], where
    // vxc_drho [g, s, B] = sum_x vxc[σ][g, x] drho_σ[g, x, s, B]
    let de_becke_full_1 = {
        let vxc_drho_α = rt::vecdot(drhoα.view(), vxc.i((.., .., α)), 1);
        let vxc_drho_β = rt::vecdot(drhoβ.view(), vxc.i((.., .., β)), 1);
        let vxc_drho = &vxc_drho_α + &vxc_drho_β;
        rt::vecdot(dw.i((.., .., None, .., None)), vxc_drho.i((.., None, .., None, ..)), 0)
    };

    // de_becke_full_2 (notebook t2): einsum("AtBsg, g, g -> ABts", ddw, exc, rhoa[0] + rhob[0])
    // via the cddw contraction above (nset = 1), naturally symmetric;
    // ddc flat is C-order [A, t, B, s, iset] == Fortran-order [iset, s, B, t, A]
    let de_becke_full_2 = rt::asarray((becke_result.ddc.unwrap(), [3, natm, 3, natm].f(), &device))
        .transpose([2, 0, 3, 1])
        .into_contig(ColMajor);

    // grid-atom parts: compact tensors for the chunk atom's row (resp. diagonal
    // block); the scatter into `[3, 3, natm, natm]` is done by
    // `make_hessian_setup_becke_uks`
    let de_becke_atom_1 =
        get_de_becke_atom_1_uks(weights.view(), prhoα.view(), prhoβ.view(), fxc.view(), drhoα.view(), drhoβ.view());
    let de_becke_atom_2 = get_de_becke_atom_2_uks(dw.view(), vxc.view(), prhoα.view(), prhoβ.view());
    let de_becke_atom_3 = get_de_becke_atom_3_uks(weights.view(), prhoα.view(), prhoβ.view(), fxc.view());

    let (de_becke_vxc_diag, de_becke_vxc_off) = get_de_becke_vxc_parts_uks(
        dao_vxc_diag_α.view(),
        dao_vxc_diag_β.view(),
        dao_vxc_off_α.view(),
        dao_vxc_off_β.view(),
        dm0α.view(),
        dm0β.view(),
        atm_idx,
        &aoslices,
    );

    let (
        vmat_becke_t1_α,
        vmat_becke_t1_β,
        vmat_becke_t2_ipip_α,
        vmat_becke_t2_ipip_β,
        vmat_becke_t2_fxc_α,
        vmat_becke_t2_fxc_β,
    ) = get_vmat_becke_parts_uks(
        xc_type,
        ao.view(),
        vxc.view(),
        fxc.view(),
        prhoα.view(),
        prhoβ.view(),
        weights.view(),
        dw.view(),
        vmat_ip_α.view(),
        vmat_ip_β.view(),
    );

    HashMap::from([
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag_a", de_vxc_diag_α),
        ("de_vxc_diag_b", de_vxc_diag_β),
        ("de_vxc_off_a", de_vxc_off_α),
        ("de_vxc_off_b", de_vxc_off_β),
        ("vmat_ip_a", vmat_ip_α),
        ("vmat_ip_b", vmat_ip_β),
        ("vmat_fxc_a", vmat_fxc_α),
        ("vmat_fxc_b", vmat_fxc_β),
        ("vmat_vxc_a", vmat_vxc_α),
        ("vmat_vxc_b", vmat_vxc_β),
        ("vmat_deriv1_a", vmat_deriv1_α),
        ("vmat_deriv1_b", vmat_deriv1_β),
        ("de_becke_full_1", de_becke_full_1),
        ("de_becke_full_2", de_becke_full_2),
        ("de_becke_atom_1", de_becke_atom_1),
        ("de_becke_atom_2", de_becke_atom_2),
        ("de_becke_atom_3", de_becke_atom_3),
        ("de_becke_vxc_diag", de_becke_vxc_diag),
        ("de_becke_vxc_off", de_becke_vxc_off),
        ("vmat_becke_T1_a", vmat_becke_t1_α),
        ("vmat_becke_T1_b", vmat_becke_t1_β),
        ("vmat_becke_T2_ipip_a", vmat_becke_t2_ipip_α),
        ("vmat_becke_T2_ipip_b", vmat_becke_t2_ipip_β),
        ("vmat_becke_T2_fxc_a", vmat_becke_t2_fxc_α),
        ("vmat_becke_T2_fxc_b", vmat_becke_t2_fxc_β),
    ])
}

/* #endregion */

/* #region parallel driver */

/// `x + x.transpose(1, 0, 3, 2)` on a `[3, 3, natm, natm]` (tsAB) tensor: the
/// `(A, t) <-> (B, s)` symmetrisation of the pyhessref reference.
fn symmetrize_ts_ab(x: Tsr) -> Tsr {
    &x + x.transpose([1, 0, 3, 2])
}

/// Parallel driver for all UKS DFT skeleton ingredients with the grid-shift
/// (pyhessref `make_hessian_setup_uks`).  Unrestricted sibling of
/// [`super::hess_rks_becke::make_hessian_setup_becke`], sharing its flat
/// chunk-level parallelization: [`quad_split_by_atom`] at `nchunk` granularity
/// produces `(atm_idx, start, end)` work units that never cross an atom
/// boundary, and each unit evaluates [`make_hessian_setup_chunk_becke_uks`]
/// (its own AO integrals included) inside one flat par_iter.
///
/// # Parameters
///
/// - `mol` : molecule.
/// - `xc_func_list` : list of `(scale, functional)` pairs.
/// - `ni` : numerical-integration driver over the full grid (only `split_batch` and the grid data
///   are read).
/// - `dm0α`, `dm0β` : shape `[nao, nao]`.  Per-spin reference density matrices in AO basis.
/// - `atm_quad_split` : shape `[natm + 1]`; atom `A` owns grids `[atm_quad_split[A],
///   atm_quad_split[A + 1])`.
/// - `quadrature_weights` : shape `[ngrids]`.  Pre-partition quadrature weights.
/// - `adjustment_factor` : `natm` row-major rows of length `natm`; the anti-symmetric Becke
///   radii-adjustment table (see [`UHessKSNIMatmulBecke::adjustment_factor`]).
/// - `hardness` : Becke switch-function hardness.
/// - `atm_list` : must be `None` or the full atom list (the grid-shift currently requires all
///   atoms).
/// - `verbose` : print per-chunk progress.
///
/// # Returns
///
/// - `result` : all keys of [`make_hessian_setup_chunk_becke_uks`] accumulated over chunks
///   (full-grid keys summed, grid-atom keys scattered into the chunk atom's column of the last (B)
///   axis), plus the assemblies: `de_xc_skeleton_no_becke [3, 3, natm, natm]` = `de_vxc_diag_a +
///   de_vxc_off_a + de_vxc_diag_b + de_vxc_off_b + de_fxc`; `de_xc_skeleton [3, 3, natm, natm]`
///   with all `de_becke_*` grid-shift parts added (translationally invariant);
///   `vmat_deriv1_grid_a/b [nao, nao, 3, natm]` = `vmat_deriv1_a/b + vmat_becke_T1_a/b +
///   T2_ipip_a/b + T2_fxc_a/b` (translationally invariant per spin).  The keys
///   `de_becke_full_1/atom_1/atom_2/vxc_diag/vxc_off` are symmetrised under `(A, t) <-> (B, s)`;
///   `de_becke_full_2` is naturally symmetric.
/// - `timing` : wall-time progress entries.
#[allow(clippy::too_many_arguments)]
pub fn make_hessian_setup_becke_uks(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0α: TsrView,
    dm0β: TsrView,
    atm_quad_split: &[usize],
    quadrature_weights: &[f64],
    adjustment_factor: &[Vec<f64>],
    hardness: usize,
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
    let device = dm0α.device().clone();

    // molecular tables of the Becke partition, built once and shared by all
    // chunks
    let tables = BeckeMolTables::new(&mol.atom_coords(), adjustment_factor, 2);

    let chunks = quad_split_by_atom(atm_quad_split, nchunk);
    let nchunks = chunks.len();

    let fxc_full: Tsr = rt::zeros(([ngrids, nvar, 2, nvar, 2], &device));
    let de_fxc: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag_α: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag_β: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off_α: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off_β: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_ip_α: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_ip_β: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_fxc_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_fxc_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_vxc_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_vxc_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_deriv1_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_deriv1_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let de_becke_full_1: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_full_2: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_1: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_2: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_atom_3: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_vxc_diag: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_becke_vxc_off: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_becke_t1_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_t1_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_t2_ipip_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_t2_ipip_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_t2_fxc_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_becke_t2_fxc_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));

    let timing = Arc::new(Mutex::new(IndexMap::from([("total", 0.0)])));
    let time_total = std::time::Instant::now();

    // serializes the `+=` reductions below
    let guard = Mutex::new(());

    // single-level parallelization by chunk; each task owns its `ni_chunk`
    // and its AO evaluation
    let ni = &*ni;
    let progress = AtomicUsize::new(0);

    chunks.into_par_iter().for_each(|(atm_idx, start, end)| {
        let mut ni_chunk = ni.split_batch(start, end);
        let result_chunk = make_hessian_setup_chunk_becke_uks(
            mol,
            xc_func_list,
            &mut ni_chunk,
            dm0α.view(),
            dm0β.view(),
            atm_idx,
            &quadrature_weights[start..end],
            &tables,
            hardness,
        );
        // fxc: disjoint grid ranges
        unsafe {
            let fxc_slc = fxc_full.i(start..end);
            let mut fxc_slc = fxc_slc.force_mut();
            fxc_slc.assign(&result_chunk["fxc"]);
        }
        // sum the full-grid keys, scatter the grid-atom keys into the chunk
        // atom's column of the last (B) axis
        unsafe {
            let _lock = guard.lock().unwrap();
            *&mut de_fxc.force_mut() += &result_chunk["de_fxc"];
            *&mut de_vxc_diag_α.force_mut() += &result_chunk["de_vxc_diag_a"];
            *&mut de_vxc_diag_β.force_mut() += &result_chunk["de_vxc_diag_b"];
            *&mut de_vxc_off_α.force_mut() += &result_chunk["de_vxc_off_a"];
            *&mut de_vxc_off_β.force_mut() += &result_chunk["de_vxc_off_b"];
            *&mut vmat_ip_α.force_mut() += &result_chunk["vmat_ip_a"];
            *&mut vmat_ip_β.force_mut() += &result_chunk["vmat_ip_b"];
            *&mut vmat_fxc_α.force_mut() += &result_chunk["vmat_fxc_a"];
            *&mut vmat_fxc_β.force_mut() += &result_chunk["vmat_fxc_b"];
            *&mut vmat_vxc_α.force_mut() += &result_chunk["vmat_vxc_a"];
            *&mut vmat_vxc_β.force_mut() += &result_chunk["vmat_vxc_b"];
            *&mut vmat_deriv1_α.force_mut() += &result_chunk["vmat_deriv1_a"];
            *&mut vmat_deriv1_β.force_mut() += &result_chunk["vmat_deriv1_b"];
            *&mut de_becke_full_1.force_mut() += &result_chunk["de_becke_full_1"];
            *&mut de_becke_full_2.force_mut() += &result_chunk["de_becke_full_2"];
            *&mut vmat_becke_t1_α.force_mut() += &result_chunk["vmat_becke_T1_a"];
            *&mut vmat_becke_t1_β.force_mut() += &result_chunk["vmat_becke_T1_b"];
            *&mut de_becke_atom_1.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_atom_1"];
            *&mut de_becke_atom_2.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_atom_2"];
            *&mut de_becke_atom_3.i((Ellipsis, atm_idx, atm_idx)).force_mut() += &result_chunk["de_becke_atom_3"];
            *&mut de_becke_vxc_diag.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_vxc_diag"];
            *&mut de_becke_vxc_off.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["de_becke_vxc_off"];
            *&mut vmat_becke_t2_ipip_α.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_T2_ipip_a"];
            *&mut vmat_becke_t2_ipip_β.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_T2_ipip_b"];
            *&mut vmat_becke_t2_fxc_α.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_T2_fxc_a"];
            *&mut vmat_becke_t2_fxc_β.i((Ellipsis, atm_idx)).force_mut() += &result_chunk["vmat_becke_T2_fxc_b"];
        }
        let ichunk = progress.fetch_add(1, Ordering::Relaxed);
        timing.lock().unwrap().insert("total", time_total.elapsed().as_secs_f64());
        if verbose {
            println!(
                "In make_hessian_setup_becke_uks, chunk {}/{} (atom {atm_idx}): grids {start}..{end}",
                ichunk + 1,
                nchunks,
            );
            println!("  Elapsed time from start (Wall time): {:.4} sec", timing.lock().unwrap()["total"]);
        }
    });

    // symmetrize on the atom indices for the becke parts; de_becke_full_2 is
    // naturally symmetric
    let de_becke_full_1 = symmetrize_ts_ab(de_becke_full_1);
    let de_becke_atom_1 = symmetrize_ts_ab(de_becke_atom_1);
    let de_becke_atom_2 = symmetrize_ts_ab(de_becke_atom_2);
    let de_becke_vxc_diag = symmetrize_ts_ab(de_becke_vxc_diag);
    let de_becke_vxc_off = symmetrize_ts_ab(de_becke_vxc_off);

    // final assemblies
    let de_xc_skeleton_no_becke = &de_vxc_diag_α + &de_vxc_off_α + &de_vxc_diag_β + &de_vxc_off_β + &de_fxc;
    let de_xc_skeleton = &de_xc_skeleton_no_becke
        + &de_becke_full_1
        + &de_becke_full_2
        + &de_becke_atom_1
        + &de_becke_atom_2
        + &de_becke_atom_3
        + &de_becke_vxc_diag
        + &de_becke_vxc_off;
    let vmat_deriv1_grid_α = &vmat_deriv1_α + &vmat_becke_t1_α + &vmat_becke_t2_ipip_α + &vmat_becke_t2_fxc_α;
    let vmat_deriv1_grid_β = &vmat_deriv1_β + &vmat_becke_t1_β + &vmat_becke_t2_ipip_β + &vmat_becke_t2_fxc_β;

    let result = HashMap::from([
        ("fxc", fxc_full),
        ("de_fxc", de_fxc),
        ("de_vxc_diag_a", de_vxc_diag_α),
        ("de_vxc_diag_b", de_vxc_diag_β),
        ("de_vxc_off_a", de_vxc_off_α),
        ("de_vxc_off_b", de_vxc_off_β),
        ("vmat_ip_a", vmat_ip_α),
        ("vmat_ip_b", vmat_ip_β),
        ("vmat_fxc_a", vmat_fxc_α),
        ("vmat_fxc_b", vmat_fxc_β),
        ("vmat_vxc_a", vmat_vxc_α),
        ("vmat_vxc_b", vmat_vxc_β),
        ("vmat_deriv1_a", vmat_deriv1_α),
        ("vmat_deriv1_b", vmat_deriv1_β),
        ("de_becke_full_1", de_becke_full_1),
        ("de_becke_full_2", de_becke_full_2),
        ("de_becke_atom_1", de_becke_atom_1),
        ("de_becke_atom_2", de_becke_atom_2),
        ("de_becke_atom_3", de_becke_atom_3),
        ("de_becke_vxc_diag", de_becke_vxc_diag),
        ("de_becke_vxc_off", de_becke_vxc_off),
        ("vmat_becke_T1_a", vmat_becke_t1_α),
        ("vmat_becke_T1_b", vmat_becke_t1_β),
        ("vmat_becke_T2_ipip_a", vmat_becke_t2_ipip_α),
        ("vmat_becke_T2_ipip_b", vmat_becke_t2_ipip_β),
        ("vmat_becke_T2_fxc_a", vmat_becke_t2_fxc_α),
        ("vmat_becke_T2_fxc_b", vmat_becke_t2_fxc_β),
        ("de_xc_skeleton_no_becke", de_xc_skeleton_no_becke),
        ("de_xc_skeleton", de_xc_skeleton),
        ("vmat_deriv1_grid_a", vmat_deriv1_grid_α),
        ("vmat_deriv1_grid_b", vmat_deriv1_grid_β),
    ]);

    let timing = timing.lock().unwrap().clone();
    (result, timing)
}

/* #endregion */

/* #region final implementation of UKS Hessian with becke grid-shift */

/// UKS Hessian XC component with the Becke grid-shift (pyhessref
/// `UHessKSNaiveBecke`), the unrestricted sibling of
/// [`super::hess_rks_becke::RHessKSNIMatmulBecke`]:
/// [`UHessElecInteractAPI`] with `make_skeleton_hess` returning the
/// translationally invariant `de_xc_skeleton` and `get_deriv1_ao` the
/// translationally invariant `vmat_deriv1_grid_a/b` per spin.
///
/// Grids must be atom-grouped (`sort_grids=False` in pyscf; the ByAtom
/// attribution scheme of `becke_partition`).
pub struct UHessKSNIMatmulBecke<'a> {
    /// Molecule.
    pub mol: CInt,
    /// List of `(scale, functional)` pairs of the XC functional.
    pub xc_func_list: &'a [(f64, LibXCFunctional)],
    /// Numerical-integration driver over the (atom-grouped) Hessian grid.
    pub ni: NIMatmul<'a>,
    /// Optional separate grid for the CP-KS response; `None` reuses `ni`.
    pub ni_cpks: Option<NIMatmul<'a>>,
    /// Pre-partition quadrature weights, shape `[ngrids]`.
    pub quadrature_weights: Vec<f64>,
    /// Cumulative per-atom grid boundaries, shape `[natm + 1]`: atom `A` owns
    /// grids `[atm_quad_split[A], atm_quad_split[A + 1])`.
    pub atm_quad_split: Vec<usize>,
    /// Anti-symmetric Becke radii-adjustment table as `natm` row-major rows of
    /// length `natm`: `adjustment_factor[A][B]` reads entry `(A, B)` (the
    /// convention of [`super::becke_partition::BeckeMolTables`]).
    pub adjustment_factor: Vec<Vec<f64>>,
    /// Becke switch-function hardness (most commonly 3).
    pub hardness: usize,
    /// Print per-chunk progress of the Hessian setup.
    pub verbose: bool,
    /// Intermediates of the Hessian setup: all keys of
    /// [`make_hessian_setup_becke_uks`] (with `fxc` renamed to `cpks_fxc`
    /// unless a CP-KS-specific grid is given), plus `mo_coeff_0/1 [nao, nmo]`,
    /// `mo_occ_0/1 [nmo]` from [`UHessElecInteractAPI::make_response_preparation`].
    pub intmd: HashMap<String, Tsr>,
}

impl<'a> UHessKSNIMatmulBecke<'a> {
    /// Create a new UKS Becke Hessian object.
    ///
    /// # Parameters
    ///
    /// - `mol` : molecule.
    /// - `xc_func_list` : list of `(scale, functional)` pairs.
    /// - `ni` : numerical-integration driver over the atom-grouped grid.
    /// - `quadrature_weights` : shape `[ngrids]`.  Pre-partition quadrature weights.
    /// - `atm_quad_split` : shape `[natm + 1]`.  Cumulative per-atom grid boundaries.
    /// - `adjustment_factor` : `natm` row-major rows of length `natm`; the anti-symmetric Becke
    ///   radii-adjustment table.
    /// - `hardness` : Becke switch-function hardness.
    /// - `verbose` : print progress.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mol: &CInt,
        xc_func_list: &'a [(f64, LibXCFunctional)],
        ni: NIMatmul<'a>,
        quadrature_weights: Vec<f64>,
        atm_quad_split: Vec<usize>,
        adjustment_factor: Vec<Vec<f64>>,
        hardness: usize,
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
            verbose,
            intmd: HashMap::new(),
        }
    }

    /// Perform the Hessian setup for UKS calculations, with the grid-shift.
    ///
    /// Mirrors [`super::hess_uks::UHessKSNIMatmul::make_hessian_setup`]: `fxc`
    /// is stored as `cpks_fxc` (unless a CP-KS-specific grid is given) for the
    /// response; `de_xc_skeleton` and `vmat_deriv1_grid_a/b` are the main
    /// results.
    pub fn make_hessian_setup(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], atm_list: Option<&[usize]>) {
        let occidx_α = mo_occ[α].view().greater(0).into_vec();
        let occidx_β = mo_occ[β].view().greater(0).into_vec();
        let mocc_α = mo_coeff[α].bool_select(-1, &occidx_α);
        let mocc_β = mo_coeff[β].bool_select(-1, &occidx_β);
        let dm0α = &mocc_α % mocc_α.t();
        let dm0β = &mocc_β % mocc_β.t();

        let (result, _timing) = make_hessian_setup_becke_uks(
            &self.mol,
            self.xc_func_list,
            &mut self.ni,
            dm0α.view(),
            dm0β.view(),
            &self.atm_quad_split,
            &self.quadrature_weights,
            &self.adjustment_factor,
            self.hardness,
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

impl<'a> HessUtilAPI for UHessKSNIMatmulBecke<'a> {}

impl<'a> UHessElecInteractAPI for UHessKSNIMatmulBecke<'a> {
    fn make_skeleton_hess(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        self.intmd["de_xc_skeleton"].to_owned()
    }

    fn get_deriv1_ao(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> [Tsr; 2] {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        [self.intmd["vmat_deriv1_grid_a"].to_owned(), self.intmd["vmat_deriv1_grid_b"].to_owned()]
    }

    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) {
        self.intmd.insert("mo_coeff_0".to_string(), mo_coeff[α].view().into_contig(ColMajor));
        self.intmd.insert("mo_coeff_1".to_string(), mo_coeff[β].view().into_contig(ColMajor));
        self.intmd.insert("mo_occ_0".to_string(), mo_occ[α].view().into_contig(ColMajor));
        self.intmd.insert("mo_occ_1".to_string(), mo_occ[β].view().into_contig(ColMajor));
    }

    fn get_response_bra(&mut self, bra: &[TsrView; 2]) -> [Tsr; 2] {
        let ni_cpks = self.ni_cpks.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff_α = self.intmd["mo_coeff_0"].view();
        let mo_coeff_β = self.intmd["mo_coeff_1"].view();
        let mo_occ_α = self.intmd["mo_occ_0"].view();
        let mo_occ_β = self.intmd["mo_occ_1"].view();
        let fxc_eff = self.intmd["cpks_fxc"].view();

        let occidx_α = mo_occ_α.view().greater(0).into_vec();
        let occidx_β = mo_occ_β.view().greater(0).into_vec();
        let mocc_α = mo_coeff_α.bool_select(-1, &occidx_α);
        let mocc_β = mo_coeff_β.bool_select(-1, &occidx_β);

        let den_type = determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec());

        let ([resp_α, resp_β], _timing) = get_uks_response_bra_batched(
            ni_cpks,
            den_type,
            fxc_eff.view(),
            bra,
            &[mocc_α.view(), mocc_β.view()],
            self.verbose,
        );
        [resp_α, resp_β]
    }
}

/* #endregion */
