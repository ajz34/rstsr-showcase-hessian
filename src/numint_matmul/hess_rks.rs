// see also pyhessref/nimatmul/rks.py

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

const fn get_hess_ao_deriv(xc_type: XCDenType) -> usize {
    match xc_type {
        RHO => 2,
        SIGMA => 3,
        TAU => 3,
        LAPL => unimplemented!(),
    }
}

const fn get_hess_ncomp_ao_dm0(xc_type: XCDenType) -> usize {
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

pub fn make_hessian_setup_batch(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> HashMap<&'static str, Tsr> {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let atm_list = atm_list.map_or_else(|| (0..mol.natm()).collect_vec(), |lst| lst.to_vec());
    // ao slices indexed by `atm_list`
    let aoslices_full = mol.aoslice_by_atom();
    let aoslices = atm_list.iter().map(|&iatm| aoslices_full[iatm]).collect_vec();
    // overall xc_type is the strictest (max nvar) one across the functionals;
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    let device = dm0.device().clone();
    let weights = rt::asarray((ni.weights.clone(), &device));

    let tic = |label: &str, t0: std::time::Instant| {
        if verbose {
            let elapsed = t0.elapsed();
            println!("{:>20} took {:.3} seconds", label, elapsed.as_secs_f64());
        }
    };

    // --- ao, rho, vxc, fxc --- //

    // ao      [ngrids, nao, ncomp]
    // ao_dm0  [ngrids, nao, ncomp_ao_dm0]
    // rho     [ngrids, nvar]
    // vxc     [ngrids, nvar]
    // fxc     [ngrids, nvar, nvar]

    let t0 = std::time::Instant::now();

    let ao = ni.get_cached_ao(get_hess_ao_deriv(xc_type));
    let ncomp_ao_dm0 = get_hess_ncomp_ao_dm0(xc_type);
    let ao_dm0 = index!(ao, ..ncomp_ao_dm0) % &dm0;

    let (rho, vxc, fxc) = get_rho_vxc_fxc(xc_func_list, ao.view(), ao_dm0.view());
    let wv = &weights * &vxc;
    let wf = &weights * &fxc;

    tic("ao, rho, vxc, fxc", t0);

    // --- drho --- //

    // drho    [ngrids, nvar, 3, natm]
    let t0 = std::time::Instant::now();
    let drho = get_drho(xc_type, ao.view(), ao_dm0.view(), &aoslices);
    tic("drho", t0);

    // --- de_fxc --- //

    // de_fxc  [3, 3, natm, natm]
    let t0 = std::time::Instant::now();
    let de_fxc = get_de_fxc(wf.view(), drho.view());
    tic("de_fxc", t0);

    // --- de_vxc_diag --- //

    // de_vxc_diag [3, 3, natm, natm]
    let t0 = std::time::Instant::now();
    let de_vxc_diag = get_de_vxc_diag(xc_type, ao.view(), ao_dm0.view(), wv.view(), &aoslices);
    tic("de_vxc_diag", t0);

    // --- de_vxc_off --- //

    // de_vxc_off [3, 3, natm, natm]
    let t0 = std::time::Instant::now();
    let de_vxc_off = get_de_vxc_off(xc_type, ao.view(), dm0.view(), wv.view(), &aoslices);
    tic("de_vxc_off", t0);

    // --- vmat_ip --- //

    // vmat_ip [nao, nao, 3]
    let t0 = std::time::Instant::now();
    let vmat_ip = get_vmat_ip(xc_type, ao.view(), wv.view());
    tic("vmat_ip", t0);

    // --- vmat_deriv1 --- //

    // vmat_deriv1 [nao, nao, 3, natm]
    let t0 = std::time::Instant::now();
    let vmat_deriv1 = get_vmat_deriv1(xc_type, ao.view(), drho.view(), wf.view(), vmat_ip.view(), &aoslices);
    tic("vmat_deriv1", t0);

    HashMap::from([
        ("rho", rho),
        ("vxc", vxc),
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag", de_vxc_diag),
        ("de_vxc_off", de_vxc_off),
        ("vmat_ip", vmat_ip),
        ("vmat_deriv1", vmat_deriv1),
    ])
}

pub fn get_rho_vxc_fxc(xc_func_list: &[(f64, LibXCFunctional)], ao: TsrView, ao_dm0: TsrView) -> (Tsr, Tsr, Tsr) {
    // see also pyhessref/nimatmul/rks.py

    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    // overall xc_type is the strictest one across the functionals; partial
    // contributions from looser families are added into the leading slice.
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

    let mut vxc = rt::zeros(([ngrids, nvar], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, nvar], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        // each sub-functional consumes only the leading `nvar_i` rho components.
        let rho_i = rho.i((.., ..nvar_i));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [_, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        // accumulate into the leading slice of the (possibly larger) global tensors.
        *&mut vxc.i_mut((.., ..nvar_i)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, ..nvar_i)) += *scale * fxc_i;
    }

    (rho, vxc, fxc)
}

pub fn get_drho(xc_type: XCDenType, ao: TsrView, ao_dm0: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    // direct transformation of `pyhessref/nimatmul/rks.py`, function `_make_drho`

    let ngrids = ao.shape()[0];
    let nvar = xc_type.num_nvar();
    let natm = aoslices.len();
    let device = ao.device().clone();

    let mut drho = rt::zeros(([ngrids, nvar, 3, natm], &device));

    // components: [rho_var, t_direction, cbra, cket]
    // tuple that contribute to each rho component
    // for symmetric components, result is multiplied by 2 at the end

    // RHO part
    let mut components = vec![(0, 0, X, O), (0, 1, Y, O), (0, 2, Z, O)];
    // SIGMA part
    if matches!(xc_type, SIGMA | TAU) {
        // bra deriv2, ket 0
        let sigma_bra2_ket0 = [
            [(1, 0, XX, O), (2, 0, XY, O), (3, 0, XZ, O)],
            [(1, 1, YX, O), (2, 1, YY, O), (3, 1, YZ, O)],
            [(1, 2, ZX, O), (2, 2, ZY, O), (3, 2, ZZ, O)],
        ];
        components.extend(sigma_bra2_ket0.concat());
        // bra deriv1, ket deriv1
        let sigma_bra1_ket1 = [
            [(1, 0, X, X), (2, 0, X, Y), (3, 0, X, Z)],
            [(1, 1, Y, X), (2, 1, Y, Y), (3, 1, Y, Z)],
            [(1, 2, Z, X), (2, 2, Z, Y), (3, 2, Z, Z)],
        ];
        components.extend(sigma_bra1_ket1.concat());
    }
    // TAU part
    if matches!(xc_type, TAU) {
        // bra deriv2, ket deriv1
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

    // RHO and SIGMA part multiply factor 2
    // TAU part does not multiply factor 2 because of the 0.5 factor in rho_tau
    match xc_type {
        RHO => *&mut drho.i_mut((.., 0..1)) *= 2.0,
        SIGMA | TAU => *&mut drho.i_mut((.., 0..4)) *= 2.0,
        LAPL => unimplemented!(),
    }
    drho
}

pub fn get_de_fxc(wf: TsrView, drho: TsrView) -> Tsr {
    // wf , drho, drho -> de_fxc
    // gxy, gxtA, gysB -> tsAB

    let [ngrids, nvar, _, natm] = drho.shape().iter().cloned().collect_array().unwrap();

    // wf    * drho  -> tmp
    // gxy.. * gx.tA -> gytA
    let tmp1 = rt::vecdot(wf.i((.., .., .., None, None)), drho.i((.., .., None, .., ..)), 1);

    // tmp1  * drho -> tmp2
    // gytA  * gysB -> tAsB
    let tmp1 = tmp1.reshape([ngrids * nvar, natm * 3]);
    let drho = drho.reshape([ngrids * nvar, natm * 3]);
    let tmp2 = tmp1.t() % drho;

    // transpose tmp2 to get de_fxc
    // tAsB -> tsAB
    tmp2.reshape([3, natm, 3, natm]).transpose([0, 2, 1, 3]).into_contig(ColMajor)
}

pub fn get_de_vxc_diag(xc_type: XCDenType, ao: TsrView, ao_dm0: TsrView, wv: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    const TRIPLE_SIGMA_DIAG: [[usize; 3]; 6] =
        [[XXX, XXY, XXZ], [XXY, XYY, XYZ], [XXZ, XYZ, XZZ], [XYY, YYY, YYZ], [XYZ, YYZ, YZZ], [XZZ, YZZ, ZZZ]];
    const TRIPLE_TAU_DIAG: [[usize; 6]; 3] =
        [[XXX, XXY, XXZ, XYY, XYZ, XZZ], [XXY, XYY, XYZ, YYY, YYZ, YZZ], [XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]];

    let natm = aoslices.len();
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

    // reduce ao-wise contributions to atom-wise contributions
    let mut de_vxc_diag = rt::zeros(([6, natm, natm], &device));
    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);
        de_vxc_diag.i_mut((.., A, A)).assign(dao_vxc_diag.i(slc).sum_axes(0));
    }
    // symmetrize and expand de_vxc_diag from [6, natm, natm] to [3, 3, natm, natm]
    de_vxc_diag.index_select(0, [0, 1, 2, 1, 3, 4, 2, 4, 5]).into_shape([3, 3, natm, natm])
}

pub fn get_de_vxc_off(xc_type: XCDenType, ao: TsrView, dm0: TsrView, wv: TsrView, aoslices: &[[usize; 4]]) -> Tsr {
    let natm = aoslices.len();
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

    // add transposition
    let dao_vxc_off = &dao_vxc_off + dao_vxc_off.transpose([1, 0, 3, 2]);

    // reduce ao-wise contributions to atom-wise contributions
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
    // direct transformation of `pyhessref/nimatmul/rks.py`, function `_vmat_ip`

    let nao = ao.shape()[1];
    let device = ao.device().clone();

    let mut vmat_ip = rt::zeros(([nao, nao, 3], &device));

    if matches!(xc_type, RHO) {
        // bra-on-A and ket-on-A halves are identical for LDA
        // (both equal 0.5 * wv[0] * ao[t+1]^T @ ao[O]); folded into a single contraction.
        let aow: Tsr = index!(wv, 0) * index!(ao, O);
        for t in 0..3 {
            index_mut!(vmat_ip, t).matmul_from(&index!(ao, t + 1).t(), &aow, 1.0, 1.0);
        }
        return vmat_ip;
    }

    // GGA + MGGA share the same SIGMA structure
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

    // MGGA tau channel
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

pub fn get_vmat_deriv1(
    xc_type: XCDenType,
    ao: TsrView,
    drho: TsrView,
    wf: TsrView,
    vmat_ip: TsrView,
    aoslices: &[[usize; 4]],
) -> Tsr {
    let natm = aoslices.len();
    let nao = ao.shape()[1];

    let mut vmat_deriv1: Tsr = rt::zeros(([nao, nao, 3, natm], ao.device()));

    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);

        if matches!(xc_type, RHO) {
            // wf_rho: [ngrids, 3]
            let wf_rho: Tsr = 0.5 * index!(wf, O, O) * drho.i((.., O, .., A));
            for t in 0..3 {
                let aow = index!(wf_rho, t) * index!(ao, O);
                index_mut!(vmat_deriv1, t, A).matmul_from(aow.t(), index!(ao, O), 1.0, 1.0);
            }
        }

        if matches!(xc_type, SIGMA | TAU) {
            // wf_rho = np.einsum("gxy, gxt -> gyt", wf, drho[A])
            // wf_rho[:, 0] *= 0.5, wf_rho[:, 4] *= 0.25
            let mut wf_rho = rt::vecdot(&wf, drho.i((.., .., None, .., A)), 1);
            *&mut wf_rho.i_mut((.., 0)) *= 0.5;
            if matches!(xc_type, TAU) {
                *&mut wf_rho.i_mut((.., 4)) *= 0.25;
            }
            for t in 0..3 {
                let aow = rt::vecdot(wf_rho.i((.., None, ..4, t)), ao.i((.., .., ..4)), 2);
                index_mut!(vmat_deriv1, t, A).matmul_from(aow.t(), index!(ao, O), 1.0, 1.0);
            }

            if matches!(xc_type, TAU) {
                for r in [X, Y, Z] {
                    for t in 0..3 {
                        let aow = wf_rho.i((.., 4, t)) * index!(ao, r);
                        index_mut!(vmat_deriv1, t, A).matmul_from(aow.t(), index!(ao, r), 1.0, 1.0);
                    }
                }
            }
        }

        *&mut vmat_deriv1.i_mut((slc, .., .., A)) -= vmat_ip.i((slc, .., ..));
    }

    &vmat_deriv1 + vmat_deriv1.swapaxes(0, 1)
}

/* #endregion */

/* #region parallel wrapper */

pub fn make_hessian_setup_with_parallel(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0: TsrView,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> HashMap<&'static str, Tsr> {
    // batch for grids
    // - except for rho, vxc, fxc; other tensors can be added (reduced)
    // - rho, vxc, fxc requires concation

    // outer iter: batch by nbatch (limit memory usage)
    // inner iter: batch by nchunk (for parallel)

    let ngrids = ni.weights.len();
    let nbatch = ni.nbatch;
    let nchunk = ni.nchunk;
    let device = dm0.device().clone();
    let nvar = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec()).num_nvar();
    let natm = atm_list.map_or_else(|| mol.natm(), |lst| lst.len());
    let nao = mol.nao();

    let rho: Tsr = rt::zeros(([ngrids, nvar], &device));
    let vxc: Tsr = rt::zeros(([ngrids, nvar], &device));
    let fxc: Tsr = rt::zeros(([ngrids, nvar, nvar], &device));
    let de_fxc: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_ip: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_deriv1: Tsr = rt::zeros(([nao, nao, 3, natm], &device));

    // atomic guard to avoid racing write
    let guard = Mutex::new(());

    for start_batch in (0..ngrids).step_by(nbatch) {
        let end_batch = (start_batch + nbatch).min(ngrids);

        (start_batch..end_batch).into_par_iter().step_by(nchunk).for_each(|start| {
            let end = (start + nchunk).min(end_batch);
            let mut ni_inner = ni.split_batch(start, end);
            let result = make_hessian_setup_batch(mol, xc_func_list, &mut ni_inner, dm0.view(), atm_list, verbose);
            // fill rho, vxc, fxc
            // this is assumed to be not racing, so no guard at here
            unsafe {
                let rho_slc = rho.i(start..end);
                let vxc_slc = vxc.i(start..end);
                let fxc_slc = fxc.i(start..end);
                let mut rho_slc = rho_slc.force_mut();
                let mut vxc_slc = vxc_slc.force_mut();
                let mut fxc_slc = fxc_slc.force_mut();
                rho_slc.assign(&result["rho"]);
                vxc_slc.assign(&result["vxc"]);
                fxc_slc.assign(&result["fxc"]);
            }
            // add up other tensors
            unsafe {
                let lock = guard.lock().unwrap();
                *&mut de_fxc.force_mut() += &result["de_fxc"];
                *&mut de_vxc_diag.force_mut() += &result["de_vxc_diag"];
                *&mut de_vxc_off.force_mut() += &result["de_vxc_off"];
                *&mut vmat_ip.force_mut() += &result["vmat_ip"];
                *&mut vmat_deriv1.force_mut() += &result["vmat_deriv1"];
                drop(lock);
            }
        });
    }

    HashMap::from([
        ("rho", rho),
        ("vxc", vxc),
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag", de_vxc_diag),
        ("de_vxc_off", de_vxc_off),
        ("vmat_ip", vmat_ip),
        ("vmat_deriv1", vmat_deriv1),
    ])
}

/* #endregion */
