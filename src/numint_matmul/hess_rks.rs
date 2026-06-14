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
    ($tsr: ident, $idx:expr) => {
        $tsr.i((Ellipsis, $idx))
    };
}

/* #endregion */

pub fn make_hessian_setup_batch(
    mol: &CInt,
    xc_func: &LibXCFunctional,
    ni: &mut NIMatmul,
    dm0: TsrView,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> HashMap<&'static str, Tsr> {
    let nao = mol.nao();
    let atm_list = atm_list.map_or_else(|| (0..mol.natm()).collect_vec(), |lst| lst.to_vec());
    let natm = atm_list.len();
    // ao slices indexed by `atm_list`
    let aoslices_full = mol.aoslice_by_atom();
    let aoslices = atm_list.iter().map(|&iatm| aoslices_full[iatm]).collect_vec();
    let xc_type = determine_den_type(xc_func);

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
    let ao_dm0 = ao.i((.., .., ..ncomp_ao_dm0)) % dm0;

    let (rho, vxc, fxc) = get_rho_vxc_fxc(xc_func, ao.view(), ao_dm0.view());
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

    HashMap::from([("rho", rho), ("vxc", vxc), ("fxc", fxc), ("de_fxc", de_fxc)])
}

pub fn get_rho_vxc_fxc(xc_func: &LibXCFunctional, ao: TsrView, ao_dm0: TsrView) -> (Tsr, Tsr, Tsr) {
    // see also pyhessref/nimatmul/rks.py

    let xc_type = determine_den_type(xc_func);
    let nvar = xc_type.num_nvar();
    let ngrids = ao.shape()[0];
    let device = ao.device().clone();

    let mut rho = rt::zeros(([ngrids, nvar], &device));
    *&mut rho.i_mut((.., 0)) += rt::vecdot(ao.i((.., .., 0)), ao_dm0.i((.., .., 0)), 1);
    if matches!(xc_type, SIGMA | TAU) {
        *&mut rho.i_mut((.., X)) += 2 * rt::vecdot(ao.i((.., .., X)), ao_dm0.i((.., .., 0)), 1);
        *&mut rho.i_mut((.., Y)) += 2 * rt::vecdot(ao.i((.., .., Y)), ao_dm0.i((.., .., 0)), 1);
        *&mut rho.i_mut((.., Z)) += 2 * rt::vecdot(ao.i((.., .., Z)), ao_dm0.i((.., .., 0)), 1);
    }
    if matches!(xc_type, TAU) {
        *&mut rho.i_mut((.., 4)) += 0.5
            * (rt::vecdot(ao.i((.., .., X)), ao_dm0.i((.., .., X)), 1)
                + rt::vecdot(ao.i((.., .., Y)), ao_dm0.i((.., .., Y)), 1)
                + rt::vecdot(ao.i((.., .., Z)), ao_dm0.i((.., .., Z)), 1))
    }

    let xc_eff = libxc_eval_eff(xc_func, rho.view(), 2, false);
    let [_, vxc, fxc] = xc_eff.into_iter().collect_array().unwrap();

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
        [[XXX, XXY, XXZ], [XXY, XYY, XYZ], [XXZ, XYZ, XZZ], [XYY, XYZ, YYY], [XYZ, YYY, YYZ], [XZZ, YZZ, ZZZ]];
    const TRIPLE_TAU_DIAG: [[usize; 6]; 3] =
        [[XXX, XXY, XXZ, XYY, XYZ, XZZ], [XXY, XYY, XYZ, YYY, YYZ, YZZ], [XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]];

    let natm = aoslices.len();
    let nao = ao.shape()[1];
    let device = ao.device().clone();

    let dao_vxc_diag = rt::zeros(([nao, 6], &device));

    // contribution 1: ao deriv2
    dao_vxc_diag
}
