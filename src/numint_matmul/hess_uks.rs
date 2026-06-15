// see also pyhessref/nimatmul/uks.py

use super::hess_rks::{get_de_vxc_diag, get_de_vxc_off, get_drho, get_vmat_ip};
use super::prelude::*;

/* #region const dimensions/indices definition */

const O: usize = 0;
const X: usize = 1;
const Y: usize = 2;
const Z: usize = 3;

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

pub fn get_rho_vxc_fxc_uks(
    xc_func_list: &[(f64, LibXCFunctional)],
    ao: TsrView,
    ao_dm0a: TsrView,
    ao_dm0b: TsrView,
) -> (Tsr, Tsr, Tsr, Tsr) {
    // Returns: (rhoa, rhob, vxc, fxc)
    // rhoa, rhob: [ngrids, nvar]
    // vxc: [ngrids, nvar, 2]
    // fxc: [ngrids, nvar, 2, nvar, 2]

    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let xc_type = xc_func_list
        .iter()
        .map(|(_, f)| determine_den_type(f))
        .max_by_key(|t| t.num_nvar())
        .expect("xc_func_list must not be empty");
    let nvar = xc_type.num_nvar();
    let ngrids = ao.shape()[0];
    let device = ao.device().clone();

    // Compute rhoa
    let mut rhoa = rt::zeros(([ngrids, nvar], &device));
    index_mut!(rhoa, 0) += rt::vecdot(index!(ao, 0), index!(ao_dm0a, O), 1);
    if matches!(xc_type, SIGMA | TAU) {
        index_mut!(rhoa, X) += 2 * rt::vecdot(index!(ao, X), index!(ao_dm0a, O), 1);
        index_mut!(rhoa, Y) += 2 * rt::vecdot(index!(ao, Y), index!(ao_dm0a, O), 1);
        index_mut!(rhoa, Z) += 2 * rt::vecdot(index!(ao, Z), index!(ao_dm0a, O), 1);
    }
    if matches!(xc_type, TAU) {
        index_mut!(rhoa, 4) += 0.5
            * (rt::vecdot(index!(ao, X), index!(ao_dm0a, X), 1)
                + rt::vecdot(index!(ao, Y), index!(ao_dm0a, Y), 1)
                + rt::vecdot(index!(ao, Z), index!(ao_dm0a, Z), 1))
    }

    // Compute rhob
    let mut rhob = rt::zeros(([ngrids, nvar], &device));
    index_mut!(rhob, 0) += rt::vecdot(index!(ao, 0), index!(ao_dm0b, O), 1);
    if matches!(xc_type, SIGMA | TAU) {
        index_mut!(rhob, X) += 2 * rt::vecdot(index!(ao, X), index!(ao_dm0b, O), 1);
        index_mut!(rhob, Y) += 2 * rt::vecdot(index!(ao, Y), index!(ao_dm0b, O), 1);
        index_mut!(rhob, Z) += 2 * rt::vecdot(index!(ao, Z), index!(ao_dm0b, O), 1);
    }
    if matches!(xc_type, TAU) {
        index_mut!(rhob, 4) += 0.5
            * (rt::vecdot(index!(ao, X), index!(ao_dm0b, X), 1)
                + rt::vecdot(index!(ao, Y), index!(ao_dm0b, Y), 1)
                + rt::vecdot(index!(ao, Z), index!(ao_dm0b, Z), 1))
    }

    // Stack into [ngrids, nvar, 2]
    let mut rho_uks = rt::zeros(([ngrids, nvar, 2], &device));
    rho_uks.i_mut((.., .., 0)).assign(&rhoa);
    rho_uks.i_mut((.., .., 1)).assign(&rhob);

    // Evaluate vxc and fxc with spin-polarized libxc
    let mut vxc = rt::zeros(([ngrids, nvar, 2], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, 2, nvar, 2], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        let rho_i = rho_uks.i((.., ..nvar_i, ..));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [_, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        *&mut vxc.i_mut((.., ..nvar_i, ..)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, .., ..nvar_i, ..)) += *scale * fxc_i;
    }

    (rhoa, rhob, vxc, fxc)
}

pub fn get_drho_uks(
    xc_type: XCDenType,
    ao: TsrView,
    ao_dm0a: TsrView,
    ao_dm0b: TsrView,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let drhoa = get_drho(xc_type, ao.view(), ao_dm0a.view(), aoslices);
    let drhob = get_drho(xc_type, ao.view(), ao_dm0b.view(), aoslices);
    (drhoa, drhob)
}

fn get_de_fxc_uks_inner(wf_block: TsrView, drho1: TsrView, drho2: TsrView) -> Tsr {
    // wf_block: [ngrids, nvar, nvar] (a single spin block of wf)
    // drho1, drho2: [ngrids, nvar, 3, natm]
    // result: [3, 3, natm, natm]

    let [ngrids, nvar, _, natm] = drho1.shape().iter().cloned().collect_array().unwrap();

    let tmp1 = rt::vecdot(wf_block.i((.., .., .., None, None)), drho1.i((.., .., None, .., ..)), 1);
    let tmp1 = tmp1.reshape([ngrids * nvar, natm * 3]);
    let drho2 = drho2.reshape([ngrids * nvar, natm * 3]);
    let tmp2 = tmp1.t() % drho2;

    tmp2.reshape([3, natm, 3, natm]).transpose([0, 2, 1, 3]).into_contig(ColMajor)
}

pub fn get_de_fxc_uks(wf: TsrView, drhoa: TsrView, drhob: TsrView) -> Tsr {
    // wf: [ngrids, nvar, 2, nvar, 2]
    // drhoa, drhob: [ngrids, nvar, 3, natm]
    // result: [3, 3, natm, natm]

    let de_aa = get_de_fxc_uks_inner(wf.i((.., .., 0, .., 0)), drhoa.view(), drhoa.view());
    let de_ab = get_de_fxc_uks_inner(wf.i((.., .., 0, .., 1)), drhoa.view(), drhob.view());
    let de_ba = get_de_fxc_uks_inner(wf.i((.., .., 1, .., 0)), drhob.view(), drhoa.view());
    let de_bb = get_de_fxc_uks_inner(wf.i((.., .., 1, .., 1)), drhob.view(), drhob.view());

    &de_aa + &de_ab + &de_ba + &de_bb
}

pub fn get_vmat_deriv1_uks(
    xc_type: XCDenType,
    ao: TsrView,
    drhoa: TsrView,
    drhob: TsrView,
    wf: TsrView,
    vmat_ip_a: TsrView,
    vmat_ip_b: TsrView,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let natm = aoslices.len();
    let nao = ao.shape()[1];
    let ngrids = ao.shape()[0];
    let nvar = xc_type.num_nvar();
    let device = ao.device();

    let mut vmata_deriv1: Tsr = rt::zeros(([nao, nao, 3, natm], device));
    let mut vmatb_deriv1: Tsr = rt::zeros(([nao, nao, 3, natm], device));

    for (A, &[_, _, p0, p1]) in aoslices.iter().enumerate() {
        let slc = rt::slice!(p0, p1);

        if matches!(xc_type, RHO) {
            // LDA: fxc is [G, 1, 2, 1, 2], extract scalar spin blocks
            // Alpha output (s2=0): fxc_aa @ drho_a + fxc_ba @ drho_b
            let wf_aa_00 = wf.i((.., 0, 0, 0, 0)); // [G], s1=0, s2=0
            let wf_ba_00 = wf.i((.., 0, 1, 0, 0)); // [G], s1=1, s2=0

            let wva_f: Tsr = 0.5
                * (wf_aa_00.i((.., None)) * drhoa.i((.., 0, .., A)) + wf_ba_00.i((.., None)) * drhob.i((.., 0, .., A)));

            // Beta output (s2=1): fxc_ab @ drho_a + fxc_bb @ drho_b
            let wf_ab_00 = wf.i((.., 0, 0, 0, 1)); // [G], s1=0, s2=1
            let wf_bb_00 = wf.i((.., 0, 1, 0, 1)); // [G], s1=1, s2=1

            let wvb_f: Tsr = 0.5
                * (wf_ab_00.i((.., None)) * drhoa.i((.., 0, .., A)) + wf_bb_00.i((.., None)) * drhob.i((.., 0, .., A)));

            for t in 0..3 {
                let aowa = wva_f.i((.., t)) * index!(ao, O);
                index_mut!(vmata_deriv1, t, A).matmul_from(aowa.t(), index!(ao, O), 1.0, 1.0);
                let aowb = wvb_f.i((.., t)) * index!(ao, O);
                index_mut!(vmatb_deriv1, t, A).matmul_from(aowb.t(), index!(ao, O), 1.0, 1.0);
            }
        }

        if matches!(xc_type, SIGMA | TAU) {
            let wf_aa = wf.i((.., .., 0, .., 0)); // [G, x, y]
            let wf_ab = wf.i((.., .., 0, .., 1)); // [G, x, y]
            let wf_ba = wf.i((.., .., 1, .., 0)); // [G, x, y]
            let wf_bb = wf.i((.., .., 1, .., 1)); // [G, x, y]

            let drhoa_A = drhoa.i((.., .., .., A)); // [G, x, 3]
            let drhob_A = drhob.i((.., .., .., A)); // [G, x, 3]

            // Python: wva_f[y,t,g] = sum_x wf_aa[x,y,g]*drhoa[A,t,x,g] + wf_ab[x,y,g]*drhob[A,t,x,g]
            // For each direction t, compute per-grid contraction:
            //   wva_f_t[g, y] = wf_aa[g, :, y]^T @ drhoa_A[g, :, t] + wf_ab[g, :, y]^T @ drhob_A[g, :, t]

            for t in 0..3 {
                // drhoa_A[:,:,t] shape: [G, x], reshape → [G, x, 1]
                let drhoa_t = drhoa_A.i((.., .., t)).into_shape([ngrids, nvar, 1]).to_owned();
                let drhob_t = drhob_A.i((.., .., t)).into_shape([ngrids, nvar, 1]).to_owned();

                // vecdot on axis 1: [G, x, y] @ [G, x, 1] → contract x
                // Remaining: [G, y] and [G, 1] → col-major broadcast → [G, y]
                // Alpha output (s2=0): fxc_aa @ drho_a + fxc_ba @ drho_b
                // Beta output (s2=1): fxc_ab @ drho_a + fxc_bb @ drho_b
                let wva_f_aa = rt::vecdot(wf_aa.view(), drhoa_t.view(), 1);
                let wva_f_ba = rt::vecdot(wf_ba.view(), drhob_t.view(), 1);
                let wvb_f_ab = rt::vecdot(wf_ab.view(), drhoa_t.view(), 1);
                let wvb_f_bb = rt::vecdot(wf_bb.view(), drhob_t.view(), 1);
                let mut wva_f_t = &wva_f_aa + &wva_f_ba; // [G, y]
                let mut wvb_f_t = &wvb_f_ab + &wvb_f_bb; // [G, y]

                *&mut wva_f_t.i_mut((.., 0)) *= 0.5;
                *&mut wvb_f_t.i_mut((.., 0)) *= 0.5;
                if matches!(xc_type, TAU) {
                    *&mut wva_f_t.i_mut((.., 4)) *= 0.25;
                    *&mut wvb_f_t.i_mut((.., 4)) *= 0.25;
                }

                // Contract with ao: aow = sum_c wva_f_t[:,c] * ao[c]
                for c in 0..4 {
                    let aowa = wva_f_t.i((.., c)).i((.., None)) * index!(ao, c); // [G, nao]
                    index_mut!(vmata_deriv1, t, A).matmul_from(aowa.t(), index!(ao, O), 1.0, 1.0);
                    let aowb = wvb_f_t.i((.., c)).i((.., None)) * index!(ao, c); // [G, nao]
                    index_mut!(vmatb_deriv1, t, A).matmul_from(aowb.t(), index!(ao, O), 1.0, 1.0);
                }

                if matches!(xc_type, TAU) {
                    for r in [X, Y, Z] {
                        let aowa = wva_f_t.i((.., 4)).i((.., None)) * index!(ao, r); // [G, nao]
                        index_mut!(vmata_deriv1, t, A).matmul_from(aowa.t(), index!(ao, r), 1.0, 1.0);
                        let aowb = wvb_f_t.i((.., 4)).i((.., None)) * index!(ao, r); // [G, nao]
                        index_mut!(vmatb_deriv1, t, A).matmul_from(aowb.t(), index!(ao, r), 1.0, 1.0);
                    }
                }
            }
        }

        *&mut vmata_deriv1.i_mut((slc, .., .., A)) -= vmat_ip_a.i((slc, .., ..));
        *&mut vmatb_deriv1.i_mut((slc, .., .., A)) -= vmat_ip_b.i((slc, .., ..));
    }

    let vmata_deriv1 = &vmata_deriv1 + vmata_deriv1.swapaxes(0, 1);
    let vmatb_deriv1 = &vmatb_deriv1 + vmatb_deriv1.swapaxes(0, 1);

    (vmata_deriv1, vmatb_deriv1)
}

pub fn make_hessian_setup_uks(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0a: TsrView,
    dm0b: TsrView,
    atm_list: Option<&[usize]>,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let atm_list = atm_list.map_or_else(|| (0..mol.natm()).collect_vec(), |lst| lst.to_vec());
    let aoslices_full = mol.aoslice_by_atom();
    let aoslices = atm_list.iter().map(|&iatm| aoslices_full[iatm]).collect_vec();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    let device = dm0a.device().clone();
    let weights = rt::asarray((ni.weights.clone(), &device));

    let mut timing = IndexMap::new();
    let mut tic = |label: &'static str, t0: std::time::Instant| {
        let elapsed = t0.elapsed().as_secs_f64();
        timing.insert(label, elapsed);
    };

    // --- ao, rho, vxc, fxc --- //
    let t0 = std::time::Instant::now();
    let ao = ni.get_cached_ao(get_hess_ao_deriv(xc_type));
    tic("ao", t0);

    let t0 = std::time::Instant::now();
    let ncomp_ao_dm0 = get_hess_ncomp_ao_dm0(xc_type);
    let ao_dm0a = index!(ao, ..ncomp_ao_dm0) % &dm0a;
    let ao_dm0b = index!(ao, ..ncomp_ao_dm0) % &dm0b;
    tic("ao_dm0", t0);

    let t0 = std::time::Instant::now();
    let (rhoa, rhob, vxc, fxc) = get_rho_vxc_fxc_uks(xc_func_list, ao.view(), ao_dm0a.view(), ao_dm0b.view());
    let wva = &weights * vxc.i((.., .., 0)); // [ngrids, nvar]
    let wvb = &weights * vxc.i((.., .., 1)); // [ngrids, nvar]
    let wf = &weights * &fxc; // [ngrids, nvar, 2, nvar, 2]
    tic("rho, vxc, fxc", t0);

    // --- drho --- //
    let t0 = std::time::Instant::now();
    let (drhoa, drhob) = get_drho_uks(xc_type, ao.view(), ao_dm0a.view(), ao_dm0b.view(), &aoslices);
    tic("drho", t0);

    // --- de_fxc --- //
    let t0 = std::time::Instant::now();
    let de_fxc = get_de_fxc_uks(wf.view(), drhoa.view(), drhob.view());
    tic("de_fxc", t0);

    // --- de_vxc_diag (per spin) --- //
    let t0 = std::time::Instant::now();
    let de_vxc_diag_a = get_de_vxc_diag(xc_type, ao.view(), ao_dm0a.view(), wva.view(), &aoslices);
    let de_vxc_diag_b = get_de_vxc_diag(xc_type, ao.view(), ao_dm0b.view(), wvb.view(), &aoslices);
    tic("de_vxc_diag", t0);

    // --- de_vxc_off (per spin) --- //
    let t0 = std::time::Instant::now();
    let de_vxc_off_a = get_de_vxc_off(xc_type, ao.view(), dm0a.view(), wva.view(), &aoslices);
    let de_vxc_off_b = get_de_vxc_off(xc_type, ao.view(), dm0b.view(), wvb.view(), &aoslices);
    tic("de_vxc_off", t0);

    // --- vmat_ip (per spin) --- //
    let t0 = std::time::Instant::now();
    let vmat_ip_a = get_vmat_ip(xc_type, ao.view(), wva.view());
    let vmat_ip_b = get_vmat_ip(xc_type, ao.view(), wvb.view());
    tic("vmat_ip", t0);

    // --- vmat_deriv1 (UKS spin-coupled) --- //
    let t0 = std::time::Instant::now();
    let (vmat_deriv1_a, vmat_deriv1_b) = get_vmat_deriv1_uks(
        xc_type,
        ao.view(),
        drhoa.view(),
        drhob.view(),
        wf.view(),
        vmat_ip_a.view(),
        vmat_ip_b.view(),
        &aoslices,
    );
    tic("vmat_deriv1", t0);

    let result = HashMap::from([
        ("rhoa", rhoa),
        ("rhob", rhob),
        ("vxc", vxc),
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag_a", de_vxc_diag_a),
        ("de_vxc_diag_b", de_vxc_diag_b),
        ("de_vxc_off_a", de_vxc_off_a),
        ("de_vxc_off_b", de_vxc_off_b),
        ("vmat_ip_a", vmat_ip_a),
        ("vmat_ip_b", vmat_ip_b),
        ("vmat_deriv1_a", vmat_deriv1_a),
        ("vmat_deriv1_b", vmat_deriv1_b),
    ]);
    (result, timing)
}

/* #endregion */

/* #region response */

pub fn get_uks_response_bra(
    ni: &mut NIMatmul,
    den_type: XCDenType,
    fxc_eff: TsrView,
    bra: &[TsrView; 2],
    mocc: &[TsrView; 2],
) -> ([Tsr; 2], IndexMap<&'static str, f64>) {
    let [α, β] = [0, 1];
    let nao = bra[α].shape()[0];
    let nocc_a = bra[α].shape()[1];
    let nocc_b = bra[β].shape()[1];
    let bra_a_shape = bra[α].shape().to_vec();
    let bra_b_shape = bra[β].shape().to_vec();
    let bra_a = bra[α].reshape((nao, nocc_a, -1));
    let bra_b = bra[β].reshape((nao, nocc_b, -1));
    let nset = bra_a.shape()[2];

    let mut timing = IndexMap::new();
    let mut tic = |label: &'static str, t0: std::time::Instant| {
        let elapsed = t0.elapsed().as_secs_f64();
        timing.insert(label, elapsed);
    };

    let t0 = std::time::Instant::now();
    ni.get_cached_ao(den_type.num_ao_deriv());
    tic("ao", t0);

    // Compute per-spin rho1
    let t0 = std::time::Instant::now();
    let bra_a_list = bra_a.axes_iter(-1).collect_vec();
    let bra_b_list = bra_b.axes_iter(-1).collect_vec();
    let rho1a = ni.make_rho_from_one_bra_mult_ket(mocc[α].view(), &bra_a_list, den_type);
    let rho1b = ni.make_rho_from_one_bra_mult_ket(mocc[β].view(), &bra_b_list, den_type);
    // Stack into [ngrids, nvar, 2, nset]
    let ngrids = rho1a.shape()[0];
    let nvar = den_type.num_nvar();
    let device = rho1a.device().clone();
    let mut rho1 = rt::zeros(([ngrids, nvar, 2, nset], &device));
    rho1.i_mut((.., .., 0, ..)).assign(&rho1a);
    rho1.i_mut((.., .., 1, ..)).assign(&rho1b);
    tic("rho1", t0);

    // Compute UKS fxc bra-trans response
    let t0 = std::time::Instant::now();
    let resp = ni.make_uks_fxc_pot_with_eff_bra_trans(fxc_eff, rho1.view(), mocc, den_type);
    tic("resp", t0);

    // UKS CPHF factor: 2.0 (hermitian symmetry only, no spin degeneracy)
    let [resp_a, resp_b] = resp;
    let resp_a = 2.0 * resp_a.into_shape(bra_a_shape);
    let resp_b = 2.0 * resp_b.into_shape(bra_b_shape);
    ([resp_a, resp_b], timing)
}

/* #endregion */

/* #region parallel/batch wrapper */

pub fn make_hessian_setup_batched_uks(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0a: TsrView,
    dm0b: TsrView,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    let ngrids = ni.weights.len();
    let nbatch = ni.nbatch;
    let nchunk = ni.nchunk;
    let device = dm0a.device().clone();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
    let nvar = xc_type.num_nvar();
    let deriv_level = get_hess_ao_deriv(xc_type);
    let natm = atm_list.map_or_else(|| mol.natm(), |lst| lst.len());
    let nao = mol.nao();

    let rhoa: Tsr = rt::zeros(([ngrids, nvar], &device));
    let rhob: Tsr = rt::zeros(([ngrids, nvar], &device));
    let vxc: Tsr = rt::zeros(([ngrids, nvar, 2], &device));
    let fxc: Tsr = rt::zeros(([ngrids, nvar, 2, nvar, 2], &device));
    let de_fxc: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag_a: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag_b: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off_a: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off_b: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_ip_a: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_ip_b: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_deriv1_a: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_deriv1_b: Tsr = rt::zeros(([nao, nao, 3, natm], &device));

    let timing = Arc::new(Mutex::new(IndexMap::from([
        ("ao", 0.0),
        ("ao_dm0", 0.0),
        ("rho, vxc, fxc", 0.0),
        ("drho", 0.0),
        ("de_fxc", 0.0),
        ("de_vxc_diag", 0.0),
        ("de_vxc_off", 0.0),
        ("vmat_ip", 0.0),
        ("vmat_deriv1", 0.0),
        ("total", 0.0),
    ])));
    let time_total = std::time::Instant::now();
    let guard = Mutex::new(());

    for start_batch in (0..ngrids).step_by(nbatch) {
        let end_batch = (start_batch + nbatch).min(ngrids);

        let t0 = std::time::Instant::now();
        let mut ni_batch = ni.split_batch(start_batch, end_batch);
        ni_batch.get_cached_ao(deriv_level);
        {
            let mut timing = timing.lock().unwrap();
            timing["ao"] += t0.elapsed().as_secs_f64();
        }

        (start_batch..end_batch).into_par_iter().step_by(nchunk).for_each(|start| {
            let end = (start + nchunk).min(end_batch);
            let mut ni_chunk = ni_batch.split_batch(start - start_batch, end - start_batch);
            let (result_chunk, timing_chunk) =
                make_hessian_setup_uks(mol, xc_func_list, &mut ni_chunk, dm0a.view(), dm0b.view(), atm_list);

            unsafe {
                let rhoa_slc = rhoa.i(start..end);
                let rhob_slc = rhob.i(start..end);
                let vxc_slc = vxc.i(start..end);
                let fxc_slc = fxc.i(start..end);
                let mut rhoa_slc = rhoa_slc.force_mut();
                let mut rhob_slc = rhob_slc.force_mut();
                let mut vxc_slc = vxc_slc.force_mut();
                let mut fxc_slc = fxc_slc.force_mut();
                rhoa_slc.assign(&result_chunk["rhoa"]);
                rhob_slc.assign(&result_chunk["rhob"]);
                vxc_slc.assign(&result_chunk["vxc"]);
                fxc_slc.assign(&result_chunk["fxc"]);
            }
            unsafe {
                let lock = guard.lock().unwrap();
                *&mut de_fxc.force_mut() += &result_chunk["de_fxc"];
                *&mut de_vxc_diag_a.force_mut() += &result_chunk["de_vxc_diag_a"];
                *&mut de_vxc_diag_b.force_mut() += &result_chunk["de_vxc_diag_b"];
                *&mut de_vxc_off_a.force_mut() += &result_chunk["de_vxc_off_a"];
                *&mut de_vxc_off_b.force_mut() += &result_chunk["de_vxc_off_b"];
                *&mut vmat_ip_a.force_mut() += &result_chunk["vmat_ip_a"];
                *&mut vmat_ip_b.force_mut() += &result_chunk["vmat_ip_b"];
                *&mut vmat_deriv1_a.force_mut() += &result_chunk["vmat_deriv1_a"];
                *&mut vmat_deriv1_b.force_mut() += &result_chunk["vmat_deriv1_b"];
                drop(lock);
            }
            {
                let mut timing = timing.lock().unwrap();
                for (key, value) in timing_chunk {
                    *timing.get_mut(key).unwrap() += value;
                }
            }
        });

        {
            let mut timing = timing.lock().unwrap();
            timing.insert("total", time_total.elapsed().as_secs_f64());
        }
        if verbose {
            let timing = timing.lock().unwrap();
            println!("In make_hessian_setup_batched_uks, Batch {start_batch}..{end_batch}");
            println!("  Elapsed time from start (Wall time): {:.4} sec", timing["total"]);
        }
    }

    let timing = timing.lock().unwrap();
    if verbose {
        println!("Finished make_hessian_setup_batched_uks");
        println!("  Total elapsed time (Wall time): {:.4} sec", timing["total"]);
        println!("  Timing breakdown:");
        for (key, value) in timing.iter() {
            if *key != "total" {
                println!("  {key:>20}: {value:.4} sec");
            }
        }
    }

    let result = HashMap::from([
        ("rhoa", rhoa),
        ("rhob", rhob),
        ("vxc", vxc),
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag_a", de_vxc_diag_a),
        ("de_vxc_diag_b", de_vxc_diag_b),
        ("de_vxc_off_a", de_vxc_off_a),
        ("de_vxc_off_b", de_vxc_off_b),
        ("vmat_ip_a", vmat_ip_a),
        ("vmat_ip_b", vmat_ip_b),
        ("vmat_deriv1_a", vmat_deriv1_a),
        ("vmat_deriv1_b", vmat_deriv1_b),
    ]);

    (result, timing.clone())
}

pub fn get_uks_response_bra_batched(
    ni: &mut NIMatmul,
    den_type: XCDenType,
    fxc_eff: TsrView,
    bra: &[TsrView; 2],
    mocc: &[TsrView; 2],
    verbose: bool,
) -> ([Tsr; 2], IndexMap<&'static str, f64>) {
    let ngrids = ni.weights.len();
    let nbatch = ni.nbatch;
    let bra_a_shape = bra[0].shape().to_vec();
    let bra_b_shape = bra[1].shape().to_vec();
    let device = bra[0].device().clone();
    let mut resp_a = rt::zeros((bra_a_shape, &device));
    let mut resp_b = rt::zeros((bra_b_shape, &device));
    let mut timing = IndexMap::from([("ao", 0.0), ("rho1", 0.0), ("resp", 0.0), ("total", 0.0)]);

    let t0 = std::time::Instant::now();
    for start in (0..ngrids).step_by(nbatch) {
        let end = (start + nbatch).min(ngrids);
        let mut ni_batch = ni.split_batch(start, end);
        let ([resp_batch_a, resp_batch_b], timing_batch) =
            get_uks_response_bra(&mut ni_batch, den_type, fxc_eff.i(start..end), bra, mocc);
        resp_a += resp_batch_a;
        resp_b += resp_batch_b;
        for (key, value) in timing_batch {
            *timing.get_mut(key).unwrap() += value;
        }
        let duration = t0.elapsed().as_secs_f64();
        timing.insert("total", duration);
        if verbose {
            println!("In get_uks_response_bra_batched, Batch {start}..{end}");
            println!("  Elapsed time from start (Wall time): {:.4} sec", duration);
        }
    }

    if verbose {
        println!("Finished get_uks_response_bra_batched");
        println!("  Total elapsed time (Wall time): {:.4} sec", timing["total"]);
    }

    ([resp_a, resp_b], timing)
}

/* #endregion */

/* #region final implementation of UKS Hessian */

pub struct UHessKSNIMatmul<'a> {
    pub mol: CInt,
    pub xc_func_list: &'a [(f64, LibXCFunctional)],
    pub ni: NIMatmul<'a>,
    pub ni_cpks: Option<NIMatmul<'a>>,
    pub verbose: bool,
    pub intmd: HashMap<String, Tsr>,
    pub result: HashMap<String, Tsr>,
}

impl<'a> UHessKSNIMatmul<'a> {
    pub fn new(mol: &CInt, xc_func_list: &'a [(f64, LibXCFunctional)], ni: NIMatmul<'a>, verbose: bool) -> Self {
        Self {
            mol: mol.clone(),
            xc_func_list,
            ni,
            ni_cpks: None,
            verbose,
            intmd: HashMap::new(),
            result: HashMap::new(),
        }
    }

    pub fn make_hessian_setup(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2], atm_list: Option<&[usize]>) {
        let [α, β] = [0, 1];
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let mocc_a = mo_coeff[α].bool_select(-1, &occidx[α]);
        let mocc_b = mo_coeff[β].bool_select(-1, &occidx[β]);
        let dm0a = &mocc_a % mocc_a.t();
        let dm0b = &mocc_b % mocc_b.t();

        let (result, _timing) = make_hessian_setup_batched_uks(
            &self.mol,
            self.xc_func_list,
            &mut self.ni,
            dm0a.view(),
            dm0b.view(),
            atm_list,
            self.verbose,
        );

        for (key, val) in result.into_iter() {
            if key == "vxc" || key == "fxc" {
                if self.ni_cpks.is_none() {
                    let key_to_store = format!("cpks_{key}");
                    self.intmd.insert(key_to_store, val);
                }
            } else if ["rhoa", "rhob"].contains(&key) {
                continue;
            } else {
                self.intmd.insert(key.to_string(), val);
            }
        }
    }

    pub fn is_hessian_setup_done(&self) -> bool {
        self.intmd.contains_key("de_fxc")
    }
}

impl<'a> HessUtilAPI for UHessKSNIMatmul<'a> {}

impl<'a> UHessElecInteractAPI for UHessKSNIMatmul<'a> {
    fn make_skeleton_hess(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> Tsr {
        if !self.is_hessian_setup_done() {
            self.make_hessian_setup(mo_coeff, mo_occ, atm_list);
        }
        &self.intmd["de_fxc"]
            + &self.intmd["de_vxc_diag_a"]
            + &self.intmd["de_vxc_off_a"]
            + &self.intmd["de_vxc_diag_b"]
            + &self.intmd["de_vxc_off_b"]
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
        [self.intmd["vmat_deriv1_a"].to_owned(), self.intmd["vmat_deriv1_b"].to_owned()]
    }

    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) {
        self.intmd.insert("mo_coeff_0".to_string(), mo_coeff[0].to_owned().into_contig(ColMajor));
        self.intmd.insert("mo_coeff_1".to_string(), mo_coeff[1].to_owned().into_contig(ColMajor));
        self.intmd.insert("mo_occ_0".to_string(), mo_occ[0].to_owned().into_contig(ColMajor));
        self.intmd.insert("mo_occ_1".to_string(), mo_occ[1].to_owned().into_contig(ColMajor));
    }

    fn get_response_bra(&mut self, bra: &[TsrView; 2]) -> [Tsr; 2] {
        let ni_cpks = self.ni_cpks.as_mut().unwrap_or(&mut self.ni);
        let mo_coeff_0 = self.intmd.get("mo_coeff_0").unwrap();
        let mo_coeff_1 = self.intmd.get("mo_coeff_1").unwrap();
        let mo_occ_0 = self.intmd.get("mo_occ_0").unwrap();
        let mo_occ_1 = self.intmd.get("mo_occ_1").unwrap();
        let fxc_eff = self.intmd.get("cpks_fxc").unwrap();

        let occidx_a = mo_occ_0.view().greater(0).into_vec();
        let occidx_b = mo_occ_1.view().greater(0).into_vec();
        let mocc_a = mo_coeff_0.bool_select(-1, &occidx_a);
        let mocc_b = mo_coeff_1.bool_select(-1, &occidx_b);

        let den_type = determine_den_type_from_list(&self.xc_func_list.iter().map(|(_, f)| f).collect_vec());

        let ([resp_a, resp_b], _timing) = get_uks_response_bra_batched(
            ni_cpks,
            den_type,
            fxc_eff.view(),
            bra,
            &[mocc_a.view(), mocc_b.view()],
            self.verbose,
        );
        [resp_a, resp_b]
    }
}

/* #endregion */
