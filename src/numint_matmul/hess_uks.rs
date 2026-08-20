// see also pyhessref/nimatmul/uks.py

use super::hess_rks::{get_de_vxc_diag, get_de_vxc_off, get_drho, get_vmat_ip, get_vmat_vxc};
use super::prelude::*;

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
    ao_dm0α: TsrView,
    ao_dm0β: TsrView,
) -> (Tsr, Tsr, Tsr) {
    // Returns: (rhoα, rhoβ, vxc, fxc)
    // rhoα, rhoβ: [ngrids, nvar]
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

    // Compute rho
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

    // Evaluate vxc and fxc with spin-polarized libxc
    let mut vxc = rt::zeros(([ngrids, nvar, 2], &device));
    let mut fxc = rt::zeros(([ngrids, nvar, 2, nvar, 2], &device));
    for (scale, xc_func) in xc_func_list {
        let xc_type_i = determine_den_type(xc_func);
        let nvar_i = xc_type_i.num_nvar();
        let rho_i = rho.i((.., ..nvar_i, ..));
        let xc_eff = libxc_eval_eff(xc_func, rho_i, 2, false);
        let [_, vxc_i, fxc_i] = xc_eff.into_iter().collect_array().unwrap();
        *&mut vxc.i_mut((.., ..nvar_i, ..)) += *scale * vxc_i;
        *&mut fxc.i_mut((.., ..nvar_i, .., ..nvar_i, ..)) += *scale * fxc_i;
    }

    (rho, vxc, fxc)
}

pub fn get_drho_uks(
    xc_type: XCDenType,
    ao: TsrView,
    ao_dm0α: TsrView,
    ao_dm0β: TsrView,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    let drhoα = get_drho(xc_type, ao.view(), ao_dm0α.view(), aoslices);
    let drhoβ = get_drho(xc_type, ao.view(), ao_dm0β.view(), aoslices);
    (drhoα, drhoβ)
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

pub fn get_de_fxc_uks(wf: TsrView, drhoα: TsrView, drhoβ: TsrView) -> Tsr {
    // wf: [ngrids, nvar, 2, nvar, 2]
    // drhoα, drhoβ: [ngrids, nvar, 3, natm]
    // result: [3, 3, natm, natm]

    let de_αα = get_de_fxc_uks_inner(wf.i((.., .., α, .., α)), drhoα.view(), drhoα.view());
    let de_αβ = get_de_fxc_uks_inner(wf.i((.., .., α, .., β)), drhoα.view(), drhoβ.view());
    let de_βα = get_de_fxc_uks_inner(wf.i((.., .., β, .., α)), drhoβ.view(), drhoα.view());
    let de_ββ = get_de_fxc_uks_inner(wf.i((.., .., β, .., β)), drhoβ.view(), drhoβ.view());

    &de_αα + &de_αβ + &de_βα + &de_ββ
}

#[allow(clippy::too_many_arguments)]
pub fn get_vmat_fxc_uks(
    xc_type: XCDenType,
    ao: TsrView,
    drhoα: TsrView,
    drhoβ: TsrView,
    wf: TsrView,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    // see also pyhessref/nimatmul/uks.py, function `_vmat_fxc_uks`
    //
    // fxc contribution to the per-atom skeleton derivative of the Vxc Fock
    // matrix for UKS - the spin-coupled part.  Unlike the spin-diagonal
    // `get_vmat_vxc` (reused from RKS per spin), the fxc contraction here
    // couples the two spin channels:
    //   wvα_f = wf_αα @ drho_α + wf_βα @ drho_β
    //   wvβ_f = wf_αβ @ drho_α + wf_ββ @ drho_β

    let natm = aoslices.len();
    let nao = ao.shape()[1];
    let device = ao.device();

    let mut vmatα_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], device));
    let mut vmatβ_fxc: Tsr = rt::zeros(([nao, nao, 3, natm], device));

    for A in 0..natm {
        if matches!(xc_type, RHO) {
            // LDA: fxc is [G, 1, 2, 1, 2], extract scalar spin blocks
            // Alpha output (s2=α): fxc_αα @ drho_α + fxc_βα @ drho_β
            let wf_αα_00: Tsr = 0.5 * wf.i((.., O, α, O, α)); // [G], s1=α, s2=α
            let wf_βα_00: Tsr = 0.5 * wf.i((.., O, β, O, α)); // [G], s1=β, s2=α

            let wvα_f: Tsr =
                wf_αα_00.i((.., None)) * drhoα.i((.., O, .., A)) + wf_βα_00.i((.., None)) * drhoβ.i((.., O, .., A));

            // Beta output (s2=β): fxc_αβ @ drho_α + fxc_ββ @ drho_β
            let wf_αβ_00: Tsr = 0.5 * wf.i((.., O, α, O, β)); // [G], s1=α, s2=β
            let wf_ββ_00: Tsr = 0.5 * wf.i((.., O, β, O, β)); // [G], s1=β, s2=β

            let wvβ_f: Tsr =
                wf_αβ_00.i((.., None)) * drhoα.i((.., O, .., A)) + wf_ββ_00.i((.., None)) * drhoβ.i((.., O, .., A));

            for t in 0..3 {
                let aowα = wvα_f.i((.., t)) * index!(ao, O);
                index_mut!(vmatα_fxc, t, A).matmul_from(aowα.t(), index!(ao, O), 1.0, 1.0);
                let aowβ = wvβ_f.i((.., t)) * index!(ao, O);
                index_mut!(vmatβ_fxc, t, A).matmul_from(aowβ.t(), index!(ao, O), 1.0, 1.0);
            }
        }

        if matches!(xc_type, SIGMA | TAU) {
            let wf_αα = wf.i((.., .., α, .., α)); // [G, x, y]
            let wf_αβ = wf.i((.., .., α, .., β)); // [G, x, y]
            let wf_βα = wf.i((.., .., β, .., α)); // [G, x, y]
            let wf_ββ = wf.i((.., .., β, .., β)); // [G, x, y]

            let drhoα_A = drhoα.i((.., .., .., A)); // [G, x, 3]
            let drhoβ_A = drhoβ.i((.., .., .., A)); // [G, x, 3]

            // Python: wvα_f[y,t,g] = sum_x wf_αα[x,y,g]*drhoα[A,t,x,g] + wf_βα[x,y,g]*drhoβ[A,t,x,g]
            // For each direction t, compute per-grid contraction:
            //   wvα_f_t[g, y] = wf_αα[g, :, y]^T @ drhoα_A[g, :, t] + wf_βα[g, :, y]^T @ drhoβ_A[g, :, t]

            for t in 0..3 {
                // drhoα_A[:,:,t] shape: [G, x], reshape → [G, x, 1]
                let drhoα_t = drhoα_A.i((.., .., t));
                let drhoβ_t = drhoβ_A.i((.., .., t));

                // vecdot on axis 1: [G, x, y] @ [G, x, 1] → contract x
                // Remaining: [G, y] and [G, 1] → col-major broadcast → [G, y]
                // Alpha output (s2=α): fxc_αα @ drho_α + fxc_βα @ drho_β
                // Beta output (s2=β): fxc_αβ @ drho_α + fxc_ββ @ drho_β
                let wf_rho_αα = rt::vecdot(wf_αα.view(), drhoα_t.view(), 1);
                let wf_rho_βα = rt::vecdot(wf_βα.view(), drhoβ_t.view(), 1);
                let wf_rho_αβ = rt::vecdot(wf_αβ.view(), drhoα_t.view(), 1);
                let wf_rho_ββ = rt::vecdot(wf_ββ.view(), drhoβ_t.view(), 1);
                let mut wf_rho_α = &wf_rho_αα + &wf_rho_βα; // [G, y]
                let mut wf_rho_β = &wf_rho_αβ + &wf_rho_ββ; // [G, y]

                *&mut wf_rho_α.i_mut((.., 0)) *= 0.5;
                *&mut wf_rho_β.i_mut((.., 0)) *= 0.5;
                if matches!(xc_type, TAU) {
                    *&mut wf_rho_α.i_mut((.., 4)) *= 0.25;
                    *&mut wf_rho_β.i_mut((.., 4)) *= 0.25;
                }

                // Contract with ao: aow = sum_c wvα_f_t[:,c] * ao[c]
                for c in 0..4 {
                    let aowα = wf_rho_α.i((.., c)) * index!(ao, c); // [G, nao]
                    index_mut!(vmatα_fxc, t, A).matmul_from(aowα.t(), index!(ao, O), 1.0, 1.0);
                    let aowβ = wf_rho_β.i((.., c)) * index!(ao, c); // [G, nao]
                    index_mut!(vmatβ_fxc, t, A).matmul_from(aowβ.t(), index!(ao, O), 1.0, 1.0);
                }

                if matches!(xc_type, TAU) {
                    for r in [X, Y, Z] {
                        let aowα = wf_rho_α.i((.., 4)) * index!(ao, r); // [G, nao]
                        index_mut!(vmatα_fxc, t, A).matmul_from(aowα.t(), index!(ao, r), 1.0, 1.0);
                        let aowβ = wf_rho_β.i((.., 4)) * index!(ao, r); // [G, nao]
                        index_mut!(vmatβ_fxc, t, A).matmul_from(aowβ.t(), index!(ao, r), 1.0, 1.0);
                    }
                }
            }
        }
    }

    let vmatα_fxc = &vmatα_fxc + vmatα_fxc.swapaxes(0, 1);
    let vmatβ_fxc = &vmatβ_fxc + vmatβ_fxc.swapaxes(0, 1);

    (vmatα_fxc, vmatβ_fxc)
}

#[allow(clippy::too_many_arguments)]
pub fn get_vmat_deriv1_uks(
    xc_type: XCDenType,
    ao: TsrView,
    drhoα: TsrView,
    drhoβ: TsrView,
    wf: TsrView,
    vmat_ip_α: TsrView,
    vmat_ip_β: TsrView,
    aoslices: &[[usize; 4]],
) -> (Tsr, Tsr) {
    // see also pyhessref/nimatmul/uks.py, function `_vmat_deriv1_uks`
    //
    // Split into a per-spin vxc contribution (the ipip basis-derivative part,
    // reused from the RKS `get_vmat_vxc`) and a spin-coupled fxc contribution
    // (`get_vmat_fxc_uks`).  Each is assembled independently and summed per spin;
    // the split is exact up to floating-point order (same as the RKS split).

    let (vmatα_fxc, vmatβ_fxc) = get_vmat_fxc_uks(xc_type, ao, drhoα, drhoβ, wf, aoslices);
    let vmatα_vxc = get_vmat_vxc(vmat_ip_α, aoslices);
    let vmatβ_vxc = get_vmat_vxc(vmat_ip_β, aoslices);
    (&vmatα_fxc + &vmatα_vxc, &vmatβ_fxc + &vmatβ_vxc)
}

pub fn make_hessian_setup_uks(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0α: TsrView,
    dm0β: TsrView,
    atm_list: Option<&[usize]>,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    assert!(!xc_func_list.is_empty(), "xc_func_list must not be empty");
    let atm_list = atm_list.map_or_else(|| (0..mol.natm()).collect_vec(), |lst| lst.to_vec());
    let aoslices_full = mol.aoslice_by_atom();
    let aoslices = atm_list.iter().map(|&iatm| aoslices_full[iatm]).collect_vec();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());

    let device = dm0α.device().clone();
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
    let ao_dm0α = index!(ao, ..ncomp_ao_dm0) % &dm0α;
    let ao_dm0β = index!(ao, ..ncomp_ao_dm0) % &dm0β;
    tic("ao_dm0", t0);

    let t0 = std::time::Instant::now();
    let (rho, vxc, fxc) = get_rho_vxc_fxc_uks(xc_func_list, ao.view(), ao_dm0α.view(), ao_dm0β.view());
    let wvα = &weights * vxc.i((.., .., α)); // [ngrids, nvar]
    let wvβ = &weights * vxc.i((.., .., β)); // [ngrids, nvar]
    let wf = &weights * &fxc; // [ngrids, nvar, 2, nvar, 2]
    tic("rho, vxc, fxc", t0);

    // --- drho --- //
    let t0 = std::time::Instant::now();
    let (drhoα, drhoβ) = get_drho_uks(xc_type, ao.view(), ao_dm0α.view(), ao_dm0β.view(), &aoslices);
    tic("drho", t0);

    // --- de_fxc --- //
    let t0 = std::time::Instant::now();
    let de_fxc = get_de_fxc_uks(wf.view(), drhoα.view(), drhoβ.view());
    tic("de_fxc", t0);

    // --- de_vxc_diag (per spin) --- //
    let t0 = std::time::Instant::now();
    let de_vxc_diag_α = get_de_vxc_diag(xc_type, ao.view(), ao_dm0α.view(), wvα.view(), &aoslices);
    let de_vxc_diag_β = get_de_vxc_diag(xc_type, ao.view(), ao_dm0β.view(), wvβ.view(), &aoslices);
    tic("de_vxc_diag", t0);

    // --- de_vxc_off (per spin) --- //
    let t0 = std::time::Instant::now();
    let de_vxc_off_α = get_de_vxc_off(xc_type, ao.view(), dm0α.view(), wvα.view(), &aoslices);
    let de_vxc_off_β = get_de_vxc_off(xc_type, ao.view(), dm0β.view(), wvβ.view(), &aoslices);
    tic("de_vxc_off", t0);

    // --- vmat_ip (per spin) --- //
    let t0 = std::time::Instant::now();
    let vmat_ip_α = get_vmat_ip(xc_type, ao.view(), wvα.view());
    let vmat_ip_β = get_vmat_ip(xc_type, ao.view(), wvβ.view());
    tic("vmat_ip", t0);

    // --- vmat_deriv1 (UKS spin-coupled) --- //
    let t0 = std::time::Instant::now();
    let (vmat_deriv1_α, vmat_deriv1_β) = get_vmat_deriv1_uks(
        xc_type,
        ao.view(),
        drhoα.view(),
        drhoβ.view(),
        wf.view(),
        vmat_ip_α.view(),
        vmat_ip_β.view(),
        &aoslices,
    );
    tic("vmat_deriv1", t0);

    let result = HashMap::from([
        ("rho", rho),
        ("vxc", vxc),
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag_a", de_vxc_diag_α),
        ("de_vxc_diag_b", de_vxc_diag_β),
        ("de_vxc_off_a", de_vxc_off_α),
        ("de_vxc_off_b", de_vxc_off_β),
        ("vmat_ip_a", vmat_ip_α),
        ("vmat_ip_b", vmat_ip_β),
        ("vmat_deriv1_a", vmat_deriv1_α),
        ("vmat_deriv1_b", vmat_deriv1_β),
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
    let nao = bra[α].shape()[0];
    let nocc_α = bra[α].shape()[1];
    let nocc_β = bra[β].shape()[1];
    let bra_α_shape = bra[α].shape().to_vec();
    let bra_β_shape = bra[β].shape().to_vec();
    let bra_α = bra[α].reshape((nao, nocc_α, -1));
    let bra_β = bra[β].reshape((nao, nocc_β, -1));
    let nset = bra_α.shape()[2];

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
    let bra_α_list = bra_α.axes_iter(-1).collect_vec();
    let bra_β_list = bra_β.axes_iter(-1).collect_vec();
    let rho1α = ni.make_rho_from_one_bra_mult_ket(mocc[α].view(), &bra_α_list, den_type);
    let rho1β = ni.make_rho_from_one_bra_mult_ket(mocc[β].view(), &bra_β_list, den_type);
    // Stack into [ngrids, nvar, 2, nset]
    let ngrids = rho1α.shape()[0];
    let nvar = den_type.num_nvar();
    let device = rho1α.device().clone();
    let mut rho1 = rt::zeros(([ngrids, nvar, 2, nset], &device));
    rho1.i_mut((.., .., α, ..)).assign(&rho1α);
    rho1.i_mut((.., .., β, ..)).assign(&rho1β);
    tic("rho1", t0);

    // Compute UKS fxc bra-trans response
    let t0 = std::time::Instant::now();
    let resp = ni.make_uks_fxc_pot_with_eff_bra_trans(fxc_eff, rho1.view(), mocc, den_type);
    tic("resp", t0);

    // UKS CPHF factor: 2.0 (hermitian symmetry only, no spin degeneracy)
    let [resp_α, resp_β] = resp;
    let resp_α = 2.0 * resp_α.into_shape(bra_α_shape);
    let resp_β = 2.0 * resp_β.into_shape(bra_β_shape);
    ([resp_α, resp_β], timing)
}

/* #endregion */

/* #region parallel/batch wrapper */

pub fn make_hessian_setup_batched_uks(
    mol: &CInt,
    xc_func_list: &[(f64, LibXCFunctional)],
    ni: &mut NIMatmul,
    dm0α: TsrView,
    dm0β: TsrView,
    atm_list: Option<&[usize]>,
    verbose: bool,
) -> (HashMap<&'static str, Tsr>, IndexMap<&'static str, f64>) {
    let ngrids = ni.weights.len();
    let nbatch = ni.nbatch;
    let nchunk = ni.nchunk;
    let device = dm0α.device().clone();
    let xc_type = determine_den_type_from_list(&xc_func_list.iter().map(|(_, f)| f).collect_vec());
    let nvar = xc_type.num_nvar();
    let deriv_level = get_hess_ao_deriv(xc_type);
    let natm = atm_list.map_or_else(|| mol.natm(), |lst| lst.len());
    let nao = mol.nao();

    let rhoα: Tsr = rt::zeros(([ngrids, nvar], &device));
    let rhoβ: Tsr = rt::zeros(([ngrids, nvar], &device));
    let vxc: Tsr = rt::zeros(([ngrids, nvar, 2], &device));
    let fxc: Tsr = rt::zeros(([ngrids, nvar, 2, nvar, 2], &device));
    let de_fxc: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag_α: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_diag_β: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off_α: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let de_vxc_off_β: Tsr = rt::zeros(([3, 3, natm, natm], &device));
    let vmat_ip_α: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_ip_β: Tsr = rt::zeros(([nao, nao, 3], &device));
    let vmat_deriv1_α: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
    let vmat_deriv1_β: Tsr = rt::zeros(([nao, nao, 3, natm], &device));

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
                make_hessian_setup_uks(mol, xc_func_list, &mut ni_chunk, dm0α.view(), dm0β.view(), atm_list);

            unsafe {
                let rhoα_slc = rhoα.i(start..end);
                let rhoβ_slc = rhoβ.i(start..end);
                let vxc_slc = vxc.i(start..end);
                let fxc_slc = fxc.i(start..end);
                let mut rhoα_slc = rhoα_slc.force_mut();
                let mut rhoβ_slc = rhoβ_slc.force_mut();
                let mut vxc_slc = vxc_slc.force_mut();
                let mut fxc_slc = fxc_slc.force_mut();
                rhoα_slc.assign(&result_chunk["rho"].i((.., .., α)));
                rhoβ_slc.assign(&result_chunk["rho"].i((.., .., β)));
                vxc_slc.assign(&result_chunk["vxc"]);
                fxc_slc.assign(&result_chunk["fxc"]);
            }
            unsafe {
                let lock = guard.lock().unwrap();
                *&mut de_fxc.force_mut() += &result_chunk["de_fxc"];
                *&mut de_vxc_diag_α.force_mut() += &result_chunk["de_vxc_diag_a"];
                *&mut de_vxc_diag_β.force_mut() += &result_chunk["de_vxc_diag_b"];
                *&mut de_vxc_off_α.force_mut() += &result_chunk["de_vxc_off_a"];
                *&mut de_vxc_off_β.force_mut() += &result_chunk["de_vxc_off_b"];
                *&mut vmat_ip_α.force_mut() += &result_chunk["vmat_ip_a"];
                *&mut vmat_ip_β.force_mut() += &result_chunk["vmat_ip_b"];
                *&mut vmat_deriv1_α.force_mut() += &result_chunk["vmat_deriv1_a"];
                *&mut vmat_deriv1_β.force_mut() += &result_chunk["vmat_deriv1_b"];
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
        ("rhoa", rhoα),
        ("rhob", rhoβ),
        ("vxc", vxc),
        ("fxc", fxc),
        ("de_fxc", de_fxc),
        ("de_vxc_diag_a", de_vxc_diag_α),
        ("de_vxc_diag_b", de_vxc_diag_β),
        ("de_vxc_off_a", de_vxc_off_α),
        ("de_vxc_off_b", de_vxc_off_β),
        ("vmat_ip_a", vmat_ip_α),
        ("vmat_ip_b", vmat_ip_β),
        ("vmat_deriv1_a", vmat_deriv1_α),
        ("vmat_deriv1_b", vmat_deriv1_β),
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
    let bra_α_shape = bra[α].shape().to_vec();
    let bra_β_shape = bra[β].shape().to_vec();
    let device = bra[α].device().clone();
    let mut resp_α = rt::zeros((bra_α_shape, &device));
    let mut resp_β = rt::zeros((bra_β_shape, &device));
    let mut timing = IndexMap::from([("ao", 0.0), ("rho1", 0.0), ("resp", 0.0), ("total", 0.0)]);

    let t0 = std::time::Instant::now();
    for start in (0..ngrids).step_by(nbatch) {
        let end = (start + nbatch).min(ngrids);
        let mut ni_batch = ni.split_batch(start, end);
        let ([resp_batch_α, resp_batch_β], timing_batch) =
            get_uks_response_bra(&mut ni_batch, den_type, fxc_eff.i(start..end), bra, mocc);
        resp_α += resp_batch_α;
        resp_β += resp_batch_β;
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
        println!("  Timing breakdown:");
        for (key, value) in timing.iter() {
            if *key != "total" {
                println!("  {key:>20}: {value:.4} sec");
            }
        }
    }

    ([resp_α, resp_β], timing)
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
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let mocc_α = mo_coeff[α].bool_select(-1, &occidx[α]);
        let mocc_β = mo_coeff[β].bool_select(-1, &occidx[β]);
        let dm0α = &mocc_α % mocc_α.t();
        let dm0β = &mocc_β % mocc_β.t();

        let (result, _timing) = make_hessian_setup_batched_uks(
            &self.mol,
            self.xc_func_list,
            &mut self.ni,
            dm0α.view(),
            dm0β.view(),
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
