//! Optimized RI-JK Hessian computation.
//!
//! **NOTE** currently this is not completed.
//!
//! Algorithm is naive and of no reference.
//!
//! The correct value is compared to PySCF (written by Qiming Sun). Reference article:
//! > Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models
//! > Chen, et al. arXiv:1906.09427
//!
//! PySCF code referenced the ORCA's implementation:
//! > Efficient implementation of the analytic second derivatives of Hartree-Fock and hybrid DFT
//! > energies: a detailed analysis of different approximations
//! > Bykov, et al. Mol Phys. 113, 1961 (2015). DOI: 10.1080/00268976.2015.1025114

use crate::prelude::*;
#[allow(unused_imports)]
use FlagSide::L as Left;
#[allow(unused_imports)]
use FlagSide::R as Right;

use crate::ri_jk::decompose::*;
use crate::ri_jk::pure_decompose::{get_j2c_decomp, solve_by_j2c, solve_by_j2c_mut};

#[allow(clippy::too_many_arguments)]
pub fn get_rijk_response_bra(
    cderi: TsrView,
    mo_coeff: TsrView,
    mo_occ: TsrView,
    bra1: TsrView,
    scale_j: f64,
    scale_k: f64,
    nbatch_aux: usize,
) -> Tsr {
    // notes on shape
    // - cderi: [nao_tp, naux]
    // - mo_coeff: [nao, nmo]
    // - mo_occ: [nmo]
    // - bra: [nao, nocc, ...]  (use nprop in program, and same to output shape)

    // derived shapes
    // - mocc: [nao, nocc]
    // - oxb: [nocc, naux, nao] (occupied, auxiliary, basis)
    // - oxo: [nocc, naux, nocc]
    // - ......

    // preparation
    let nao = mo_coeff.shape()[0];
    let naux = cderi.shape()[1];
    let nao_tp = nao * (nao + 1) / 2;
    assert_eq!(cderi.shape()[0], nao_tp);
    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let nocc = occidx.iter().filter(|&&x| x).count();

    // reshape bra
    let shape_bra = bra1.shape();
    assert_eq!(bra1.shape()[0], nao);
    assert_eq!(bra1.shape()[1], nocc);
    let bra1 = bra1.reshape((nao, nocc, -1));
    let nprop = bra1.shape()[2];
    let mut resp = rt::zeros_like(&bra1);

    // --- J contribution --- //

    if scale_j != 0.0 {
        // - mo1_dm: [nao, nao, nprop]
        // - mp1_dm_tp: [nao_tp, nprop]
        // - itm_j_aux: [naux, nprop]
        // - resp_tp_j: [nao_tp, nprop]
        // - resp_ao_j: [nao, nao, nprop]
        let dm1 = &bra1 % &mocc.t();
        let dm1 = &dm1 + &dm1.swapaxes(0, 1);
        let dm1_tp = pack_triu_tilde(dm1.view());
        let itm_j_aux = cderi.t() % &dm1_tp;
        let resp_tp_j: Tsr = 2.0 * &cderi % itm_j_aux;
        let resp_ao_j = resp_tp_j.unpack_tri(Upper, FlagSymm::Sy);
        let resp_bra_j = scale_j * (resp_ao_j % &mocc);
        resp += resp_bra_j;
    }

    // --- K contribution --- //

    if scale_k != 0.0 {
        let mut resp_bra_k = rt::zeros_like(&resp);
        for iaux_start in (0..naux).step_by(nbatch_aux) {
            let iaux_end = (iaux_start + nbatch_aux).min(naux);
            let slc = rt::slice!(iaux_start, iaux_end);
            // Please note following `naux` is actually in batch, just overwriting the outside `naux` for
            // convenience.
            let naux = iaux_end - iaux_start;

            // - cderi: [nao, nao, naux]
            // - cderi_bxo: [nao, naux, nocc]
            // - cderi_oxo: [nocc, naux, nocc]
            // - cderi_box: [nao, nocc, naux]
            let cderi = cderi.i((.., slc)).unpack_tri(Upper, FlagSymm::Sy);
            let cderi_bxo = (cderi.reshape([nao, nao * naux]).t() % &mocc).into_shape([nao, naux, nocc]);
            let cderi_oxo = (&mocc.t() % cderi_bxo.reshape([nao, naux * nocc])).into_shape([nocc, naux, nocc]);

            for A in 0..nprop {
                let bra1A = bra1.i((.., .., A));
                let mut respkA = resp_bra_k.i_mut((.., .., A));
                // k contribution part 0
                // - cderi_bxo_1: [nao, naux, nocc]
                // - einsum progress: uPj, iPj -> ui
                let cderi_bxo_1 = (cderi.reshape([nao, nao * naux]).t() % &bra1A).into_shape([nao, naux, nocc]);
                respkA -= cderi_bxo_1.reshape([nao, naux * nocc]) % cderi_oxo.reshape([nocc, naux * nocc]).t();
                // k contribution part 1
                // - cderi_oxo_1: [nocc, naux, nocc] (note it is iPj, where j from bra1, i from mocc)
                // - einsum progress: uPj, iPj -> ui
                let cderi_oxo_1 = (mocc.t() % cderi_bxo_1.reshape([nao, naux * nocc])).into_shape([nocc, naux, nocc]);
                respkA -= cderi_bxo.reshape([nao, naux * nocc]) % cderi_oxo_1.reshape([nocc, naux * nocc]).t();
            }
        }
        resp += scale_k * resp_bra_k;
    }

    resp.into_shape(shape_bra)
}

/* #region skeleton */

macro_rules! tic {
    ($timing:expr, $t0:expr, $msg:expr) => {
        let t1 = std::time::Instant::now();
        let dt = t1.duration_since($t0).as_secs_f64();
        $timing.push(($msg.to_string(), dt));
    };
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn get_rijk_skeleton_decomposed_separated(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: &[TsrView],
    mo_occ: &[TsrView],
    cderi: TsrView,
    j2c_decomp: &J2CDecompose,
    do_j: bool,
    do_k: bool,
    nbatch_aux: usize,
    atm_list: Option<&[usize]>,
    dm0: Option<TsrView>,
) -> (Option<HashMap<&'static str, Tsr>>, Vec<HashMap<&'static str, Tsr>>, Vec<(String, f64)>) {
    let device = cderi.device().clone();
    let mut timing = vec![];
    let time_full = std::time::Instant::now();

    // --- basic checks --- //
    if mo_coeff.is_empty() || mo_occ.is_empty() {
        panic!("mo_coeff and mo_occ must be non-empty");
    }
    if mo_coeff.len() != mo_occ.len() {
        panic!("mo_coeff and mo_occ must have the same length");
    }
    let nset = mo_coeff.len();

    // --- prepare shared --- //
    let t0 = std::time::Instant::now();
    let (dims, aoslices, auxslices, aux_ranges, shared, solve_aux) =
        prepare_shared(mol, aux, j2c_decomp, nbatch_aux, atm_list, &device);
    tic!(timing, t0, "prepare_shared");

    // --- prepare j --- //
    let j_in = do_j.then(|| {
        let t0 = std::time::Instant::now();
        // - dm0: [nao, nao]; note this density is total density instead of spin-separated density
        let dm0 = dm0.map_or_else(
            || {
                let nao = dims["nao"];
                let mut dm0 = rt::zeros(([nao, nao], &device));
                for iset in 0..nset {
                    dm0 += get_dm0_restricted(mo_coeff[iset].view(), mo_occ[iset].view())
                }
                dm0
            },
            |dm0| dm0.to_owned(),
        );
        let j_in = prepare_j(&solve_aux, &dims, dm0.view(), cderi.view());
        tic!(timing, t0, "prepare_j");
        j_in
    });
    let mut j_out = do_j.then(HashMap::new);

    // --- prepare k --- //

    let mut k_ins = vec![];
    if do_k {
        for iset in 0..nset {
            let t0 = std::time::Instant::now();
            let k_in = prepare_k(&solve_aux, &dims, mo_coeff[iset].view(), mo_occ[iset].view(), cderi.view());
            k_ins.push(k_in);
            tic!(timing, t0, &format!("prepare_k iset_{iset}"));
        }
    }
    let mut k_outs = (0..k_ins.len()).map(|_| HashMap::new()).collect_vec();

    // --- evaluate oneshot --- //

    let t0 = std::time::Instant::now();
    let timing_oneshot = evaluate_oneshot(
        &dims,
        mol,
        aux,
        &aoslices,
        &auxslices,
        &aux_ranges,
        &device,
        j_in.as_ref(),
        &k_ins,
        j_out.as_mut(),
        &mut k_outs,
    );
    timing.extend(timing_oneshot);
    tic!(timing, t0, "evaluate_oneshot");

    // --- evaluate j2c-derivatives-only terms --- //

    let t0 = std::time::Instant::now();
    let timing_j2c_deriv_only = evaluate_j2c_deriv_only(
        &dims,
        &shared,
        aux,
        &auxslices,
        &device,
        j_in.as_ref(),
        &k_ins,
        j_out.as_mut(),
        &mut k_outs,
    );
    timing.extend(timing_j2c_deriv_only);
    tic!(timing, t0, "evaluate_j2c_deriv_only");

    // --- evaluate jk1 j2c-skeleton terms --- //

    let t0 = std::time::Instant::now();
    let timing_jk1_j2c_deriv = evaluate_jk1_j2c_deriv(
        &dims,
        &shared,
        cderi,
        &auxslices,
        &device,
        &solve_aux,
        j_in.as_ref(),
        &k_ins,
        j_out.as_mut(),
        &mut k_outs,
    );
    timing.extend(timing_jk1_j2c_deriv);
    tic!(timing, t0, "evaluate_jk1_j2c_deriv");

    tic!(timing, time_full, "get_rijk_skeleton_decomposed_separated");
    (j_out, k_outs, timing)
}

pub type FnSolveAux<'a> = Box<dyn Fn(TsrMut, FlagSide, bool) + 'a>;
type PrepareSharedOutput<'a> = (
    HashMap<&'static str, usize>, // dims
    Vec<[usize; 4]>,              // aoslices
    Vec<[usize; 4]>,              // auxslices
    Vec<[usize; 4]>,              // aux_ranges
    HashMap<&'static str, Tsr>,   // shared (intermediates)
    FnSolveAux<'a>,               // solve_aux(tsr_mut, left/right, do_flip)
);

pub fn prepare_shared<'a>(
    mol: &CInt,
    aux: &CInt,
    j2c_decomp: &'a J2CDecompose,
    nbatch_aux: usize,
    atm_list: Option<&[usize]>,
    device: &DeviceTsr,
) -> PrepareSharedOutput<'a> {
    // aoslices, auxslices
    let atm_list = atm_list.map_or_else(|| (0..mol.natm()).collect_vec(), |list| list.to_vec());
    let natm = atm_list.len();
    let aoslices = mol.aoslice_by_atom();
    let auxslices = aux.aoslice_by_atom();
    let aoslices = atm_list.iter().map(|&i| aoslices[i]).collect_vec();
    let auxslices = atm_list.iter().map(|&i| auxslices[i]).collect_vec();

    // aux_ranges
    let aux_balance = aux.balance_partition(nbatch_aux);
    let mut p0 = 0;
    let aux_ranges = aux_balance
        .into_iter()
        .map(|[sh0, sh1, size]| {
            let range = [sh0, sh1, p0, p0 + size];
            p0 += size;
            range
        })
        .collect_vec();

    let solve_aux =
        |tsr_mut: TsrMut, side: FlagSide, do_flip: bool| solve_by_j2c_mut(tsr_mut, j2c_decomp, side, do_flip);

    let j2c_ip1 = hess_intor(aux, "int2c2e_ip1", "s1", None, device);
    let mut rcd_j2c_ip1 = j2c_ip1.to_owned();
    rcd_j2c_ip1.axes_iter_mut(-1).for_each(|m| solve_aux(m, Right, false));
    let mut rrcd_j2c_ip1 = rcd_j2c_ip1.to_owned();
    rrcd_j2c_ip1.axes_iter_mut(-1).for_each(|m| solve_aux(m, Right, true));

    let naux = aux.nao();
    let j2c_inv = {
        let mut eye = rt::eye((naux, device));
        solve_aux(eye.view_mut(), Right, true);
        let mut out = eye.t().into_contig(ColMajor);
        solve_aux(out.view_mut(), Right, true);
        out
    };

    let dims = HashMap::from([("natm", natm), ("nao", mol.nao()), ("naux", aux.nao())]);
    let shared = HashMap::from([
        ("j2c_ip1", j2c_ip1),
        ("rcd_j2c_ip1", rcd_j2c_ip1),
        ("rrcd_j2c_ip1", rrcd_j2c_ip1),
        ("j2c_inv", j2c_inv),
    ]);

    (dims, aoslices, auxslices, aux_ranges, shared, Box::new(solve_aux))
}

pub fn prepare_j(
    solve_aux: &FnSolveAux,
    dims: &HashMap<&'static str, usize>,
    dm0: TsrView,
    cderi: TsrView,
) -> HashMap<&'static str, Tsr> {
    let nao = dims["nao"];
    check_shape!(dm0.shape(), [nao, nao], "dm0 in prepare_j must be a single square matrix of shape [nao, nao]");

    // pack density matrix with scaled (off-diag, diag)
    let dm0_tp = pack_triu_tilde(dm0.view());
    let rrcd_eri_aux = {
        let mut out = dm0_tp.view() % cderi;
        solve_aux(out.view_mut(), Right, true);
        out
    };
    HashMap::from([("dm0", dm0.to_owned()), ("dm0_tp", dm0_tp), ("rrcd_eri_aux", rrcd_eri_aux)])
}

pub fn prepare_k(
    solve_aux: &FnSolveAux,
    dims: &HashMap<&'static str, usize>,
    mo_coeff: TsrView,
    mo_occ: TsrView,
    cderi: TsrView,
) -> HashMap<&'static str, Tsr> {
    let nao = dims["nao"];
    let naux = dims["naux"];
    let nmo = mo_coeff.shape()[1];
    let device = cderi.device().clone();
    check_shape!(mo_coeff.shape(), [nao, nmo], "mo_coeff in prepare_k must be of shape [nao, nmo]");
    check_shape!(mo_occ.shape(), [nmo], "mo_occ in prepare_k must be of shape [nmo]");

    let occidx = mo_occ.view().greater(0).into_vec();
    let nocc = occidx.iter().filter(|&&x| x).count();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    let mocc_2 = &mocc * occ.view().sqrt().i((None, ..));
    let occ_invsqrt = occ.view().pow(-0.5);

    // orbital transformation
    let rcd_eri_bra: Tsr = rt::zeros(([nao, nocc, naux], &device));
    let rcd_eri_occ: Tsr = rt::zeros(([nocc, nocc, naux], &device));
    (0..naux).into_par_iter().for_each(|p| {
        let rcd_eri_bra_p = rcd_eri_bra.i((.., .., p));
        let rcd_eri_occ_p = rcd_eri_occ.i((.., .., p));
        let mut rcd_eri_bra_p = unsafe { rcd_eri_bra_p.force_mut() };
        let mut rcd_eri_occ_p = unsafe { rcd_eri_occ_p.force_mut() };
        let cderi_p = cderi.i((.., p)).unpack_tri(Upper, FlagSymm::Sy);
        rcd_eri_bra_p.matmul_from(&cderi_p, &mocc_2, 1.0, 0.0);
        rcd_eri_occ_p.matmul_from(&mocc_2.t(), &rcd_eri_bra_p, 1.0, 0.0);
    });

    // j2c solve (consumes previous results in-place)
    let mut rrcd_eri_bra = rcd_eri_bra;
    solve_aux(rrcd_eri_bra.view_mut(), Right, true);
    let mut rrcd_eri_occ = rcd_eri_occ;
    solve_aux(rrcd_eri_occ.view_mut(), Right, true);

    HashMap::from([
        ("mocc", mocc),
        ("mocc_2", mocc_2),
        ("occ_invsqrt", occ_invsqrt),
        ("rrcd_eri_bra", rrcd_eri_bra),
        ("rrcd_eri_occ", rrcd_eri_occ),
    ])
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_oneshot(
    dims: &HashMap<&'static str, usize>,
    mol: &CInt,
    aux: &CInt,
    aoslices: &[[usize; 4]],
    auxslices: &[[usize; 4]],
    aux_ranges: &[[usize; 4]],
    device: &DeviceTsr,
    j_in: Option<&HashMap<&'static str, Tsr>>,
    k_ins: &[HashMap<&'static str, Tsr>],
    j_out: Option<&mut HashMap<&'static str, Tsr>>,
    k_outs: &mut [HashMap<&'static str, Tsr>],
) -> Vec<(String, f64)> {
    let mut timing = vec![];

    let nao = dims["nao"];
    let naux = dims["naux"];
    let natm = dims["natm"];
    let do_j = j_in.is_some();
    let nset_k = k_outs.len();

    // --- integral generators --- //

    let gen_j3c_ipvip1 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ipvip1", "s1", device);
    let gen_j3c_ipip1 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ipip1", "s1", device);
    let gen_j3c_ip1ip2 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ip1ip2", "s1", device);
    let gen_j3c_ipip2 = generator_hess_intor_j3c_by_aux(mol, aux, "int3c2e_ipip2", "s1", device);

    // --- dbas allocations --- //

    let mut dbas_j: HashMap<&str, Tsr> = HashMap::new();
    let mut dbas_ks: Vec<HashMap<&str, Tsr>> = (0..nset_k).map(|_| HashMap::new()).collect_vec();
    if do_j {
        dbas_j.insert("J20_2", rt::zeros(([nao, nao, 3, 3], device)));
        dbas_j.insert("J20_3", rt::zeros(([nao, nao, 3, 3], device)));
        dbas_j.insert("J11_1", rt::zeros(([nao, naux, 3, 3], device)));
        dbas_j.insert("J02_1", rt::zeros(([naux, 3, 3], device)));
    }
    for i in 0..nset_k {
        let dbas_k = &mut dbas_ks[i];
        dbas_k.insert("K20_2", rt::zeros(([nao, nao, 3, 3], device)));
        dbas_k.insert("K20_3", rt::zeros(([nao, nao, 3, 3], device)));
        dbas_k.insert("K11_1", rt::zeros(([nao, naux, 3, 3], device)));
        dbas_k.insert("K02_1", rt::zeros(([naux, 3, 3], device)));
    }

    // --- dbas evaluation --- //

    for &[sh0, sh1, p0, p1] in aux_ranges {
        // --- common tensors --- //

        let t0 = std::time::Instant::now();

        // j-part
        let rrcd_eri_aux = j_in.map(|j_in| j_in["rrcd_eri_aux"].i(p0..p1));

        // k-part
        let mut tmps_k_ao = vec![];
        for iset in 0..nset_k {
            let k_in = &k_ins[iset];
            let mocc_2 = &k_in["mocc_2"];
            let rrcd_eri_occ = k_in["rrcd_eri_occ"].i((.., .., p0..p1));
            tmps_k_ao.push(mocc_2 % rrcd_eri_occ % mocc_2.t());
        }

        tic!(timing, t0, &format!("evaluate_oneshot, common aux({p0}:{p1})"));

        // --- 20-2 (ipvip1) --- //

        let t0 = std::time::Instant::now();
        let j3c_ipvip1 = gen_j3c_ipvip1([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_oneshot, gen_j3c_ipvip1 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let rrcd_eri_aux = rrcd_eri_aux.as_ref().unwrap().view();
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let tmp1 = rt::vecdot(&j3c_ipvip1, rrcd_eri_aux.i((None, None, ..)), 2);
            *dbas_j.get_mut("J20_2").unwrap() += tmp1 * dm0;
        }
        for iset in 0..nset_k {
            let tmp_k_ao = &tmps_k_ao[iset];
            let tmp1 = rt::vecdot(&j3c_ipvip1, tmp_k_ao, 2);
            *dbas_ks[iset].get_mut("K20_2").unwrap() += tmp1;
        }
        tic!(timing, t0, &format!("evaluate_oneshot, dbas 20-2 aux({p0}:{p1})"));
        drop(j3c_ipvip1);

        // --- 20-3 (ipip1) --- //

        let t0 = std::time::Instant::now();
        let j3c_ipip1 = gen_j3c_ipip1([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_oneshot, gen_j3c_ipip1 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let rrcd_eri_aux = rrcd_eri_aux.as_ref().unwrap().view();
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let tmp1 = rt::vecdot(&j3c_ipip1, rrcd_eri_aux.i((None, None, ..)), 2);
            *dbas_j.get_mut("J20_3").unwrap() += tmp1 * dm0;
        }
        for iset in 0..nset_k {
            let tmp_k_ao = &tmps_k_ao[iset];
            let tmp1 = rt::vecdot(&j3c_ipip1, tmp_k_ao, 2);
            *dbas_ks[iset].get_mut("K20_3").unwrap() += tmp1;
        }
        tic!(timing, t0, &format!("evaluate_oneshot, dbas 20-3 aux({p0}:{p1})"));
        drop(j3c_ipip1);

        // --- 11-1 (ip1ip2) --- //

        let t0 = std::time::Instant::now();
        let j3c_ip1ip2 = gen_j3c_ip1ip2([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_oneshot, gen_j3c_ip1ip2 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let rrcd_eri_aux = rrcd_eri_aux.as_ref().unwrap().view();
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let tmp1 = rt::vecdot(&j3c_ip1ip2, &dm0, 1) * rrcd_eri_aux.i((None, ..));
            dbas_j.get_mut("J11_1").unwrap().i_mut((.., p0..p1)).assign(tmp1);
        }
        for iset in 0..nset_k {
            let tmp_k_ao = &tmps_k_ao[iset];
            let tmp1 = rt::vecdot(&j3c_ip1ip2, tmp_k_ao, 1);
            dbas_ks[iset].get_mut("K11_1").unwrap().i_mut((.., p0..p1)).assign(tmp1);
        }
        tic!(timing, t0, &format!("evaluate_oneshot, dbas 11-1 aux({p0}:{p1})"));
        drop(j3c_ip1ip2);

        // --- 02-1 (ipip2) --- //

        let t0 = std::time::Instant::now();
        let j3c_ipip2 = gen_j3c_ipip2([sh0, sh1]);
        tic!(timing, t0, &format!("evaluate_oneshot, gen_j3c_ipip2 aux({p0}:{p1})"));

        let t0 = std::time::Instant::now();
        if do_j {
            let rrcd_eri_aux = rrcd_eri_aux.as_ref().unwrap().view();
            let dm0 = j_in.as_ref().unwrap()["dm0"].view();
            let tmp1 = rt::vecdot(&j3c_ipip2, &dm0, ([0, 1], [0, 1])) * rrcd_eri_aux;
            dbas_j.get_mut("J02_1").unwrap().i_mut(p0..p1).assign(tmp1);
        }
        for iset in 0..nset_k {
            let tmp_k_ao = &tmps_k_ao[iset];
            let tmp1 = rt::vecdot(&j3c_ipip2, tmp_k_ao, ([0, 1], [0, 1]));
            dbas_ks[iset].get_mut("K02_1").unwrap().i_mut(p0..p1).assign(tmp1);
        }
        tic!(timing, t0, &format!("evaluate_oneshot, dbas 02-1 aux({p0}:{p1})"));
        drop(j3c_ipip2);
    }

    // --- reduce to hessian contribution --- //

    let t0 = std::time::Instant::now();

    if do_j {
        let j_out = j_out.unwrap();

        let mut de_J20_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J20_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J11_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_1: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            let tmp = dbas_j["J20_3"].i(slcA).sum_axes([0, 1]);
            de_J20_3.i_mut((.., .., A, A)).assign(&tmp);

            for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);
                let tmp = dbas_j["J20_2"].i((slcA, slcB)).sum_axes([0, 1]);
                de_J20_2.i_mut((.., .., B, A)).assign(&tmp);
            }

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);
                let tmp = dbas_j["J11_1"].i((slcA, slcB)).sum_axes([0, 1]);
                de_J11_1.i_mut((.., .., B, A)).assign(&tmp);
            }
        }

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            let tmp = dbas_j["J02_1"].i(slcA).sum_axes(0);
            de_J02_1.i_mut((.., .., A, A)).assign(&tmp);
        }

        let scale_J20_2 = 1.0;
        let scale_J20_3 = 1.0;
        let scale_J11_1 = 2.0;
        let scale_J02_1 = 0.5;
        j_out.insert("de_J20_2", scale_J20_2 * (&de_J20_2 + &de_J20_2.transpose([1, 0, 3, 2])));
        j_out.insert("de_J20_3", scale_J20_3 * (&de_J20_3 + &de_J20_3.transpose([1, 0, 3, 2])));
        j_out.insert("de_J11_1", scale_J11_1 * (&de_J11_1 + &de_J11_1.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_1", scale_J02_1 * (&de_J02_1 + &de_J02_1.transpose([1, 0, 3, 2])));
    }

    for iset in 0..nset_k {
        let k_out = &mut k_outs[iset];

        let mut de_K20_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K20_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K11_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_1: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            let tmp = dbas_ks[iset]["K20_3"].i(slcA).sum_axes([0, 1]);
            de_K20_3.i_mut((.., .., A, A)).assign(&tmp);

            for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);
                let tmp = dbas_ks[iset]["K20_2"].i((slcA, slcB)).sum_axes([0, 1]);
                de_K20_2.i_mut((.., .., B, A)).assign(&tmp);
            }

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);
                let tmp = dbas_ks[iset]["K11_1"].i((slcA, slcB)).sum_axes([0, 1]);
                de_K11_1.i_mut((.., .., B, A)).assign(&tmp);
            }
        }

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            let tmp = dbas_ks[iset]["K02_1"].i(slcA).sum_axes(0);
            de_K02_1.i_mut((.., .., A, A)).assign(&tmp);
        }

        let scale_K20_2 = 1.0;
        let scale_K20_3 = 1.0;
        let scale_K11_1 = 2.0;
        let scale_K02_1 = 0.5;
        k_out.insert("de_K20_2", scale_K20_2 * (&de_K20_2 + &de_K20_2.transpose([1, 0, 3, 2])));
        k_out.insert("de_K20_3", scale_K20_3 * (&de_K20_3 + &de_K20_3.transpose([1, 0, 3, 2])));
        k_out.insert("de_K11_1", scale_K11_1 * (&de_K11_1 + &de_K11_1.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_1", scale_K02_1 * (&de_K02_1 + &de_K02_1.transpose([1, 0, 3, 2])));
    }

    tic!(timing, t0, "evaluate_oneshot, reduce to hessian");

    timing
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_j2c_deriv_only(
    dims: &HashMap<&'static str, usize>,
    shared: &HashMap<&'static str, Tsr>,
    aux: &CInt,
    auxslices: &[[usize; 4]],
    device: &DeviceTsr,
    j_in: Option<&HashMap<&'static str, Tsr>>,
    k_ins: &[HashMap<&'static str, Tsr>],
    j_out: Option<&mut HashMap<&'static str, Tsr>>,
    k_outs: &mut [HashMap<&'static str, Tsr>],
) -> Vec<(String, f64)> {
    let mut timing = vec![];

    let naux = dims["naux"];
    let natm = dims["natm"];
    let do_j = j_in.is_some();
    let nset_k = k_outs.len();

    let j2c_ip1 = shared["j2c_ip1"].view();
    let rcd_j2c_ip1 = shared["rcd_j2c_ip1"].view();
    let rrcd_j2c_ip1 = shared["rrcd_j2c_ip1"].view();
    let j2c_inv = shared["j2c_inv"].view();

    // --- integral evaluation --- //

    let t0 = std::time::Instant::now();
    let j2c_ipip1 = hess_intor(aux, "int2c2e_ipip1", "s1", None, device);
    let j2c_ip1ip2 = hess_intor(aux, "int2c2e_ip1ip2", "s1", None, device);
    tic!(timing, t0, "evaluate_j2c_deriv_only, integration");

    // --- evaluation j-part --- //

    let t0 = std::time::Instant::now();
    if do_j {
        let j_out = j_out.unwrap();
        let rrcd_eri_aux = j_in.as_ref().unwrap()["rrcd_eri_aux"].view();

        // --- dbas evaluation --- //

        let dbas_J02_2 = rt::vecdot(&j2c_ipip1, &rrcd_eri_aux, 0) * &rrcd_eri_aux;
        let dbas_J02_3a = &j2c_ip1ip2 * &rrcd_eri_aux * rrcd_eri_aux.i((None, ..));
        let tmp1 = &rcd_j2c_ip1 * &rrcd_eri_aux;
        let dbas_J02_3b = tmp1.i((.., .., None, ..)) % tmp1.i((.., .., .., None)).swapaxes(0, 1);
        let tmp1 = rt::vecdot(&j2c_ip1, &rrcd_eri_aux, 0);
        let dbas_J02_6 = tmp1.i((.., None, None, ..)) * &j2c_inv * tmp1.i((None, .., .., None));
        let tmp2 = &rrcd_j2c_ip1 * &rrcd_eri_aux;
        let dbas_J02_8 = tmp1.i((None, .., .., None)) * tmp2.i((.., .., None, ..));

        // --- reduce to hessian contribution --- //

        let mut de_J02_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_3a: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_3b: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_6: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_J02_8: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            let tmp = dbas_J02_2.i(slcA).sum_axes(0);
            de_J02_2.i_mut((.., .., A, A)).assign(&tmp);

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_J02_3a.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_3a.i_mut((.., .., B, A)).assign(tmp);
                let tmp = dbas_J02_3b.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_3b.i_mut((.., .., B, A)).assign(tmp);
                let tmp = dbas_J02_6.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_6.i_mut((.., .., B, A)).assign(tmp);
                let tmp = dbas_J02_8.i((slcA, slcB)).sum_axes([0, 1]);
                de_J02_8.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        let scale_J02_2 = -0.5;
        let scale_J02_3a = -0.5;
        let scale_J02_3b = 0.5;
        let scale_J02_6 = 0.5;
        let scale_J02_8 = -1;
        j_out.insert("de_J02_2", scale_J02_2 * (&de_J02_2 + &de_J02_2.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_3a", scale_J02_3a * (&de_J02_3a + &de_J02_3a.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_3b", scale_J02_3b * (&de_J02_3b + &de_J02_3b.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_6", scale_J02_6 * (&de_J02_6 + &de_J02_6.transpose([1, 0, 3, 2])));
        j_out.insert("de_J02_8", scale_J02_8 * (&de_J02_8 + &de_J02_8.transpose([1, 0, 3, 2])));
    }
    tic!(timing, t0, "evaluate_j2c_deriv_only, evaluate j-part");

    // --- evaluation (K) --- //

    for iset in 0..nset_k {
        let t0 = std::time::Instant::now();

        let k_in = &k_ins[iset];
        let k_out = &mut k_outs[iset];
        let rrcd_eri_occ = k_in["rrcd_eri_occ"].view();
        let fold_eri_aux = rrcd_eri_occ.reshape((-1, naux)).t() % rrcd_eri_occ.reshape((-1, naux));

        // --- dbas evaluation --- //

        let dbas_K02_2 = rt::vecdot(&j2c_ipip1, &fold_eri_aux, 0);
        let dbas_K02_3a = &j2c_ip1ip2 * &fold_eri_aux;
        let dbas_K02_3b =
            rcd_j2c_ip1.i((.., .., None, ..)) % rcd_j2c_ip1.i((.., .., .., None)).swapaxes(0, 1) * &fold_eri_aux;
        let dbas_K02_6 = j2c_ip1.i((.., .., None, ..)) % &fold_eri_aux % j2c_ip1.i((.., .., .., None)) * &j2c_inv;
        let dbas_K02_8 = fold_eri_aux % j2c_ip1.i((.., .., .., None)) * rrcd_j2c_ip1.i((.., .., None, ..));

        // --- reduce to hessian contribution --- //

        let mut de_K02_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_3a: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_3b: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_6: Tsr = rt::zeros(([3, 3, natm, natm], device));
        let mut de_K02_8: Tsr = rt::zeros(([3, 3, natm, natm], device));

        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);

            let tmp = dbas_K02_2.i(slcA).sum_axes(0);
            de_K02_2.i_mut((.., .., A, A)).assign(&tmp);

            for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
                let slcB = rt::slice!(p0B, p1B);

                let tmp = dbas_K02_3a.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_3a.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K02_3b.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_3b.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K02_6.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_6.i_mut((.., .., B, A)).assign(tmp);

                let tmp = dbas_K02_8.i((slcA, slcB)).sum_axes([0, 1]);
                de_K02_8.i_mut((.., .., B, A)).assign(tmp);
            }
        }

        let scale_K02_2 = -0.5;
        let scale_K02_3a = -0.5;
        let scale_K02_3b = 0.5;
        let scale_K02_6 = -0.5;
        let scale_K02_8 = -1.0;
        k_out.insert("de_K02_2", scale_K02_2 * (&de_K02_2 + &de_K02_2.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_3a", scale_K02_3a * (&de_K02_3a + &de_K02_3a.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_3b", scale_K02_3b * (&de_K02_3b + &de_K02_3b.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_6", scale_K02_6 * (&de_K02_6 + &de_K02_6.transpose([1, 0, 3, 2])));
        k_out.insert("de_K02_8", scale_K02_8 * (&de_K02_8 + &de_K02_8.transpose([1, 0, 3, 2])));

        tic!(timing, t0, &format!("evaluate_j2c_deriv_only, evaluate k-part {iset}"));
    }

    timing
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_jk1_j2c_deriv(
    dims: &HashMap<&'static str, usize>,
    shared: &HashMap<&'static str, Tsr>,
    cderi: TsrView,
    auxslices: &[[usize; 4]],
    device: &DeviceTsr,
    solve_aux: &FnSolveAux,
    j_in: Option<&HashMap<&'static str, Tsr>>,
    k_ins: &[HashMap<&'static str, Tsr>],
    j_out: Option<&mut HashMap<&'static str, Tsr>>,
    k_outs: &mut [HashMap<&'static str, Tsr>],
) -> Vec<(String, f64)> {
    let mut timing = vec![];

    let nao = dims["nao"];
    let naux = dims["naux"];
    let natm = dims["natm"];
    let do_j = j_in.is_some();
    let nset_k = k_outs.len();
    let j2c_ip1 = shared["j2c_ip1"].view();
    let rcd_j2c_ip1 = shared["rcd_j2c_ip1"].view();

    // --- evaluation j-part --- //

    if do_j {
        let t0 = std::time::Instant::now();

        let j_out = j_out.unwrap();
        let rrcd_eri_aux = j_in.as_ref().unwrap()["rrcd_eri_aux"].view();

        // --- j1ao_aux1_3 --- //

        let tmp1 = -rt::vecdot(&j2c_ip1, &rrcd_eri_aux, 0);
        let mut tmp2: Tsr = rt::zeros(([naux, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            tmp2.i_mut((slcA, .., A)).assign(tmp1.i(slcA));
        }
        solve_aux(tmp2.view_mut(), Left, true);
        let j1ao_aux1_3_tp = &cderi % tmp2.reshape((naux, -1));
        let j1ao_aux1_3 = j1ao_aux1_3_tp.reshape((-1, 3, natm)).unpack_tri(Upper, FlagSymm::Sy);
        j_out.insert("j1ao_aux1_3", j1ao_aux1_3);

        // --- j1ao_aux1_4 --- //

        let tmp1 = &rcd_j2c_ip1 * &rrcd_eri_aux;
        let mut tmp2: Tsr = rt::zeros(([naux, 3, natm], device));
        for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
            let slcA = rt::slice!(p0A, p1A);
            tmp2.i_mut((.., .., A)).assign(tmp1.i(slcA).sum_axes(0));
        }
        let j1ao_aux1_4_tp = &cderi % tmp2.reshape((naux, -1));
        let j1ao_aux1_4 = j1ao_aux1_4_tp.reshape((-1, 3, natm)).unpack_tri(Upper, FlagSymm::Sy);
        j_out.insert("j1ao_aux1_4", j1ao_aux1_4);

        tic!(timing, t0, "evaluate_jk1_j2c_deriv, j-part");
    }

    // --- evaluation k-part --- //

    for iset in 0..nset_k {
        let t0 = std::time::Instant::now();

        let k_in = &k_ins[iset];
        let k_out = &mut k_outs[iset];
        let rrcd_eri_occ = k_in["rrcd_eri_occ"].view();
        let rrcd_eri_bra = k_in["rrcd_eri_bra"].view();
        let occ_invsqrt = k_in["occ_invsqrt"].view();
        let nocc = occ_invsqrt.shape()[0];

        // --- k1bra_aux1_3 --- //

        let mut k1bra_aux1_3 = rt::zeros(([nao, nocc, 3, natm], device));
        for t in 0..3 {
            let tmp1 = rrcd_eri_bra.reshape([nao * nocc, naux]) % j2c_ip1.i((.., .., t)).t();
            let tmp1 = tmp1.into_shape([nao, nocc, naux]);
            for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
                let slcA = rt::slice!(p0A, p1A);
                let tmp =
                    tmp1.i((.., .., slcA)).reshape((nao, -1)) % rrcd_eri_occ.i((.., .., slcA)).reshape((nocc, -1)).t();
                k1bra_aux1_3.i_mut((.., .., t, A)).assign(tmp);
            }
        }
        k1bra_aux1_3 *= occ_invsqrt.i((None, ..));
        k_out.insert("k1bra_aux1_3", k1bra_aux1_3);

        // --- k1bra_aux1_4 --- //

        let mut k1bra_aux1_4 = rt::zeros(([nao, nocc, 3, natm], device));
        for t in 0..3 {
            let tmp1 = rrcd_eri_occ.reshape([nocc * nocc, naux]) % j2c_ip1.i((.., .., t)).t();
            let tmp1 = tmp1.into_shape([nocc, nocc, naux]);
            for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
                let slcA = rt::slice!(p0A, p1A);
                let tmp =
                    rrcd_eri_bra.i((.., .., slcA)).reshape((nao, -1)) % tmp1.i((.., .., slcA)).reshape((nocc, -1)).t();
                k1bra_aux1_4.i_mut((.., .., t, A)).assign(tmp);
            }
        }
        k1bra_aux1_4 *= occ_invsqrt.i((None, ..));
        k_out.insert("k1bra_aux1_4", k1bra_aux1_4);

        tic!(timing, t0, &format!("evaluate_jk1_j2c_deriv, k-part {iset}"));
    }

    timing
}

/* #endregion */

/// Generate cderi and decomposition.
pub fn generate_cderi_with_decomp(
    mol: &CInt,
    aux: &CInt,
    j2c_decomp_option: J2CDecompOption,
    device: &DeviceTsr,
) -> (Tsr, J2CDecompose) {
    let j3c = hess_intor_cross(&[mol, mol, aux], "int3c2e", "s2ij", None, device);
    let j2c_decomp = get_j2c_decomp(aux, device, j2c_decomp_option);
    let cderi = solve_by_j2c(j3c, &j2c_decomp, Right, false);
    (cderi, j2c_decomp)
}

pub struct RHessRIJK<'a> {
    pub mol: CInt,
    pub aux: CInt,
    pub scale_j: f64,
    pub scale_k: f64,
    pub cderi: TsrCow<'a>,
    pub j2c_decomp: J2CDecompose,
    pub intmd: HashMap<&'static str, Tsr>, // intermediates
    pub result: HashMap<&'static str, Tsr>,
}

impl<'a> RHessRIJK<'a> {
    pub fn new_without_cderi(mol: &CInt, aux: &CInt, scale_j: f64, scale_k: f64) -> Self {
        let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: Some(1e-14), uplo: Upper };
        // note: the following two options are also valid
        // let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: Some(1e-14),
        // uplo: Lower };
        // let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Eig, threshold: Some(1e-14),
        // uplo: Upper };
        let device = DeviceTsr::default();
        let (cderi, j2c_decomp) = generate_cderi_with_decomp(mol, aux, j2c_decomp_option, &device);
        Self {
            mol: mol.clone(),
            aux: aux.clone(),
            scale_j,
            scale_k,
            cderi: cderi.into_cow(),
            j2c_decomp,
            intmd: HashMap::new(),
            result: HashMap::new(),
        }
    }

    pub fn new_with_cderi(
        mol: &CInt,
        aux: &CInt,
        scale_j: f64,
        scale_k: f64,
        cderi: TsrView<'a>,
        j2c_decomp: J2CDecompose,
    ) -> Self {
        Self {
            mol: mol.clone(),
            aux: aux.clone(),
            scale_j,
            scale_k,
            cderi: cderi.into_cow(),
            j2c_decomp,
            intmd: HashMap::new(),
            result: HashMap::new(),
        }
    }
}

impl<'a> HessUtilAPI for RHessRIJK<'a> {}

impl<'a> RHessElecInteractAPI for RHessRIJK<'a> {
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        use crate::ri_jk::hess_r_naive::{
            get_decomposed_rij_skeleton_deriv2_naive, get_decomposed_rik_skeleton_deriv2_naive,
        };
        let de_J_skeleton_dict =
            get_decomposed_rij_skeleton_deriv2_naive(&self.mol, &self.aux, mo_coeff.view(), mo_occ.view(), atm_list);
        let de_K_skeleton_dict =
            get_decomposed_rik_skeleton_deriv2_naive(&self.mol, &self.aux, mo_coeff.view(), mo_occ.view(), atm_list);
        let result = &mut self.result;
        result.extend(de_J_skeleton_dict);
        result.extend(de_K_skeleton_dict);
        let de_J = &result["de_J20"] + &result["de_J11"] + &result["de_J02"];
        let de_K = &result["de_K20"] + &result["de_K11"] + &result["de_K02"];
        self.scale_j * de_J - 0.5 * self.scale_k * de_K
    }

    fn get_deriv1_ao(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        use crate::ri_jk::hess_r_naive::{get_rij_deriv1_ao_naive, get_rik_deriv1_ao_naive};
        let j1ao_dict = get_rij_deriv1_ao_naive(&self.mol, &self.aux, mo_coeff.view(), mo_occ.view(), atm_list);
        let k1ao_dict = get_rik_deriv1_ao_naive(&self.mol, &self.aux, mo_coeff.view(), mo_occ.view(), atm_list);
        let result = &mut self.result;
        result.extend(j1ao_dict);
        result.extend(k1ao_dict);
        let j1ao = &result["j1ao_aux0"] + &result["j1ao_aux1"];
        let k1ao = &result["k1ao_aux0"] + &result["k1ao_aux1"];
        self.scale_j * j1ao - 0.5 * self.scale_k * k1ao
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        self.intmd.insert("mo_coeff", mo_coeff.into_contig(RowMajor));
        self.intmd.insert("mo_occ", mo_occ.to_owned());
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        let mo_coeff = self.intmd["mo_coeff"].view();
        let mo_occ = self.intmd["mo_occ"].view();
        let cderi = self.cderi.view();
        // TODO: batch size `72` should be tunable by max-memory.
        get_rijk_response_bra(cderi, mo_coeff, mo_occ, bra, self.scale_j, self.scale_k, 72)
    }
}
