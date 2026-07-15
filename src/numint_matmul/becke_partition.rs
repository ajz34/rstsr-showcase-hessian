//! Standalone Becke partitioning implementation for DFT numerical integration.
//!
//! This file is intended not to interfere to other parts of the codebase.
//!
//! This code tries to somehow optimize with SIMD. However, this requires manually setting
//! `target-cpu` or something similar when cargo build (only release is not enough). The efficiency
//! is not tested currently, but theoretically should not be too pitiful I believe.

use itertools::Itertools;
use libcint::gto::deriv_util::FpSimd;
use rayon::prelude::*;
use std::sync::Mutex;

#[allow(non_camel_case_types)]
type f64simd = FpSimd<f64, SIMDD>;

const SIMDD: usize = 8;
const INVTOL: f64 = 1e-14;

const SIMD_0: f64simd = f64simd::splat(0.0);
const SIMD_0_5: f64simd = f64simd::splat(0.5);
const SIMD_1_0: f64simd = f64simd::splat(1.0);
const SIMD_1_5: f64simd = f64simd::splat(1.5);
const SIMD_2_0: f64simd = f64simd::splat(2.0);
const SIMD_3_0: f64simd = f64simd::splat(3.0);

/// Output of Becke partitioning function.
///
/// - `w`: Partition weights for each grid point. Shape `(ngrids, )`.
/// - `dw`: 1st derivative of partition weights for each grid point. Shape `(natm, 3, ngrids)` in
///   row-major or `(ngrids, 3, natm)` in column-major. We do not provide shape information here,
///   since it can be derived from length of variable `w`.
/// - `ddw`: 2nd derivative of partition weights for each grid point. Shape `(natm, 3, natm, 3,
///   ngrids)` in row-major or `(ngrids, 3, natm, 3, natm)` in column-major.
/// - `c`: Contracted weights for each grid point. Shape `(nset, )`.
/// - `dc`: 1st derivative of contracted weights for each grid point. Shape `(natm, 3, nset)` in
///   row-major or `(nset, 3, natm)` in column-major.
/// - `ddc`: 2nd derivative of contracted weights for each grid point. Shape `(natm, 3, natm, 3,
///   nset)` in row-major or `(nset, 3, natm, 3, natm)` in column-major.
#[derive(Debug, Clone, Default)]
pub struct BeckePartitionOutput {
    pub w: Option<Vec<f64>>,
    pub dw: Option<Vec<f64>>,
    pub ddw: Option<Vec<f64>>,
    pub c: Option<Vec<f64>>,
    pub dc: Option<Vec<f64>>,
    pub ddc: Option<Vec<f64>>,
}

/// Arguments for Becke partitioning function.
///
/// For usual usage, just default should work. You can contract the weights manually outside becke
/// partition function.
///
/// For advanced usage, user may want not only to compute the partition weights, but also to
/// contract them (especially 2nd derivative that generate a large $O(N^3)$ tensor of `ddw`, and the
/// user may want to avoid it). For this case, the `contract` field will provide a slice, of shape
/// `(nset, ngrids)` in row-major or `(ngrids, nset)` in column-major, to contract the weights. For
/// the output shape, see [`BeckePartitionOutput`].
///
/// The user may want to specify if they want to output `w`, `dw`, `ddw`. However,
/// - If deriv level is 0, `dw` and `ddw` will still be None, even `output_dw/ddw` is true. Same
///   applies to deriv level 1.
/// - If `contract_w/dw/ddw` is provided, `c/dc/ddc` will be computed with the given deriv level by
///   default. But if it is not provided, `c/dc/ddc` will be None, even `output_c/dc/ddc` is true.
#[derive(Debug)]
pub struct BeckeDerivArg<'a> {
    pub output_w: bool,
    pub output_dw: bool,
    pub output_ddw: bool,
    pub contract_w: Option<&'a [f64]>,
    pub contract_dw: Option<&'a [f64]>,
    pub contract_ddw: Option<&'a [f64]>,
}

impl<'a> Default for BeckeDerivArg<'a> {
    fn default() -> Self {
        Self {
            output_w: true,
            output_dw: true,
            output_ddw: true,
            contract_w: None,
            contract_dw: None,
            contract_ddw: None,
        }
    }
}

impl<'a> BeckeDerivArg<'a> {
    /// Set the contraction weights for the Becke partitioning.
    pub fn set_contract(mut self, deriv: usize, contract: &'a [f64]) -> Self {
        match deriv {
            0 => self.contract_w = Some(contract),
            1 => self.contract_dw = Some(contract),
            2 => self.contract_ddw = Some(contract),
            _ => panic!("Unsupported derivative order: {}", deriv),
        }
        self
    }

    /// Set the output flags for the Becke partitioning.
    pub fn set_output_weights(mut self, deriv: usize, value: bool) -> Self {
        match deriv {
            0 => self.output_w = value,
            1 => self.output_dw = value,
            2 => self.output_ddw = value,
            _ => panic!("Unsupported derivative order: {}", deriv),
        }
        self
    }
}

/// Becke partitioning implementation for DFT numerical integration.
///
/// # Arguments
///
/// - `grid_coords`: A slice of 3D coordinates representing the grid points. Length `ngrids`.
/// - `atm_coords`: A slice of 3D coordinates representing the atomic positions. Length `natm`.
/// - `atm_indices`: A slice of indices mapping each grid point to its corresponding atom (that
///   generates the Lebedev angular grids). Length `ngrids`. Usually values should not exceed `natm
///   - 1`, but will treat as padding atom if exceeds.
/// - `quadrature_weights`: A slice of original quadrature weights (generated by Lebedev angular
///   quadrature) for each grid point. Length `ngrids`.
/// - `adjustment_factor`: A slice of adjustment factors (usually adjusted by atomic radii) for each
///   grid point. Shape `(natm, natm)`, flattened. cf Becke 1988 eqs (A2, A5).
///   - **Note**: this pairwise matrix is anti-symmetric, in column-major order. If you use this
///     function with row-major, you may need to first transpose it.
/// - `hardness`: Cutoff hardness of screen function. cf Becke 1988 eq (20) and FIG 1. Most commonly
///   used value is 3, and we have manually dispatched the case of 3 in code implementation.
/// - `nbatch`: Batch size for parallel processing. Must be a multiple of `SIMDD` (8 for AVX-512).
/// - `deriv`: Derivative order.
/// - `deriv_arg`: Optional arguments for derivative output and contraction. If None, default values
///   will be used, which outputs all weights and derivatives without contraction.
///
/// # Reference
///
/// A multicenter numerical integration scheme for polyatomic molecules
/// A. D. Becke
/// J. Chem. Phys. 88, 2547 (1988), doi: 10.1063/1.454033
#[allow(clippy::too_many_arguments)]
pub fn becke_partition<'a>(
    grid_coords: &[[f64; 3]],
    atm_coords: &[[f64; 3]],
    atm_indices: &[usize],
    quadrature_weights: &[f64],
    adjustment_factor: &[f64],
    hardness: usize,
    nbatch: usize,
    deriv: usize,
    deriv_arg: Option<BeckeDerivArg<'a>>,
) -> BeckePartitionOutput {
    // dimensions
    let ngrids = grid_coords.len();
    let natm = atm_coords.len();
    assert!(nbatch % SIMDD == 0, "nbatch must be a multiple of {SIMDD}");
    assert!(adjustment_factor.len() == natm * natm, "adjustment_factor must have length natm * natm");
    assert!(atm_indices.len() == ngrids, "atm_indices must have length ngrids");
    assert!(quadrature_weights.len() == ngrids, "quadrature_weights must have length ngrids");

    let deriv_arg = deriv_arg.unwrap_or_default();
    assert!(deriv <= 2, "deriv must be 0, 1, or 2 at current time");

    let adjustment_factor = adjustment_factor.chunks_exact(natm).collect_vec();

    // check if contraction is requested, and split the contraction weights into
    // per-set grid slices (shape `(nset, ngrids)` row-major).  Done before
    // building the output struct so the buffers can be allocated in one literal.
    let contract_w = deriv_arg.contract_w.as_ref().map(|c| {
        assert!(c.len() % ngrids == 0, "contract_w length must be a multiple of ngrids");
        c.chunks_exact(ngrids).collect_vec()
    });
    let contract_dw = deriv_arg.contract_dw.as_ref().map(|c| {
        assert!(c.len() % ngrids == 0, "contract_dw length must be a multiple of ngrids");
        c.chunks_exact(ngrids).collect_vec()
    });
    let contract_ddw = deriv_arg.contract_ddw.as_ref().map(|c| {
        assert!(c.len() % ngrids == 0, "contract_ddw length must be a multiple of ngrids");
        c.chunks_exact(ngrids).collect_vec()
    });
    let nset_w = contract_w.as_ref().map(|c| c.len());
    let nset_dw = (deriv >= 1 && contract_dw.is_some()).then(|| contract_dw.as_ref().unwrap().len());
    let nset_ddw = (deriv >= 2 && contract_ddw.is_some()).then(|| contract_ddw.as_ref().unwrap().len());
    // whether any 2nd-order output (the full `ddw` tensor or its contraction) is
    // actually requested.  When false the entire deriv2 machinery is skipped even
    // if `deriv == 2`, since neither `ddw` nor `ddc` would be consumed.
    let need_ddw = deriv_arg.output_ddw || contract_ddw.is_some();

    // prepare output (buffers mutated later through `cast_mut_slice`, so `output`
    // itself is not declared `mut`).
    let output = BeckePartitionOutput {
        w: deriv_arg.output_w.then(|| vec![0.0; ngrids]),
        dw: (deriv_arg.output_dw && deriv >= 1).then(|| vec![0.0; natm * 3 * ngrids]),
        ddw: (deriv_arg.output_ddw && deriv >= 2).then(|| vec![0.0; natm * 3 * natm * 3 * ngrids]),
        c: nset_w.map(|nset| vec![0.0; nset]),
        dc: nset_dw.map(|nset| vec![0.0; natm * 3 * nset]),
        ddc: nset_ddw.map(|nset| vec![0.0; natm * 3 * natm * 3 * nset]),
    };

    // generate atm_dist before iteration to grid coordinates
    // since it is not bottleneck, we duplicate the calculation for A > B.
    let atm_dist: Vec<f64> = (0..natm * natm)
        .into_par_iter()
        .map(|idx| {
            let (A, B) = (idx / natm, idx % natm);
            if A == B {
                f64::INFINITY
            } else {
                dist3_naive(&atm_coords[A], &atm_coords[B])
            }
        })
        .collect();
    let atm_dist = atm_dist.chunks_exact(natm).collect_vec();

    // compute derivative of atm_dist if deriv >= 1
    let dR_atm_dist: Option<Vec<[f64; 3]>> = (deriv >= 1).then(|| {
        (0..natm * natm)
            .into_par_iter()
            .map(|idx| {
                let (A, B) = (idx / natm, idx % natm);
                (0..3).map(|t| (atm_coords[A][t] - atm_coords[B][t]) / atm_dist[A][B]).collect_array().unwrap()
            })
            .collect()
    });
    let dR_atm_dist = dR_atm_dist.as_ref().map(|v| v.chunks_exact(natm).collect_vec());

    // par-iter over grid coordinates in batches
    let ntasks = ngrids.div_ceil(nbatch);
    // guards the cross-task reduction into the shared c/dc/ddc buffers.  The
    // contraction is a sum over grids into a per-set output, so (unlike the
    // disjoint grid-range writes of w/dw/ddw) concurrent `+=` would race.  Each
    // task accumulates a private partial lock-free, then takes this lock once to
    // add it in - the cast_mut_slice += inside the guard is sound because the
    // mutex grants exclusive access.
    let contract_guard = Mutex::new(());
    (0..ntasks).into_par_iter().for_each(|itask| {
        let g0 = itask * nbatch;
        let g1 = (g0 + nbatch).min(ngrids);
        let nlane = (g1 - g0).div_ceil(SIMDD);

        // per-task contraction partials.  Each task accumulates its own grid range
        // into a private buffer (no cross-task sharing) during the lane loop, then
        // adds the partial into the shared output under `contract_mutex`.  This is
        // what lets the caller obtain a contracted `ddc` without ever materializing
        // the full `O(natm^2 * ngrids)` `ddw` tensor.
        let mut c_partial = nset_w.map(|n| vec![0.0; n]);
        let mut dc_partial = nset_dw.map(|n| vec![0.0; natm * 3 * n]);
        let mut ddc_partial = nset_ddw.map(|n| vec![0.0; natm * 3 * natm * 3 * n]);

        // reusable scratch buffers holding this lane's contraction weights (one
        // SIMD per set).  Allocated once per task and refilled each lane, so the
        // dc/ddc contraction does not allocate inside the lane loop.
        let mut cw_lanes_dw: Vec<f64simd> = nset_dw.map_or_else(Vec::new, |n| vec![SIMD_0; n]);
        let mut cw_lanes_ddw: Vec<f64simd> = nset_ddw.map_or_else(Vec::new, |n| vec![SIMD_0; n]);

        // batched vectors preparation
        let mut coords_lanes = vec![[SIMD_0; 3]; nlane];
        let mut wquad_lanes = vec![SIMD_0; nlane];
        let mut atm_idx_lanes = vec![[0; SIMDD]; nlane];
        for lane in 0..nlane {
            let glane = g0 + lane * SIMDD;
            for g in 0..SIMDD {
                if glane + g < g1 {
                    for t in 0..3 {
                        coords_lanes[lane][t][g] = grid_coords[glane + g][t];
                    }
                    wquad_lanes[lane][g] = quadrature_weights[glane + g];
                    atm_idx_lanes[lane][g] = atm_indices[glane + g];
                } else {
                    // padding atom index set to natm (out of range)
                    atm_idx_lanes[lane][g] = natm;
                }
            }
        }

        for lane in 0..nlane {
            let coords = coords_lanes[lane];
            let wquad = wquad_lanes[lane];
            let atm_idx = atm_idx_lanes[lane];

            // --- deriv 0 --- //

            // partition output
            let mut P = vec![SIMD_1_0; natm];

            // evaluate grid distance to atom
            let mut dist = vec![SIMD_0; natm];
            for A in 0..natm {
                dist[A] = dist3_hybrid(&coords, &atm_coords[A]);
            }

            // 1st pass of switch function (without derivative)
            for A in 0..natm {
                for B in 0..A {
                    let a_factor = adjustment_factor[B][A]; // column-major order
                    let mu = (dist[A] - dist[B]) / atm_dist[A][B];
                    let f3 = match hardness {
                        3 => switch_f3(mu, a_factor),
                        _ => switch_f_hardness(mu, a_factor, hardness),
                    };
                    P[A] *= SIMD_0_5 * (SIMD_1_0 - f3);
                    P[B] *= SIMD_0_5 * (SIMD_1_0 + f3);
                }
            }

            // compute partition function and weights
            let mut Pg = SIMD_0;
            let mut Z = SIMD_0;
            for A in 0..natm {
                Z += P[A];
                let mask = atm_idx.map(|a| a == A);
                Pg = P[A].mask_select(mask, Pg);
            }
            let partition = Pg / Z;
            let w = wquad * partition;

            // write back to output buffer
            let g_start = g0 + lane * SIMDD;
            let g_end = (g_start + SIMDD).min(g1);
            let nlane_g = g_end - g_start;
            if let Some(w_buf) = output.w.as_ref() {
                let wslc = unsafe { cast_mut_slice(&w_buf[g_start..g_end]) };
                wslc[..nlane_g].copy_from_slice(&w.0[..nlane_g]);
            }

            // contract w -> c:  c[iset] += sum_g contract_w[iset, g] * w[g]
            if let Some(cw) = contract_w.as_ref() {
                let cp = c_partial.as_mut().unwrap();
                for iset in 0..nset_w.unwrap() {
                    let cw_lane = load_simd_pad(&cw[iset][g_start..g_end]);
                    cp[iset] += sum_lanes(w * cw_lane, nlane_g);
                }
            }

            // --- deriv 1 --- //

            if deriv >= 1 {
                let dR_atm_dist = dR_atm_dist.as_ref().unwrap();

                // partition output
                let mut dR_Z = vec![[SIMD_0; 3]; natm];
                let mut dR_Pg = vec![[SIMD_0; 3]; natm];

                // evaluate derivative of grid distance to atom
                let mut dR_dist = vec![[SIMD_0; 3]; natm];
                for A in 0..natm {
                    for t in 0..3 {
                        dR_dist[A][t] = (-coords[t] + atm_coords[A][t]) / dist[A];
                    }
                }

                // 2nd-order intermediates (only materialized for deriv >= 2).
                // `dR_log_P[M][A][t]` (4D) is the minimal cross-term intermediate; the 6D
                // `ddR_log_P`/`ddR_P` of the vectorized reference are never materialized - the
                // 2nd log-deriv (L2) contributions are accumulated pair-by-pair directly into the
                // 5D outputs `ddR_Z`/`ddR_Pg` indexed `[A][B][t][s]`.
                let do_deriv2 = deriv >= 2 && need_ddw;
                let mut dR_log_P = do_deriv2.then(|| vec![vec![[SIMD_0; 3]; natm]; natm]); // [M][A][t]
                let mut ddR_Z = do_deriv2.then(|| vec![vec![[[SIMD_0; 3]; 3]; natm]; natm]); // [A][B][t][s]
                let mut ddR_Pg = do_deriv2.then(|| vec![vec![[[SIMD_0; 3]; 3]; natm]; natm]); // [A][B][t][s]

                // per-atom projection matrix PrM[M] = Proj(r_M)/|r_M| (depends only on the atom,
                // not the pair partner, so precomputed once per batch instead of per pair).
                let PrM: Option<Vec<[[f64simd; 3]; 3]>> = do_deriv2.then(|| {
                    (0..natm)
                        .map(|M| {
                            let inv_d = SIMD_1_0 / dist[M];
                            let mut pm = [[SIMD_0; 3]; 3];
                            for t in 0..3 {
                                for s in 0..3 {
                                    let delta = if t == s { SIMD_1_0 } else { SIMD_0 };
                                    pm[t][s] = (delta - dR_dist[M][t] * dR_dist[M][s]) * inv_d;
                                }
                            }
                            pm
                        })
                        .collect_vec()
                });

                // 2nd pass of switch function (with 1st derivative)
                // variable `P` is required to be generated in the 1st pass
                // so two passes cannot merge for first derivative
                for A in 0..natm {
                    for B in 0..A {
                        let a_factor = adjustment_factor[B][A]; // column-major order
                        let inv_atm_dist_AB = 1.0 / atm_dist[A][B];
                        let mu = (dist[A] - dist[B]) * inv_atm_dist_AB;
                        // switch value + 1st nu-deriv always; the 2nd nu-deriv (f3pp) is only
                        // needed for deriv >= 2, so the deriv == 1 path uses the cheaper
                        // 1st-order-only switch and avoids computing f3''.
                        let (f3, df3, ddf3): (f64simd, f64simd, Option<f64simd>) = if do_deriv2 {
                            let (f3, df3, ddf3) = match hardness {
                                3 => switch_d2nu_f3(mu, a_factor),
                                _ => switch_d2nu_f_hardness(mu, a_factor, hardness),
                            };
                            (f3, df3, Some(ddf3))
                        } else {
                            let (f3, df3) = match hardness {
                                3 => switch_dnu_f3(mu, a_factor),
                                _ => switch_dnu_f_hardness(mu, a_factor, hardness),
                            };
                            (f3, df3, None)
                        };
                        let sA = SIMD_0_5 * (SIMD_1_0 - f3);
                        let sB = SIMD_0_5 * (SIMD_1_0 + f3);
                        let dmu_nu = SIMD_1_0 - SIMD_2_0 * mu * a_factor;
                        let dmu_sA = -SIMD_0_5 * df3 * dmu_nu;
                        let dmu_sB = SIMD_0_5 * df3 * dmu_nu;
                        let sA_safe = sA.max_compare(INVTOL);
                        let sB_safe = sB.max_compare(INVTOL);
                        let dmu_log_sA = dmu_sA / sA_safe;
                        let dmu_log_sB = dmu_sB / sB_safe;

                        let common_Z = P[A] * dmu_log_sA + P[B] * dmu_log_sB;
                        let maskA = atm_idx.map(|a| a == A);
                        let maskB = atm_idx.map(|a| a == B);
                        let common_Pg = (P[A] * dmu_log_sA).mask_select(maskA, SIMD_0)
                            + (P[B] * dmu_log_sB).mask_select(maskB, SIMD_0);

                        let mut dR_mu_roleA = [SIMD_0; 3];
                        let mut dR_mu_roleB = [SIMD_0; 3];
                        let dR_atm_dist_AB = dR_atm_dist[A][B];
                        for t in 0..3 {
                            dR_mu_roleA[t] = (dR_dist[A][t] - mu * dR_atm_dist_AB[t]) * inv_atm_dist_AB;
                            dR_mu_roleB[t] = (-dR_dist[B][t] + mu * dR_atm_dist_AB[t]) * inv_atm_dist_AB;
                            dR_Z[A][t] += common_Z * dR_mu_roleA[t];
                            dR_Z[B][t] += common_Z * dR_mu_roleB[t];
                            dR_Pg[A][t] += common_Pg * dR_mu_roleA[t];
                            dR_Pg[B][t] += common_Pg * dR_mu_roleB[t];
                        }

                        // --- deriv 2 (per-pair L2 accumulation) --- //

                        if do_deriv2 {
                            let ddf3 = ddf3.unwrap();
                            let dR_log_P = dR_log_P.as_mut().unwrap();
                            let ddR_Z = ddR_Z.as_mut().unwrap();
                            let ddR_Pg = ddR_Pg.as_mut().unwrap();

                            // 2nd mu-derivatives of s(mu); ddmu_sB = -ddmu_sA (s_BA = 1 - s_AB)
                            let ddmu_nu = -SIMD_2_0 * a_factor; // nu'' = -2a
                            let ddmu_sA = -SIMD_0_5 * ddf3 * dmu_nu * dmu_nu - SIMD_0_5 * df3 * ddmu_nu;
                            let ddmu_sB = -ddmu_sA;
                            // ddmu_log_s = s''/s - (s'/s)^2
                            let ddmu_log_sA = ddmu_sA / sA_safe - dmu_log_sA * dmu_log_sA;
                            let ddmu_log_sB = ddmu_sB / sB_safe - dmu_log_sB * dmu_log_sB;

                            // 2nd role derivatives of mu_{AB} (3 blocks) via quotient rule
                            //   d2(f/g) = [f_xy g - (f_x g_y + g_x f_y) - f g_xy]/g^2 + 2 f g_x g_y/g^3
                            // rA, rB = unit vec (R_atom - r_g)/|r|; Uvec = unit vec (R_A - R_B)/|R_AB|;
                            // PrA = Proj(rA)/|r_A|, PrB = Proj(rB)/|r_B|, PU = Proj(Uvec)/|R_AB|.
                            let rA = dR_dist[A];
                            let rB = dR_dist[B];
                            let Uvec = [
                                f64simd::splat(dR_atm_dist_AB[0]),
                                f64simd::splat(dR_atm_dist_AB[1]),
                                f64simd::splat(dR_atm_dist_AB[2]),
                            ];
                            // PrA/PrB are per-atom (precomputed in PrM above); only PU is per-pair.
                            let PrA = PrM.as_ref().unwrap()[A];
                            let PrB = PrM.as_ref().unwrap()[B];
                            let mut PU = [[SIMD_0; 3]; 3];
                            for t in 0..3 {
                                for s in 0..3 {
                                    let delta = if t == s { SIMD_1_0 } else { SIMD_0 };
                                    PU[t][s] = (delta - Uvec[t] * Uvec[s]) * inv_atm_dist_AB;
                                }
                            }
                            let g_ab = atm_dist[A][B];
                            let f_ab = dist[A] - dist[B]; // f = |r_A| - |r_B| (= mu * g_ab)
                            let inv_g2 = inv_atm_dist_AB * inv_atm_dist_AB;
                            let inv_g3 = inv_g2 * inv_atm_dist_AB;
                            let zero_ts = [[SIMD_0; 3]; 3];
                            let nrB = neg3(rB);
                            let nUv = neg3(Uvec);
                            let nPrB = neg33(PrB);
                            let nPU = neg33(PU);
                            // d2mu(fX, fY, fXY, gX, gY, gXY) -> [[f64simd;3];3] over (t, s)
                            let d2mu = |fX: &[f64simd; 3],
                                        fY: &[f64simd; 3],
                                        fXY: &[[f64simd; 3]; 3],
                                        gX: &[f64simd; 3],
                                        gY: &[f64simd; 3],
                                        gXY: &[[f64simd; 3]; 3]|
                             -> [[f64simd; 3]; 3] {
                                let mut out = [[SIMD_0; 3]; 3];
                                for t in 0..3 {
                                    for s in 0..3 {
                                        let ofg = fX[t] * gY[s] + gX[t] * fY[s];
                                        let ogg = gX[t] * gY[s];
                                        out[t][s] = (fXY[t][s] * g_ab - ofg - f_ab * gXY[t][s]) * inv_g2
                                            + SIMD_2_0 * f_ab * ogg * inv_g3;
                                    }
                                }
                                out
                            };
                            let ddR_mu_roleAA = d2mu(&rA, &rA, &PrA, &Uvec, &Uvec, &PU);
                            let ddR_mu_roleAB = d2mu(&rA, &nrB, &zero_ts, &Uvec, &nUv, &nPU);
                            let ddR_mu_roleBB = d2mu(&nrB, &nrB, &nPrB, &nUv, &nUv, &PU);
                            // role BA = role AB transposed in (t, s)
                            let mut ddR_mu_roleBA = [[SIMD_0; 3]; 3];
                            for t in 0..3 {
                                for s in 0..3 {
                                    ddR_mu_roleBA[t][s] = ddR_mu_roleAB[s][t];
                                }
                            }

                            // accumulate dR_log_P (4D): convention dmu_log_sA = +nat[A,B],
                            // dmu_log_sB = -nat[B,A]; all contributions below are plus.
                            for t in 0..3 {
                                dR_log_P[A][A][t] += dmu_log_sA * dR_mu_roleA[t]; // M=A, role A
                                dR_log_P[A][B][t] += dmu_log_sA * dR_mu_roleB[t]; // M=A, role B
                                dR_log_P[B][A][t] += dmu_log_sB * dR_mu_roleA[t]; // M=B, role B
                                dR_log_P[B][B][t] += dmu_log_sB * dR_mu_roleB[t]; // M=B, role A
                            }

                            // L2 (2nd log-deriv) into ddR_Z and ddR_Pg.  The w*(d_A mu)(d_B mu)
                            // term uses the FIRST role derivatives dR_mu_roleA/B (NOT the unit
                            // vectors rA/rB).  The 4 role outer products are formed once per (t,s)
                            // and reused for both ddR_Z and ddR_Pg.
                            let common_dd = P[A] * ddmu_log_sA + P[B] * ddmu_log_sB;
                            let coef_A = P[A].mask_select(maskA, SIMD_0);
                            let coef_B = P[B].mask_select(maskB, SIMD_0);
                            let c1_Pg = coef_A * dmu_log_sA + coef_B * dmu_log_sB;
                            let cdd_Pg = coef_A * ddmu_log_sA + coef_B * ddmu_log_sB;
                            for t in 0..3 {
                                for s in 0..3 {
                                    let ooAA = dR_mu_roleA[t] * dR_mu_roleA[s];
                                    let ooAB = dR_mu_roleA[t] * dR_mu_roleB[s];
                                    let ooBA = dR_mu_roleB[t] * dR_mu_roleA[s];
                                    let ooBB = dR_mu_roleB[t] * dR_mu_roleB[s];
                                    ddR_Z[A][A][t][s] += common_dd * ooAA + common_Z * ddR_mu_roleAA[t][s];
                                    ddR_Z[A][B][t][s] += common_dd * ooAB + common_Z * ddR_mu_roleAB[t][s];
                                    ddR_Z[B][A][t][s] += common_dd * ooBA + common_Z * ddR_mu_roleBA[t][s];
                                    ddR_Z[B][B][t][s] += common_dd * ooBB + common_Z * ddR_mu_roleBB[t][s];
                                    ddR_Pg[A][A][t][s] += cdd_Pg * ooAA + c1_Pg * ddR_mu_roleAA[t][s];
                                    ddR_Pg[A][B][t][s] += cdd_Pg * ooAB + c1_Pg * ddR_mu_roleAB[t][s];
                                    ddR_Pg[B][A][t][s] += cdd_Pg * ooBA + c1_Pg * ddR_mu_roleBA[t][s];
                                    ddR_Pg[B][B][t][s] += cdd_Pg * ooBB + c1_Pg * ddR_mu_roleBB[t][s];
                                }
                            }
                        }
                    }
                }

                // fill derivatives
                let mut dw = vec![[SIMD_0; 3]; natm];
                let inv_Z = SIMD_1_0 / Z;
                for A in 0..natm {
                    for t in 0..3 {
                        dw[A][t] = wquad * inv_Z * (dR_Pg[A][t] - Pg * inv_Z * dR_Z[A][t]);
                    }
                }

                // apply translation invariance
                let mut dw_g = [SIMD_0; 3];
                let mut dw_neg_sum = [SIMD_0; 3];
                for A in 0..natm {
                    let mask = atm_idx.map(|a| a == A);
                    for t in 0..3 {
                        dw_neg_sum[t] -= dw[A][t];
                        dw_g[t] = dw[A][t].mask_select(mask, dw_g[t]);
                    }
                }
                for g in 0..SIMDD {
                    let atm_g = atm_idx[g];
                    if atm_g < natm {
                        for t in 0..3 {
                            dw[atm_g][t][g] = dw_neg_sum[t][g] + dw_g[t][g];
                        }
                    }
                }

                // write back to output buffer
                let g_start = g0 + lane * SIMDD;
                let g_end = (g_start + SIMDD).min(g1);
                let nlane_g = g_end - g_start;
                if let Some(dw_buf) = output.dw.as_ref() {
                    let dweights = unsafe { cast_mut_slice(dw_buf) };
                    for A in 0..natm {
                        for t in 0..3 {
                            let base = A * 3 * ngrids + t * ngrids + g_start;
                            dweights[base..base + nlane_g].copy_from_slice(&dw[A][t].0[..nlane_g]);
                        }
                    }
                }

                // contract dw -> dc:  dc[A, t, iset] += sum_g contract_dw[iset, g] * dw[A, t, g]
                if let Some(cdw) = contract_dw.as_ref() {
                    let dcp = dc_partial.as_mut().unwrap();
                    let nset = nset_dw.unwrap();
                    // load this lane's contraction weights once per set, reuse across (A, t)
                    for iset in 0..nset {
                        cw_lanes_dw[iset] = load_simd_pad(&cdw[iset][g_start..g_end]);
                    }
                    for A in 0..natm {
                        for t in 0..3 {
                            let dwv = dw[A][t];
                            for iset in 0..nset {
                                dcp[(A * 3 + t) * nset + iset] += sum_lanes(dwv * cw_lanes_dw[iset], nlane_g);
                            }
                        }
                    }
                }

                // --- deriv 2 (cross term, quotient rule, translation invariance) --- //

                if do_deriv2 {
                    let dR_log_P = dR_log_P.as_ref().unwrap();
                    let ddR_Z = ddR_Z.as_mut().unwrap();
                    let ddR_Pg = ddR_Pg.as_mut().unwrap();

                    // gather M = A_g for the ddR_Pg cross term (dlog_Ag[A][t], P_Ag) first, so the
                    // two independent cross-term accumulations below can share one (A,B,t,s) loop.
                    let mut dlog_Ag = vec![[SIMD_0; 3]; natm]; // [A][t]
                    let mut P_Ag = SIMD_0;
                    for A in 0..natm {
                        for t in 0..3 {
                            let mut v = SIMD_0;
                            for M in 0..natm {
                                v = dR_log_P[M][A][t].mask_select(atm_idx.map(|a| a == M), v);
                            }
                            dlog_Ag[A][t] = v;
                        }
                        P_Ag = P[A].mask_select(atm_idx.map(|a| a == A), P_Ag);
                    }
                    // cross terms (one shared (A,B,t,s) loop):
                    //   ddR_Z  += sum_M P_M (dlog_M_A)(dlog_M_B)
                    //   ddR_Pg += P_Ag (dlog_Ag_A)(dlog_Ag_B)
                    for A in 0..natm {
                        for B in 0..natm {
                            for t in 0..3 {
                                for s in 0..3 {
                                    let mut acc = SIMD_0;
                                    for M in 0..natm {
                                        acc += P[M] * dR_log_P[M][A][t] * dR_log_P[M][B][s];
                                    }
                                    ddR_Z[A][B][t][s] += acc;
                                    ddR_Pg[A][B][t][s] += P_Ag * dlog_Ag[A][t] * dlog_Ag[B][s];
                                }
                            }
                        }
                    }

                    // quotient rule for ddw (r_g fixed): q = Pg / Z,
                    //   d2q = (ddR_Pg - (dq_B)(dZ_A) - q ddR_Z) / Z - (dq_A)(dZ_B) / Z
                    let inv_Z = SIMD_1_0 / Z;
                    let q = Pg * inv_Z;
                    let mut dq = vec![[SIMD_0; 3]; natm]; // [A][t]
                    for A in 0..natm {
                        for t in 0..3 {
                            dq[A][t] = (dR_Pg[A][t] - q * dR_Z[A][t]) * inv_Z;
                        }
                    }
                    // ddw_partial[A][B][t][s] = wquad * d2q; the translation-invariance axis sums
                    // (fullA = sum_A, fullB = sum_B, fullAB = sum_{A,B}) are accumulated in the
                    // same pass over (A,B,t,s) so no separate sum loop is needed.
                    let mut ddw = vec![vec![[[SIMD_0; 3]; 3]; natm]; natm]; // [A][B][t][s]
                    let mut fullA = vec![[[SIMD_0; 3]; 3]; natm]; // [B][t][s] = sum_A ddw[A][B][t][s]
                    let mut fullB = vec![[[SIMD_0; 3]; 3]; natm]; // [A][t][s] = sum_B ddw[A][B][t][s]
                    let mut fullAB = [[SIMD_0; 3]; 3]; // [t][s] = sum_A sum_B ddw[A][B][t][s]
                    for A in 0..natm {
                        for B in 0..natm {
                            for t in 0..3 {
                                for s in 0..3 {
                                    let term1 = dq[B][s] * dR_Z[A][t];
                                    let term2 = dq[A][t] * dR_Z[B][s];
                                    let d2q =
                                        (ddR_Pg[A][B][t][s] - term1 - q * ddR_Z[A][B][t][s]) * inv_Z - term2 * inv_Z;
                                    let v = wquad * d2q;
                                    ddw[A][B][t][s] = v;
                                    fullA[B][t][s] += v;
                                    fullB[A][t][s] += v;
                                    fullAB[t][s] += v;
                                }
                            }
                        }
                    }

                    // translation-invariance fix (mirrors the deriv1 pattern): the quotient-rule
                    // partial value is already correct for A,B != A_g; only the A=A_g row and
                    // B=A_g column need the fix.  The axis sums are accumulated in f64simd and the
                    // per-lane `for g` only writes back that row/column (minimal scalar work).
                    //   row : ddw[A_g, t, B, s]   = -sum_{A'!=A_g} = -fullA[B,t,s] + ddw_partial[A_g,B,t,s]
                    //   col : ddw[A, t, A_g, s]   = -sum_{B'!=A_g} = -fullB[A,t,s] + ddw_partial[A,A_g,t,s]
                    //   corner: ddw[A_g,t,A_g,s] =  sum_{A'!=A_g,B'!=A_g}
                    //                            = fullAB - fullB[A_g] - fullA[A_g] + ddw_partial[A_g,A_g]
                    // per-lane write-back of only the A=A_g row and B=A_g column
                    for g in 0..SIMDD {
                        let atm_g = atm_idx[g];
                        if atm_g >= natm {
                            continue;
                        }
                        // row A = atm_g: B == atm_g is the corner, otherwise the row value
                        //   (reads ddw[atm_g][B][t][s][g] = ddw_partial, still unmodified here)
                        for B in 0..natm {
                            for t in 0..3 {
                                for s in 0..3 {
                                    ddw[atm_g][B][t][s][g] = if B == atm_g {
                                        fullAB[t][s][g] - fullB[atm_g][t][s][g] - fullA[atm_g][t][s][g]
                                            + ddw[atm_g][atm_g][t][s][g]
                                    } else {
                                        -fullA[B][t][s][g] + ddw[atm_g][B][t][s][g]
                                    };
                                }
                            }
                        }
                        // column B = atm_g (A != atm_g; the A = atm_g corner is already written
                        //   above, and ddw[A][atm_g] is still ddw_partial since the row only
                        //   touched A = atm_g)
                        for A in 0..natm {
                            if A == atm_g {
                                continue;
                            }
                            for t in 0..3 {
                                for s in 0..3 {
                                    ddw[A][atm_g][t][s][g] -= fullB[A][t][s][g];
                                }
                            }
                        }
                    }

                    // write back to output buffer; flat index is C-order for [A, t, B, s, g].
                    if let Some(ddw_buf) = output.ddw.as_ref() {
                        let ddweights = unsafe { cast_mut_slice(ddw_buf) };
                        for A in 0..natm {
                            for t in 0..3 {
                                for B in 0..natm {
                                    for s in 0..3 {
                                        let base = ((A * 3 + t) * natm + B) * (3 * ngrids) + s * ngrids + g_start;
                                        ddweights[base..base + nlane_g].copy_from_slice(&ddw[A][B][t][s].0[..nlane_g]);
                                    }
                                }
                            }
                        }
                    }

                    // contract ddw -> ddc:
                    //   ddc[A, t, B, s, iset] += sum_g contract_ddw[iset, g] * ddw[A, t, B, s, g]
                    // This is the contraction that lets the caller obtain a 2nd-order contracted
                    // weight without ever materializing the full ddw tensor.
                    if let Some(cddw) = contract_ddw.as_ref() {
                        let ddcp = ddc_partial.as_mut().unwrap();
                        let nset = nset_ddw.unwrap();
                        // load this lane's contraction weights once per set, reuse across (A, t, B, s)
                        for iset in 0..nset {
                            cw_lanes_ddw[iset] = load_simd_pad(&cddw[iset][g_start..g_end]);
                        }
                        for A in 0..natm {
                            for t in 0..3 {
                                for B in 0..natm {
                                    for s in 0..3 {
                                        let ddwv = ddw[A][B][t][s];
                                        for iset in 0..nset {
                                            ddcp[((A * 3 + t) * natm + B) * 3 * nset + s * nset + iset] +=
                                                sum_lanes(ddwv * cw_lanes_ddw[iset], nlane_g);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // reduce this task's private partials into the shared output buffers.
        // Held under `contract_mutex` because c/dc/ddc are sums over grids into a
        // per-set output that every task writes - the guard serializes the `+=` so
        // no updates are lost.  The cast_mut_slice write is sound under the guard:
        // the mutex grants exclusive access (no concurrent reader/writer), and the
        // &mut-from-& derivation is the same pattern the w/dw/ddw write-back uses.
        if c_partial.is_some() || dc_partial.is_some() || ddc_partial.is_some() {
            let _guard = contract_guard.lock().unwrap();
            if let (Some(c_buf), Some(cp)) = (output.c.as_ref(), c_partial.as_ref()) {
                let cslc = unsafe { cast_mut_slice(c_buf) };
                for (o, v) in cslc.iter_mut().zip(cp.iter()) {
                    *o += v;
                }
            }
            if let (Some(dc_buf), Some(dcp)) = (output.dc.as_ref(), dc_partial.as_ref()) {
                let dcslc = unsafe { cast_mut_slice(dc_buf) };
                for (o, v) in dcslc.iter_mut().zip(dcp.iter()) {
                    *o += v;
                }
            }
            if let (Some(ddc_buf), Some(ddcp)) = (output.ddc.as_ref(), ddc_partial.as_ref()) {
                let ddcslc = unsafe { cast_mut_slice(ddc_buf) };
                for (o, v) in ddcslc.iter_mut().zip(ddcp.iter()) {
                    *o += v;
                }
            }
        }
    });

    output
}

/* #region simple utilities */

fn dist3_naive(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn dist3_hybrid(a: &[f64simd; 3], b: &[f64; 3]) -> f64simd {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).map(f64::sqrt)
}

/// Lane-wise negation of a length-3 SIMD vector.
fn neg3(v: [f64simd; 3]) -> [f64simd; 3] {
    [-v[0], -v[1], -v[2]]
}

/// Lane-wise negation of a 3x3 SIMD matrix (over `(t, s)`).
fn neg33(m: [[f64simd; 3]; 3]) -> [[f64simd; 3]; 3] {
    [[-m[0][0], -m[0][1], -m[0][2]], [-m[1][0], -m[1][1], -m[1][2]], [-m[2][0], -m[2][1], -m[2][2]]]
}

/// Load up to `SIMDD` elements from `slc` into a SIMD register, zero-padding the
/// remaining lanes.  Used to contract a per-grid SIMD intermediate against the
/// contraction-weight slice for the current lane (whose tail may be shorter than
/// `SIMDD` at the final grid batch).
#[inline(always)]
fn load_simd_pad(slc: &[f64]) -> f64simd {
    let mut s = SIMD_0;
    for i in 0..slc.len() {
        s[i] = slc[i];
    }
    s
}

/// Horizontal sum of the first `n` lanes of a SIMD register (`n <= SIMDD`).  The
/// trailing lanes are ignored; the caller passes `nlane_g` so padding lanes
/// (already zero in the contraction weight) are never touched.
#[inline(always)]
fn sum_lanes(s: f64simd, n: usize) -> f64 {
    // `n` is always <= SIMDD (== 8); hint this to the optimizer so the lane
    // accumulation loop can be fully unrolled.  Maintained by the caller, which
    // passes `nlane_g = g_end - g_start <= SIMDD`.
    unsafe { std::hint::assert_unchecked(n <= SIMDD) };
    let mut acc = 0.0;
    for i in 0..n {
        acc += s[i];
    }
    acc
}

/* #endregion */

/* #region switch function utilities */

fn switch_f3(mu: f64simd, a_factor: f64) -> f64simd {
    let nu = mu + (SIMD_1_0 - mu * mu) * a_factor; // eq (A2)
    let f1 = (SIMD_1_5 - SIMD_0_5 * nu * nu) * nu; // eq (19)
    let f2 = (SIMD_1_5 - SIMD_0_5 * f1 * f1) * f1; // eq (19)
    let f3 = (SIMD_1_5 - SIMD_0_5 * f2 * f2) * f2; // eq (19)
    f3
}

fn switch_f_hardness(mu: f64simd, a_factor: f64, hardness: usize) -> f64simd {
    let nu = mu + (SIMD_1_0 - mu * mu) * a_factor; // eq (A2)
    let mut f = nu;
    for _ in 0..hardness {
        f = (SIMD_1_5 - SIMD_0_5 * f * f) * f; // eq (19)
    }
    f
}

fn switch_dnu_f3(mu: f64simd, a_factor: f64) -> (f64simd, f64simd) {
    let nu = mu + (SIMD_1_0 - mu * mu) * a_factor; // eq (A2)
    let f1 = (SIMD_1_5 - SIMD_0_5 * nu * nu) * nu; // eq (19)
    let f2 = (SIMD_1_5 - SIMD_0_5 * f1 * f1) * f1; // eq (19)
    let f3 = (SIMD_1_5 - SIMD_0_5 * f2 * f2) * f2; // eq (19)

    let df1 = SIMD_1_5 * (SIMD_1_0 - nu * nu);
    let df2 = SIMD_1_5 * (SIMD_1_0 - f1 * f1) * df1;
    let df3 = SIMD_1_5 * (SIMD_1_0 - f2 * f2) * df2;
    (f3, df3)
}

fn switch_dnu_f_hardness(mu: f64simd, a_factor: f64, hardness: usize) -> (f64simd, f64simd) {
    let nu = mu + (SIMD_1_0 - mu * mu) * a_factor; // eq (A2)
    let mut f = nu;
    let mut df = SIMD_1_0;
    for _ in 0..hardness {
        df = SIMD_1_5 * (SIMD_1_0 - f * f) * df;
        f = (SIMD_1_5 - SIMD_0_5 * f * f) * f; // eq (19)
    }
    (f, df)
}

/// Switch function `f3(nu)` together with its 1st and 2nd derivatives wrt `nu`, where
/// `nu = mu + a(1 - mu^2)` and `f3 = p∘p∘p(nu)`, `p(x) = 3/2 x − 1/2 x^3` (hardness = 3).
///
/// With `g_i = p'(f_{i-1})` (`f_0 = nu`), `p'(x) = 3/2(1 − x^2)`, `p''(x) = −3x`:
/// `f3'(nu) = g2 g1 g0`, `f3''(nu) = −3 [f2 (g1 g0)^2 + f1 g2 g0^2 + nu g2 g1]`.
fn switch_d2nu_f3(mu: f64simd, a_factor: f64) -> (f64simd, f64simd, f64simd) {
    let nu = mu + (SIMD_1_0 - mu * mu) * a_factor; // eq (A2)
    let f1 = (SIMD_1_5 - SIMD_0_5 * nu * nu) * nu; // eq (19)
    let f2 = (SIMD_1_5 - SIMD_0_5 * f1 * f1) * f1; // eq (19)
    let f3 = (SIMD_1_5 - SIMD_0_5 * f2 * f2) * f2; // eq (19)

    let g0 = SIMD_1_5 * (SIMD_1_0 - nu * nu);
    let g1 = SIMD_1_5 * (SIMD_1_0 - f1 * f1);
    let g2 = SIMD_1_5 * (SIMD_1_0 - f2 * f2);
    let f3p = g2 * g1 * g0;
    let f3pp = -SIMD_3_0 * (f2 * (g1 * g0) * (g1 * g0) + f1 * g2 * g0 * g0 + nu * g2 * g1);
    (f3, f3p, f3pp)
}

/// Arbitrary-hardness variant of [`switch_d2nu_f3`]: returns `f, f'(nu), f''(nu)` for
/// `f = p^hardness(nu)`. Loop recurrence (compute `ddf` first with old `f, df`, then `df`,
/// then `f`), `g = p'(f) = 3/2(1 − f^2)`, `p''(x) = −3x`: `ddf = −3 f df^2 + g ddf`.
fn switch_d2nu_f_hardness(mu: f64simd, a_factor: f64, hardness: usize) -> (f64simd, f64simd, f64simd) {
    let nu = mu + (SIMD_1_0 - mu * mu) * a_factor; // eq (A2)
    let mut f = nu;
    let mut df = SIMD_1_0;
    let mut ddf = SIMD_0;
    for _ in 0..hardness {
        let g = SIMD_1_5 * (SIMD_1_0 - f * f);
        ddf = -SIMD_3_0 * f * df * df + g * ddf; // f'' recurrence (old f, df, ddf)
        df = g * df; // f'  recurrence (old df)
        f = (SIMD_1_5 - SIMD_0_5 * f * f) * f; // f = p(f) (old f)
    }
    (f, df, ddf)
}

/* #endregion */

/* #region enhancement to FpSimd */

trait FpSimdEnhanceAPI<T> {
    fn mask_select(self, mask: [bool; SIMDD], other: Self) -> Self;
    fn max_compare(self, val: T) -> Self
    where
        T: PartialOrd;
}

impl<T: Copy> FpSimdEnhanceAPI<T> for FpSimd<T> {
    fn mask_select(self, mask: [bool; SIMDD], other: Self) -> Self {
        let mut result = self;
        for i in 0..SIMDD {
            match mask[i] {
                true => result[i] = self[i],
                false => result[i] = other[i],
            }
        }
        result
    }

    fn max_compare(mut self, val: T) -> Self
    where
        T: PartialOrd,
    {
        for i in 0..SIMDD {
            self[i] = if self[i] > val { self[i] } else { val };
        }
        self
    }
}

/* #endregion */

/* #region other utilities */

#[allow(clippy::mut_from_ref)]
unsafe fn cast_mut_slice<T>(slc: &[T]) -> &mut [T] {
    let len = slc.len();
    let ptr = slc.as_ptr() as *mut T;
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/* #endregion */
