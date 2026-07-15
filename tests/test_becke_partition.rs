mod test_util;

use rstsr::prelude::*;
use rstsr_showcase_hessian::numint_matmul::becke_partition::becke_partition;

use test_util::{fp, read_npz_dict};

#[test]
pub fn test_becke_partition_0() {
    let d = read_npz_dict("becke_deriv1_dict.npz");
    println!("keys: {:?}", d.keys());
    let atm_coords = d["atm_coords"].to_owned().into_pack_array::<3>(-1).into_vec();
    let grid_coords = d["grids"].to_owned().into_pack_array::<3>(-1).into_vec();
    let atm_indices = d["atm_indices"].to_owned().mapv(|i| i as usize).into_vec();
    let quadrature_weights = d["wquad"].to_owned().into_vec();
    // be careful about col/row-major order; adjustment_factor is anti-symmetric.
    let adjustment_factor = d["radii_table"].to_owned().into_shape(-1).into_vec();
    let w_ref = d["weights"].to_owned();
    let dw_ref = d["dw_ref"].to_owned();
    let device = w_ref.device().clone();

    let time = std::time::Instant::now();
    let res = becke_partition(
        &grid_coords,
        &atm_coords,
        &atm_indices,
        &quadrature_weights,
        &adjustment_factor,
        3,
        512,
        1,
        None,
    );
    let w = res.w.unwrap();
    let dw = res.dw;
    println!("becke_partition time: {:?}", time.elapsed());

    let natm = atm_coords.len();
    let ngrids = grid_coords.len();
    let w = rt::asarray((w, &device));
    let dw = rt::asarray((dw.unwrap(), [natm, 3, ngrids].c(), &device));

    assert!(rt::allclose(&w, &w_ref, None));
    assert!(rt::allclose(&dw, &dw_ref, None));
}

#[test]
pub fn test_becke_partition_2() {
    // inputs (same molecule/grid as becke_deriv1_dict.npz) + 1st-order reference
    let d = read_npz_dict("becke_deriv1_dict.npz");
    let atm_coords = d["atm_coords"].to_owned().into_pack_array::<3>(-1).into_vec();
    let grid_coords = d["grids"].to_owned().into_pack_array::<3>(-1).into_vec();
    let atm_indices = d["atm_indices"].to_owned().mapv(|i| i as usize).into_vec();
    let quadrature_weights = d["wquad"].to_owned().into_vec();
    // be careful about col/row-major order; adjustment_factor is anti-symmetric.
    let adjustment_factor = d["radii_table"].to_owned().into_shape(-1).into_vec();
    let w_ref = d["weights"].to_owned();
    let dw_ref = d["dw_ref"].to_owned();
    // 2nd-order analytical reference (produced by 10-5-becke_rsprep_deriv2.ipynb)
    let d2 = read_npz_dict("becke_deriv2_dict.npz");
    let ddw_ref = d2["ddw_ref"].to_owned();
    let device = w_ref.device().clone();

    let time = std::time::Instant::now();
    let res = becke_partition(
        &grid_coords,
        &atm_coords,
        &atm_indices,
        &quadrature_weights,
        &adjustment_factor,
        3,
        512,
        2,
        None,
    );
    let w = res.w.unwrap();
    let dw = res.dw.unwrap();
    let ddw = res.ddw.unwrap();
    println!("becke_partition (deriv=2) time: {:?}", time.elapsed());

    let natm = atm_coords.len();
    let ngrids = grid_coords.len();
    let w = rt::asarray((w, &device));
    let dw = rt::asarray((dw, [natm, 3, ngrids].c(), &device));
    let ddw = rt::asarray((ddw, [natm, 3, natm, 3, ngrids].c(), &device));

    // sanity: w and dw still match (deriv=2 path uses the s_safe=1.0 regularization for dw)
    assert!(rt::allclose(&w, &w_ref, None));
    assert!(rt::allclose(&dw, &dw_ref, None));
    // 2nd derivative: analytical-vs-analytical (same algorithm as the notebook), expect
    // near machine-precision agreement.  Default allclose (atol=1e-8, rtol=1e-5) only rules out
    // gross errors, so also fingerprint-compare (sensitive to any systematic term/sign mistake).
    let fp_ddw = fp(ddw.view());
    let fp_ref = fp(ddw_ref.view());
    println!("ddw fp(ddw)={:.10e}  fp(ref)={:.10e}  |fp diff|={:.3e}", fp_ddw, fp_ref, (fp_ddw - fp_ref).abs());
    assert!(rt::allclose(&ddw, &ddw_ref, None));
}
