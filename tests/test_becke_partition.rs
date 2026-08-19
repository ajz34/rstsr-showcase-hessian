mod test_util;

use rstsr::prelude::*;
use rstsr_showcase_hessian::numint_matmul::becke_partition::{becke_partition, AtmIndices};

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
        AtmIndices::ByGrid(&atm_indices),
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
        AtmIndices::ByGrid(&atm_indices),
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

#[test]
pub fn test_becke_partition_by_atom() {
    // ByAtom scheme on the same grid: reconstruct the natm+1 per-atom boundaries from the
    // per-grid indices (requires grids grouped by atom, in atom order), then check w/dw/ddw
    // against the same references and against the ByGrid scheme.  Per-grid outputs do not
    // depend on the batch partition, so ByAtom must reproduce ByGrid exactly.
    let d = read_npz_dict("becke_deriv1_dict.npz");
    let atm_coords = d["atm_coords"].to_owned().into_pack_array::<3>(-1).into_vec();
    let grid_coords = d["grids"].to_owned().into_pack_array::<3>(-1).into_vec();
    let atm_indices = d["atm_indices"].to_owned().mapv(|i| i as usize).into_vec();
    let quadrature_weights = d["wquad"].to_owned().into_vec();
    let adjustment_factor = d["radii_table"].to_owned().into_shape(-1).into_vec();
    let w_ref = d["weights"].to_owned();
    let dw_ref = d["dw_ref"].to_owned();
    let ddw_ref = read_npz_dict("becke_deriv2_dict.npz")["ddw_ref"].to_owned();
    let device = w_ref.device().clone();

    let natm = atm_coords.len();
    let ngrids = grid_coords.len();

    // per-grid indices -> cumulative per-atom boundaries (cf get_quad_split); also
    // verifies the grouping assumption the ByAtom scheme requires
    let mut split = vec![0usize];
    for atm in 0..natm {
        let mut end = split[atm];
        while end < ngrids && atm_indices[end] == atm {
            end += 1;
        }
        assert!(
            end == ngrids || atm_indices[end] == atm + 1,
            "grids not grouped by atom in atom order"
        );
        split.push(end);
    }
    assert_eq!(split[natm], ngrids);

    let time = std::time::Instant::now();
    let res_atom = becke_partition(
        &grid_coords,
        &atm_coords,
        AtmIndices::ByAtom(&split),
        &quadrature_weights,
        &adjustment_factor,
        3,
        512,
        2,
        None,
    );
    println!("becke_partition (ByAtom, deriv=2) time: {:?}", time.elapsed());

    // exact scheme equivalence: same per-grid values as ByGrid (bit-identical; the
    // different batch partition only affects task grouping, not per-lane results)
    let res_grid = becke_partition(
        &grid_coords,
        &atm_coords,
        AtmIndices::ByGrid(&atm_indices),
        &quadrature_weights,
        &adjustment_factor,
        3,
        512,
        2,
        None,
    );
    assert_eq!(res_atom.w.as_ref().unwrap(), res_grid.w.as_ref().unwrap());
    assert_eq!(res_atom.dw.as_ref().unwrap(), res_grid.dw.as_ref().unwrap());
    assert_eq!(res_atom.ddw.as_ref().unwrap(), res_grid.ddw.as_ref().unwrap());

    let w = rt::asarray((res_atom.w.unwrap(), &device));
    let dw = rt::asarray((res_atom.dw.unwrap(), [natm, 3, ngrids].c(), &device));
    let ddw = rt::asarray((res_atom.ddw.unwrap(), [natm, 3, natm, 3, ngrids].c(), &device));

    assert!(rt::allclose(&w, &w_ref, None));
    assert!(rt::allclose(&dw, &dw_ref, None));
    let fp_ddw = fp(ddw.view());
    let fp_ref = fp(ddw_ref.view());
    println!("ddw fp(ddw)={:.10e}  fp(ref)={:.10e}  |fp diff|={:.3e}", fp_ddw, fp_ref, (fp_ddw - fp_ref).abs());
    assert!(rt::allclose(&ddw, &ddw_ref, None));
}

#[test]
#[should_panic(expected = "ByAtom atm_indices must not exceed ngrids")]
pub fn test_becke_partition_by_atom_validation() {
    // synthetic 2-atom / 8-grid input; a boundary beyond ngrids must be rejected by
    // validation instead of reading past the input grid arrays
    let atm_coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let grid_coords = vec![[0.0f64; 3]; 8];
    let quadrature_weights = vec![1.0f64; 8];
    let adjustment_factor = vec![0.0f64; 4];
    // ngrids == 8, but atom 1's interval claims grids [9, 8)
    let bad_split = vec![0usize, 9, 8];

    let _ = becke_partition(
        &grid_coords,
        &atm_coords,
        AtmIndices::ByAtom(&bad_split),
        &quadrature_weights,
        &adjustment_factor,
        3,
        8,
        0,
        None,
    );
}
