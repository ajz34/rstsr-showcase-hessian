mod test_util;

use rstsr::prelude::*;
use rstsr_showcase_hessian::numint_matmul::becke_partition::becke_partition;

use test_util::read_npz_dict;

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
    let device = w_ref.device().clone();

    let time = std::time::Instant::now();
    let w = becke_partition(&grid_coords, &atm_coords, &atm_indices, &quadrature_weights, &adjustment_factor, 3, 512);
    println!("becke_partition time: {:?}", time.elapsed());

    let w = rt::asarray((w, &device));
    assert!(rt::allclose(&w, &w_ref, None));
}
