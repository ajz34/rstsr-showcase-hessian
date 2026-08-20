#![allow(clippy::deref_addrof)]

pub mod test_b3lyp;
pub mod test_svwn;
pub mod test_tpss0;

use crate::test_util::{read_npz, read_npz_dict};
use libcint::prelude::*;
use rstest::fixture;
use rstsr_showcase_hessian::prelude_dev::*;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CaseAmoniaRKSBecke {
    pub mol: CInt,
    pub aux: CInt,
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    pub mo_energy: Tsr,
    pub grid_coords: Vec<[f64; 3]>,
    pub grid_weights: Vec<f64>,
    pub quadrature_weights: Vec<f64>,
    pub atm_quad_split: Vec<usize>,
    pub adjustment_factor: Vec<f64>,
    pub xc: String,
    pub ref_dict: HashMap<String, Tsr>,
}

/// NH3 case with atom-grouped grids (prototype/12-1-export_becke_ref.ipynb).
///
/// Grid/becke data is xc-independent and comes from the shared
/// `nh3_grid_becke.npz`; `mo_*` and the grid-order-independent references
/// come from the existing `nh3_r_{xc}.npz` / `nh3_r_{xc}_decomp.npz`.
pub fn hess_case_becke(xc: &str) -> CaseAmoniaRKSBecke {
    let toml_token = r#"
        atom = """
            N  0   0   0
            H  1.0 0.1 0.2
            H  0.3 1.1 0.2
            H  0.1 0.1 1.2
        """
        basis = "BASIS"
    "#;
    let mol = CIntMol::from_toml(toml_token.replace("BASIS", "def2-TZVP").as_str()).cint;
    let aux = CIntMol::from_toml(toml_token.replace("BASIS", "def2-universal-jkfit").as_str()).cint;

    let mo_coeff = read_npz(&format!("nh3_r_{}.npz", xc), "mo_coeff").into_contig(ColMajor);
    let mo_occ = read_npz(&format!("nh3_r_{}.npz", xc), "mo_occ").into_contig(ColMajor);
    let mo_energy = read_npz(&format!("nh3_r_{}.npz", xc), "mo_energy").into_contig(ColMajor);
    let ref_dict = read_npz_dict(&format!("nh3_r_{}_decomp.npz", xc));

    let d = read_npz_dict("nh3_grid_becke.npz");
    let grid_coords = d["grid_coords"].to_owned().into_pack_array::<3>(-1).into_vec();
    let grid_weights = d["grid_weights"].to_owned().into_vec();
    let quadrature_weights = d["quadrature_weights"].to_owned().into_vec();
    // atm_idx is stored as float64 (read_npz loads f64 only); grids are atom-grouped
    let atm_idx = d["atm_idx"].to_owned().mapv(|i| i as usize).into_vec();
    let mut atm_quad_split = vec![0usize];
    for i in 1..atm_idx.len() {
        if atm_idx[i] != atm_idx[i - 1] {
            atm_quad_split.push(i);
        }
    }
    atm_quad_split.push(atm_idx.len());
    // be careful about col/row-major order; adjustment_factor is anti-symmetric.
    let adjustment_factor = d["radii_table"].to_owned().into_shape(-1).into_vec();

    CaseAmoniaRKSBecke {
        mol,
        aux,
        mo_coeff,
        mo_occ,
        mo_energy,
        grid_coords,
        grid_weights,
        quadrature_weights,
        atm_quad_split,
        adjustment_factor,
        xc: xc.to_string(),
        ref_dict,
    }
}

#[fixture]
#[once]
pub fn hess_case_svwn() -> CaseAmoniaRKSBecke {
    hess_case_becke("svwn")
}

#[fixture]
#[once]
pub fn hess_case_b3lyp() -> CaseAmoniaRKSBecke {
    hess_case_becke("b3lyp")
}

#[fixture]
#[once]
pub fn hess_case_tpss0() -> CaseAmoniaRKSBecke {
    hess_case_becke("tpss0")
}
