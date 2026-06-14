#![allow(clippy::deref_addrof)]

pub mod test_b3lyp;
pub mod test_svwn;
pub mod test_tpss0;

use crate::test_util::{read_npz, read_npz_dict};
use libcint::prelude::*;
use rstest::fixture;
use rstsr_showcase_hessian::prelude_dev::*;

#[derive(Debug, Clone)]
pub struct CaseAmoniaRKS {
    pub mol: CInt,
    pub aux: CInt,
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    pub mo_energy: Tsr,
    pub grid_coords: Vec<[f64; 3]>,
    pub grid_weights: Vec<f64>,
    pub xc: String,
    pub ref_dict: HashMap<String, Tsr>,
}

pub fn hess_case(xc: &str) -> CaseAmoniaRKS {
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

    let grid_coords = read_npz(&format!("nh3_r_{}.npz", xc), "grid_coords").into_pack_array::<3>(-1).into_vec();
    let grid_weights = read_npz(&format!("nh3_r_{}.npz", xc), "grid_weights").into_vec();

    CaseAmoniaRKS { mol, aux, mo_coeff, mo_occ, mo_energy, grid_coords, grid_weights, xc: xc.to_string(), ref_dict }
}

#[fixture]
#[once]
pub fn hess_case_svwn() -> CaseAmoniaRKS {
    hess_case("svwn")
}

#[fixture]
#[once]
pub fn hess_case_b3lyp() -> CaseAmoniaRKS {
    hess_case("b3lyp")
}

#[fixture]
#[once]
pub fn hess_case_tpss0() -> CaseAmoniaRKS {
    hess_case("tpss0")
}
