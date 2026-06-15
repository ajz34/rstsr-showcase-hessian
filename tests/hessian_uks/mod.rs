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
pub struct CaseAmoniaUKS {
    pub mol: CInt,
    pub aux: CInt,
    pub mo_coeff: [Tsr; 2],
    pub mo_occ: [Tsr; 2],
    pub mo_energy: [Tsr; 2],
    pub grid_coords: Vec<[f64; 3]>,
    pub grid_weights: Vec<f64>,
    pub xc: String,
    pub ref_dict: HashMap<String, Tsr>,
}

pub fn hess_case(xc: &str) -> CaseAmoniaUKS {
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

    let ref_dict = read_npz_dict(&format!("nh3_u_{}_decomp.npz", xc));

    // UKS mo_coeff has shape [2, nao, nmo] in Python → transposed to [nao, nmo] per spin in Rust
    let mo_coeff_np = read_npz(&format!("nh3_u_{}.npz", xc), "mo_coeff");
    let mo_occ_np = read_npz(&format!("nh3_u_{}.npz", xc), "mo_occ");
    let mo_energy_np = read_npz(&format!("nh3_u_{}.npz", xc), "mo_energy");

    // Python shape: [2, nao, nmo] → in numpy row-major → loaded as [2, nao, nmo]
    // We need [nao, nmo] per spin, in column-major
    let nao = mol.nao();
    let nmo = mo_coeff_np.shape()[1];
    let mo_coeff_a = mo_coeff_np.i((0, .., ..)).reshape([nao, nmo]).into_contig(ColMajor);
    let mo_coeff_b = mo_coeff_np.i((1, .., ..)).reshape([nao, nmo]).into_contig(ColMajor);

    // Python shape: [2, nmo] → [nmo] per spin
    let mo_occ_a = mo_occ_np.i((0, ..)).into_contig(ColMajor);
    let mo_occ_b = mo_occ_np.i((1, ..)).into_contig(ColMajor);
    let mo_energy_a = mo_energy_np.i((0, ..)).into_contig(ColMajor);
    let mo_energy_b = mo_energy_np.i((1, ..)).into_contig(ColMajor);

    let grid_coords = read_npz(&format!("nh3_u_{}.npz", xc), "grid_coords").into_pack_array::<3>(-1).into_vec();
    let grid_weights = read_npz(&format!("nh3_u_{}.npz", xc), "grid_weights").into_vec();

    CaseAmoniaUKS {
        mol,
        aux,
        mo_coeff: [mo_coeff_a, mo_coeff_b],
        mo_occ: [mo_occ_a, mo_occ_b],
        mo_energy: [mo_energy_a, mo_energy_b],
        grid_coords,
        grid_weights,
        xc: xc.to_string(),
        ref_dict,
    }
}

#[fixture]
#[once]
pub fn hess_case_svwn() -> CaseAmoniaUKS {
    hess_case("svwn")
}

#[fixture]
#[once]
pub fn hess_case_b3lyp() -> CaseAmoniaUKS {
    hess_case("b3lyp")
}

#[fixture]
#[once]
pub fn hess_case_tpss0() -> CaseAmoniaUKS {
    hess_case("tpss0")
}
