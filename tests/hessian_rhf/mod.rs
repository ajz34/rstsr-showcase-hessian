pub mod test_core;

use crate::test_util::{read_npz, read_npz_dict};
use libcint::prelude::*;
use rstest::fixture;
use rstsr_showcase_hessian::prelude_dev::*;

#[derive(Debug, Clone)]
pub struct CaseAmoniaRHF {
    pub mol: CInt,
    pub aux: CInt,
    pub mo_coeff: Tsr<f64>,
    pub mo_occ: Tsr<f64>,
    pub ref_dict: HashMap<String, Tsr<f64>>,
}

#[fixture]
pub fn hess_case() -> CaseAmoniaRHF {
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

    let mo_coeff = read_npz("nh3_r_hf.npz", "mo_coeff").into_contig(ColMajor);
    let mo_occ = read_npz("nh3_r_hf.npz", "mo_occ").into_contig(ColMajor);
    let ref_dict = read_npz_dict("nh3_r_hf_decomp.npz");

    CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, ref_dict }
}
