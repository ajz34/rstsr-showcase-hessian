mod test_component_core;
mod test_component_rijk_naive;

use crate::test_util::{read_npz, read_npz_dict};
use libcint::prelude::*;
use rstest::fixture;
use rstsr_showcase_hessian::prelude_dev::*;

#[derive(Debug, Clone)]
pub struct CaseAmoniaUHF {
    pub mol: CInt,
    pub aux: CInt,
    pub mo_coeff: [Tsr; 2],
    pub mo_occ: [Tsr; 2],
    pub mo_energy: [Tsr; 2],
    pub ref_dict: HashMap<String, Tsr>,
}

#[fixture]
#[once]
pub fn hess_case() -> CaseAmoniaUHF {
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

    let ref_dict = read_npz_dict("nh3_u_hf_decomp.npz");

    let mo_coeff = read_npz("nh3_u_hf.npz", "mo_coeff").into_contig(ColMajor);
    let mo_occ = read_npz("nh3_u_hf.npz", "mo_occ").into_contig(ColMajor);
    let mo_energy = read_npz("nh3_u_hf.npz", "mo_energy").into_contig(ColMajor);

    let [α, β] = [0, 1];
    let mo_coeff = [mo_coeff.i(α).into_contig(ColMajor), mo_coeff.i(β).into_contig(ColMajor)];
    let mo_occ = [mo_occ.i(α).to_owned(), mo_occ.i(β).to_owned()];
    let mo_energy = [mo_energy.i(α).to_owned(), mo_energy.i(β).to_owned()];

    CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, mo_energy, ref_dict }
}
