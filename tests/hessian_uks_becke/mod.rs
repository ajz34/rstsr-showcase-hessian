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
pub struct CaseAmoniaUKSBecke {
    pub mol: CInt,
    pub aux: CInt,
    pub mo_coeff: [Tsr; 2],
    pub mo_occ: [Tsr; 2],
    pub mo_energy: [Tsr; 2],
    pub grid_coords: Vec<[f64; 3]>,
    pub grid_weights: Vec<f64>,
    pub quadrature_weights: Vec<f64>,
    pub atm_quad_split: Vec<usize>,
    pub adjustment_factor: Vec<Vec<f64>>,
    pub xc: String,
    pub ref_dict: HashMap<String, Tsr>,
}

/// NH3 UKS case with atom-grouped grids (prototype/12-1-export_becke_ref.ipynb).
///
/// Grid/becke data is xc-independent and comes from the shared
/// `nh3_grid_becke.npz`; `mo_*` and the grid-order-independent references come
/// from the existing `nh3_u_{xc}.npz` / `nh3_u_{xc}_decomp.npz`.  Note the
/// reference SCF solutions carry an exact-exchange admixture (0.1*HF + SVWN
/// for the LDA case) — the hybrid part is handled by the RIJK object in the
/// end-to-end tests, not by the xc_func_list.
pub fn hess_case_becke_uks(xc: &str) -> CaseAmoniaUKSBecke {
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

    let nao = mol.nao();
    let nmo = mo_coeff_np.shape()[1];
    let mo_coeff_a = mo_coeff_np.i((0, .., ..)).reshape([nao, nmo]).into_contig(ColMajor);
    let mo_coeff_b = mo_coeff_np.i((1, .., ..)).reshape([nao, nmo]).into_contig(ColMajor);
    let mo_occ_a = mo_occ_np.i((0, ..)).into_contig(ColMajor);
    let mo_occ_b = mo_occ_np.i((1, ..)).into_contig(ColMajor);
    let mo_energy_a = mo_energy_np.i((0, ..)).into_contig(ColMajor);
    let mo_energy_b = mo_energy_np.i((1, ..)).into_contig(ColMajor);

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
    // `into_shape(-1)` flattens the C-order table column-major; un-transpose it to
    // row-major rows (rows[A][B] = table entry (A, B)).
    let natm = mol.natm();
    let radii_flat = d["radii_table"].to_owned().into_shape(-1).into_vec();
    let adjustment_factor: Vec<Vec<f64>> =
        (0..natm).map(|a| (0..natm).map(|b| radii_flat[b * natm + a]).collect()).collect();

    CaseAmoniaUKSBecke {
        mol,
        aux,
        mo_coeff: [mo_coeff_a, mo_coeff_b],
        mo_occ: [mo_occ_a, mo_occ_b],
        mo_energy: [mo_energy_a, mo_energy_b],
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
pub fn hess_case_svwn() -> CaseAmoniaUKSBecke {
    hess_case_becke_uks("svwn")
}

#[fixture]
#[once]
pub fn hess_case_b3lyp() -> CaseAmoniaUKSBecke {
    hess_case_becke_uks("b3lyp")
}

#[fixture]
#[once]
pub fn hess_case_tpss0() -> CaseAmoniaUKSBecke {
    hess_case_becke_uks("tpss0")
}
