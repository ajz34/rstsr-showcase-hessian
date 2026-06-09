use crate::hessian::ri_jk_restricted_naive;
use crate::prelude::*;

pub fn get_decomposed_rij_skeleton_deriv2_unrestricted_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: &[TsrView; 2],
    mo_occ: &[TsrView; 2],
    atm_list: Option<&[usize]>,
) -> HashMap<&'static str, Tsr> {
    // concatenate mo_coeff and mo_occ for total density
    // type conversion due to rstsr not impl &[T; N] concate
    let mo_coeff: [TsrView; 2] = mo_coeff.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_occ: [TsrView; 2] = mo_occ.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_coeff_stack: Tsr = rt::concatenate((mo_coeff, -1));
    let mo_occ_stack: Tsr = rt::concatenate((mo_occ, -1));

    ri_jk_restricted_naive::get_decomposed_rij_skeleton_deriv2_naive(
        mol,
        aux,
        mo_coeff_stack.view(),
        mo_occ_stack.view(),
        atm_list,
    )
}

pub fn get_decomposed_rik_skeleton_deriv2_unrestricted_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: &[TsrView; 2],
    mo_occ: &[TsrView; 2],
    atm_list: Option<&[usize]>,
) -> HashMap<&'static str, Tsr> {
    // compute alpha and beta separately, then sum
    let [α, β] = [0, 1];

    let de_rik_alpha = ri_jk_restricted_naive::get_decomposed_rik_skeleton_deriv2_naive(
        mol,
        aux,
        mo_coeff[α].view(),
        mo_occ[α].view(),
        atm_list,
    );
    let de_rik_beta = ri_jk_restricted_naive::get_decomposed_rik_skeleton_deriv2_naive(
        mol,
        aux,
        mo_coeff[β].view(),
        mo_occ[β].view(),
        atm_list,
    );

    let mut result = HashMap::new();
    for &key in de_rik_alpha.keys() {
        let de_alpha = &de_rik_alpha[key];
        let de_beta = &de_rik_beta[key];
        result.insert(key, de_alpha + de_beta);
    }
    result
}

/// Get the first-order skeleton derivative of the Coulomb interaction in AO basis (UHF).
///
/// Spin-independent: depends only on total density.
///
/// # Returns
///
/// - `HashMap<&'static str, Tsr>`. Same keys as restricted version. Each value has shape `[nao,
///   nao, 3, natm]`.
pub fn get_rij_deriv1_ao_unrestricted_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: &[TsrView; 2],
    mo_occ: &[TsrView; 2],
    atm_list: Option<&[usize]>,
) -> HashMap<&'static str, Tsr> {
    // J is spin-independent: concatenate mo_coeff and mo_occ for total density
    // type conversion due to rstsr not impl &[T; N] concate
    let mo_coeff: [TsrView; 2] = mo_coeff.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_occ: [TsrView; 2] = mo_occ.iter().map(|x| x.view()).collect_array().unwrap();
    let mo_coeff_stack: Tsr = rt::concatenate((mo_coeff, -1));
    let mo_occ_stack: Tsr = rt::concatenate((mo_occ, -1));
    ri_jk_restricted_naive::get_rij_deriv1_ao_naive(mol, aux, mo_coeff_stack.view(), mo_occ_stack.view(), atm_list)
}

/// Get the first-order skeleton derivative of the exchange interaction in AO basis (UHF).
///
/// Spin-resolved: returns per-spin results.
///
/// # Returns
///
/// - `HashMap<&'static str, Tsr>`. Each value has shape `[nao, nao, 3, natm, 2]` (leading dimension
///   indexes spin).
pub fn get_rik_deriv1_ao_unrestricted_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: &[TsrView; 2],
    mo_occ: &[TsrView; 2],
    atm_list: Option<&[usize]>,
) -> HashMap<&'static str, Tsr> {
    let k1ao_alpha =
        ri_jk_restricted_naive::get_rik_deriv1_ao_naive(mol, aux, mo_coeff[0].view(), mo_occ[0].view(), atm_list);
    let k1ao_beta =
        ri_jk_restricted_naive::get_rik_deriv1_ao_naive(mol, aux, mo_coeff[1].view(), mo_occ[1].view(), atm_list);

    let mut result = HashMap::new();
    for &key in k1ao_alpha.keys() {
        let val_alpha = &k1ao_alpha[key];
        let val_beta = &k1ao_beta[key];
        result.insert(key, rt::stack(([val_alpha.view(), val_beta.view()], -1)));
    }
    result
}

/// UHF response of RI-JK given bra (per-spin perturbed coefficients).
///
/// J response couples spin channels (sees total density); K response is same-spin only.
pub fn get_uijk_response_bra_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: &[TsrView; 2],
    mo_occ: &[TsrView; 2],
    bra: &[TsrView; 2],
) -> [Tsr; 2] {
    let nao = mol.nao();
    let device = bra[0].device().clone();

    let occidx = [mo_occ[0].view().greater(0).into_vec(), mo_occ[1].view().greater(0).into_vec()];
    let mocc = [mo_coeff[0].bool_select(-1, &occidx[0]), mo_coeff[1].bool_select(-1, &occidx[1])];
    let nocc = [mocc[0].shape()[1], mocc[1].shape()[1]];
    let in_shapes = [bra[0].shape(), bra[1].shape()];
    let bra = [bra[0].reshape((nao, nocc[0], -1)), bra[1].reshape((nao, nocc[1], -1))];

    let int2c2e = hess_intor(aux, "int2c2e", "s1", None, &device);
    let int2c2e_inv = rt::linalg::inv(int2c2e.view());
    let int3c2e = hess_intor_cross(&[mol, mol, aux], "int3c2e", "s1", None, &device);

    let mut resp = [None, None];

    for s in [0, 1] {
        let bra_s = bra[s].view();

        let mut r = rt::zeros_like(&bra_s);

        // J contribution (sees total density): sum over spin channel tau
        for tau in 0..2 {
            let subscripts = "uvP, PQ, klQ, kjA, lj, vi -> uiA";
            let operands =
                [int3c2e.view(), int2c2e_inv.view(), int3c2e.view(), bra[tau].view(), mocc[tau].view(), mocc[s].view()];
            r += 2.0 * rt::tblis::einsum(subscripts, operands, true, None);
        }

        // K contribution (same-spin only), two terms
        let subscripts = "uvP, PQ, klQ, vjA, lj, ki -> uiA";
        let operands =
            [int3c2e.view(), int2c2e_inv.view(), int3c2e.view(), bra_s.view(), mocc[s].view(), mocc[s].view()];
        r -= rt::tblis::einsum(subscripts, operands, true, None);

        let subscripts = "uvP, PQ, klQ, kjA, vj, li -> uiA";
        let operands =
            [int3c2e.view(), int2c2e_inv.view(), int3c2e.view(), bra_s.view(), mocc[s].view(), mocc[s].view()];
        r -= rt::tblis::einsum(subscripts, operands, true, None);

        resp[s] = Some(r);
    }

    [resp[0].take().unwrap().into_shape(in_shapes[0]), resp[1].take().unwrap().into_shape(in_shapes[1])]
}

pub struct UHessRIJKNaive {
    pub mol: CInt,
    pub aux: CInt,
    pub scale_j: f64,
    pub scale_k: f64,
    pub intmd: HashMap<&'static str, Tsr>,
    pub result: HashMap<&'static str, Tsr>,
}

impl UHessRIJKNaive {
    pub fn new(mol: &CInt, aux: &CInt, scale_j: f64, scale_k: f64) -> Self {
        Self { mol: mol.clone(), aux: aux.clone(), scale_j, scale_k, intmd: HashMap::new(), result: HashMap::new() }
    }
}

impl UHessElecInteractAPI for UHessRIJKNaive {
    fn make_skeleton_hess(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> Tsr {
        let de_J_skeleton_dict =
            get_decomposed_rij_skeleton_deriv2_unrestricted_naive(&self.mol, &self.aux, mo_coeff, mo_occ, atm_list);
        let de_K_skeleton_dict =
            get_decomposed_rik_skeleton_deriv2_unrestricted_naive(&self.mol, &self.aux, mo_coeff, mo_occ, atm_list);
        let result = &mut self.result;
        result.extend(de_J_skeleton_dict);
        result.extend(de_K_skeleton_dict);
        let de_J = &result["de_J20"] + &result["de_J11"] + &result["de_J02"];
        let de_K = &result["de_K20"] + &result["de_K11"] + &result["de_K02"];
        // UHF: K coefficient is -1 (not -0.5 as in RHF) because de_K already includes spin sum.
        self.scale_j * de_J - self.scale_k * de_K
    }

    fn get_deriv1_ao(
        &mut self,
        mo_coeff: &[TsrView; 2],
        mo_occ: &[TsrView; 2],
        atm_list: Option<&[usize]>,
    ) -> [Tsr; 2] {
        let j1ao_dict = get_rij_deriv1_ao_unrestricted_naive(&self.mol, &self.aux, mo_coeff, mo_occ, atm_list);
        let k1ao_dict = get_rik_deriv1_ao_unrestricted_naive(&self.mol, &self.aux, mo_coeff, mo_occ, atm_list);
        let result = &mut self.result;
        result.extend(j1ao_dict);
        result.extend(k1ao_dict);

        // J is spin-independent [nao, nao, 3, natm]; K is spin-resolved [2, nao, nao, 3, natm]
        let j1ao = &result["j1ao_aux0"] + &result["j1ao_aux1"];
        let k1ao = &result["k1ao_aux0"] + &result["k1ao_aux1"];

        // Broadcast J to both spins: [2, nao, nao, 3, natm], then subtract per-spin K
        let [α, β] = [0, 1];
        let deriv1_ao_α = self.scale_j * &j1ao - self.scale_k * k1ao.i((Ellipsis, α));
        let deriv1_ao_β = self.scale_j * &j1ao - self.scale_k * k1ao.i((Ellipsis, β));
        [deriv1_ao_α, deriv1_ao_β]
    }

    fn make_response_preparation(&mut self, mo_coeff: &[TsrView; 2], mo_occ: &[TsrView; 2]) {
        self.intmd.insert("mo_coeff_0", mo_coeff[0].view().into_contig(RowMajor));
        self.intmd.insert("mo_coeff_1", mo_coeff[1].view().into_contig(RowMajor));
        self.intmd.insert("mo_occ_0", mo_occ[0].to_owned());
        self.intmd.insert("mo_occ_1", mo_occ[1].to_owned());
    }

    fn get_response_bra(&self, bra: &[TsrView; 2]) -> [Tsr; 2] {
        let mo_coeff = [self.intmd["mo_coeff_0"].view(), self.intmd["mo_coeff_1"].view()];
        let mo_occ = [self.intmd["mo_occ_0"].view(), self.intmd["mo_occ_1"].view()];
        get_uijk_response_bra_naive(&self.mol, &self.aux, &mo_coeff, &mo_occ, bra)
    }
}
