use crate::prelude::*;

const TOL_OCC: f64 = 1e-15;

/// Get the skeleton of the second derivative of the Coulomb interaction.
///
/// This is naive implementation:
///
/// - computes all integrals in full, not optimized for memory;
/// - extensively use einsum, easy for equation-code translation but not fully efficient;
/// - not extensively combined contribution terms;
/// - in principle, RI-J should be evaluated alongwith RI-K, but here we only compute RI-J part;
/// - we evaluated all auxiliary basis derivative contributions, which is sometimes not necessary
///   for hessian computation.
///
/// This function not only returns the summed hessian, but also all the separated contributions,
/// useful for debugging and understanding the code.
///
/// # Parameters
///
/// - `mol` : [CInt]. The molecule object.
/// - `aux` : [CInt]. The auxiliary basis set molecule object.
/// - `mo_coeff` : shape `[nao, nmo]`. The molecular orbital coefficients.
/// - `mo_occ` : shape `[nmo]`. The molecular orbital occupation numbers.
///
/// Returns
/// -------
/// - `de_J_skeleton` : `HashMap<&'static str, Tsr>`.
///
///   The skeleton of the second derivative of the Coulomb interaction, separated by different
///   contributions. Each contribution is `[3, 3, natm, natm]` array. The contributions are denoted
///   as `de_J<bas_deriv><aux_deriv>_<contrib_idx>`, e.g. `de_J20_2`. Sometimes the contribution idx
///   will be number with alphabet like `2a` and `2b`. Meaning of these contribution may not be
///   fully documented in returned keys. See code comments for details.
pub fn get_decomposed_rij_skeleton_deriv2_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: TsrView,
    mo_occ: TsrView,
) -> HashMap<&'static str, Tsr> {
    // some elementary information
    let natm = mol.natm();
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let aoslices = mol.aoslice_by_atom();
    let auxslices = aux.aoslice_by_atom();
    let device = mo_coeff.device();

    // integrals we need
    let int3c2e = hess_intor_cross(&[mol, mol, aux], "int3c2e", "s1", None, device);
    let int3c2e_ip1 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip1", "s1", None, device);
    let int3c2e_ip2 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip2", "s1", None, device);
    let int3c2e_ipip1 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ipip1", "s1", None, device);
    let int3c2e_ipvip1 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ipvip1", "s1", None, device);
    let int3c2e_ip1ip2 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip1ip2", "s1", None, device);
    let int3c2e_ipip2 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ipip2", "s1", None, device);
    let int2c2e = hess_intor(aux, "int2c2e", "s1", None, device);
    let int2c2e_inv = rt::linalg::inv(int2c2e.view());
    let int2c2e_ip1 = hess_intor(aux, "int2c2e_ip1", "s1", None, device);
    let int2c2e_ipip1 = hess_intor(aux, "int2c2e_ipip1", "s1", None, device);
    let int2c2e_ip1ip2 = hess_intor(aux, "int2c2e_ip1ip2", "s1", None, device);

    // --- J20 (basis deriv 2, aux deriv 0) --- //

    // (10|0)(0|10)
    let subscripts = "uvPt, PQ, klQs, uv, kl -> ukst";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int3c2e_ip1, &dm0, &dm0];
    let dbas_J20_1 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J20_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            *&mut de_J20_1.i_mut((.., .., B, A)) += 4.0 * dbas_J20_1.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }

    // (11|0)(0|00)
    let subscripts = "uvPst, PQ, klQ, uv, kl -> uvst";
    let operands = [&int3c2e_ipvip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J20_2 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J20_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            *&mut de_J20_2.i_mut((.., .., B, A)) += 2.0 * dbas_J20_2.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }

    // (20|0)(0|00)
    let subscripts = "uvPst, PQ, klQ, uv, kl -> uvst";
    let operands = [&int3c2e_ipip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J20_3 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J20_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        *&mut de_J20_3.i_mut((.., .., A, A)) += 2.0 * dbas_J20_3.i(p0A..p1A).sum_axes([0, 1]);
    }

    let de_J20 = &de_J20_1 + &de_J20_2 + &de_J20_3;

    // --- J11 (basis deriv 1, aux deriv 1) --- //

    // (10|1)(0|0)(0|00)
    let subscripts = "uvPst, PQ, klQ, uv, kl -> uPst";
    let operands = [&int3c2e_ip1ip2, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J11_1 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J11_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J11_1.i_mut((.., .., B, A)) += 2.0 * dbas_J11_1.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J11_1 = &de_J11_1 + de_J11_1.view().into_transpose([1, 0, 3, 2]);

    // (10|0)(0|1)(0|00)
    let subscripts = "uvPt, PQ, QRs, RS, klS, uv, kl -> uRst";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J11_2 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J11_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J11_2.i_mut((.., .., B, A)) += 2.0 * dbas_J11_2.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J11_2 = &de_J11_2 + de_J11_2.view().into_transpose([1, 0, 3, 2]);

    // (10|0)(1|0)(0|00)
    let subscripts = "uvPt, PQ, QRs, RS, klS, uv, kl -> uQst";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J11_3 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J11_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J11_3.i_mut((.., .., B, A)) += -2.0 * dbas_J11_3.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J11_3 = &de_J11_3 + de_J11_3.view().into_transpose([1, 0, 3, 2]);

    // (10|0)(0|0)(1|00)
    let subscripts = "uvPt, PQ, klQs, uv, kl -> uQst";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int3c2e_ip2, &dm0, &dm0];
    let dbas_J11_4 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J11_4: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J11_4.i_mut((.., .., B, A)) += 2.0 * dbas_J11_4.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J11_4 = &de_J11_4 + de_J11_4.view().into_transpose([1, 0, 3, 2]);

    let de_J11 = &de_J11_1 + &de_J11_2 + &de_J11_3 + &de_J11_4;

    // --- J02 (basis deriv 0, aux deriv 2) --- //

    // (00|2)(0|00)
    let subscripts = "uvPst, PQ, klQ, uv, kl -> Pst";
    let operands = [&int3c2e_ipip2, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J02_1 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        *&mut de_J02_1.i_mut((.., .., A, A)) += dbas_J02_1.i(p0A..p1A).sum_axes([0]);
    }

    // (00|0)(2|0)(0|00)
    let subscripts = "uvP, PQ, QRst, RS, klS, uv, kl -> Qst";
    let operands = [&int3c2e, &int2c2e_inv, &int2c2e_ipip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J02_2 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        *&mut de_J02_2.i_mut((.., .., A, A)) += -1.0 * dbas_J02_2.i(p0A..p1A).sum_axes([0]);
    }

    // (00|0)(1|1)(0|00)
    let subscripts = "uvP, PQ, QRst, RS, klS, uv, kl -> QRst";
    let operands = [&int3c2e, &int2c2e_inv, &int2c2e_ip1ip2, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J02_3a = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_3a: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J02_3a.i_mut((.., .., B, A)) += -0.5 * dbas_J02_3a.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J02_3a = &de_J02_3a + de_J02_3a.view().into_transpose([1, 0, 3, 2]);

    // (00|0)(1|0)(0|1)(0|00)
    let subscripts = "uvP, PQ, QRt, RS, STs, TU, klU, uv, kl -> QTst";
    let operands =
        [&int3c2e, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J02_3b = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_3b: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J02_3b.i_mut((.., .., B, A)) += -0.5 * dbas_J02_3b.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J02_3b = &de_J02_3b + de_J02_3b.view().into_transpose([1, 0, 3, 2]);

    // (00|1)(1|0)(0|00)
    let subscripts = "uvPt, PQ, QRs, RS, klS, uv, kl -> PQst";
    let operands = [&int3c2e_ip2, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J02_4 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_4: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J02_4.i_mut((.., .., B, A)) += -1.0 * dbas_J02_4.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J02_4 = &de_J02_4 + de_J02_4.view().into_transpose([1, 0, 3, 2]);

    // (00|1)(1|00)
    let subscripts = "uvPt, PQ, klQs, uv, kl -> PQst";
    let operands = [&int3c2e_ip2, &int2c2e_inv, &int3c2e_ip2, &dm0, &dm0];
    let dbas_J02_5 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_5: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J02_5.i_mut((.., .., B, A)) += 0.5 * dbas_J02_5.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J02_5 = &de_J02_5 + de_J02_5.view().into_transpose([1, 0, 3, 2]);

    // (00|0)(0|1)(1|0)(0|00)
    let subscripts = "uvP, PQ, RQt, RS, STs, TU, klU, uv, kl -> RSst";
    let operands =
        [&int3c2e, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J02_6 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_6: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J02_6.i_mut((.., .., B, A)) += 0.5 * dbas_J02_6.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J02_6 = &de_J02_6 + de_J02_6.view().into_transpose([1, 0, 3, 2]);

    // (00|1)(0|1)(0|00)
    let subscripts = "uvPt, PQ, RQs, RS, klS, uv, kl -> PRst";
    let operands = [&int3c2e_ip2, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J02_7 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_7: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J02_7.i_mut((.., .., B, A)) += -1.0 * dbas_J02_7.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J02_7 = &de_J02_7 + de_J02_7.view().into_transpose([1, 0, 3, 2]);

    // (00|0)(1|0)(1|0)(0|00)
    let subscripts = "uvP, PQ, QRt, RS, STs, TU, klU, uv, kl -> RTst";
    let operands =
        [&int3c2e, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J02_8 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J02_8: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_J02_8.i_mut((.., .., B, A)) += 1.0 * dbas_J02_8.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_J02_8 = &de_J02_8 + de_J02_8.view().into_transpose([1, 0, 3, 2]);

    let de_J02 =
        &de_J02_1 + &de_J02_2 + &de_J02_3a + &de_J02_3b + &de_J02_4 + &de_J02_5 + &de_J02_6 + &de_J02_7 + &de_J02_8;

    HashMap::from([
        // de_J20
        ("de_J20_1", de_J20_1),
        ("de_J20_2", de_J20_2),
        ("de_J20_3", de_J20_3),
        // de_J11
        ("de_J11_1", de_J11_1),
        ("de_J11_2", de_J11_2),
        ("de_J11_3", de_J11_3),
        ("de_J11_4", de_J11_4),
        // de_J02
        ("de_J02_1", de_J02_1),
        ("de_J02_2", de_J02_2),
        ("de_J02_3a", de_J02_3a),
        ("de_J02_3b", de_J02_3b),
        ("de_J02_4", de_J02_4),
        ("de_J02_5", de_J02_5),
        ("de_J02_6", de_J02_6),
        ("de_J02_7", de_J02_7),
        ("de_J02_8", de_J02_8),
        // total
        ("de_J20", de_J20),
        ("de_J11", de_J11),
        ("de_J02", de_J02),
    ])
}

/// Get the skeleton of the second derivative of the exchange interaction.
///
/// This is naive implementation, see [`get_decomposed_rij_skeleton_deriv2_naive`] for details and
/// returned keys documentation.
///
/// # Parameters
///
/// - `mol` : [CInt]. The molecule object.
/// - `aux` : [CInt]. The auxiliary basis set molecule object.
/// - `mo_coeff` : shape `[nao, nmo]`. The molecular orbital coefficients.
/// - `mo_occ` : shape `[nmo]`. The molecular orbital occupation numbers.
///
/// Returns
/// -------
/// - `de_K_skeleton` : `HashMap<&'static str, Tsr>`.
///
///   The skeleton of the second derivative of the exchange interaction, separated by different
///   contributions. Each contribution is `[3, 3, natm, natm]` array. The contributions are denoted
///   as `de_K<bas_deriv><aux_deriv>_<contrib_idx>`, e.g. `de_K20_2`.
pub fn get_decomposed_rik_skeleton_deriv2_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: TsrView,
    mo_occ: TsrView,
) -> HashMap<&'static str, Tsr> {
    // some elementary information
    let natm = mol.natm();
    let aoslices = mol.aoslice_by_atom();
    let auxslices = aux.aoslice_by_atom();
    let device = mo_coeff.device();

    // occupation: mocc_2 = mocc * sqrt(occ)
    let occidx = mo_occ.view().greater(TOL_OCC).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    let mocc_2 = &mocc * occ.sqrt().i((None, ..));

    // integrals we need
    let int3c2e = hess_intor_cross(&[mol, mol, aux], "int3c2e", "s1", None, device);
    let int3c2e_ip1 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip1", "s1", None, device);
    let int3c2e_ip2 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip2", "s1", None, device);
    let int3c2e_ipip1 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ipip1", "s1", None, device);
    let int3c2e_ipvip1 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ipvip1", "s1", None, device);
    let int3c2e_ip1ip2 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip1ip2", "s1", None, device);
    let int3c2e_ipip2 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ipip2", "s1", None, device);
    let int2c2e = hess_intor(aux, "int2c2e", "s1", None, device);
    let int2c2e_inv = rt::linalg::inv(int2c2e.view());
    let int2c2e_ip1 = hess_intor(aux, "int2c2e_ip1", "s1", None, device);
    let int2c2e_ipip1 = hess_intor(aux, "int2c2e_ipip1", "s1", None, device);
    let int2c2e_ip1ip2 = hess_intor(aux, "int2c2e_ip1ip2", "s1", None, device);

    // --- K20 (basis deriv 2, aux deriv 0) --- //

    // (10|0)(0|10), part a
    // python: tuvP, PQ, sklQ, ui, vj, ki, lj -> tsuk
    let subscripts = "uvPt, PQ, klQs, ui, vj, ki, lj -> ukst";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int3c2e_ip1, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K20_1a = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K20_1a: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            *&mut de_K20_1a.i_mut((.., .., B, A)) += 2.0 * dbas_K20_1a.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }

    // (10|0)(0|10), part b
    // python: tuvP, PQ, sklQ, ui, vj, kj, li -> tsuk
    let subscripts = "uvPt, PQ, klQs, ui, vj, kj, li -> ukst";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int3c2e_ip1, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K20_1b = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K20_1b: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            *&mut de_K20_1b.i_mut((.., .., B, A)) += 2.0 * dbas_K20_1b.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }

    // (11|0)(0|00)
    // python: tsuvP, PQ, klQ, ui, vj, ki, lj -> tsuv
    let subscripts = "uvPst, PQ, klQ, ui, vj, ki, lj -> uvst";
    let operands = [&int3c2e_ipvip1, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K20_2 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K20_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            *&mut de_K20_2.i_mut((.., .., B, A)) += 2.0 * dbas_K20_2.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }

    // (20|0)(0|00)
    // python: tsuvP, PQ, klQ, ui, vj, ki, lj -> tsuv
    let subscripts = "uvPst, PQ, klQ, ui, vj, ki, lj -> uvst";
    let operands = [&int3c2e_ipip1, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K20_3 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K20_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        *&mut de_K20_3.i_mut((.., .., A, A)) += 2.0 * dbas_K20_3.i(p0A..p1A).sum_axes([0, 1]);
    }

    let de_K20 = &de_K20_1a + &de_K20_1b + &de_K20_2 + &de_K20_3;

    // --- K11 (basis deriv 1, aux deriv 1) --- //

    // (10|1)(0|0)(0|00)
    // python: tsuvP, PQ, klQ, vi, li, uj, kj -> tsuP
    let subscripts = "uvPst, PQ, klQ, vi, li, uj, kj -> uPst";
    let operands = [&int3c2e_ip1ip2, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K11_1 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K11_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K11_1.i_mut((.., .., B, A)) += 2.0 * dbas_K11_1.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K11_1 = &de_K11_1 + de_K11_1.view().into_transpose([1, 0, 3, 2]);

    // (10|0)(0|1)(0|00)
    // python: tuvP, PQ, sQR, RS, klS, ui, vj, ki, lj -> tsuR
    let subscripts = "uvPt, PQ, QRs, RS, klS, ui, vj, ki, lj -> uRst";
    let operands =
        [&int3c2e_ip1, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K11_2 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K11_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K11_2.i_mut((.., .., B, A)) += 2.0 * dbas_K11_2.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K11_2 = &de_K11_2 + de_K11_2.view().into_transpose([1, 0, 3, 2]);

    // (10|0)(1|0)(0|00)
    // python: tuvP, PQ, sQR, RS, klS, ui, vj, ki, lj -> tsuQ
    let subscripts = "uvPt, PQ, QRs, RS, klS, ui, vj, ki, lj -> uQst";
    let operands =
        [&int3c2e_ip1, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K11_3 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K11_3: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K11_3.i_mut((.., .., B, A)) += -2.0 * dbas_K11_3.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K11_3 = &de_K11_3 + de_K11_3.view().into_transpose([1, 0, 3, 2]);

    // (10|0)(0|0)(1|00)
    // python: tuvP, PQ, sklQ, ui, vj, ki, lj -> tsuQ
    let subscripts = "uvPt, PQ, klQs, ui, vj, ki, lj -> uQst";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int3c2e_ip2, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K11_4 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K11_4: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K11_4.i_mut((.., .., B, A)) += 2.0 * dbas_K11_4.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K11_4 = &de_K11_4 + de_K11_4.view().into_transpose([1, 0, 3, 2]);

    let de_K11 = &de_K11_1 + &de_K11_2 + &de_K11_3 + &de_K11_4;

    // --- K02 (basis deriv 0, aux deriv 2) --- //

    // (00|2)(0|00)
    // python: tsuvP, PQ, klQ, ui, vj, ki, lj -> tsP
    let subscripts = "uvPst, PQ, klQ, ui, vj, ki, lj -> Pst";
    let operands = [&int3c2e_ipip2, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K02_1 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_1: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        *&mut de_K02_1.i_mut((.., .., A, A)) += dbas_K02_1.i(p0A..p1A).sum_axes([0]);
    }

    // (00|0)(2|0)(0|00)
    // python: uvP, PQ, tsQR, RS, klS, ui, vj, ki, lj -> tsQ
    let subscripts = "uvP, PQ, QRst, RS, klS, ui, vj, ki, lj -> Qst";
    let operands = [&int3c2e, &int2c2e_inv, &int2c2e_ipip1, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K02_2 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_2: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        *&mut de_K02_2.i_mut((.., .., A, A)) += -1.0 * dbas_K02_2.i(p0A..p1A).sum_axes([0]);
    }

    // (00|0)(1|1)(0|00)
    // python: uvP, PQ, tsQR, RS, klS, ui, vj, ki, lj -> tsQR
    let subscripts = "uvP, PQ, QRst, RS, klS, ui, vj, ki, lj -> QRst";
    let operands =
        [&int3c2e, &int2c2e_inv, &int2c2e_ip1ip2, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K02_3a = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_3a: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K02_3a.i_mut((.., .., B, A)) += -0.5 * dbas_K02_3a.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K02_3a = &de_K02_3a + de_K02_3a.view().into_transpose([1, 0, 3, 2]);

    // (00|0)(1|0)(0|1)(0|00)
    // python: uvP, PQ, tQR, RS, sST, TU, klU, ui, vj, ki, lj -> tsQT
    let subscripts = "uvP, PQ, QRt, RS, STs, TU, klU, ui, vj, ki, lj -> QTst";
    let operands = [
        &int3c2e,
        &int2c2e_inv,
        &int2c2e_ip1,
        &int2c2e_inv,
        &int2c2e_ip1,
        &int2c2e_inv,
        &int3c2e,
        &mocc_2,
        &mocc_2,
        &mocc_2,
        &mocc_2,
    ];
    let dbas_K02_3b = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_3b: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K02_3b.i_mut((.., .., B, A)) += -0.5 * dbas_K02_3b.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K02_3b = &de_K02_3b + de_K02_3b.view().into_transpose([1, 0, 3, 2]);

    // (00|1)(1|0)(0|00)
    // python: tuvP, PQ, sQR, RS, klS, ui, vj, ki, lj -> tsPQ
    let subscripts = "uvPt, PQ, QRs, RS, klS, ui, vj, ki, lj -> PQst";
    let operands =
        [&int3c2e_ip2, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K02_4 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_4: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K02_4.i_mut((.., .., B, A)) += -1.0 * dbas_K02_4.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K02_4 = &de_K02_4 + de_K02_4.view().into_transpose([1, 0, 3, 2]);

    // (00|1)(1|00)
    // python: tuvP, PQ, sklQ, ui, vj, ki, lj -> tsPQ
    let subscripts = "uvPt, PQ, klQs, ui, vj, ki, lj -> PQst";
    let operands = [&int3c2e_ip2, &int2c2e_inv, &int3c2e_ip2, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K02_5 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_5: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K02_5.i_mut((.., .., B, A)) += 0.5 * dbas_K02_5.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K02_5 = &de_K02_5 + de_K02_5.view().into_transpose([1, 0, 3, 2]);

    // (00|0)(0|1)(1|0)(0|00)
    // python: uvP, PQ, tRQ, RS, sST, TU, klU, ui, vj, ki, lj -> tsRS
    let subscripts = "uvP, PQ, RQt, RS, STs, TU, klU, ui, vj, ki, lj -> RSst";
    let operands = [
        &int3c2e,
        &int2c2e_inv,
        &int2c2e_ip1,
        &int2c2e_inv,
        &int2c2e_ip1,
        &int2c2e_inv,
        &int3c2e,
        &mocc_2,
        &mocc_2,
        &mocc_2,
        &mocc_2,
    ];
    let dbas_K02_6 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_6: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K02_6.i_mut((.., .., B, A)) += 0.5 * dbas_K02_6.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K02_6 = &de_K02_6 + de_K02_6.view().into_transpose([1, 0, 3, 2]);

    // (00|1)(0|1)(0|00)
    // python: tuvP, PQ, sRQ, RS, klS, ui, vj, ki, lj -> tsPR
    let subscripts = "uvPt, PQ, RQs, RS, klS, ui, vj, ki, lj -> PRst";
    let operands =
        [&int3c2e_ip2, &int2c2e_inv, &int2c2e_ip1, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2, &mocc_2, &mocc_2];
    let dbas_K02_7 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_7: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K02_7.i_mut((.., .., B, A)) += -1.0 * dbas_K02_7.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K02_7 = &de_K02_7 + de_K02_7.view().into_transpose([1, 0, 3, 2]);

    // (00|0)(1|0)(1|0)(0|00)
    // python: uvP, PQ, tQR, RS, sST, TU, klU, ui, vj, ki, lj -> tsQS
    let subscripts = "uvP, PQ, QRt, RS, STs, TU, klU, ui, vj, ki, lj -> QSst";
    let operands = [
        &int3c2e,
        &int2c2e_inv,
        &int2c2e_ip1,
        &int2c2e_inv,
        &int2c2e_ip1,
        &int2c2e_inv,
        &int3c2e,
        &mocc_2,
        &mocc_2,
        &mocc_2,
        &mocc_2,
    ];
    let dbas_K02_8 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_K02_8: Tsr = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in auxslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in auxslices.iter().enumerate() {
            *&mut de_K02_8.i_mut((.., .., B, A)) += 1.0 * dbas_K02_8.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }
    let de_K02_8 = &de_K02_8 + de_K02_8.view().into_transpose([1, 0, 3, 2]);

    let de_K02 =
        &de_K02_1 + &de_K02_2 + &de_K02_3a + &de_K02_3b + &de_K02_4 + &de_K02_5 + &de_K02_6 + &de_K02_7 + &de_K02_8;

    HashMap::from([
        // de_K20
        ("de_K20_1a", de_K20_1a),
        ("de_K20_1b", de_K20_1b),
        ("de_K20_2", de_K20_2),
        ("de_K20_3", de_K20_3),
        // de_K11
        ("de_K11_1", de_K11_1),
        ("de_K11_2", de_K11_2),
        ("de_K11_3", de_K11_3),
        ("de_K11_4", de_K11_4),
        // de_K02
        ("de_K02_1", de_K02_1),
        ("de_K02_2", de_K02_2),
        ("de_K02_3a", de_K02_3a),
        ("de_K02_3b", de_K02_3b),
        ("de_K02_4", de_K02_4),
        ("de_K02_5", de_K02_5),
        ("de_K02_6", de_K02_6),
        ("de_K02_7", de_K02_7),
        ("de_K02_8", de_K02_8),
        // total
        ("de_K20", de_K20),
        ("de_K11", de_K11),
        ("de_K02", de_K02),
    ])
}
