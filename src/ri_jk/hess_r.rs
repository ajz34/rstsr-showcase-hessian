//! Optimized RI-JK Hessian computation.
//!
//! **NOTE** currently this is not completed.
//!
//! Algorithm is naive and of no reference.
//!
//! The correct value is compared to PySCF (written by Qiming Sun). Reference article:
//! > Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models
//! > Chen, et al. arXiv:1906.09427
//!
//! PySCF code referenced the ORCA's implementation:
//! > Efficient implementation of the analytic second derivatives of Hartree-Fock and hybrid DFT
//! > energies: a detailed analysis of different approximations
//! > Bykov, et al. Mol Phys. 113, 1961 (2015). DOI: 10.1080/00268976.2015.1025114

use crate::prelude::*;
use crate::ri_jk::decompose::*;
use crate::ri_jk::pure_decompose::{get_j2c_decomp, solve_by_j2c};

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
    atm_list: Option<&[usize]>,
) -> HashMap<String, Tsr> {
    // some elementary information
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
    let (aoslices, _) = filter_aoslices(mol, atm_list);
    let (auxslices, _) = filter_aoslices(aux, atm_list);
    let natm = aoslices.len();
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

    HashMap::from(
        [
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
        ]
        .map(|(k, v)| (k.to_string(), v)),
    )
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
    atm_list: Option<&[usize]>,
) -> HashMap<String, Tsr> {
    // some elementary information
    let (aoslices, _) = filter_aoslices(mol, atm_list);
    let (auxslices, _) = filter_aoslices(aux, atm_list);
    let natm = aoslices.len();
    let device = mo_coeff.device();

    // occupation: mocc_2 = mocc * sqrt(occ)
    let occidx = mo_occ.view().greater(0).into_vec();
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

    HashMap::from(
        [
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
        ]
        .map(|(k, v)| (k.to_string(), v)),
    )
}

/// Get the first derivative of the Coulomb interaction in AO basis.
///
/// # Parameters
///
/// - `mol` : [CInt]. The molecule object.
/// - `aux` : [CInt]. The auxiliary basis set molecule object.
/// - `mo_coeff` : shape `[nao, nmo]`. The molecular orbital coefficients.
/// - `mo_occ` : shape `[nmo]`. The molecular orbital occupation numbers.
///
/// # Returns
///
/// - `HashMap<&'static str, Tsr>`. Storing contribution by auxiliary basis derivative order. Output
///   shape is `[nao, nao, 3, natm]`.
pub fn get_rij_deriv1_ao_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: TsrView,
    mo_occ: TsrView,
    atm_list: Option<&[usize]>,
) -> HashMap<String, Tsr> {
    // some elementary information
    let nao = mol.nao();
    let (aoslices, _) = filter_aoslices(mol, atm_list);
    let (auxslices, _) = filter_aoslices(aux, atm_list);
    let natm = aoslices.len();
    let device = mo_coeff.device();
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());

    // integrals we need
    let int3c2e = hess_intor_cross(&[mol, mol, aux], "int3c2e", "s1", None, device);
    let int3c2e_ip1 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip1", "s1", None, device);
    let int3c2e_ip2 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip2", "s1", None, device);
    let int2c2e = hess_intor(aux, "int2c2e", "s1", None, device);
    let int2c2e_inv = rt::linalg::inv(int2c2e);
    let int2c2e_ip1 = hess_intor(aux, "int2c2e_ip1", "s1", None, device);

    // --- aux deriv 0 --- //

    let subscripts = "uvPt, PQ, klQ, kl -> uvt";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int3c2e, &dm0];
    let scr1 = rt::tblis::einsum(subscripts, operands, true, None);

    let mut j1ao_aux0 = rt::zeros(([nao, nao, 3, natm], device));
    for A in 0..natm {
        let &[_, _, p0, p1] = &aoslices[A];
        let slc = rt::slice!(p0, p1);
        // (10|0)(0|00)
        *&mut j1ao_aux0.i_mut((slc, .., .., A)) -= scr1.i(slc);
        // (01|0)(0|00), can be symmetrized
        *&mut j1ao_aux0.i_mut((.., slc, .., A)) -= scr1.i(slc).swapaxes(0, 1);
        // (00|0)(0|10), (00|0)(0|01)
        let subscripts = "klPt, PQ, uvQ, kl -> uvt";
        let operands = [int3c2e_ip1.i(slc), int2c2e_inv.view(), int3c2e.view(), dm0.i(slc)];
        let scr2 = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut j1ao_aux0.i_mut((.., .., .., A)) -= 2 * scr2;
    }

    // --- aux deriv 1 --- //

    let mut j1ao_aux1 = rt::zeros(([nao, nao, 3, natm], device));
    for A in 0..natm {
        let &[_, _, p0, p1] = &auxslices[A];
        let slc = rt::slice!(p0, p1);

        // (00|1)(0|00)
        let subscripts = "uvPt, PQ, klQ, kl -> uvt";
        let operands = [int3c2e_ip2.i((.., .., slc)), int2c2e_inv.i(slc), int3c2e.view(), dm0.view()];
        let scr = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut j1ao_aux1.i_mut((Ellipsis, A)) -= scr;

        // (00|0)(1|00)
        let subscripts = "uvP, PQ, klQt, kl -> uvt";
        let operands = [int3c2e.view(), int2c2e_inv.i((.., slc)), int3c2e_ip2.i((.., .., slc)), dm0.view()];
        let scr = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut j1ao_aux1.i_mut((Ellipsis, A)) -= scr;

        // (00|0)(1|0)(0|00)
        let subscripts = "uvP, PQ, QRt, RS, klS, kl -> uvt";
        let operands = [
            int3c2e.view(),
            int2c2e_inv.i((.., slc)),
            int2c2e_ip1.i(slc),
            int2c2e_inv.view(),
            int3c2e.view(),
            dm0.view(),
        ];
        let scr = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut j1ao_aux1.i_mut((Ellipsis, A)) += scr;

        // (00|0)(0|1)(0|00)
        let subscripts = "uvP, PQ, RQt, RS, klS, kl -> uvt";
        let operands =
            [int3c2e.view(), int2c2e_inv.view(), int2c2e_ip1.i(slc), int2c2e_inv.i(slc), int3c2e.view(), dm0.view()];
        let scr = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut j1ao_aux1.i_mut((Ellipsis, A)) += scr;
    }

    HashMap::from([("j1ao_aux0", j1ao_aux0), ("j1ao_aux1", j1ao_aux1)].map(|(k, v)| (k.to_string(), v)))
}

/// Get the first derivative of the exchange interaction in AO basis.
///
/// # Parameters
///
/// - `mol` : [CInt]. The molecule object.
/// - `aux` : [CInt]. The auxiliary basis set molecule object.
/// - `mo_coeff` : shape `[nao, nmo]`. The molecular orbital coefficients.
/// - `mo_occ` : shape `[nmo]`. The molecular orbital occupation numbers.
///
/// # Returns
///
/// - `HashMap<&'static str, Tsr>`. Storing contribution by auxiliary basis derivative order. Output
///   shape is `[nao, nao, 3, natm]`.
pub fn get_rik_deriv1_ao_naive(
    mol: &CInt,
    aux: &CInt,
    mo_coeff: TsrView,
    mo_occ: TsrView,
    atm_list: Option<&[usize]>,
) -> HashMap<String, Tsr> {
    // some elementary information
    let nao = mol.nao();
    let (aoslices, _) = filter_aoslices(mol, atm_list);
    let (auxslices, _) = filter_aoslices(aux, atm_list);
    let natm = aoslices.len();
    let device = mo_coeff.device();

    // occupation: mocc_2 = mocc * sqrt(occ)
    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let occ = mo_occ.bool_select(-1, &occidx);
    let mocc_2 = &mocc * occ.sqrt().i((None, ..));

    // integrals we need
    let int3c2e = hess_intor_cross(&[mol, mol, aux], "int3c2e", "s1", None, device);
    let int3c2e_ip1 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip1", "s1", None, device);
    let int3c2e_ip2 = hess_intor_cross(&[mol, mol, aux], "int3c2e_ip2", "s1", None, device);
    let int2c2e = hess_intor(aux, "int2c2e", "s1", None, device);
    let int2c2e_inv = rt::linalg::inv(int2c2e);
    let int2c2e_ip1 = hess_intor(aux, "int2c2e_ip1", "s1", None, device);

    // --- aux deriv 0 --- //

    // python: tuvP, PQ, klQ, vi, li -> tuk
    let subscripts = "uvPt, PQ, klQ, vi, li -> ukt";
    let operands = [&int3c2e_ip1, &int2c2e_inv, &int3c2e, &mocc_2, &mocc_2];
    let scr1 = rt::tblis::einsum(subscripts, operands, true, None);

    let mut k1ao_aux0 = rt::zeros(([nao, nao, 3, natm], device));
    for A in 0..natm {
        let &[_, _, p0, p1] = &aoslices[A];
        let slc = rt::slice!(p0, p1);
        // (10|0)(0|00)
        *&mut k1ao_aux0.i_mut((slc, .., .., A)) -= scr1.i(slc);
        // (01|0)(0|00)
        *&mut k1ao_aux0.i_mut((.., slc, .., A)) -= scr1.i(slc).swapaxes(0, 1);
        // (00|0)(0|10), (00|0)(0|01)
        // python: tklP, PQ, uvQ, ki, ui -> tlv
        let subscripts = "klPt, PQ, uvQ, ki, ui -> lvt";
        let operands = [int3c2e_ip1.i(slc), int2c2e_inv.view(), int3c2e.view(), mocc_2.i(slc), mocc_2.view()];
        let scr2 = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut k1ao_aux0.i_mut((.., .., .., A)) -= &scr2 + scr2.swapaxes(0, 1);
    }

    // --- aux deriv 1 --- //

    let mut k1ao_aux1 = rt::zeros(([nao, nao, 3, natm], device));
    for A in 0..natm {
        let &[_, _, p0, p1] = &auxslices[A];
        let slc = rt::slice!(p0, p1);

        // (00|1)(0|00)
        // python: tuvP, PQ, klQ, vi, li -> tuk
        let subscripts = "uvPt, PQ, klQ, vi, li -> ukt";
        let operands = [int3c2e_ip2.i((.., .., slc)), int2c2e_inv.i(slc), int3c2e.view(), mocc_2.view(), mocc_2.view()];
        let scr = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut k1ao_aux1.i_mut((Ellipsis, A)) -= scr;

        // (00|0)(1|00)
        // python: uvP, PQ, tklQ, vi, li -> tuk
        let subscripts = "uvP, PQ, klQt, vi, li -> ukt";
        let operands =
            [int3c2e.view(), int2c2e_inv.i((.., slc)), int3c2e_ip2.i((.., .., slc)), mocc_2.view(), mocc_2.view()];
        let scr = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut k1ao_aux1.i_mut((Ellipsis, A)) -= scr;

        // (00|0)(1|0)(0|00)
        // python: uvP, PQ, tQR, RS, klS, vi, li -> tuk
        let subscripts = "uvP, PQ, QRt, RS, klS, vi, li -> ukt";
        let operands = [
            int3c2e.view(),
            int2c2e_inv.i((.., slc)),
            int2c2e_ip1.i(slc),
            int2c2e_inv.view(),
            int3c2e.view(),
            mocc_2.view(),
            mocc_2.view(),
        ];
        let scr = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut k1ao_aux1.i_mut((Ellipsis, A)) += scr;

        // (00|0)(0|1)(0|00)
        // python: uvP, PQ, tRQ, RS, klS, vi, li -> tuk
        let subscripts = "uvP, PQ, RQt, RS, klS, vi, li -> ukt";
        let operands = [
            int3c2e.view(),
            int2c2e_inv.view(),
            int2c2e_ip1.i(slc),
            int2c2e_inv.i(slc),
            int3c2e.view(),
            mocc_2.view(),
            mocc_2.view(),
        ];
        let scr = rt::tblis::einsum(subscripts, operands, true, None);
        *&mut k1ao_aux1.i_mut((Ellipsis, A)) += scr;
    }

    HashMap::from([("k1ao_aux0", k1ao_aux0), ("k1ao_aux1", k1ao_aux1)].map(|(k, v)| (k.to_string(), v)))
}

#[allow(clippy::too_many_arguments)]
pub fn get_rijk_response_bra(
    cderi: TsrView,
    mo_coeff: TsrView,
    mo_occ: TsrView,
    bra1: TsrView,
    scale_j: f64,
    scale_k: f64,
    nbatch_aux: usize,
) -> Tsr {
    // notes on shape
    // - cderi: [nao_tp, naux]
    // - mo_coeff: [nao, nmo]
    // - mo_occ: [nmo]
    // - bra: [nao, nocc, ...]  (use nprop in program, and same to output shape)

    // derived shapes
    // - mocc: [nao, nocc]
    // - oxb: [nocc, naux, nao] (occupied, auxiliary, basis)
    // - oxo: [nocc, naux, nocc]
    // - ......

    // preparation
    let nao = mo_coeff.shape()[0];
    let naux = cderi.shape()[1];
    let nao_tp = nao * (nao + 1) / 2;
    assert_eq!(cderi.shape()[0], nao_tp);
    let occidx = mo_occ.view().greater(0).into_vec();
    let mocc = mo_coeff.bool_select(-1, &occidx);
    let nocc = occidx.iter().filter(|&&x| x).count();

    // reshape bra
    let shape_bra = bra1.shape();
    assert_eq!(bra1.shape()[0], nao);
    assert_eq!(bra1.shape()[1], nocc);
    let bra1 = bra1.reshape((nao, nocc, -1));
    let nprop = bra1.shape()[2];
    let mut resp = rt::zeros_like(&bra1);

    // --- J contribution --- //

    if scale_j != 0.0 {
        // - mo1_dm: [nao, nao, nprop]
        // - mp1_dm_tp: [nao_tp, nprop]
        // - itm_j_aux: [naux, nprop]
        // - resp_tp_j: [nao_tp, nprop]
        // - resp_ao_j: [nao, nao, nprop]
        let dm1 = &bra1 % &mocc.t();
        let dm1 = &dm1 + &dm1.swapaxes(0, 1);
        let dm1_tp = pack_triu_tilde(dm1.view());
        let itm_j_aux = cderi.t() % &dm1_tp;
        let resp_tp_j: Tsr = 2.0 * &cderi % itm_j_aux;
        let resp_ao_j = resp_tp_j.unpack_tri(Upper, FlagSymm::Sy);
        let resp_bra_j = scale_j * (resp_ao_j % &mocc);
        resp += resp_bra_j;
    }

    // --- K contribution --- //

    if scale_k != 0.0 {
        let mut resp_bra_k = rt::zeros_like(&resp);
        for iaux_start in (0..naux).step_by(nbatch_aux) {
            let iaux_end = (iaux_start + nbatch_aux).min(naux);
            let slc = rt::slice!(iaux_start, iaux_end);
            // Please note following `naux` is actually in batch, just overwriting the outside `naux` for
            // convenience.
            let naux = iaux_end - iaux_start;

            // - cderi: [nao, nao, naux]
            // - cderi_bxo: [nao, naux, nocc]
            // - cderi_oxo: [nocc, naux, nocc]
            // - cderi_box: [nao, nocc, naux]
            let cderi = cderi.i((.., slc)).unpack_tri(Upper, FlagSymm::Sy);
            let cderi_bxo = (cderi.reshape([nao, nao * naux]).t() % &mocc).into_shape([nao, naux, nocc]);
            let cderi_oxo = (&mocc.t() % cderi_bxo.reshape([nao, naux * nocc])).into_shape([nocc, naux, nocc]);

            for A in 0..nprop {
                let bra1A = bra1.i((.., .., A));
                let mut respkA = resp_bra_k.i_mut((.., .., A));
                // k contribution part 0
                // - cderi_bxo_1: [nao, naux, nocc]
                // - einsum progress: uPj, iPj -> ui
                let cderi_bxo_1 = (cderi.reshape([nao, nao * naux]).t() % &bra1A).into_shape([nao, naux, nocc]);
                respkA -= cderi_bxo_1.reshape([nao, naux * nocc]) % cderi_oxo.reshape([nocc, naux * nocc]).t();
                // k contribution part 1
                // - cderi_oxo_1: [nocc, naux, nocc] (note it is iPj, where j from bra1, i from mocc)
                // - einsum progress: uPj, iPj -> ui
                let cderi_oxo_1 = (mocc.t() % cderi_bxo_1.reshape([nao, naux * nocc])).into_shape([nocc, naux, nocc]);
                respkA -= cderi_bxo.reshape([nao, naux * nocc]) % cderi_oxo_1.reshape([nocc, naux * nocc]).t();
            }
        }
        resp += scale_k * resp_bra_k;
    }

    resp.into_shape(shape_bra)
}

/// Generate cderi and decomposition.
pub fn generate_cderi_with_decomp(
    mol: &CInt,
    aux: &CInt,
    j2c_decomp_option: J2CDecompOption,
    device: &DeviceTsr,
) -> (Tsr, J2CDecompose) {
    let j3c = hess_intor_cross(&[mol, mol, aux], "int3c2e", "s2ij", None, device);
    let j2c_decomp = get_j2c_decomp(aux, device, j2c_decomp_option);
    let cderi = solve_by_j2c(j3c, &j2c_decomp, false);
    (cderi, j2c_decomp)
}

pub struct RHessRIJK<'a> {
    pub mol: CInt,
    pub aux: CInt,
    pub scale_j: f64,
    pub scale_k: f64,
    pub cderi: TsrCow<'a>,
    pub j2c_decomp: J2CDecompose,
    pub intmd: HashMap<String, Tsr>, // intermediates
    pub result: HashMap<String, Tsr>,
}

impl<'a> RHessRIJK<'a> {
    pub fn new_without_cderi(mol: &CInt, aux: &CInt, scale_j: f64, scale_k: f64) -> Self {
        let j2c_decomp_option = J2CDecompOption::default();
        let device = DeviceTsr::default();
        let (cderi, j2c_decomp) = generate_cderi_with_decomp(mol, aux, j2c_decomp_option, &device);
        Self {
            mol: mol.clone(),
            aux: aux.clone(),
            scale_j,
            scale_k,
            cderi: cderi.into_cow(),
            j2c_decomp,
            intmd: HashMap::new(),
            result: HashMap::new(),
        }
    }

    pub fn new_with_cderi(
        mol: &CInt,
        aux: &CInt,
        scale_j: f64,
        scale_k: f64,
        cderi: TsrView<'a>,
        j2c_decomp: J2CDecompose,
    ) -> Self {
        Self {
            mol: mol.clone(),
            aux: aux.clone(),
            scale_j,
            scale_k,
            cderi: cderi.into_cow(),
            j2c_decomp,
            intmd: HashMap::new(),
            result: HashMap::new(),
        }
    }
}

impl<'a> HessUtilAPI for RHessRIJK<'a> {}

impl<'a> RHessElecInteractAPI for RHessRIJK<'a> {
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        let de_J_skeleton_dict =
            get_decomposed_rij_skeleton_deriv2_naive(&self.mol, &self.aux, mo_coeff.view(), mo_occ.view(), atm_list);
        let de_K_skeleton_dict =
            get_decomposed_rik_skeleton_deriv2_naive(&self.mol, &self.aux, mo_coeff.view(), mo_occ.view(), atm_list);
        let result = &mut self.result;
        result.extend(de_J_skeleton_dict);
        result.extend(de_K_skeleton_dict);
        let de_J = &result["de_J20"] + &result["de_J11"] + &result["de_J02"];
        let de_K = &result["de_K20"] + &result["de_K11"] + &result["de_K02"];
        self.scale_j * de_J - 0.5 * self.scale_k * de_K
    }

    fn get_deriv1_ao(&mut self, mo_coeff: TsrView, mo_occ: TsrView, atm_list: Option<&[usize]>) -> Tsr {
        let j1ao_dict = get_rij_deriv1_ao_naive(&self.mol, &self.aux, mo_coeff.view(), mo_occ.view(), atm_list);
        let k1ao_dict = get_rik_deriv1_ao_naive(&self.mol, &self.aux, mo_coeff.view(), mo_occ.view(), atm_list);
        let result = &mut self.result;
        result.extend(j1ao_dict);
        result.extend(k1ao_dict);
        let j1ao = &result["j1ao_aux0"] + &result["j1ao_aux1"];
        let k1ao = &result["k1ao_aux0"] + &result["k1ao_aux1"];
        self.scale_j * j1ao - 0.5 * self.scale_k * k1ao
    }

    fn make_response_preparation(&mut self, mo_coeff: TsrView, mo_occ: TsrView) {
        self.intmd.insert("mo_coeff".to_string(), mo_coeff.into_contig(RowMajor));
        self.intmd.insert("mo_occ".to_string(), mo_occ.to_owned());
    }

    fn get_response_bra(&mut self, bra: TsrView) -> Tsr {
        let mo_coeff = self.intmd["mo_coeff"].view();
        let mo_occ = self.intmd["mo_occ"].view();
        let cderi = self.cderi.view();
        // TODO: batch size `72` should be tunable by max-memory.
        get_rijk_response_bra(cderi, mo_coeff, mo_occ, bra, self.scale_j, self.scale_k, 72)
    }
}
