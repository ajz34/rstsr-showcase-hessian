use crate::prelude::*;

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
/// - `mo_coeff` : [TsrView]. The molecular orbital coefficients, shape `[nao, nmo]`.
/// - `mo_occ` : [TsrView]. The molecular orbital occupation numbers, shape `[nmo]`.
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
    let nao = mol.nao();
    let natm = mol.natm();
    let naux = aux.nao();
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
    let mut de_J20_1 = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            *&mut de_J20_1.i_mut((.., .., B, A)) += 4.0 * dbas_J20_1.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }

    // (11|0)(0|00)
    let subscripts = "uvPst, PQ, klQ, uv, kl -> uvst";
    let operands = [&int3c2e_ipvip1, &int2c2e_inv, &int3c2e, &dm0, &dm0];
    let dbas_J20_2 = rt::tblis::einsum(subscripts, operands, true, None);
    let mut de_J20_2 = rt::zeros(([3, 3, natm, natm], device));
    for (A, &[_, _, p0A, p1A]) in aoslices.iter().enumerate() {
        for (B, &[_, _, p0B, p1B]) in aoslices.iter().enumerate() {
            *&mut de_J20_2.i_mut((.., .., B, A)) += 2.0 * dbas_J20_2.i((p0A..p1A, p0B..p1B)).sum_axes([0, 1]);
        }
    }

    HashMap::from([("de_J20_1", de_J20_1), ("de_J20_2", de_J20_2)])
}
