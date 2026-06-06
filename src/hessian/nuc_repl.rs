use crate::prelude::*;

/// Hessian contribution from nuclear repulsion.
///
/// # Parameters
///
/// - `mol` : [`CInt`]. The molecule object.
/// - `device` : [`DeviceTsr`]. The device on which the returned tensor is allocated.
///
/// # Returns
/// -------
/// - `de_nuc` : shape [3, 3, natm, natm]. The nuclear repulsion Hessian.
pub fn get_nuc_repl_hess(mol: &CInt, device: &DeviceTsr) -> Tsr {
    let natm = mol.natm();

    // de_nuc: shape [3, 3, natm, natm]. The nuclear repulsion Hessian.
    let mut de_nuc: Tsr = rt::zeros(([3, 3, natm, natm], device));
    // qs: shape [natm]. The atomic charges.
    let qs = rt::asarray((mol.atom_charges(), device));
    // rs: shape [3, natm]. The atomic coordinates.
    let rs = rt::asarray((mol.atom_coords(), device)).into_unpack_array(0);

    for A in 0..natm {
        // r12: shape [3, natm]. The vector from atom A to other atoms.
        let r12 = rs.i((.., A)) - &rs;
        // s12: shape [natm]. The distance from atom A to other atoms.
        //      this value will be divided, so we set zero distance to inf to avoid NaN.
        let mut s12 = r12.l2_norm_axes(0);
        s12[[A]] = f64::INFINITY;

        // tmp1: shape [natm]
        let tmp1 = qs[[A]] * &qs / s12.pow(3);
        // prefactor: shape [natm]
        let prefactor = -3.0 * qs[[A]] * &qs / s12.pow(5);
        // tmp2: shape [3, 3, natm]
        let tmp2 = prefactor.i((None, None, ..)) * r12.i((.., None, ..)) * r12.i((None, .., ..));

        // diagonal block
        let tmp1_sum = tmp1.sum(); // scalar
        let tmp2_sum = tmp2.sum_axes(-1); // shape [3, 3]
        de_nuc[[0, 0, A, A]] -= tmp1_sum;
        de_nuc[[1, 1, A, A]] -= tmp1_sum;
        de_nuc[[2, 2, A, A]] -= tmp1_sum;
        *&mut de_nuc.i_mut((.., .., A, A)) -= tmp2_sum;

        // off-diagonal blocks
        *&mut de_nuc.i_mut((0, 0, A, ..)) += &tmp1;
        *&mut de_nuc.i_mut((1, 1, A, ..)) += &tmp1;
        *&mut de_nuc.i_mut((2, 2, A, ..)) += &tmp1;
        *&mut de_nuc.i_mut((.., .., A, ..)) += &tmp2;
    }

    de_nuc
}

/// Hessian contribution from nuclear repulsion.
pub struct HessNucRepl {
    pub mol: CInt,
}

impl HessNucRepl {
    pub fn new(mol: &CInt) -> Self {
        Self { mol: mol.clone() }
    }
}

impl RHessCoreAPI for HessNucRepl {
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, _mo_occ: TsrView) -> Tsr {
        get_nuc_repl_hess(&self.mol, mo_coeff.device())
    }

    fn generator_deriv1(&self) -> Option<Box<dyn FnMut(usize) -> Tsr>> {
        None
    }
}
