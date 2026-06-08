use crate::prelude::*;

/// Hessian contribution from overlap matrix derivative.
///
/// # Notes
///
/// Please be aware that the overlap matrix derivative is **NOT skeleton derivative**.
///
/// It's true origin is the application of Hellmann-Feynman theorem, that converts part of the
/// response of density matrix to the response of basis functions.
///
/// # Parameters
///
/// - `mol` : [`CInt`]. The molecule object.
/// - `dme0` : shape `[nao, nao]`. The energy-weighted density matrix for current SCF component.
///
/// Returns
/// -------
/// - `de_ovlp` : shape `[3, 3, natm, natm]`. The Hessian contribution from the overlap matrix
///   derivative.
pub fn get_hess_ovlp(mol: &CInt, dme0: TsrView) -> Tsr {
    let device = dme0.device();
    let natm = mol.natm();
    let nao = mol.nao();
    let aoslices = mol.aoslice_by_atom();

    check_shape!(dme0.shape(), [nao, nao], "density matrix shape not correct.");

    let s2_aa = hess_intor(mol, "int1e_ipipovlp", "s1", None, device);
    let s2_ab = hess_intor(mol, "int1e_ipovlpip", "s1", None, device);

    let mut de_ovlp = rt::zeros(([3, 3, natm, natm], device));
    for A in 0..natm {
        let [_, _, p0A, p1A] = aoslices[A];
        let slcA = rt::slice!(p0A, p1A);
        let scr = -2 * (s2_aa.i(slcA) * dme0.i(slcA)).sum_axes([0, 1]);
        *&mut de_ovlp.i_mut((.., .., A, A)) += scr;

        for B in 0..=A {
            let [_, _, p0B, p1B] = aoslices[B];
            let slcB = rt::slice!(p0B, p1B);
            let scr = -2 * (s2_ab.i((slcA, slcB)) * dme0.i((slcA, slcB))).sum_axes([0, 1]);
            *&mut de_ovlp.i_mut((.., .., B, A)) += scr;
        }
        for B in 0..A {
            let de_to_copy = de_ovlp.i((.., .., B, A)).t().to_owned();
            *&mut de_ovlp.i_mut((.., .., A, B)) += de_to_copy;
        }
    }
    de_ovlp
}

/// Generator for the first derivative of overlap matrix.
///
/// # Parameters
///
/// - `mol` : [`CInt`]. The molecule object.
/// - `device` : [`DeviceTsr`]. The device on which the returned tensor is allocated.
///
/// # Returns
///
/// - `FnMut(A: usize, B: usize) -> Tsr`. A function that computes the first derivative of the
///   overlap matrix with respect to the nuclear coordinates. Input is the atom index A. The
///   returned array has shape [3, nao, nao].
pub fn generator_ovlp_deriv1(mol: &CInt, device: &DeviceTsr) -> impl FnMut(usize) -> Tsr {
    // preparation
    let device = device.clone();
    let nao = mol.nao();
    let aoslices = mol.aoslice_by_atom();

    let int1e_ipovlpip = hess_intor(mol, "int1e_ipovlpip", "s1", None, &device);

    move |A: usize| {
        let [_, _, p0, p1] = aoslices[A];
        let slc = rt::slice!(p0, p1);
        let mut s1ao = rt::zeros(([nao, nao, 3], &device));
        *&mut s1ao.i_mut((slc, ..)) -= int1e_ipovlpip.i(slc);
        *&mut s1ao.i_mut((.., slc)) -= int1e_ipovlpip.i(slc).swapaxes(0, 1);
        s1ao
    }
}

/// Hessian contribution from overlap matrix derivative.
///
/// Note that overlap is special to the SCF part, in that
/// - The contribution of hessian from overlap is not skeleton, so we do not derive this class from
///   [`RHessCoreAPI`].
/// - The CP-HF requires both first order derivative of hcore and ovlp, but their roles are
///   different.
///
/// Due to these reasons, although it has the similar interface to [`RHessCoreAPI`],
/// [`RHessOvlp`] is designed as a standalone class, without inheriting from any abstract class.
pub struct RHessOvlp {
    pub mol: CInt,
    pub device: DeviceTsr,
}

impl RHessOvlp {
    pub fn new(mol: &CInt, device: &DeviceTsr) -> Self {
        Self { mol: mol.clone(), device: device.clone() }
    }

    pub fn make_hess(&self, dme0: TsrView) -> Tsr {
        get_hess_ovlp(&self.mol, dme0)
    }

    pub fn generator_deriv1(&self) -> impl FnMut(usize) -> Tsr {
        generator_ovlp_deriv1(&self.mol, &self.device)
    }
}
