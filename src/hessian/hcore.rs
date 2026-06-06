use crate::prelude::*;

/// Generator for second derivatives of the core Hamiltonian (skeleton derivative).
///
/// # Parameters
///
/// - `mol` : [`CInt`]. The molecule object.
/// - `device` : [`DeviceTsr`]. The device on which the returned tensor is allocated.
///
/// Returns
/// -------
/// - `get_hcore_deriv_at_atoms` : `Fn(A: usize, B: usize) -> Tsr`. A function that computes the
///   second derivative of the core Hamiltonian with respect to the nuclear coordinates. The
///   returned array has shape [nao, nao, 3, 3].
pub fn generator_hcore_deriv2(mol: &CInt, device: &DeviceTsr) -> impl FnMut(usize, usize) -> Tsr {
    // preparation
    let device = device.clone();
    let mut mol = mol.clone();
    let nao = mol.nao();
    let nbas = mol.nbas();
    let ecp_atoms = get_ecp_atoms(&mol);
    let aoslices = mol.aoslice_by_atom();

    // we need to prepare some integrals, to somehow avoid redundant calculations in the loop
    // - aa: Hamiltonian derivative to only the first basis
    // - ab: Hamiltonian derivative to the first and second basis
    // all integrals are of shape [nao, nao, 3, 3]
    let mut h2_aa = hess_intor(&mol, "int1e_ipipkin", "s1", None, &device);
    let mut h2_ab = hess_intor(&mol, "int1e_ipkinip", "s1", None, &device);
    h2_aa += hess_intor(&mol, "int1e_ipipnuc", "s1", None, &device);
    h2_ab += hess_intor(&mol, "int1e_ipnucip", "s1", None, &device);
    if mol.has_ecp() {
        h2_aa += hess_intor(&mol, "ECPscalar_ipipnuc", "s1", None, &device);
        h2_ab += hess_intor(&mol, "ECPscalar_ipnucip", "s1", None, &device);
    }

    move |A: usize, B: usize| {
        let [sh0A, sh1A, p0A, p1A] = aoslices[A];
        let [sh0B, sh1B, p0B, p1B] = aoslices[B];
        let slcA = rt::slice!(p0A, p1A);
        let slcB = rt::slice!(p0B, p1B);
        let zA = mol.atom_charge(A);
        let zB = mol.atom_charge(B);

        let mut hcore_deriv: Tsr = rt::zeros(([nao, nao, 3, 3], &device));

        if A == B {
            mol.with_rinv_at_nucleus(A, |mol| {
                let mut rinv_aa = -zA * hess_intor(mol, "int1e_ipiprinv", "s1", None, &device);
                let mut rinv_ab = -zA * hess_intor(mol, "int1e_iprinvip", "s1", None, &device);
                if ecp_atoms.contains(&A) {
                    rinv_aa += hess_intor(mol, "ECPscalar_ipiprinv", "s1", None, &device);
                    rinv_ab += hess_intor(mol, "ECPscalar_iprinvip", "s1", None, &device);
                }
                hcore_deriv += &rinv_aa;
                hcore_deriv += &rinv_ab;
                *&mut hcore_deriv.i_mut((slcA, ..)) -= rinv_aa.i((slcA, ..));
                *&mut hcore_deriv.i_mut((slcA, ..)) -= rinv_ab.i((slcA, ..));
                *&mut hcore_deriv.i_mut((.., slcA)) -= rinv_aa.i((slcA, ..)).swapaxes(0, 1);
                *&mut hcore_deriv.i_mut((.., slcA)) -= rinv_ab.i((.., slcA));
            });
            *&mut hcore_deriv.i_mut((slcA, ..)) += h2_aa.i((slcA, ..));
            *&mut hcore_deriv.i_mut((slcA, slcA)) += h2_ab.i((slcA, slcA));
        } else {
            *&mut hcore_deriv.i_mut((slcA, slcB)) += h2_ab.i((slcA, slcB));
            mol.with_rinv_at_nucleus(A, |mol| {
                let shls_slice = [[sh0B, sh1B], [0, nbas]];
                let mut rinv_atom_aa = -zA * hess_intor(mol, "int1e_ipiprinv", "s1", shls_slice, &device);
                let mut rinv_atom_ab = -zA * hess_intor(mol, "int1e_iprinvip", "s1", shls_slice, &device);
                if ecp_atoms.contains(&A) {
                    rinv_atom_aa += hess_intor(mol, "ECPscalar_ipiprinv", "s1", shls_slice, &device);
                    rinv_atom_ab += hess_intor(mol, "ECPscalar_iprinvip", "s1", shls_slice, &device);
                }
                *&mut hcore_deriv.i_mut((slcB, ..)) -= &rinv_atom_aa;
                *&mut hcore_deriv.i_mut((slcB, ..)) -= &rinv_atom_ab.swapaxes(-1, -2);
            });
            mol.with_rinv_at_nucleus(B, |mol| {
                let shls_slice = [[sh0A, sh1A], [0, nbas]];
                let mut rinv_atom_aa = -zB * hess_intor(mol, "int1e_ipiprinv", "s1", shls_slice, &device);
                let mut rinv_atom_ab = -zB * hess_intor(mol, "int1e_iprinvip", "s1", shls_slice, &device);
                if ecp_atoms.contains(&B) {
                    rinv_atom_aa += hess_intor(mol, "ECPscalar_ipiprinv", "s1", shls_slice, &device);
                    rinv_atom_ab += hess_intor(mol, "ECPscalar_iprinvip", "s1", shls_slice, &device);
                }
                *&mut hcore_deriv.i_mut((slcA, ..)) -= &rinv_atom_aa;
                *&mut hcore_deriv.i_mut((slcA, ..)) -= &rinv_atom_ab;
            });
        }

        &hcore_deriv + hcore_deriv.swapaxes(0, 1)
    }
}

/// Hessian contribution from the core Hamiltonian.
///
/// # Parameters
///
/// - `mol` : [`CInt`]. The molecule object.
/// - `dm0` : shape [nao, nao]. The SCF density matrix.
///
/// # Returns
/// - `de_hcore` : shape [3, 3, natm, natm]. The Hessian contribution from the core Hamiltonian.
pub fn get_hess_hcore(mol: &CInt, dm0: TsrView) -> Tsr {
    let device = dm0.device();
    let natm = mol.natm();
    let nao = mol.nao();

    check_shape!(dm0.shape(), [nao, nao], "density matrix shape not correct.");

    let mut de_hcore = rt::zeros(([3, 3, natm, natm], device));
    let mut gen_hcore_deriv2 = generator_hcore_deriv2(mol, device);

    for A in 0..natm {
        for B in 0..=A {
            let hcore_deriv2 = gen_hcore_deriv2(A, B);
            *&mut de_hcore.i_mut((.., .., B, A)) += (hcore_deriv2 * &dm0).sum_axes((0, 1));
        }
        for B in 0..A {
            let de_to_copy = de_hcore.i((.., .., B, A)).t().to_owned();
            *&mut de_hcore.i_mut((.., .., A, B)) += de_to_copy;
        }
    }
    de_hcore
}

/// Hessian contribution from the core Hamiltonian.
pub struct HessHcore {
    pub mol: CInt,
}

impl HessHcore {
    pub fn new(mol: CInt) -> Self {
        Self { mol }
    }
}

impl RHessCoreAPI for HessHcore {
    fn make_skeleton_hess(&mut self, mo_coeff: TsrView, mo_occ: TsrView) -> Tsr {
        let dm0 = get_dm0_restricted(mo_coeff, mo_occ);
        get_hess_hcore(&self.mol, dm0.view())
    }

    fn generator_deriv1(&self) -> Option<Box<dyn FnMut(usize) -> Tsr>> {
        todo!()
    }
}
