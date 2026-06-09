//! Hessian implementations for unrestricted SCF.

use crate::prelude::*;
/// Working solver and maintainer of all hessian components for unrestricted SCF method.
pub struct UHessSCF<'a> {
    pub mo_coeff: [Tsr; 2],
    pub mo_occ: [Tsr; 2],
    pub mo_energy: [Tsr; 2],
    pub ovlp_obj: UHessOvlp,
    pub nuc_list: Vec<&'a mut dyn HessNucAPI>,
    pub core_list: Vec<&'a mut dyn UHessCoreAPI>,
    pub el_list: Vec<&'a mut dyn UHessElecInteractAPI>,
    pub config: HessSCFConfig,
    pub atm_list: Option<Vec<usize>>,
}

impl<'a> UHessSCF<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mo_coeff: [Tsr; 2],
        mo_occ: [Tsr; 2],
        mo_energy: [Tsr; 2],
        ovlp_obj: UHessOvlp,
        nuc_list: Vec<&'a mut dyn HessNucAPI>,
        core_list: Vec<&'a mut dyn UHessCoreAPI>,
        el_list: Vec<&'a mut dyn UHessElecInteractAPI>,
        config: HessSCFConfig,
        atm_list: Option<Vec<usize>>,
    ) -> Self {
        Self { mo_coeff, mo_occ, mo_energy, ovlp_obj, nuc_list, core_list, el_list, config, atm_list }
    }

    /// Number of atoms over which the Hessian is computed. This is `atm_list.len()` if
    /// `atm_list` is `Some`, otherwise the total number of atoms in the molecule.
    pub fn natm(&self) -> usize {
        match &self.atm_list {
            Some(list) => list.len(),
            None => self.ovlp_obj.natm(),
        }
    }

    /// Return the list of (global) atom indices the Hessian is computed for, ordered the same
    /// way as the local indexing used in the returned Hessian.
    pub fn atm_indices(&self) -> Vec<usize> {
        match &self.atm_list {
            Some(list) => list.clone(),
            None => (0..self.ovlp_obj.natm()).collect(),
        }
    }

    /// Compute the dimensionless CPHF right-hand side, along with necessary intermediates for later
    /// steps.
    ///
    /// # Returns
    ///     
    /// A dictionary containing:
    /// - `rhs : shape `[nmo, nocc_alpha, 3, natm]` and `[nmo, nocc_beta, 3, natm]`. The
    ///   dimensionless CPHF right-hand side.
    /// - `f1mo` : shape `[nmo, nocc_alpha, 3, natm]` and `[nmo, nocc_beta, 3, natm]`. The
    ///   first-order derivative of the Fock matrix in MO basis.
    /// - `s1mo` : shape `[nmo, nocc_alpha, 3, natm]` and `[nmo, nocc_beta, 3, natm]`. The
    ///   first-order derivative of the overlap matrix in MO basis.
    pub fn compute_dimless_cphf_rhs(&mut self) -> HashMap<&'static str, Tsr> {
        let [α, β] = [0, 1];
        let mo_coeff = [self.mo_coeff[α].view(), self.mo_coeff[β].view()];
        let mo_occ = [self.mo_occ[α].view(), self.mo_occ[β].view()];
        let mo_energy = [self.mo_energy[α].view(), self.mo_energy[β].view()];
        let level_shift = self.config.level_shift;
        let device = mo_coeff[α].device().clone();

        let nao = mo_coeff[α].shape()[0];
        let nmo = [mo_coeff[α].shape()[1], mo_coeff[β].shape()[1]];
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let viridx = [occidx[α].iter().map(|&x| !x).collect_vec(), occidx[β].iter().map(|&x| !x).collect_vec()];
        let mocc = [mo_coeff[α].bool_select(-1, &occidx[α]), mo_coeff[β].bool_select(-1, &occidx[β])];
        let eocc = [mo_energy[α].bool_select(-1, &occidx[α]), mo_energy[β].bool_select(-1, &occidx[β])];
        let evir = [mo_energy[α].bool_select(-1, &viridx[α]), mo_energy[β].bool_select(-1, &viridx[β])];
        let nocc = [mocc[α].shape()[1], mocc[β].shape()[1]];
        let natm = self.natm();
        let atm_indices = self.atm_indices();
        let atm_list = self.atm_list.as_deref();

        let e_ai = [evir[α].i((.., None)) - eocc[α].i((None, ..)), evir[β].i((.., None)) - eocc[β].i((None, ..))];
        let e_ai_shift = [&e_ai[α] + level_shift, &e_ai[β] + level_shift];

        // --- f1mo --- //

        // fock skeleton derivative (core contribution)
        let mut f1ao_core: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
        for core_obj in self.core_list.iter() {
            let mut gen_core_deriv1 = core_obj.generator_deriv1();
            for (A_loc, &A_glob) in atm_indices.iter().enumerate() {
                *&mut f1ao_core.i_mut((Ellipsis, A_loc)) += gen_core_deriv1(A_glob);
            }
        }

        // fock skeleton derivative (electron interaction contribution, half-transformed to bra)
        let mut f1bra_el: [Tsr; 2] =
            [rt::zeros(([nao, nocc[α], 3, natm], &device)), rt::zeros(([nao, nocc[β], 3, natm], &device))];
        for el_obj in self.el_list.iter_mut() {
            let bra = el_obj.get_deriv1_bra(&mo_coeff, &mo_occ, atm_list);
            f1bra_el[α] += &bra[α];
            f1bra_el[β] += &bra[β];
        }

        // construct whole f1mo
        let f1mo_α = mo_coeff[α].t() % (&f1ao_core % &mocc[α] + &f1bra_el[α]);
        let f1mo_β = mo_coeff[β].t() % (&f1ao_core % &mocc[β] + &f1bra_el[β]);

        // --- s1mo --- //

        let mut gen_ovlp_deriv1 = self.ovlp_obj.generator_deriv1();
        let mut s1ao: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
        for (A_loc, &A_glob) in atm_indices.iter().enumerate() {
            *&mut s1ao.i_mut((Ellipsis, A_loc)) += gen_ovlp_deriv1(A_glob);
        }
        let s1mo_α = mo_coeff[α].t() % (&s1ao % &mocc[α]);
        let s1mo_β = mo_coeff[β].t() % (&s1ao % &mocc[β]);

        // --- dimensionless rhs --- //

        let so = [rt::slice!(0, nocc[α]), rt::slice!(0, nocc[β])];
        let sv = [rt::slice!(nocc[α], nmo[α]), rt::slice!(nocc[β], nmo[β])];
        let b1mo_α = &f1mo_α - &s1mo_α * eocc[α].i((None, ..));
        let b1mo_β = &f1mo_β - &s1mo_β * eocc[β].i((None, ..));
        let mut rhs_α = rt::zeros(([nmo[α], nocc[α], 3, natm], &device));
        let mut rhs_β = rt::zeros(([nmo[β], nocc[β], 3, natm], &device));
        *&mut rhs_α.i_mut(sv[α]) += -b1mo_α.i(sv[α]) / &e_ai_shift[α];
        *&mut rhs_β.i_mut(sv[β]) += -b1mo_β.i(sv[β]) / &e_ai_shift[β];
        *&mut rhs_α.i_mut(so[α]) += -0.5 * s1mo_α.i(so[α]);
        *&mut rhs_β.i_mut(so[β]) += -0.5 * s1mo_β.i(so[β]);

        HashMap::from([
            ("f1mo_0", f1mo_α),
            ("f1mo_1", f1mo_β),
            ("s1mo_0", s1mo_α),
            ("s1mo_1", s1mo_β),
            ("rhs_0", rhs_α),
            ("rhs_1", rhs_β),
        ])
    }
}
