//! Hessian implementations for unrestricted SCF.

use crate::prelude::*;

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
        todo!()
    }
}
