//! Hessian implementations for restricted SCF.

use crate::prelude::*;

const TOL_OCC: f64 = 1e-15;

#[derive(Debug, Clone, PartialEq)]
pub struct RHessSCFConfig {
    pub level_shift: f64,
    pub cphf_tol: f64,
    pub cphf_max_cycle: usize,
    pub cphf_max_space: usize,
}

impl Default for RHessSCFConfig {
    fn default() -> Self {
        Self { level_shift: 0.0, cphf_tol: 1e-8, cphf_max_cycle: 42, cphf_max_space: 14 }
    }
}

/// Working solver and maintainer of all hessian components for restricted SCF method.
pub struct RHessSCF<'a> {
    pub mo_coeff: Tsr,
    pub mo_occ: Tsr,
    pub mo_energy: Tsr,
    pub ovlp_obj: RHessOvlp,
    pub core_list: Vec<&'a mut dyn RHessCoreAPI>,
    pub el_list: Vec<&'a mut dyn RHessElecInteractAPI>,
    pub config: RHessSCFConfig,
}

impl<'a> RHessSCF<'a> {
    pub fn new(
        mo_coeff: Tsr,
        mo_occ: Tsr,
        mo_energy: Tsr,
        ovlp_obj: RHessOvlp,
        core_list: Vec<&'a mut dyn RHessCoreAPI>,
        el_list: Vec<&'a mut dyn RHessElecInteractAPI>,
        config: RHessSCFConfig,
    ) -> Self {
        Self { mo_coeff, mo_occ, mo_energy, ovlp_obj, core_list, el_list, config }
    }

    /// Number of atoms in the molecule, for which the hessian is computed.
    pub fn natm(&self) -> usize {
        self.ovlp_obj.natm()
    }

    /// Compute the dimensionless CPHF right-hand side, along with necessary intermediates for later
    /// steps.
    ///
    /// Note there are some differences compared to usual CP-HF:
    /// - Usual CP-HF is `(ea - ei) U - AU = B`, where now we handle something like `U + (A / (ea -
    ///   ei)) U = - B / (ea - ei)`
    /// - We now handle the U in all-occ block, instead of standard vir-occ block; this will omit
    ///   the response evaluation during rhs (B), making the rhs evaluation cheap, but we also need
    ///   to carefully handle the CP-HF equation. this behavior should be similar to PySCF's
    ///   `solve_withs1`.
    ///
    /// # Returns
    ///     
    /// A dictionary containing:
    /// - `rhs : shape `[nmo, nocc, 3, natm]`. The dimensionless CPHF right-hand side.
    /// - `f1mo` : shape `[nmo, nocc, 3, natm]`. The first-order derivative of the Fock matrix in MO
    ///   basis.
    /// - `s1mo` : shape `[nmo, nocc, 3, natm]`. The first-order derivative of the overlap matrix in
    ///   MO basis.
    pub fn compute_dimless_cphf_rhs(&mut self) -> HashMap<&'static str, Tsr> {
        // setups
        let mo_coeff = &self.mo_coeff;
        let mo_occ = &self.mo_occ;
        let mo_energy = &self.mo_energy;
        let level_shift = self.config.level_shift;
        let device = mo_coeff.device().clone();

        let [nao, nmo] = mo_coeff.shape().to_vec().try_into().unwrap();
        let occidx = mo_occ.view().greater(TOL_OCC).into_vec();
        let viridx = occidx.iter().map(|&x| !x).collect_vec();
        let mocc = mo_coeff.bool_select(-1, &occidx);
        let eocc = mo_energy.bool_select(-1, &occidx);
        let evir = mo_energy.bool_select(-1, &viridx);
        let nocc = occidx.iter().filter(|&&x| x).count();
        let natm = self.natm();

        let e_ai = evir.i((.., None)) - eocc.i((None, ..));
        let e_ai_shift = &e_ai + level_shift;

        // --- f1mo --- //

        // fock skeleton derivative (core contribution)
        let mut f1ao_core: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
        for core_obj in self.core_list.iter() {
            if let Some(mut gen_core_deriv1) = core_obj.generator_deriv1() {
                for A in 0..natm {
                    *&mut f1ao_core.i_mut((Ellipsis, A)) += gen_core_deriv1(A);
                }
            }
        }

        // fock skeleton derivative (electron interaction contribution, half-transformed to bra)
        let mut f1bra_el: Tsr = rt::zeros(([nao, nocc, 3, natm], &device));
        for el_obj in self.el_list.iter_mut() {
            f1bra_el += el_obj.get_deriv1_bra(mo_coeff.view(), mo_occ.view());
        }

        // construct whole f1mo
        let f1bra = f1bra_el + f1ao_core % &mocc;
        let f1mo = mo_coeff.t() % f1bra;

        // --- s1mo --- //

        let mut gen_ovlp_deriv1 = self.ovlp_obj.generator_deriv1();
        let mut s1ao: Tsr = rt::zeros(([nao, nao, 3, natm], &device));
        for A in 0..natm {
            *&mut s1ao.i_mut((Ellipsis, A)) += gen_ovlp_deriv1(A);
        }
        let s1mo = mo_coeff.t() % &s1ao % &mocc;

        // --- dimensionless cphf rhs --- //

        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);
        let b1mo = &f1mo - &s1mo * eocc.i((None, ..));
        let mut rhs = rt::zeros(([nmo, nocc, 3, natm], &device));
        *&mut rhs.i_mut(sv) += -b1mo.i(sv) / &e_ai_shift;
        *&mut rhs.i_mut(so) += -0.5 * s1mo.i(so);

        HashMap::from([("f1mo", f1mo), ("s1mo", s1mo), ("rhs", rhs)])
    }
}
