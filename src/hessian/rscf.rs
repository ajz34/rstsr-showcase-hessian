//! Hessian implementations for restricted SCF.

use crate::prelude::*;

const TOL_OCC: f64 = 1e-15;

#[derive(Debug, Clone, PartialEq)]
pub struct RHessSCFConfig {
    pub level_shift: f64,
    pub cphf_tol: f64,
    pub cphf_max_cycle: usize,
    pub cphf_max_space: usize,
    pub cphf_lindep: f64,
}

impl Default for RHessSCFConfig {
    fn default() -> Self {
        Self { level_shift: 0.0, cphf_tol: 1e-8, cphf_max_cycle: 42, cphf_max_space: 14, cphf_lindep: 1e-14 }
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

    /// Prepare the response for CPHF calculation.
    ///
    /// This involves all electron-interaction objects.
    pub fn make_response_preparation(&mut self) {
        for el_obj in self.el_list.iter_mut() {
            el_obj.make_response_preparation(self.mo_coeff.view(), self.mo_occ.view());
        }
    }
    /// Compute the response of the system to a given perturbation in MO space (mo1), which is
    /// needed for CPHF.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo, nocc, ...]`. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[..., nmo, nocc]`. The response in MO space.
    pub fn response_mo(&self, mo1: TsrView) -> Tsr {
        let mo_coeff = self.mo_coeff.view();
        let ubra = &mo_coeff % &mo1;
        let mut resp = rt::zeros_like(&mo1);
        for el_obj in self.el_list.iter() {
            resp += mo_coeff.t() % el_obj.get_response_bra(ubra.view());
        }
        resp
    }
    /// Compute the dimensionless response for CP-HF calculation.
    ///
    /// Compared to usual CP-HF response, this additionally handles
    /// - the level shift in denominator
    /// - the zeroing of occupied-part response (we use `mo1[occ, occ]` part for evaluating
    ///   `resp[vir, occ]`, but we actually only want to solve the `mo1[vir, occ]` part and freeze
    ///   `mo1[occ, occ]` part to always be 0.5 times of ovlp_deriv1).
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo, nocc, ...]`. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo, nocc, ...]`. The dimensionless response in MO space.
    pub fn response_dimless_cphf(&self, mo1: TsrView) -> Tsr {
        let mo_occ = self.mo_occ.view();
        let mo_energy = self.mo_energy.view();
        let level_shift = self.config.level_shift;
        let occidx = mo_occ.view().greater(TOL_OCC).into_vec();
        let viridx = occidx.iter().map(|&x| !x).collect_vec();
        let nocc = occidx.iter().filter(|&&x| x).count();
        let nmo = mo_occ.shape()[0];
        let eocc = mo_energy.bool_select(-1, &occidx);
        let evir = mo_energy.bool_select(-1, &viridx);
        let so = rt::slice!(0, nocc);
        let sv = rt::slice!(nocc, nmo);
        let e_ai = evir.i((.., None)) - eocc.i((None, ..));
        let e_ai_shift = &e_ai + level_shift;

        let mut resp = self.response_mo(mo1.view());

        // handle dimensionless denominator and force handle virtual-part only
        if level_shift != 0.0 {
            resp -= level_shift * &mo1;
        }
        *&mut resp.i_mut(sv) /= &e_ai_shift;
        resp.i_mut(so).fill(0.0);
        resp
    }

    /// Solve the dimensionless CP-HF equation using a Krylov solver.
    ///
    /// This should solves `U + resp(U) = rhs`. Note difference of standard CP-HF equation as
    /// mentioned in functions above.
    ///
    /// # Parameters
    ///
    /// - `rhs` : shape `[nmo, nocc, 3, natm]`. Dimensionless right-hand side.
    ///
    /// # Returns
    ///
    /// - `mo1` : shape `[nmo, nocc, 3, natm]`. Perturbation in MO space that solves the
    ///   dimensionless CP-HF equation.
    pub fn solve_dimless_cphf(&self, rhs: TsrView) -> Tsr {
        let rhs_shape = rhs.shape().to_vec();
        let nmo = rhs.shape()[0];
        let nocc = rhs.shape()[1];
        let rhs = rhs.reshape((nmo * nocc, -1));

        let response_cphf_flattened = |x: TsrView| -> Tsr {
            let x = x.reshape((nmo, nocc, -1));
            let y = self.response_dimless_cphf(x.view());
            y.into_shape((nmo * nocc, -1))
        };

        let tol = self.config.cphf_tol;
        let max_cycle = self.config.cphf_max_cycle;
        let max_space = self.config.cphf_max_space;
        let lindep = self.config.cphf_lindep;
        let mo1 = krylov_block(response_cphf_flattened, rhs.view(), None, tol, max_cycle, max_space, lindep);
        mo1.into_shape(rhs_shape)
    }
}
