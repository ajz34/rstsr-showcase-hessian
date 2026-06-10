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
    /// - `rhs : shape `[nmo, nocc_α, 3, natm]` and `[nmo, nocc_β, 3, natm]`. The dimensionless CPHF
    ///   right-hand side.
    /// - `f1mo` : shape `[nmo, nocc_α, 3, natm]` and `[nmo, nocc_β, 3, natm]`. The first-order
    ///   derivative of the Fock matrix in MO basis.
    /// - `s1mo` : shape `[nmo, nocc_α, 3, natm]` and `[nmo, nocc_β, 3, natm]`. The first-order
    ///   derivative of the overlap matrix in MO basis.
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

    /// Prepare the response for CPHF calculation.
    ///
    /// This involves all electron-interaction objects.
    pub fn make_response_preparation(&mut self) {
        let mo_coeff = [self.mo_coeff[0].view(), self.mo_coeff[1].view()];
        let mo_occ = [self.mo_occ[0].view(), self.mo_occ[1].view()];
        for el_obj in self.el_list.iter_mut() {
            el_obj.make_response_preparation(&mo_coeff, &mo_occ);
        }
    }

    /// Compute the response of the system to a given perturbation in MO space (mo1), which is
    /// needed for CPHF.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. The response in MO space.
    pub fn response_mo(&self, mo1: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        let ubra_α = &self.mo_coeff[α] % &mo1[α];
        let ubra_β = &self.mo_coeff[β] % &mo1[β];
        let mut resp_α = rt::zeros_like(&ubra_α);
        let mut resp_β = rt::zeros_like(&ubra_β);

        for el_obj in self.el_list.iter() {
            let el_resp = el_obj.get_response_bra(&[ubra_α.view(), ubra_β.view()]);
            resp_α += self.mo_coeff[α].t() % &el_resp[α];
            resp_β += self.mo_coeff[β].t() % &el_resp[β];
        }
        [resp_α, resp_β]
    }

    /// Compute the dimensionless response for CP-HF calculation.
    ///
    /// # Parameters
    ///
    /// - `mo1` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. The perturbation in MO space.
    ///
    /// # Returns
    ///
    /// - `resp` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. The dimensionless response
    ///   in MO space.
    pub fn response_dimless_cphf(&self, mo1: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        let mo_occ = [self.mo_occ[α].view(), self.mo_occ[β].view()];
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let viridx = [occidx[α].iter().map(|&x| !x).collect_vec(), occidx[β].iter().map(|&x| !x).collect_vec()];
        let nocc = [occidx[α].iter().filter(|&&x| x).count(), occidx[β].iter().filter(|&&x| x).count()];
        let nmo = [mo_occ[α].shape()[0], mo_occ[β].shape()[0]];
        let eocc = [
            self.mo_energy[α].view().bool_select(-1, &occidx[α]),
            self.mo_energy[β].view().bool_select(-1, &occidx[β]),
        ];
        let evir = [
            self.mo_energy[α].view().bool_select(-1, &viridx[α]),
            self.mo_energy[β].view().bool_select(-1, &viridx[β]),
        ];
        let e_ai = [evir[α].i((.., None)) - eocc[α].i((None, ..)), evir[β].i((.., None)) - eocc[β].i((None, ..))];
        let level_shift = self.config.level_shift;
        let e_ai_shift = [&e_ai[0] + level_shift, &e_ai[1] + level_shift];
        let so = [rt::slice!(0, nocc[α]), rt::slice!(0, nocc[β])];
        let sv = [rt::slice!(nocc[α], nmo[α]), rt::slice!(nocc[β], nmo[β])];

        let mut resp = self.response_mo(mo1);

        // handle dimension less denominator and occupied response part
        if level_shift != 0.0 {
            *&mut resp[α] -= level_shift * &mo1[α];
            *&mut resp[β] -= level_shift * &mo1[β];
        }
        *&mut resp[α].i_mut(sv[α]) /= &e_ai_shift[α];
        *&mut resp[β].i_mut(sv[β]) /= &e_ai_shift[β];
        resp[α].i_mut(so[α]).fill(0.0);
        resp[β].i_mut(so[β]).fill(0.0);
        resp
    }

    /// Solve the dimensionless CP-HF equation using a Krylov solver.
    ///
    /// # Parameters
    ///
    /// - `rhs` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. Dimensionless right-hand
    ///   side.
    ///
    /// # Returns
    ///
    /// - `mo1` : shape `[nmo, nocc_α, ...]` and `[nmo, nocc_β, ...]`. Perturbation in MO space that
    ///   solves the dimensionless CP-HF equation.
    pub fn solve_dimless_cphf(&self, rhs: &[TsrView; 2]) -> [Tsr; 2] {
        let [α, β] = [0, 1];
        let rhs_shape = [rhs[α].shape().to_vec(), rhs[β].shape().to_vec()];
        let nmo = [rhs[α].shape()[0], rhs[β].shape()[0]];
        let nocc = [rhs[α].shape()[1], rhs[β].shape()[1]];
        let rhs = [rhs[α].reshape((nmo[α], nocc[α], -1)), rhs[β].reshape((nmo[β], nocc[β], -1))];
        let device = rhs[α].device().clone();

        let pack_flattened = |x: &[TsrView; 2]| -> Tsr {
            // original: [nmo_α, nocc_α, nprop] and [nmo_β, nocc_β, nprop]
            // target: [nmo_α * nocc_α + nmo_β * nocc_β, nprop]
            check_shape!(x[α].ndim(), 3, "Expected x[α] to have shape [nmo_α, nocc_α, nprop]");
            check_shape!(x[β].ndim(), 3, "Expected x[β] to have shape [nmo_β, nocc_β, nprop]");
            let nprop = x[α].shape()[2];
            let mut x_flattened = rt::zeros(([nmo[α] * nocc[α] + nmo[β] * nocc[β], nprop], &device));
            for A in 0..nprop {
                x_flattened.i_mut((..nmo[α] * nocc[α], A)).assign(x[α].i((.., .., A)).reshape(-1));
                x_flattened.i_mut((nmo[α] * nocc[α].., A)).assign(x[β].i((.., .., A)).reshape(-1));
            }
            x_flattened
        };

        let unpack_flattened = |x: TsrView| -> [Tsr; 2] {
            // original: [nmo_α * nocc_α + nmo_β * nocc_β, nprop]
            // target: [nmo_α, nocc_α, nprop] and [nmo_β, nocc_β, nprop]
            check_shape!(x.ndim(), 2, "Expected x to have shape [nmo_α * nocc_α + nmo_β * nocc_β, nprop]");
            let nprop = x.shape()[1];
            let idx_split = nmo[α] * nocc[α];
            let mut x_α = rt::zeros(([nmo[α], nocc[α], nprop], &device));
            let mut x_β = rt::zeros(([nmo[β], nocc[β], nprop], &device));
            for A in 0..nprop {
                x_α.i_mut((.., .., A)).assign(x.i((..idx_split, A)).reshape((nmo[α], nocc[α])));
                x_β.i_mut((.., .., A)).assign(x.i((idx_split.., A)).reshape((nmo[β], nocc[β])));
            }
            [x_α, x_β]
        };

        let response_cphf_flattened = |x: TsrView| -> Tsr {
            // split x by spin and reshape to original shape
            let [x_α, x_β] = unpack_flattened(x);
            // compute response by usual means
            let resp = self.response_dimless_cphf(&[x_α.view(), x_β.view()]);
            // flatten resp to shape (nmo*nocc, nprop)
            let resp_view = resp.iter().map(|r| r.view()).collect_array().unwrap();
            pack_flattened(&resp_view)
        };

        let tol = self.config.cphf_tol;
        let max_cycle = self.config.cphf_max_cycle;
        let max_space = self.config.cphf_max_space;
        let lindep = self.config.cphf_lindep;
        let rhs_view = rhs.iter().map(|r| r.view()).collect_array().unwrap();
        let rhs_packed = pack_flattened(&rhs_view);
        let mo1_flattened =
            krylov_block(response_cphf_flattened, rhs_packed.view(), None, tol, max_cycle, max_space, lindep);
        let [mo1_α, mo1_β] = unpack_flattened(mo1_flattened.view());
        [mo1_α.into_shape(rhs_shape[α].to_vec()), mo1_β.into_shape(rhs_shape[β].to_vec())]
    }

    /// Finalize the CP-HF calculation by computing necessary intermediates for Hessian assembly.
    ///
    ///
    /// # Parameters
    ///
    /// - `f1mo` : shape `[nmo_α, nocc_α, 3, natm]` and `[nmo_β, nocc_β, 3, natm]`. The first-order
    ///   derivative of the Fock matrix in MO basis, obtained from
    ///   [`Self::compute_dimless_cphf_rhs`].
    /// - `s1mo` : shape `[nmo_α, nocc_α, 3, natm]` and `[nmo_β, nocc_β, 3, natm]`. The first-order
    ///   derivative of the overlap matrix in MO basis, obtained from
    ///   [`Self::compute_dimless_cphf_rhs`].
    /// - `mo1` : shape `[nmo_α, nocc_α, 3, natm]` and `[nmo_β, nocc_β, 3, natm]`. The perturbation
    ///   in MO space obtained from Krylov solver.
    ///
    /// # Returns
    ///
    /// `HashMap<&str, Tsr>`
    ///
    /// - `mo1_0`, `mo1_1` : shape `[nmo_α, nocc_α, 3, natm]` and `[nmo_β, nocc_β, 3, natm]`. The
    ///   finalized perturbation in MO space.
    /// - `mo_e1_0`, `mo_e1_1` : shape `[nocc_α, nocc_α, 3, natm]` and `[nocc_β, nocc_β, 3, natm]`.
    ///   The derivative of occupied orbital energies (Fock matrix) with respect to perturbation.
    pub fn finalize_cphf(
        &self,
        f1mo: &[TsrView; 2],
        s1mo: &[TsrView; 2],
        mo1: &[TsrView; 2],
    ) -> HashMap<&'static str, Tsr> {
        let [α, β] = [0, 1];
        let mo_occ = [self.mo_occ[α].view(), self.mo_occ[β].view()];
        let occidx = [mo_occ[α].view().greater(0).into_vec(), mo_occ[β].view().greater(0).into_vec()];
        let viridx = [occidx[α].iter().map(|&x| !x).collect_vec(), occidx[β].iter().map(|&x| !x).collect_vec()];
        let nocc = [occidx[α].iter().filter(|&&x| x).count(), occidx[β].iter().filter(|&&x| x).count()];
        let nmo = [mo_occ[α].shape()[0], mo_occ[β].shape()[0]];
        let eocc = [
            self.mo_energy[α].view().bool_select(-1, &occidx[α]),
            self.mo_energy[β].view().bool_select(-1, &occidx[β]),
        ];
        let evir = [
            self.mo_energy[α].view().bool_select(-1, &viridx[α]),
            self.mo_energy[β].view().bool_select(-1, &viridx[β]),
        ];
        let so = [rt::slice!(0, nocc[α]), rt::slice!(0, nocc[β])];
        let sv = [rt::slice!(nocc[α], nmo[α]), rt::slice!(nocc[β], nmo[β])];
        let e_ai = [evir[α].i((.., None)) - eocc[α].i((None, ..)), evir[β].i((.., None)) - eocc[β].i((None, ..))];
        let e_ij = [eocc[α].i((.., None)) - eocc[α].i((None, ..)), eocc[β].i((.., None)) - eocc[β].i((None, ..))];

        // last-iter the cp-hf equation, and remove the level-shift
        let last_resp = self.response_mo(mo1);
        let b1mo_α = &f1mo[α] - &s1mo[α] * eocc[α].i((None, ..)) + &last_resp[α];
        let b1mo_β = &f1mo[β] - &s1mo[β] * eocc[β].i((None, ..)) + &last_resp[β];
        let mut mo1_α = mo1[α].to_owned();
        let mut mo1_β = mo1[β].to_owned();
        mo1_α.i_mut(sv[α]).assign(-b1mo_α.i(sv[α]) / &e_ai[α]);
        mo1_β.i_mut(sv[β]).assign(-b1mo_β.i(sv[β]) / &e_ai[β]);

        // get the derivative of fock matrix in occ-occ block (derivative of orbital energy with rotation)
        let mo_e1_α = b1mo_α.i(so[α]) + &mo1_α.i(so[α]) * &e_ij[α];
        let mo_e1_β = b1mo_β.i(so[β]) + &mo1_β.i(so[β]) * &e_ij[β];

        HashMap::from([("mo1_0", mo1_α), ("mo1_1", mo1_β), ("mo_e1_0", mo_e1_α), ("mo_e1_1", mo_e1_β)])
    }
}
