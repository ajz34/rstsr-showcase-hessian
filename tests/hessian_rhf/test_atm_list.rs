//! Tests for the `atm_list` argument: results for a selected subset of atoms must equal the
//! corresponding sub-block of the full Hessian.

use crate::hessian_rhf::*;
use crate::test_util::{Tsr, TsrView};
use rstest::rstest;
use rstsr_showcase_hessian::hessian::ri_jk_restricted_naive::*;
use rstsr_showcase_hessian::prelude::*;
use rstsr_showcase_hessian::util::density_matrices::get_dme0_restricted;

/// Subset of atoms used across all atm_list tests. NH3 has 4 atoms (indices 0..=3).
const ATM_LIST: &[usize] = &[1, 3];

/// Build a `[3, 3, len(atm_list), len(atm_list)]` view by selecting rows/cols of a full Hessian.
fn select_atoms(full: TsrView, atm_list: &[usize]) -> Tsr {
    full.index_select(-1, atm_list).index_select(-2, atm_list)
}

#[rstest]
fn test_atm_list_nuc_repl(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, .. } = hess_case;
    let mut hess = HessNucRepl::new(mol, &DeviceTsr::default());
    let full = hess.make_skeleton_hess(None);
    let sel = hess.make_skeleton_hess(Some(ATM_LIST));
    let expected = select_atoms(full.view(), ATM_LIST);

    println!("full:\n{expected:12.6}");
    println!("sel:\n{sel:12.6}");
    assert!(rt::allclose(sel.view(), expected.view(), (1e-10, 1e-12)));
}

#[rstest]
fn test_atm_list_hcore(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, mo_coeff, mo_occ, .. } = hess_case;
    let mut hess = RHessHcore::new(mol, &DeviceTsr::default());
    let full = hess.make_skeleton_hess(mo_coeff.view(), mo_occ.view(), None);
    let sel = hess.make_skeleton_hess(mo_coeff.view(), mo_occ.view(), Some(ATM_LIST));
    let expected = select_atoms(full.view(), ATM_LIST);
    assert!(rt::allclose(sel.view(), expected.view(), (1e-10, 1e-12)));
}

#[rstest]
fn test_atm_list_ovlp(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, mo_coeff, mo_occ, mo_energy, .. } = hess_case;
    let dme0 = get_dme0_restricted(mo_coeff.view(), mo_occ.view(), mo_energy.view());
    let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
    let full = ovlp_obj.make_hess(dme0.view(), None);
    let sel = ovlp_obj.make_hess(dme0.view(), Some(ATM_LIST));
    let expected = select_atoms(full.view(), ATM_LIST);
    assert!(rt::allclose(sel.view(), expected.view(), (1e-10, 1e-12)));
}

#[rstest]
fn test_atm_list_rijk_skeleton(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;

    let full_j = get_decomposed_rij_skeleton_deriv2_naive(mol, aux, mo_coeff.view(), mo_occ.view(), None);
    let sel_j = get_decomposed_rij_skeleton_deriv2_naive(mol, aux, mo_coeff.view(), mo_occ.view(), Some(ATM_LIST));
    for (key, full_val) in &full_j {
        let expected = select_atoms(full_val.view(), ATM_LIST);
        assert!(
            rt::allclose(sel_j[key].view(), expected.view(), (1e-10, 1e-12)),
            "RI-J skeleton mismatch on key {key}"
        );
    }

    let full_k = get_decomposed_rik_skeleton_deriv2_naive(mol, aux, mo_coeff.view(), mo_occ.view(), None);
    let sel_k = get_decomposed_rik_skeleton_deriv2_naive(mol, aux, mo_coeff.view(), mo_occ.view(), Some(ATM_LIST));
    for (key, full_val) in &full_k {
        let expected = select_atoms(full_val.view(), ATM_LIST);
        assert!(
            rt::allclose(sel_k[key].view(), expected.view(), (1e-10, 1e-12)),
            "RI-K skeleton mismatch on key {key}"
        );
    }
}

#[rstest]
fn test_atm_list_rijk_deriv1_ao(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;

    let full_j = get_rij_deriv1_ao_naive(mol, aux, mo_coeff.view(), mo_occ.view(), None);
    let sel_j = get_rij_deriv1_ao_naive(mol, aux, mo_coeff.view(), mo_occ.view(), Some(ATM_LIST));
    for (key, full_val) in &full_j {
        // Output shape is [nao, nao, 3, natm]; select last axis on atm_list.
        let expected = full_val.index_select(-1, ATM_LIST);
        assert!(
            rt::allclose(sel_j[key].view(), expected.view(), (1e-10, 1e-12)),
            "RI-J deriv1_ao mismatch on key {key}"
        );
    }

    let full_k = get_rik_deriv1_ao_naive(mol, aux, mo_coeff.view(), mo_occ.view(), None);
    let sel_k = get_rik_deriv1_ao_naive(mol, aux, mo_coeff.view(), mo_occ.view(), Some(ATM_LIST));
    for (key, full_val) in &full_k {
        let expected = full_val.index_select(-1, ATM_LIST);
        assert!(
            rt::allclose(sel_k[key].view(), expected.view(), (1e-10, 1e-12)),
            "RI-K deriv1_ao mismatch on key {key}"
        );
    }
}

#[rstest]
fn test_atm_list_make_hess(hess_case: &CaseAmoniaRHF) {
    let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, mo_energy, .. } = hess_case;

    let mo_coeff_c = mo_coeff.view().into_contig(ColMajor);
    let mo_occ_c = mo_occ.view().into_contig(ColMajor);
    let mo_energy_c = mo_energy.view().into_contig(ColMajor);

    // full hessian
    let full = {
        let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
        let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
        let mut hcore_obj = RHessHcore::new(mol, &DeviceTsr::default());
        let mut rijk_obj = RHessRIJKNaive::new(mol, aux, 1.0, 1.0);
        let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
        let core_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        let config = HessSCFConfig::default();
        let mut hess_scf = RHessSCF::new(
            mo_coeff_c.clone(),
            mo_occ_c.clone(),
            mo_energy_c.clone(),
            ovlp_obj,
            nuc_list,
            core_list,
            el_list,
            config,
            None,
        );
        hess_scf.make_hess()
    };

    // selected-atom hessian
    let sel = {
        let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
        let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
        let mut hcore_obj = RHessHcore::new(mol, &DeviceTsr::default());
        let mut rijk_obj = RHessRIJKNaive::new(mol, aux, 1.0, 1.0);
        let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
        let core_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        let config = HessSCFConfig::default();
        let mut hess_scf = RHessSCF::new(
            mo_coeff_c.clone(),
            mo_occ_c.clone(),
            mo_energy_c.clone(),
            ovlp_obj,
            nuc_list,
            core_list,
            el_list,
            config,
            Some(ATM_LIST),
        );
        assert_eq!(hess_scf.natm(), ATM_LIST.len());
        hess_scf.make_hess()
    };

    // CPHF iteration only converges to ~1e-8; the full vs. selected solves use different
    // Krylov subspaces, so per-element differences of that magnitude are expected.
    let expected = select_atoms(full.view(), ATM_LIST);
    assert!(rt::allclose(sel.view(), expected.view(), (1e-6, 1e-7)));
}
