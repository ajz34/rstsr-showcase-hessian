use crate::hessian_rks_becke::*;
use crate::test_util::*;
use libxc::prelude::*;
use rstest::rstest;
use rstsr_showcase_hessian::numint_matmul::hess_rks::make_hessian_setup_batched;
use rstsr_showcase_hessian::numint_matmul::hess_rks_becke::RHessKSNIMatmulBecke;
use rstsr_showcase_hessian::prelude::*;
use rstsr_showcase_hessian::util::density_matrices::get_dm0_restricted;

#[rstest]
fn test_setup(hess_case_svwn: &CaseAmoniaRKSBecke) {
    let CaseAmoniaRKSBecke {
        mol,
        mo_coeff,
        mo_occ,
        grid_coords,
        grid_weights,
        quadrature_weights,
        atm_quad_split,
        adjustment_factor,
        ref_dict,
        ..
    } = hess_case_svwn.clone();

    let xc_func_list = [
        (1.0, LibXCFunctional::from_identifier("LDA_X", LibXCSpin::Unpolarized)),
        (1.0, LibXCFunctional::from_identifier("LDA_C_VWN", LibXCSpin::Unpolarized)),
    ];
    let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());

    let ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let mut becke_obj = RHessKSNIMatmulBecke::new(
        &mol,
        &xc_func_list,
        ni,
        quadrature_weights,
        atm_quad_split,
        adjustment_factor,
        3,
        16384,
        false,
    );
    becke_obj.make_hessian_setup(mo_coeff.view(), mo_occ.view(), None);

    // (1) without-becke parts vs the non-becke path on the same grids (duplication check)
    let mut ni_ref = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let (result_ref, _) = make_hessian_setup_batched(&mol, &xc_func_list, &mut ni_ref, dm0.view(), None, false);
    // (vmat_fxc/vmat_vxc are not returned by the non-becke driver; they are covered by their sum
    // vmat_deriv1)
    for key in ["de_fxc", "de_vxc_diag", "de_vxc_off", "vmat_ip", "vmat_deriv1"] {
        let diff = &becke_obj.intmd[key] - &result_ref[key];
        let maxdiff = diff.abs().max();
        assert!(maxdiff < 1e-8, "key {key}: max diff vs non-becke path = {maxdiff}");
    }

    // (2) without-becke parts vs the python reference (grid-order independent sums)
    for key in ["de_fxc", "de_vxc_diag", "de_vxc_off", "vmat_deriv1"] {
        let diff = &becke_obj.intmd[key] - &ref_dict[key].t();
        let maxdiff = diff.abs().max();
        assert!(maxdiff < 1e-4, "key {key}: max diff vs python ref = {maxdiff}");
    }
    {
        let diff = &becke_obj.intmd["vmat_ip"] - &ref_dict["vmat_ip"].transpose([1, 2, 0]);
        let maxdiff = diff.abs().max();
        assert!(maxdiff < 1e-4, "key vmat_ip: max diff vs python ref = {maxdiff}");
    }

    // (3) translational invariance of the grid-shifted quantities (< 1e-9)
    let inv1 = becke_obj.intmd["de_xc_skeleton"].sum_axes([-1, -2]);
    let m1 = inv1.abs().max();
    println!("de_xc_skeleton invariance max: {m1:.3e}");
    assert!(m1 < 1e-9, "de_xc_skeleton translational invariance = {m1}");

    let inv2 = becke_obj.intmd["vmat_deriv1_grid"].sum_axes(-1);
    let m2 = inv2.abs().max();
    println!("vmat_deriv1_grid invariance max: {m2:.3e}");
    assert!(m2 < 1e-9, "vmat_deriv1_grid translational invariance = {m2}");

    for key in [
        "de_becke_full_1",
        "de_becke_full_2",
        "de_becke_atom_1",
        "de_becke_atom_2",
        "de_becke_atom_3",
        "de_becke_vxc_diag",
        "de_becke_vxc_off",
        "vmat_becke_T1",
        "vmat_becke_T2_ipip",
        "vmat_becke_T2_fxc",
    ] {
        println!("fp {key}: {}", fp(becke_obj.intmd[key].view()));
    }
}

#[rstest]
fn test_make_hess(hess_case_svwn: &CaseAmoniaRKSBecke) {
    let CaseAmoniaRKSBecke {
        mol,
        aux,
        mo_coeff,
        mo_occ,
        mo_energy,
        grid_coords,
        grid_weights,
        quadrature_weights,
        atm_quad_split,
        adjustment_factor,
        ref_dict,
        ..
    } = hess_case_svwn.clone();

    let mo_coeff = mo_coeff.view().into_contig(ColMajor);
    let mo_occ = mo_occ.view().into_contig(ColMajor);
    let mo_energy = mo_energy.view().into_contig(ColMajor);
    let ovlp_obj = RHessOvlp::new(&mol, &DeviceTsr::default());
    let mut nuc_repl_obj = HessNucRepl::new(&mol, &DeviceTsr::default());
    let mut hcore_obj = RHessHcore::new(&mol, &DeviceTsr::default());
    // SVWN is pure DFT, no exact exchange contribution
    let mut rijk_obj = RHessRIJK::new_without_cderi(&mol, &aux, 1.0, 0.0);

    let ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let xc_func_list = [
        (1.0, LibXCFunctional::from_identifier("LDA_X", LibXCSpin::Unpolarized)),
        (1.0, LibXCFunctional::from_identifier("LDA_C_VWN", LibXCSpin::Unpolarized)),
    ];
    let mut nimatmul_becke_obj = RHessKSNIMatmulBecke::new(
        &mol,
        &xc_func_list,
        ni,
        quadrature_weights,
        atm_quad_split,
        adjustment_factor,
        3,
        16384,
        true,
    );

    let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
    let hcore_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
    let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj, &mut nimatmul_becke_obj];
    let config = HessSCFConfig::default();
    let mut hess_scf =
        RHessSCF::new(mo_coeff, mo_occ, mo_energy, ovlp_obj, nuc_list, hcore_list, el_list, config, None);

    let de_hess = hess_scf.make_hess();

    // vs the (grid-fixed) PySCF reference: grid-shift magnitude for LDA
    // (element-loop comparison; the de_ref layout is python [A, B, t, s])
    let de_ref_4d = ref_dict["de_ref"].to_owned();
    let maxdiff = (de_hess.t() - &de_ref_4d).abs().max();
    println!("max|de_hess - de_ref| (grid-shift corrected vs PySCF ref): {maxdiff}");
    assert!(maxdiff < 1e-3, "max abs diff = {maxdiff}");

    // full Hessian translational invariance (< 1e-7)
    let inv = de_hess.sum_axes([-1, -2]);
    let m = inv.abs().max();
    println!("full Hessian translational invariance max: {m:.3e}");
    assert!(m < 1e-7, "translational invariance = {m}");
}
