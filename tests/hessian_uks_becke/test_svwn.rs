use crate::hessian_uks_becke::*;
use crate::test_util::*;
use libxc::prelude::*;
use rstest::rstest;
use rstsr_showcase_hessian::numint_matmul::hess_uks::make_hessian_setup_batched_uks;
use rstsr_showcase_hessian::numint_matmul::hess_uks_becke::UHessKSNIMatmulBecke;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_setup(hess_case_svwn: &CaseAmoniaUKSBecke) {
    let CaseAmoniaUKSBecke {
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
        (1.0, LibXCFunctional::from_identifier("LDA_X", LibXCSpin::Polarized)),
        (1.0, LibXCFunctional::from_identifier("LDA_C_VWN", LibXCSpin::Polarized)),
    ];

    let occidx_a = mo_occ[0].view().greater(0).into_vec();
    let occidx_b = mo_occ[1].view().greater(0).into_vec();
    let mocc_a = mo_coeff[0].bool_select(-1, &occidx_a);
    let mocc_b = mo_coeff[1].bool_select(-1, &occidx_b);
    let dm0a = &mocc_a % mocc_a.t();
    let dm0b = &mocc_b % mocc_b.t();

    let ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let mut becke_obj = UHessKSNIMatmulBecke::new(
        &mol,
        &xc_func_list,
        ni,
        quadrature_weights,
        atm_quad_split,
        adjustment_factor,
        3,
        false,
    );
    becke_obj.make_hessian_setup(
        &[mo_coeff[0].view(), mo_coeff[1].view()],
        &[mo_occ[0].view(), mo_occ[1].view()],
        None,
    );

    // (1) without-becke parts vs the non-becke path on the same grids (duplication check)
    let mut ni_ref = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let (result_ref, _) =
        make_hessian_setup_batched_uks(&mol, &xc_func_list, &mut ni_ref, dm0a.view(), dm0b.view(), None, false);
    // (vmat_fxc_a/vmat_vxc_a etc. are not returned by the non-becke driver; they are covered by
    // their sum vmat_deriv1)
    for key in [
        "de_fxc",
        "de_vxc_diag_a",
        "de_vxc_diag_b",
        "de_vxc_off_a",
        "de_vxc_off_b",
        "vmat_ip_a",
        "vmat_ip_b",
        "vmat_deriv1_a",
        "vmat_deriv1_b",
    ] {
        let diff = &becke_obj.intmd[key] - &result_ref[key];
        let maxdiff = diff.abs().max();
        assert!(maxdiff < 1e-8, "key {key}: max diff vs non-becke path = {maxdiff}");
    }

    // (2) without-becke parts vs the python reference (grid-order independent sums)
    for key in
        ["de_fxc", "de_vxc_diag_a", "de_vxc_diag_b", "de_vxc_off_a", "de_vxc_off_b", "vmat_deriv1_a", "vmat_deriv1_b"]
    {
        let diff = &becke_obj.intmd[key] - &ref_dict[key].t();
        let maxdiff = diff.abs().max();
        assert!(maxdiff < 1e-4, "key {key}: max diff vs python ref = {maxdiff}");
    }
    for key in ["vmat_ip_a", "vmat_ip_b"] {
        let diff = &becke_obj.intmd[key] - &ref_dict[key].transpose([1, 2, 0]);
        let maxdiff = diff.abs().max();
        assert!(maxdiff < 1e-4, "key {key}: max diff vs python ref = {maxdiff}");
    }

    // (3) translational invariance of the grid-shifted quantities (< 1e-9)
    let inv1 = becke_obj.intmd["de_xc_skeleton"].sum_axes([-1, -2]);
    let m1 = inv1.abs().max();
    println!("de_xc_skeleton invariance max: {m1:.3e}");
    assert!(m1 < 1e-9, "de_xc_skeleton translational invariance = {m1}");

    for key in ["vmat_deriv1_grid_a", "vmat_deriv1_grid_b"] {
        let inv = becke_obj.intmd[key].sum_axes(-1);
        let m = inv.abs().max();
        println!("{key} invariance max: {m:.3e}");
        assert!(m < 1e-9, "{key} translational invariance = {m}");
    }

    for key in [
        "de_becke_full_1",
        "de_becke_full_2",
        "de_becke_atom_1",
        "de_becke_atom_2",
        "de_becke_atom_3",
        "de_becke_vxc_diag",
        "de_becke_vxc_off",
        "vmat_becke_dw_a",
        "vmat_becke_dw_b",
        "vmat_becke_vxc_a",
        "vmat_becke_vxc_b",
        "vmat_becke_fxc_a",
        "vmat_becke_fxc_b",
    ] {
        println!("fp {key}: {}", fp(becke_obj.intmd[key].view()));
    }
}

#[rstest]
fn test_make_hess(hess_case_svwn: &CaseAmoniaUKSBecke) {
    let CaseAmoniaUKSBecke {
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

    let mo_coeff = [mo_coeff[0].view().into_contig(ColMajor), mo_coeff[1].view().into_contig(ColMajor)];
    let mo_occ = [mo_occ[0].view().into_contig(ColMajor), mo_occ[1].view().into_contig(ColMajor)];
    let mo_energy = [mo_energy[0].view().into_contig(ColMajor), mo_energy[1].view().into_contig(ColMajor)];
    let ovlp_obj = UHessOvlp::new(&mol, &DeviceTsr::default());
    let mut nuc_repl_obj = HessNucRepl::new(&mol, &DeviceTsr::default());
    let mut hcore_obj = UHessHcore::new(&mol, &DeviceTsr::default());
    // 0.1*HF + SVWN, hybrid coefficient = 0.1 (baked into the reference solution)
    let mut rijk_obj = UHessRIJK::new_without_cderi(&mol, &aux, 1.0, 0.1);

    let ni = NIMatmul::new(&mol, &grid_coords, &grid_weights);
    let xc_func_list = [
        (1.0, LibXCFunctional::from_identifier("LDA_X", LibXCSpin::Polarized)),
        (1.0, LibXCFunctional::from_identifier("LDA_C_VWN", LibXCSpin::Polarized)),
    ];
    let mut nimatmul_becke_obj = UHessKSNIMatmulBecke::new(
        &mol,
        &xc_func_list,
        ni,
        quadrature_weights,
        atm_quad_split,
        adjustment_factor,
        3,
        true,
    );

    let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
    let core_list: Vec<&mut dyn UHessCoreAPI> = vec![&mut hcore_obj];
    let el_list: Vec<&mut dyn UHessElecInteractAPI> = vec![&mut rijk_obj, &mut nimatmul_becke_obj];
    let config = HessSCFConfig::default();
    let mut hess_scf = UHessSCF::new(
        mo_coeff.map(|c| c.to_owned()),
        mo_occ.map(|c| c.to_owned()),
        mo_energy.map(|c| c.to_owned()),
        ovlp_obj,
        nuc_list,
        core_list,
        el_list,
        config,
        None,
    );

    let de_hess = hess_scf.make_hess();

    // vs the (grid-fixed) PySCF reference: grid-shift magnitude for LDA
    // (element-loop comparison; the de_ref layout is python [A, B, t, s])
    let de_ref_4d = ref_dict["de_ref"].transpose([2, 3, 0, 1]);
    let maxdiff = (de_hess.view() - &de_ref_4d).abs().max();
    println!("max|de_hess - de_ref| (grid-shift corrected vs PySCF ref): {maxdiff}");
    assert!(maxdiff < 1e-3, "max abs diff = {maxdiff}");

    // full Hessian translational invariance (< 1e-6).  The XC skeleton itself is invariant to
    // ~1e-13; the residual is set by the remaining (grid-free) machinery — the pyhessref
    // reference shows the same behaviour at 6.0e-8 for tpss0.
    let inv = de_hess.sum_axes([-1, -2]);
    let m = inv.abs().max();
    println!("full Hessian translational invariance max: {m:.3e}");
    assert!(m < 1e-6, "translational invariance = {m}");
}
