use crate::hessian_uhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::prelude::*;

#[rstest]
fn test_dimensionless_cphf_rhs(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, mo_energy, .. } = hess_case;

    let ovlp_obj = UHessOvlp::new(mol, &DeviceTsr::default());
    let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
    let mut hcore_obj = UHessHcore::new(mol, &DeviceTsr::default());
    let mut rijk_obj = UHessRIJKNaive::new(mol, aux, 1.0, 1.0);
    let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
    let hcore_list: Vec<&mut dyn UHessCoreAPI> = vec![&mut hcore_obj];
    let el_list: Vec<&mut dyn UHessElecInteractAPI> = vec![&mut rijk_obj];
    let config = HessSCFConfig::default();
    let mut hess_scf = UHessSCF::new(
        mo_coeff.clone(),
        mo_occ.clone(),
        mo_energy.clone(),
        ovlp_obj,
        nuc_list,
        hcore_list,
        el_list,
        config,
        None,
    );

    // before krylov, first obtain dimensionless rhs part
    let pre_cphf_dict = hess_scf.compute_dimless_cphf_rhs();
    assert_abs_diff_eq!(fp(pre_cphf_dict["rhs_0"].swapaxes(0, 1)), -0.01785256539468953, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(pre_cphf_dict["rhs_1"].swapaxes(0, 1)), 0.14550989432158085, epsilon = 1e-5);

    // solve cphf
    let rhs = [pre_cphf_dict["rhs_0"].view(), pre_cphf_dict["rhs_1"].view()];
    hess_scf.make_response_preparation();
    let mo1 = hess_scf.solve_dimless_cphf(&rhs);

    let ref_mo1_0 = hess_case.ref_dict["mo1_a"].transpose((2, 3, 1, 0));
    let ref_mo1_1 = hess_case.ref_dict["mo1_b"].transpose((2, 3, 1, 0));
    assert!(rt::allclose(mo1[0].view(), ref_mo1_0.view(), (1e-3, 1e-4)));
    assert!(rt::allclose(mo1[1].view(), ref_mo1_1.view(), (1e-3, 1e-4)));
    assert_abs_diff_eq!(fp(mo1[0].swapaxes(0, 1)), 0.04797427280601669, epsilon = 1e-4);
    assert_abs_diff_eq!(fp(mo1[1].swapaxes(0, 1)), -1.1346573239117455, epsilon = 1e-4);

    // finalize cphf
    let f1mo = [pre_cphf_dict["f1mo_0"].view(), pre_cphf_dict["f1mo_1"].view()];
    let s1mo = [pre_cphf_dict["s1mo_0"].view(), pre_cphf_dict["s1mo_1"].view()];
    let mo1 = [mo1[0].view(), mo1[1].view()];
    let result_cphf = hess_scf.finalize_cphf(&f1mo, &s1mo, &mo1);
    let ref_mo_e1_0 = hess_case.ref_dict["mo_e1_a"].transpose([2, 3, 1, 0]);
    let ref_mo_e1_1 = hess_case.ref_dict["mo_e1_b"].transpose([2, 3, 1, 0]);
    let mo1_fin_0 = result_cphf["mo1_0"].view();
    let mo1_fin_1 = result_cphf["mo1_1"].view();
    let mo_e1_0 = result_cphf["mo_e1_0"].view();
    let mo_e1_1 = result_cphf["mo_e1_1"].view();

    assert!(rt::allclose(mo1_fin_0.view(), ref_mo1_0.view(), (1e-4, 1e-5)));
    assert!(rt::allclose(mo1_fin_1.view(), ref_mo1_1.view(), (1e-4, 1e-5)));
    assert_abs_diff_eq!(fp(mo1_fin_0.swapaxes(0, 1)), 0.04797427280601669, epsilon = 1e-5);
    assert_abs_diff_eq!(fp(mo1_fin_1.swapaxes(0, 1)), -1.1346573239117455, epsilon = 1e-5);
    assert!(rt::allclose(mo_e1_0.view(), ref_mo_e1_0.view(), (1e-3, 1e-4)));
    assert!(rt::allclose(mo_e1_1.view(), ref_mo_e1_1.view(), (1e-3, 1e-4)));
    assert_abs_diff_eq!(fp(mo_e1_0.swapaxes(0, 1)), -1.1979763394388616, epsilon = 1e-4);
    assert_abs_diff_eq!(fp(mo_e1_1.swapaxes(0, 1)), -0.20920766550023265, epsilon = 1e-4);

    // compute de_cphf
    let mo1 = [mo1_fin_0.view(), mo1_fin_1.view()];
    let mo_e1 = [mo_e1_0.view(), mo_e1_1.view()];
    let de_cphf = hess_scf.get_cphf_hess(&f1mo, &s1mo, &mo1, &mo_e1);
    let ref_de_cphf = hess_case.ref_dict["de_cphf"].transpose([2, 3, 0, 1]);
    assert!(rt::allclose(de_cphf.view(), ref_de_cphf.view(), (1e-4, 1e-6)));
    assert_abs_diff_eq!(fp(de_cphf.view()), -0.40949468934990596, epsilon = 1e-5);
}

#[rstest]
fn test_make_hess(hess_case: &CaseAmoniaUHF) {
    let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, mo_energy, ref_dict } = hess_case;

    let ovlp_obj = UHessOvlp::new(mol, &DeviceTsr::default());
    let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
    let mut hcore_obj = UHessHcore::new(mol, &DeviceTsr::default());
    let mut rijk_obj = UHessRIJKNaive::new(mol, aux, 1.0, 1.0);
    let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
    let hcore_list: Vec<&mut dyn UHessCoreAPI> = vec![&mut hcore_obj];
    let el_list: Vec<&mut dyn UHessElecInteractAPI> = vec![&mut rijk_obj];
    let config = HessSCFConfig::default();
    let mut hess_scf = UHessSCF::new(
        mo_coeff.clone(),
        mo_occ.clone(),
        mo_energy.clone(),
        ovlp_obj,
        nuc_list,
        hcore_list,
        el_list,
        config,
        None,
    );

    let de_hess = hess_scf.make_hess();
    let de_hess_ref = ref_dict["de_ref"].transpose([2, 3, 0, 1]);
    assert!(rt::allclose(de_hess.view(), de_hess_ref.view(), (1e-4, 1e-5)));
    assert_abs_diff_eq!(fp(de_hess.view()), 0.6241806384454698, epsilon = 1e-4);

    println!("Result keys of hessian object: {:?}", hess_scf.result.keys());
    println!("Timing of hessian");
    for (key, value) in hess_scf.timing.iter() {
        println!("    {:60}: {:10.6} seconds", key, value);
    }
}

#[cfg(test)]
mod test_uhf_optimized {
    use super::*;
    use rstsr_showcase_hessian::prelude_dev::Tsr;
    #[rstest]
    fn test_response_bra_compare(hess_case: &CaseAmoniaUHF) {
        use rstsr_showcase_hessian::ri_jk::hess_r::get_rijk_response_bra_separated;
        use rstsr_showcase_hessian::ri_jk::hess_u_naive::get_uijk_response_bra_naive;

        let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;
        let device = DeviceTsr::default();
        let nao = mol.nao();
        let occidx = [mo_occ[0].view().greater(0.0).into_vec(), mo_occ[1].view().greater(0.0).into_vec()];
        let nocc = [occidx[0].iter().filter(|&&x| x).count(), occidx[1].iter().filter(|&&x| x).count()];

        // build a deterministic bra per spin [nao, nocc_s, 3, natm] from arange
        let natm = mol.natm();
        let mk_bra = |nocc_s: usize| -> Tsr {
            let n = nao * nocc_s * 3 * natm;
            let raw = rt::arange((0.0, n as f64, &device)).into_shape([nao, nocc_s, 3, natm]);
            &raw * 0.01
        };
        let bra: [Tsr; 2] = [mk_bra(nocc[0]), mk_bra(nocc[1])];
        let bra_v = [bra[0].view(), bra[1].view()];
        let mo_coeff_v = [mo_coeff[0].view(), mo_coeff[1].view()];
        let mo_occ_v = [mo_occ[0].view(), mo_occ[1].view()];

        // naive reference (full)
        let ref_resp = get_uijk_response_bra_naive(mol, aux, &mo_coeff_v, &mo_occ_v, &bra_v, 1.0, 1.0);

        // optimized separated core (reuse cderi from a fresh UHessRIJK)
        let rijk_obj = UHessRIJK::new_without_cderi(mol, aux, 1.0, 1.0);
        let cderi = rijk_obj.cderi.view();
        let mocc = [
            mo_coeff[0].view().bool_select(-1, &occidx[0]).into_contig(ColMajor),
            mo_coeff[1].view().bool_select(-1, &occidx[1]).into_contig(ColMajor),
        ];

        let assemble = |do_j: bool, do_k: bool| -> [Tsr; 2] {
            let (j_ao, k_bras) =
                get_rijk_response_bra_separated(cderi.view(), &mo_coeff_v, &mo_occ_v, &bra_v, do_j, do_k, 72);
            let mut out: [Option<Tsr>; 2] = [None, None];
            for s in 0..2 {
                let shape = bra[s].shape().to_vec();
                let nprop: usize = shape[2..].iter().product();
                let mut r = rt::zeros(([nao, nocc[s], nprop], &device));
                // J: shared AO operator; UHF carries an extra 0.5 prefactor (occ = 1 vs RHF occ = 2).
                if let Some(j_ao) = j_ao.as_ref() {
                    r += 0.5 * (j_ao.view() % &mocc[s]);
                }
                // K: same-spin bra form; core already bakes in the exchange sign.
                if let Some(k_bra) = k_bras.get(s) {
                    r += k_bra.view().reshape((nao, nocc[s], nprop));
                }
                out[s] = Some(r.into_shape(shape));
            }
            [out[0].take().unwrap(), out[1].take().unwrap()]
        };

        // Compare J-only, K-only, and full response against the naive UHF reference.
        assert!(rt::allclose(
            assemble(true, false)[0].view(),
            get_uijk_response_bra_naive(mol, aux, &mo_coeff_v, &mo_occ_v, &bra_v, 1.0, 0.0)[0].view(),
            (1e-6, 1e-8)
        ));
        assert!(rt::allclose(
            assemble(false, true)[1].view(),
            get_uijk_response_bra_naive(mol, aux, &mo_coeff_v, &mo_occ_v, &bra_v, 0.0, 1.0)[1].view(),
            (1e-6, 1e-8)
        ));
        let opt_full = assemble(true, true);
        assert!(rt::allclose(opt_full[0].view(), ref_resp[0].view(), (1e-6, 1e-8)));
        assert!(rt::allclose(opt_full[1].view(), ref_resp[1].view(), (1e-6, 1e-8)));
    }

    #[rstest]
    fn test_make_hess_faster(hess_case: &CaseAmoniaUHF) {
        let CaseAmoniaUHF { mol, aux, mo_coeff, mo_occ, mo_energy, ref_dict } = hess_case;

        let ovlp_obj = UHessOvlp::new(mol, &DeviceTsr::default());
        let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
        let mut hcore_obj = UHessHcore::new(mol, &DeviceTsr::default());
        let mut rijk_obj = UHessRIJK::new_without_cderi(mol, aux, 1.0, 1.0);
        let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
        let hcore_list: Vec<&mut dyn UHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn UHessElecInteractAPI> = vec![&mut rijk_obj];
        let config = HessSCFConfig::default();
        let mut hess_scf = UHessSCF::new(
            mo_coeff.clone(),
            mo_occ.clone(),
            mo_energy.clone(),
            ovlp_obj,
            nuc_list,
            hcore_list,
            el_list,
            config,
            None,
        );

        let de_hess = hess_scf.make_hess();
        let de_hess_ref = ref_dict["de_ref"].transpose([2, 3, 0, 1]);
        // print timing
        println!("Hessian computation timing:");
        for (key, value) in hess_scf.timing.iter() {
            println!("    {:60}: {:10.6} seconds", key, value);
        }
        println!("max deviation of Hessian: {:16.10e}", (de_hess.view() - de_hess_ref.view()).abs().max());
        assert!(rt::allclose(de_hess.view(), de_hess_ref.view(), (1e-4, 1e-5)));
        assert_abs_diff_eq!(fp(de_hess.view()), 0.6241806384454698, epsilon = 1e-4);
    }
}
