use crate::hessian_rhf::*;
use crate::test_util::*;
use approx::assert_abs_diff_eq;
use rstest::rstest;
use rstsr_showcase_hessian::prelude::*;

#[cfg(test)]
mod test_rhf_naive {
    use super::*;

    #[rstest]
    fn test_f1ao(hess_case: &CaseAmoniaRHF) {
        let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;

        let hess_hcore_obj = RHessHcore::new(mol, &DeviceTsr::default());
        let mut hess_rijk_obj = RHessRIJKNaive::new(mol, aux, 1.0, 1.0);

        let natm = mol.natm();
        let gen_h1ao = hess_hcore_obj.generator_deriv1();
        let h1ao_list = (0..natm).map(gen_h1ao).collect_vec();
        let h1ao = rt::stack((h1ao_list, -1));
        let jk1ao = hess_rijk_obj.get_deriv1_ao(mo_coeff.view(), mo_occ.view(), None);
        let f1ao = &h1ao + &jk1ao;
        assert_abs_diff_eq!(fp(f1ao.view().swapaxes(0, 1)), 0.03306328817997084, epsilon = 1e-6);
    }

    #[rstest]
    fn test_dimensionless_cphf_rhs(hess_case: &CaseAmoniaRHF) {
        let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, mo_energy, ref_dict } = hess_case;

        let mo_coeff = mo_coeff.view().into_contig(ColMajor);
        let mo_occ = mo_occ.view().into_contig(ColMajor);
        let mo_energy = mo_energy.view().into_contig(ColMajor);
        let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
        let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
        let mut hcore_obj = RHessHcore::new(mol, &DeviceTsr::default());
        let mut rijk_obj = RHessRIJKNaive::new(mol, aux, 1.0, 1.0);
        let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
        let hcore_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        let config = HessSCFConfig::default();
        let mut hess_scf =
            RHessSCF::new(mo_coeff, mo_occ, mo_energy, ovlp_obj, nuc_list, hcore_list, el_list, config, None);

        // before krylov, first obtain dimensionless rhs part
        let pre_cphf_dict = hess_scf.compute_dimless_cphf_rhs();
        assert_abs_diff_eq!(fp(pre_cphf_dict["rhs"].swapaxes(0, 1)), -0.027755691019085788, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(pre_cphf_dict["f1mo"].swapaxes(0, 1)), 9.624352641672411, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(pre_cphf_dict["s1mo"].swapaxes(0, 1)), -3.0146480401818847, epsilon = 1e-6);

        // solve cphf
        let rhs = pre_cphf_dict["rhs"].view();
        hess_scf.make_response_preparation();
        let mo1 = hess_scf.solve_dimless_cphf(rhs);
        let ref_mo1 = ref_dict["mo1"].transpose((2, 3, 1, 0));
        assert!(rt::allclose(mo1.view(), ref_mo1.view(), (1e-4, 1e-6)));
        assert_abs_diff_eq!(fp(mo1.swapaxes(0, 1)), -0.02385155247256418, epsilon = 1e-6);

        // finalize cphf
        let f1mo = pre_cphf_dict["f1mo"].view();
        let s1mo = pre_cphf_dict["s1mo"].view();
        let result_cphf = hess_scf.finalize_cphf(f1mo.view(), s1mo.view(), mo1.view());
        let ref_mo_e1 = ref_dict["mo_e1"].transpose([2, 3, 1, 0]);
        let mo1 = result_cphf["mo1"].view();
        let mo_e1 = result_cphf["mo_e1"].view();
        assert!(rt::allclose(mo1.view(), ref_mo1.view(), (1e-4, 1e-6)));
        assert!(rt::allclose(mo_e1.view(), ref_mo_e1.view(), (1e-4, 1e-6)));
        assert_abs_diff_eq!(fp(mo1.swapaxes(0, 1)), -0.02385155247256418, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(mo_e1.swapaxes(0, 1)), 0.2961618130386303, epsilon = 1e-6);

        // compute de_cphf
        let de_cphf = hess_scf.get_cphf_hess(f1mo.view(), s1mo.view(), mo1.view(), mo_e1.view());
        let ref_de_cphf = ref_dict["de_cphf"].transpose([2, 3, 0, 1]);
        assert!(rt::allclose(de_cphf.view(), ref_de_cphf.view(), (1e-4, 1e-6)));
        assert_abs_diff_eq!(fp(de_cphf.view()), 1.0888788930763051, epsilon = 1e-6);
    }

    #[rstest]
    fn test_make_hess(hess_case: &CaseAmoniaRHF) {
        let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, mo_energy, ref_dict } = hess_case;

        let mo_coeff = mo_coeff.view().into_contig(ColMajor);
        let mo_occ = mo_occ.view().into_contig(ColMajor);
        let mo_energy = mo_energy.view().into_contig(ColMajor);
        let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
        let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
        let mut hcore_obj = RHessHcore::new(mol, &DeviceTsr::default());
        let mut rijk_obj = RHessRIJKNaive::new(mol, aux, 1.0, 1.0);
        let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
        let hcore_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        let config = HessSCFConfig::default();
        let mut hess_scf =
            RHessSCF::new(mo_coeff, mo_occ, mo_energy, ovlp_obj, nuc_list, hcore_list, el_list, config, None);

        let de_hess = hess_scf.make_hess();
        let de_hess_ref = ref_dict["de_ref"].transpose([2, 3, 0, 1]);
        assert!(rt::allclose(de_hess.view(), de_hess_ref.view(), (1e-4, 1e-6)));
        assert_abs_diff_eq!(fp(de_hess.view()), 1.4704252379360374, epsilon = 1e-5);

        // print timing
        println!("Hessian computation timing:");
        for (key, value) in hess_scf.timing.iter() {
            println!("    {:60}: {:10.6} seconds", key, value);
        }
    }
}

#[cfg(test)]
mod test_rhf_optimized {
    use super::*;
    #[rstest]
    fn test_make_hess_faster(hess_case: &CaseAmoniaRHF) {
        let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, mo_energy, ref_dict } = hess_case;

        let mo_coeff = mo_coeff.view().into_contig(ColMajor);
        let mo_occ = mo_occ.view().into_contig(ColMajor);
        let mo_energy = mo_energy.view().into_contig(ColMajor);
        let ovlp_obj = RHessOvlp::new(mol, &DeviceTsr::default());
        let mut nuc_repl_obj = HessNucRepl::new(mol, &DeviceTsr::default());
        let mut hcore_obj = RHessHcore::new(mol, &DeviceTsr::default());
        let mut rijk_obj = RHessRIJK::new_without_cderi(mol, aux, 1.0, 1.0);
        let nuc_list: Vec<&mut dyn HessNucAPI> = vec![&mut nuc_repl_obj];
        let hcore_list: Vec<&mut dyn RHessCoreAPI> = vec![&mut hcore_obj];
        let el_list: Vec<&mut dyn RHessElecInteractAPI> = vec![&mut rijk_obj];
        let config = HessSCFConfig::default();
        let mut hess_scf =
            RHessSCF::new(mo_coeff, mo_occ, mo_energy, ovlp_obj, nuc_list, hcore_list, el_list, config, None);

        let de_hess = hess_scf.make_hess();
        let de_hess_ref = ref_dict["de_ref"].transpose([2, 3, 0, 1]);
        // print timing
        println!("Hessian computation timing:");
        for (key, value) in hess_scf.timing.iter() {
            println!("    {:60}: {:10.6} seconds", key, value);
        }
        println!("max deviation of Hessian: {:16.10e}", (de_hess.view() - de_hess_ref.view()).abs().max());
        assert!(rt::allclose(de_hess.view(), de_hess_ref.view(), (1e-4, 1e-6)));
        assert_abs_diff_eq!(fp(de_hess.view()), 1.4704252379360374, epsilon = 1e-5);
    }

    #[rstest]
    fn print_prepare(hess_case: &CaseAmoniaRHF) {
        use rstsr_showcase_hessian::ri_jk::decompose::*;
        use rstsr_showcase_hessian::ri_jk::hess_r::*;
        use rstsr_showcase_hessian::util::cint_handling::*;
        use rstsr_showcase_hessian::util::density_matrices::*;

        let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, .. } = hess_case;
        let device = DeviceTsr::default();

        let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: Some(1e-14), uplo: Upper };
        let (cderi, j2c_decomp) = generate_cderi_with_decomp(mol, aux, j2c_decomp_option, &device);

        // --- shared --- //
        let result = prepare_shared(mol, aux, &j2c_decomp, 72, None, &DeviceTsr::default());
        let (dims, _aoslices, _auxslices, aux_ranges, shared, solve_aux) = result;
        println!("dims: {:?}", dims);
        println!("aux_ranges: {:?}", aux_ranges);
        let j2c = hess_intor(aux, "int2c2e", "s1", None, &device);
        assert!(rt::allclose(shared["j2c_inv"].view(), rt::linalg::inv(&j2c), None));
        for key in shared.keys() {
            println!("shared   {key:<20}, fp {:>16.10}, shape {:?}", fp(shared[key].view()), shared[key].shape());
        }

        // --- check j2c_inv --- //
        // By the following print, we know that
        // - j2c is usually not very that well-conditioned, and some value may be large
        // - using cholesky decomposition is more accurate then direct "exact" inversion (which is also
        //   based on svd or something else).
        // - In rust impl, j2c_inv deviation at 1e-10, exact_inv at 1e-9
        // - In python impl, j2c_inv deviation at 6e-10, exact_inv at 5e-8 (numpy) 1e-9 (scipy), emmm weird.
        //   Anyway, scipy (directly LAPACK) is probably better than numpy (customized LAPACK) in terms of
        //   inversion accuracy.
        let j2c_inv = shared["j2c_inv"].view();
        let j2c = hess_intor(aux, "int2c2e", "s1", None, &device);
        let mut recap = &j2c_inv % &j2c;
        *&mut recap.diagonal_mut(None) -= 1.0;
        let x_diag = recap.diagonal(None).abs().max();
        let x = recap.abs().max();
        println!("j2c_inv deviation: max abs {x:16.10e}, max abs diag {x_diag:16.10e}");
        let exact_inv = rt::linalg::inv(&j2c);
        let mut recap_exact = exact_inv % j2c;
        *&mut recap_exact.diagonal_mut(None) -= 1.0;
        let x_diag_exact = recap_exact.diagonal(None).abs().max();
        let x_exact = recap_exact.abs().max();
        println!("exact_inv deviation: max abs {x_exact:16.10e}, max abs diag {x_diag_exact:16.10e}");

        // --- prepare_j --- //
        let dm0 = get_dm0_restricted(mo_coeff.view(), mo_occ.view());
        let j_in = prepare_j(&solve_aux, &dims, dm0.view(), cderi.view());
        for key in j_in.keys() {
            println!("j_in     {key:<20}, fp {:>16.10}, shape {:?}", fp(j_in[key].view()), j_in[key].shape());
        }

        // --- prepare_k --- //
        let k_in = prepare_k(&solve_aux, &dims, mo_coeff.view(), mo_occ.view(), cderi.view());
        for key in k_in.keys() {
            println!("k_in     {key:<20}, fp {:>16.10}, shape {:?}", fp(k_in[key].view()), k_in[key].shape());
        }
    }

    #[rstest]
    fn test_get_rijk_skeleton_decomposed_separated(hess_case: &CaseAmoniaRHF) {
        use rstsr_showcase_hessian::ri_jk::decompose::*;
        use rstsr_showcase_hessian::ri_jk::hess_r::*;

        let CaseAmoniaRHF { mol, aux, mo_coeff, mo_occ, ref_dict, .. } = hess_case;
        let device = DeviceTsr::default();

        let j2c_decomp_option = J2CDecompOption { policy: J2CDecompPolicy::Cd, threshold: Some(1e-14), uplo: Upper };
        let (cderi, j2c_decomp) = generate_cderi_with_decomp(mol, aux, j2c_decomp_option, &device);

        let (j_out, k_outs, timing) = get_rijk_skeleton_decomposed_separated(
            mol,
            aux,
            &[mo_coeff.view()],
            &[mo_occ.view()],
            cderi.view(),
            &j2c_decomp,
            true,
            true,
            72,
            None,
            None,
        );
        // rhf only has one k_out, so we can unwrap it
        let j_out = j_out.unwrap();
        let k_out = &k_outs[0];

        println!("get_rijk_skeleton_decomposed_separated timing:");
        for (key, value) in timing.iter() {
            println!("    {:60}: {:10.6} seconds", key, value);
        }

        println!("j_out");
        for &key in j_out.keys().sorted() {
            if key.starts_with("de") {
                println!(
                    "j_out    {key:<20}, fp {:>16.10}, fp ref {:>16.10}, shape {:?}",
                    fp(j_out[key].view()),
                    fp(ref_dict[key].t()),
                    j_out[key].shape()
                );
                let ref_val = ref_dict[key].t();
                assert!(rt::allclose(j_out[key].view(), ref_val.view(), (1e-4, 1e-6)));
            } else {
                println!("j_out    {key:<20}, fp {:>16.10}, shape {:?}", fp(j_out[key].view()), j_out[key].shape());
            }
        }
        // special terms check
        assert_abs_diff_eq!(fp(j_out["j1ao_aux0"].view()), 35.385559919002, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(j_out["j1ao_aux1_1"].view()), -4.623388594699, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(j_out["j1ao_aux1_2"].view()), 2.521062525444, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(j_out["j1ao_aux1_3"].view()), -2.522442283815, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(j_out["j1ao_aux1_4"].view()), 4.739420467750, epsilon = 1e-6);

        println!("k_out");
        for &key in k_out.keys().sorted() {
            if key.starts_with("de") {
                println!(
                    "k_out    {key:<20}, fp {:>16.10}, fp ref {:>16.10}, shape {:?}",
                    fp(k_out[key].view()),
                    fp(ref_dict[key].t()),
                    k_out[key].shape()
                );
                let ref_val = ref_dict[key].t();
                assert!(rt::allclose(k_out[key].view(), ref_val.view(), (1e-4, 1e-6)));
            } else {
                println!("k_out    {key:<20}, fp {:>16.10}, shape {:?}", fp(k_out[key].view()), k_out[key].shape());
            }
        }
        // special terms check
        assert_abs_diff_eq!(fp(k_out["k1bra_aux0_1"].view()), 18.802914150861, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(k_out["k1bra_aux0_2"].view()), -25.362976629706, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(k_out["k1bra_aux0_3"].view()), -9.746782950535, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(k_out["k1bra_aux0_4"].view()), 17.639880529349, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(k_out["k1bra_aux1_1"].view()), -8.642371202611, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(k_out["k1bra_aux1_2"].view()), 8.757631723726, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(k_out["k1bra_aux1_3"].view()), -8.749320624840, epsilon = 1e-6);
        assert_abs_diff_eq!(fp(k_out["k1bra_aux1_4"].view()), 8.635554865181, epsilon = 1e-6);
    }
}
