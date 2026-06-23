//! Vibrational analysis tests — Rust port validation against the Python
//! `pyhessref.vib` reference for NH₃ / HF / def2-TZVP.
//!
//! Reference Hessian: `prototype/nh3_r_hf_decomp.npz`, key `de_ref`
//! (shape `[4,4,3,3]`, C-order). Geometry and isotope-averaged masses match
//! the PySCF molecule used to generate the reference. Geometry is in **bohr**.

mod test_util;

use rstsr::prelude::*;
use rstsr_showcase_hessian::hessian::vib::*;
use rstsr_showcase_hessian::prelude_dev::*;
use test_util::{read_npz, Tsr};

/// Hardcoded NH₃ geometry [3, natm] (bohr, column-major) and isotope-avg masses.
fn nh3_geom(device: &DeviceTsr) -> Tsr {
    rt::asarray((
        vec![
            0.0, 1.88972612, 0.56691784, 0.18897261, // x: N, H, H, H
            0.0, 0.18897261, 2.07869874, 0.18897261, // y
            0.0, 0.37794522, 0.37794522, 2.26767135, // z
        ],
        [3, 4].c(),
        device,
    ))
}

fn nh3_mass(device: &DeviceTsr) -> Tsr {
    rt::asarray((vec![14.007_f64, 1.008, 1.008, 1.008], device))
}

/// Build the flat `[12, 12]` Hessian (col-major) from the stored `de_ref`
/// `[4,4,3,3]` C-order array: `hess[(a,i), (b,j)] = de_ref[a,b,i,j]`.
///
/// `de_ref` is loaded as a row-major-flagged `[4,4,3,3]` tensor (C-order data).
/// Transpose to logical `[a,i,b,j]`, materialize col-major, then reshape to
/// `[12,12]` so that `hess[a*3+i, b*3+j] = de_ref[a,b,i,j]`.
fn build_hess_flat(device: &DeviceTsr) -> Tsr {
    let de_ref = read_npz("nh3_r_hf_decomp.npz", "de_ref"); // logical [a,b,i,j]
                                                            // hess[(a,i), (b,j)] = de_ref[a,b,i,j]. Index de_ref directly (read_npz
                                                            // gives correct logical indexing for C-order npy) and assemble the flat
                                                            // [12,12] col-major Hessian.
    let mut hess = rt::asarray((vec![0.0_f64; 144], [12, 12].c(), device));
    for a in 0..4 {
        for b in 0..4 {
            for i in 0..3 {
                for j in 0..3 {
                    hess[[a * 3 + i, b * 3 + j]] = de_ref[[a, b, i, j]];
                }
            }
        }
    }
    hess
}

/// Mass-centred geometry `[3, natm]`.
fn mass_centred_geom(geom: &Tsr, mass: &Tsr, device: &DeviceTsr) -> Tsr {
    let mass_sum = mass.to_vec().iter().sum::<f64>();
    let mc = (geom * mass.i((None, ..))).sum_axes(1) / mass_sum; // [3]
    let mc_vec = mc.to_vec();
    geom - rt::asarray((mc_vec, [3, 1].c(), device))
}

// ============================================================================
//  TR space
// ============================================================================

#[test]
fn test_tr_space_shapes() {
    let device = DeviceTsr::default();

    // single atom → 3 dof
    let m1 = rt::asarray((vec![1.0_f64], &device));
    let g1 = rt::asarray((vec![0.0, 0.0, 0.0], [3, 1].c(), &device));
    assert_eq!(get_tr_space(m1.view(), g1.view(), "TR").shape().as_slice(), &[3, 3]);

    // linear 2-atom → 5 dof
    let m2 = rt::asarray((vec![1.0_f64, 1.0], &device));
    let g2 = rt::asarray((vec![1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [3, 2].c(), &device));
    assert_eq!(get_tr_space(m2.view(), g2.view(), "TR").shape().as_slice(), &[6, 5]);

    // nonlinear 4-atom → 6 dof
    let m4 = rt::asarray((vec![1.0_f64; 4], &device));
    let g4 = rt::asarray((vec![1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0], [3, 4].c(), &device));
    let tr = get_tr_space(m4.view(), g4.view(), "TR");
    assert_eq!(tr.shape().as_slice(), &[12, 6]);

    // orthonormality: tr^T tr = I
    let ovlp = tr.t() % &tr;
    let eye_vec: Vec<f64> = (0..6).flat_map(|i| (0..6).map(move |j| if i == j { 1.0 } else { 0.0 })).collect();
    let eye = rt::asarray((eye_vec, [6, 6].c(), &device));
    assert!(rt::allclose(ovlp.view(), eye.view(), (1e-12, 1e-12)));
}

#[test]
fn test_tr_space_nh3_orthonormal() {
    let device = DeviceTsr::default();
    let geom = nh3_geom(&device);
    let mass = nh3_mass(&device);
    let tr = get_tr_space(mass.view(), geom.view(), "TR");
    assert_eq!(tr.shape().as_slice(), &[12, 6]);
    let ovlp = tr.t() % &tr;
    let eye_vec: Vec<f64> = (0..6).flat_map(|i| (0..6).map(move |j| if i == j { 1.0 } else { 0.0 })).collect();
    let eye = rt::asarray((eye_vec, [6, 6].c(), &device));
    assert!(rt::allclose(ovlp.view(), eye.view(), (1e-12, 1e-12)));
}

// ============================================================================
//  Rotation constants / rotor type
// ============================================================================

#[test]
fn test_rotation_const_nh3() {
    let device = DeviceTsr::default();
    let geom = nh3_geom(&device);
    let mass = nh3_mass(&device);
    let geom_c = mass_centred_geom(&geom, &mass, &device);

    let rc_ghz = rotation_const(mass.view(), geom_c.view(), "GHz");
    let rc_cm = rotation_const(mass.view(), geom_c.view(), "wavenumber");
    let ghz = rc_ghz.to_vec();
    let cm = rc_cm.to_vec();

    // Python reference values
    assert!((ghz[0] - 285.07591188).abs() < 1e-5);
    assert!((ghz[1] - 205.77561468).abs() < 1e-5);
    assert!((ghz[2] - 181.01823364).abs() < 1e-5);
    assert!((cm[0] - 9.50910889).abs() < 1e-7);
    assert!((cm[1] - 6.86393565).abs() < 1e-7);
    assert!((cm[2] - 6.03811833).abs() < 1e-7);

    assert_eq!(get_rotor_type(rc_ghz.view()), "REGULAR");
}

#[test]
fn test_rotor_type_classification() {
    let device = DeviceTsr::default();
    // atom
    let rc_atom = rt::asarray((vec![1e9_f64, 1e9, 1e9], &device));
    assert_eq!(get_rotor_type(rc_atom.view()), "ATOM");
    // linear
    let rc_lin = rt::asarray((vec![5e8_f64, 2.0, 2.0], &device));
    assert_eq!(get_rotor_type(rc_lin.view()), "LINEAR");
    // regular
    let rc_reg = rt::asarray((vec![300.0_f64, 200.0, 180.0], &device));
    assert_eq!(get_rotor_type(rc_reg.view()), "REGULAR");
}

// ============================================================================
//  harmonic_analysis
// ============================================================================

#[test]
fn test_harmonic_analysis_frequencies() {
    let device = DeviceTsr::default();
    let geom = nh3_geom(&device);
    let mass = nh3_mass(&device);
    let hess = build_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);

    // all 12 modes returned
    assert_eq!(vib.ndof(), 12);
    // 6 TR + 6 V
    let n_tr = vib.trv.iter().filter(|&&t| t == "TR").count();
    let n_v = vib.trv.iter().filter(|&&t| t == "V").count();
    assert_eq!(n_tr, 6);
    assert_eq!(n_v, 6);

    // TR modes near zero frequency
    for i in 0..12 {
        if vib.trv[i] == "TR" {
            assert!(vib.omega[i].abs() < 1.0, "TR mode {} freq {} not near zero", i, vib.omega[i]);
        }
    }

    // vibrational frequencies match Python reference
    let vib_idx = vib.vib_indices();
    let ref_freqs = [1263.343780, 1367.102321, 1424.072405, 2132.997526, 2443.140863, 3517.051480];
    assert_eq!(vib_idx.len(), 6);
    for (k, &i) in vib_idx.iter().enumerate() {
        assert!((vib.omega[i] - ref_freqs[k]).abs() < 1e-3, "vib mode {}: {} vs {}", k, vib.omega[i], ref_freqs[k]);
    }
}

#[test]
fn test_harmonic_analysis_properties() {
    let device = DeviceTsr::default();
    let geom = nh3_geom(&device);
    let mass = nh3_mass(&device);
    let hess = build_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);
    let vib_idx = vib.vib_indices();

    // reduced masses (vib only)
    let ref_mu = [1.062044, 1.101637, 1.082919, 1.019235, 1.023446, 1.056792];
    for (k, &i) in vib_idx.iter().enumerate() {
        assert!((vib.mu[i] - ref_mu[k]).abs() < 1e-5, "mu[{}] {} vs {}", k, vib.mu[i], ref_mu[k]);
    }

    // force constants (vib only)
    let ref_k = [0.998701, 1.213084, 1.293928, 2.732155, 3.599250, 7.701887];
    for (k, &i) in vib_idx.iter().enumerate() {
        assert!((vib.k[i] - ref_k[k]).abs() < 1e-5, "k[{}] {} vs {}", k, vib.k[i], ref_k[k]);
    }

    // characteristic temperatures (vib only)
    let ref_theta = [1817.670, 1966.956, 2048.923, 3068.909, 3515.136, 5060.254];
    for (k, &i) in vib_idx.iter().enumerate() {
        assert!(
            (vib.theta_vib[i] - ref_theta[k]).abs() < 1e-2,
            "theta[{}] {} vs {}",
            k,
            vib.theta_vib[i],
            ref_theta[k]
        );
    }

    // turning points (vib only)
    let ref_xtp0 = [0.299557, 0.282743, 0.279414, 0.235331, 0.219435, 0.179982];
    let ref_dq0 = [0.218291, 0.209844, 0.205604, 0.167997, 0.156972, 0.130830];
    for (k, &i) in vib_idx.iter().enumerate() {
        assert!((vib.xtp0[i] - ref_xtp0[k]).abs() < 1e-5, "xtp0[{}] {} vs {}", k, vib.xtp0[i], ref_xtp0[k]);
        assert!((vib.dq0[i] - ref_dq0[k]).abs() < 1e-5, "dq0[{}] {} vs {}", k, vib.dq0[i], ref_dq0[k]);
    }
}

#[test]
fn test_harmonic_analysis_q_orthonormal() {
    let device = DeviceTsr::default();
    let geom = nh3_geom(&device);
    let mass = nh3_mass(&device);
    let hess = build_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);

    // q^T q = I  (mass-weighted normal modes are orthonormal)
    let q = &vib.q;
    let ovlp = q.t() % q;
    let n = 12;
    let eye_vec: Vec<f64> = (0..n).flat_map(|i| (0..n).map(move |j| if i == j { 1.0 } else { 0.0 })).collect();
    let eye = rt::asarray((eye_vec, [n, n].c(), &device));
    assert!(rt::allclose(ovlp.view(), eye.view(), (1e-10, 1e-12)));

    // x columns have unit norm: ||x_i||^2 = 1
    let x_norms = vib.x.l2_norm_axes(0); // [ndof], norm over rows per column
    let xnv = x_norms.to_vec();
    for &nrm in xnv.iter() {
        assert!((nrm - 1.0).abs() < 1e-9, "x column norm {} != 1", nrm);
    }
}

#[test]
fn test_harmonic_analysis_degeneracy() {
    let device = DeviceTsr::default();
    let geom = nh3_geom(&device);
    let mass = nh3_mass(&device);
    let hess = build_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);

    // degeneracy label matches group count from rounding omega_real to 0.1 cm⁻¹
    let omega_real: Vec<f64> = (0..12).map(|i| if vib.imag[i] { 0.0 } else { vib.omega[i] }).collect();
    let mut keys: Vec<(i64, usize)> = (0..12).map(|i| ((omega_real[i] * 10.0).round() as i64, i)).collect();
    keys.sort_by_key(|&(k, _)| k);
    let mut expected = vec![0i64; 12];
    let mut start = 0;
    while start < keys.len() {
        let k = keys[start].0;
        let mut end = start + 1;
        while end < keys.len() && keys[end].0 == k {
            end += 1;
        }
        let count = (end - start) as i64;
        for j in start..end {
            expected[keys[j].1] = count;
        }
        start = end;
    }
    for i in 0..12 {
        assert_eq!(vib.degeneracy[i], expected[i], "degeneracy[{}]", i);
    }
}

// ============================================================================
//  Thermochemistry
// ============================================================================

#[test]
fn test_thermo_nh3() {
    let device = DeviceTsr::default();
    let geom = nh3_geom(&device);
    let mass = nh3_mass(&device);
    let hess = build_hess_flat(&device);
    let mass_sum = mass.to_vec().iter().sum::<f64>();

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);

    let geom_c = mass_centred_geom(&geom, &mass, &device);
    let rc_cm = rotation_const(mass.view(), geom_c.view(), "wavenumber");
    let rc_ghz = rotation_const(mass.view(), geom_c.view(), "GHz");
    let rotor = RotorType::from_rot_const_ghz(rc_ghz.view());
    let rc_cm_vec = rc_cm.to_vec();

    let th = thermo(&vib, 298.15, 101325.0, 1, mass_sum, -56.0, 3, &rc_cm_vec, rotor);

    // ZPE
    assert!((th.zpe[VIB] - 0.027674515751).abs() < 1e-8);
    assert!((th.zpe_tot - (-55.972325484249)).abs() < 1e-7);
    // component entropies [mEh/K]
    assert!((th.s[TRANS] - 0.054886124656).abs() < 1e-9);
    assert!((th.s[ROT] - 0.018956988041).abs() < 1e-9);
    assert!((th.s[VIB] - 0.000110929605).abs() < 1e-9);
    assert!((th.s_tot - 0.073954042302).abs() < 1e-9);
    // component energies [Eh]
    assert!((th.e[VIB] - 0.027703154873).abs() < 1e-8);
    assert!((th.e[TRANS] - 0.001416276822).abs() < 1e-9);
    assert!((th.e[ROT] - 0.001416276822).abs() < 1e-9);
    // totals
    assert!((th.e_tot - (-55.969464291483)).abs() < 1e-7);
    assert!((th.h_tot - (-55.968520106935)).abs() < 1e-7);
    assert!((th.g_tot - (-55.990569504648)).abs() < 1e-7);
    // heat capacities [mEh/K]
    assert!((th.cv_tot - 0.010125123641).abs() < 1e-8);
    assert!((th.cp_tot - 0.013291934132).abs() < 1e-8);
}

#[test]
fn test_thermo_gibbs_relation() {
    // G = H - T*S for each component and total
    let device = DeviceTsr::default();
    let geom = nh3_geom(&device);
    let mass = nh3_mass(&device);
    let hess = build_hess_flat(&device);
    let mass_sum = mass.to_vec().iter().sum::<f64>();

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);
    let geom_c = mass_centred_geom(&geom, &mass, &device);
    let rc_cm = rotation_const(mass.view(), geom_c.view(), "wavenumber").to_vec();
    let rotor = RotorType::Regular;

    let t = 298.15;
    let th = thermo(&vib, t, 101325.0, 1, mass_sum, -56.0, 3, &rc_cm, rotor);

    for i in 0..4 {
        let g_expected = th.h[i] - t * th.s[i] / 1000.0; // S in mEh/K → Eh/K
        assert!((th.g[i] - g_expected).abs() < 1e-12, "G[{}] mismatch", i);
    }
    let g_tot_expected = th.h_tot - t * th.s_tot / 1000.0;
    assert!((th.g_tot - g_tot_expected).abs() < 1e-12);
}

// ============================================================================
//  Acetaldehyde (CH3CHO) / HF / def2-TZVP
//  9 atoms → 27 dof = 6 TR + 21 vibrational modes.
//  Geometry in Ångström in the source; stored below in bohr (col-major [3,9]).
//  Hessian key is `ref_de` (note: NOT `de_ref`).
// ============================================================================

/// Acetaldehyde geometry `[3, 9]` (bohr, column-major).
fn et_geom(device: &DeviceTsr) -> Tsr {
    rt::asarray((
        vec![
            // x: C, C, O, H, H, H, H, H, H
            1.7384157537711373,
            4.59999022967752,
            5.510252406619264,
            1.0112491410385016,
            0.9976431129416331,
            1.0099641272737974,
            5.339685726616021,
            5.331295342622953,
            4.86853920923946,
            // y
            -0.01859490506572021,
            -0.02263891897228944,
            -1.9603451898400581,
            1.4538418966728848,
            0.3039813443975359,
            -1.8510623280564606,
            1.7931989141222784,
            -0.37431695075384747,
            -1.6204590490757862,
            // z
            0.08785336753102972,
            0.1009113750517743,
            1.7184413486344845,
            -1.1665279366940127,
            1.9927161983538577,
            -0.5392711441671317,
            0.7583092992654681,
            -1.8003042843506432,
            3.406382517618489,
        ],
        [3, 9].c(),
        device,
    ))
}

fn et_mass(device: &DeviceTsr) -> Tsr {
    rt::asarray((vec![12.011_f64, 12.011, 15.999, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008], device))
}

/// Build the flat `[27, 27]` Hessian (col-major) from the stored `ref_de`
/// `[9,9,3,3]` C-order array: `hess[(a,i), (b,j)] = ref_de[a,b,i,j]`.
fn build_et_hess_flat(device: &DeviceTsr) -> Tsr {
    let ref_de = read_npz("et_r_hf.npz", "ref_de"); // logical [a,b,i,j]
    let n = 27;
    let mut hess = rt::asarray((vec![0.0_f64; n * n], [n, n].c(), device));
    for a in 0..9 {
        for b in 0..9 {
            for i in 0..3 {
                for j in 0..3 {
                    hess[[a * 3 + i, b * 3 + j]] = ref_de[[a, b, i, j]];
                }
            }
        }
    }
    hess
}

#[test]
fn test_et_trv_counts() {
    let device = DeviceTsr::default();
    let geom = et_geom(&device);
    let mass = et_mass(&device);
    let hess = build_et_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);

    assert_eq!(vib.ndof(), 27);
    let n_tr = vib.trv.iter().filter(|&&t| t == "TR").count();
    let n_v = vib.trv.iter().filter(|&&t| t == "V").count();
    assert_eq!(n_tr, 6, "expected 6 TR modes");
    assert_eq!(n_v, 21, "expected 21 vibrational modes");
}

#[test]
fn test_et_tr_frequencies() {
    let device = DeviceTsr::default();
    let geom = et_geom(&device);
    let mass = et_mass(&device);
    let hess = build_et_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);

    // 6 TR modes, all with |frequency| well below 1 cm⁻¹ (they are projected
    // out, so their eigenvalues are numerical noise; some are imaginary, for
    // which `omega` holds the magnitude). Python reference real parts are
    // [0, 0, 0.037, 0.073, 0.092, 0.104] — all < 0.2 cm⁻¹.
    let tr_freqs: Vec<f64> = (0..27).filter(|&i| vib.trv[i] == "TR").map(|i| vib.omega[i]).collect();
    assert_eq!(tr_freqs.len(), 6, "expected exactly 6 TR modes");
    for (k, &f) in tr_freqs.iter().enumerate() {
        assert!(f.abs() < 0.5, "TR freq {} not near zero: {}", k, f);
    }
}

#[test]
fn test_et_vib_frequencies_first10() {
    let device = DeviceTsr::default();
    let geom = et_geom(&device);
    let mass = et_mass(&device);
    let hess = build_et_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);
    let vib_idx = vib.vib_indices();
    assert_eq!(vib_idx.len(), 21);

    // first 10 vibrational frequencies [cm⁻¹]
    let ref_freq = [
        397.693146,
        482.146314,
        748.727643,
        909.980504,
        966.500675,
        1128.802224,
        1198.255774,
        1254.376121,
        1404.826430,
        1513.104741,
    ];
    for k in 0..10 {
        let i = vib_idx[k];
        assert!((vib.omega[i] - ref_freq[k]).abs() < 1e-3, "vib freq {}: {} vs {}", k, vib.omega[i], ref_freq[k]);
    }
}

#[test]
fn test_et_vib_properties_first10() {
    let device = DeviceTsr::default();
    let geom = et_geom(&device);
    let mass = et_mass(&device);
    let hess = build_et_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);
    let vib_idx = vib.vib_indices();

    // reduced masses [u]
    let ref_mu = [1.114579, 2.671366, 1.116921, 1.079670, 2.425550, 3.653739, 1.566218, 1.431222, 1.196174, 1.233885];
    // characteristic temperatures [K]
    let ref_theta =
        [572.1919, 693.7012, 1077.2524, 1309.2593, 1390.5793, 1624.0951, 1724.0233, 1804.7680, 2021.2325, 2177.0208];
    for k in 0..10 {
        let i = vib_idx[k];
        assert!((vib.mu[i] - ref_mu[k]).abs() < 1e-4, "mu {}: {} vs {}", k, vib.mu[i], ref_mu[k]);
        assert!(
            (vib.theta_vib[i] - ref_theta[k]).abs() < 1e-2,
            "theta {}: {} vs {}",
            k,
            vib.theta_vib[i],
            ref_theta[k]
        );
    }
}

#[test]
fn test_et_q_orthonormal() {
    let device = DeviceTsr::default();
    let geom = et_geom(&device);
    let mass = et_mass(&device);
    let hess = build_et_hess_flat(&device);

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);
    let q = &vib.q;
    let ovlp = q.t() % q;
    let n = 27;
    let eye_vec: Vec<f64> = (0..n).flat_map(|i| (0..n).map(move |j| if i == j { 1.0 } else { 0.0 })).collect();
    let eye = rt::asarray((eye_vec, [n, n].c(), &device));
    assert!(rt::allclose(ovlp.view(), eye.view(), (1e-10, 1e-12)));
}

#[test]
fn test_et_rotation_constants() {
    let device = DeviceTsr::default();
    let geom = et_geom(&device);
    let mass = et_mass(&device);
    let geom_c = mass_centred_geom(&geom, &mass, &device);

    let rc_cm = rotation_const(mass.view(), geom_c.view(), "wavenumber").to_vec();
    let rc_ghz = rotation_const(mass.view(), geom_c.view(), "GHz").to_vec();
    // Python: cm⁻¹ [1.11175601, 0.31790423, 0.27739508], GHz [33.32960657, 9.53052918, 8.31609533]
    assert!((rc_cm[0] - 1.11175601).abs() < 1e-7);
    assert!((rc_cm[1] - 0.31790423).abs() < 1e-7);
    assert!((rc_cm[2] - 0.27739508).abs() < 1e-7);
    assert!((rc_ghz[0] - 33.32960657).abs() < 1e-6);
    assert!((rc_ghz[1] - 9.53052918).abs() < 1e-6);
    assert!((rc_ghz[2] - 8.31609533).abs() < 1e-6);
    assert_eq!(get_rotor_type(rt::asarray((rc_ghz.clone(), &device)).view()), "REGULAR");
}

#[test]
fn test_et_thermo() {
    let device = DeviceTsr::default();
    let geom = et_geom(&device);
    let mass = et_mass(&device);
    let hess = build_et_hess_flat(&device);
    let mass_sum = mass.to_vec().iter().sum::<f64>();

    let vib = harmonic_analysis(hess.view(), geom.view(), mass.view(), true, true);
    let geom_c = mass_centred_geom(&geom, &mass, &device);
    let rc_cm = rotation_const(mass.view(), geom_c.view(), "wavenumber").to_vec();
    let rc_ghz = rotation_const(mass.view(), geom_c.view(), "GHz");
    let rotor = RotorType::from_rot_const_ghz(rc_ghz.view());

    // acetaldehyde: sigma=1, E0=-153.0 (arbitrary well bottom), multiplicity=1
    let th = thermo(&vib, 298.15, 101325.0, 1, mass_sum, -153.0, 1, &rc_cm, rotor);

    // ZPE / energies [Eh]
    assert!((th.zpe[VIB] - 0.084945609427).abs() < 1e-8);
    assert!((th.e[VIB] - 0.085764093805).abs() < 1e-8);
    assert!((th.zpe_tot - (-152.915054390573)).abs() < 1e-7);
    assert!((th.e_tot - (-152.911403352551)).abs() < 1e-7);
    assert!((th.h_tot - (-152.910459168003)).abs() < 1e-7);
    assert!((th.g_tot - (-152.939964870917)).abs() < 1e-7);
    // entropies [mEh/K]
    assert!((th.s[TRANS] - 0.059613088371).abs() < 1e-9);
    assert!((th.s[ROT] - 0.035576765773).abs() < 1e-9);
    assert!((th.s[VIB] - 0.003772758345).abs() < 1e-9);
    assert!((th.s_tot - 0.098962612490).abs() < 1e-9);
    // heat capacities [mEh/K]
    assert!((th.cv_tot - 0.018232060490).abs() < 1e-8);
    assert!((th.cp_tot - 0.021398870982).abs() < 1e-8);
}
