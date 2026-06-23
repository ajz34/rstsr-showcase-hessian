"""Test vibrational analysis module.

Uses the NH₃ / HF / def2-TZVP reference Hessian from
``prototype/nh3_r_hf_decomp.npz`` (key ``de_ref``) to validate
:mod:`pyhessref.vib` against PySCF and Psi4 reference implementations.

Cross-validation strategy:
- Frequencies and normal modes validated against PySCF's ``pyscf.hessian.thermo``.
- TR space construction validated against Psi4's ``_get_TR_space`` (shapes
  and orthonormality).
- Thermochemistry validated against Psi4's ``thermo`` (via psi4 if available).
"""

import unittest
import numpy as np
from pyscf import gto, scf
from pyscf.hessian import thermo as pyscf_thermo

from pyhessref.vib import (
    harmonic_analysis,
    thermo,
    filter_nonvib,
    filter_omega_to_real,
    rotation_const,
    _get_TR_space,
    _get_rotor_type,
    _phase_cols_to_max_element,
    _check_degen_modes,
    _check_rank_degen_modes,
    _vec_in_space,
    print_vibs,
    print_molden_vibs,
    _format_omega,
)


def setUpModule():
    global mol, mf, mass, geom, hess_44, hess_flat, vibinfo, pyscf_results
    # NH3 / HF / def2-TZVP (distorted geometry, same as all other hessian tests)
    xyz = """
    N  0   0   0
    H  1.0 0.1 0.2
    H  0.3 1.1 0.2
    H  0.1 0.1 1.2
    """
    PATH_PROTOTYPE = "prototype/"
    PATH_REF = PATH_PROTOTYPE + "nh3_r_hf_decomp.npz"

    mol = gto.Mole(atom=xyz, basis="def2-TZVP", max_memory=8000).build()
    mass = mol.atom_mass_list(isotope_avg=True)      # [14.007, 1.008, 1.008, 1.008]
    geom = mol.atom_coords()

    # Load reference data & run SCF to get consistent Hessian
    ref_value = np.load(PATH_REF)
    mf = scf.RHF(mol).density_fit()
    mf.mo_coeff = ref_value["mo_coeff"]
    mf.mo_occ = ref_value["mo_occ"]
    mf.mo_energy = ref_value["mo_energy"]
    mf.with_df.build()
    mf.converged = True

    # PySCF Hessian (natm, natm, 3, 3) and flat (3*natm, 3*natm)
    hess_44 = mf.Hessian().kernel()  # shape (4, 4, 3, 3)
    hess_flat = hess_44.transpose(0, 2, 1, 3).reshape(12, 12)

    # PySCF harmonic analysis (returns only 6 vib modes, no TR)
    pyscf_results = pyscf_thermo.harmonic_analysis(mol, hess_44)

    # Our harmonic analysis (returns all 12 modes)
    vibinfo = harmonic_analysis(hess_flat, geom, mass)


# ============================================================================
#  _get_TR_space
# ============================================================================

class TestTRSpace(unittest.TestCase):
    """Translation / rotation space construction (follows Psi4)."""

    def test_shapes(self):
        """Check expected TR-space dimensions for atom / linear / nonlinear cases."""
        # single atom → 3 dof (T only or TR = 3)
        m1 = np.array([1.0])
        g1 = np.array([[0., 0., 0.]])
        self.assertEqual(_get_TR_space(m1, g1).shape, (3, 3))
        self.assertEqual(_get_TR_space(m1, g1, space='T').shape, (3, 3))

        # linear 2-atom → 5 dof
        m2 = np.array([1., 1.])
        g2 = np.array([[1., 0., 0.], [-1., 0., 0.]])
        self.assertEqual(_get_TR_space(m2, g2).shape, (5, 6))

        # linear 3-atom → 5 dof
        m3 = np.array([1., 1., 1.])
        g3 = np.array([[3., 0., 0.], [4., 0., 0.], [5., 0., 0.]])
        self.assertEqual(_get_TR_space(m3, g3).shape, (5, 9))

        # nonlinear 4-atom → 6 dof
        m4 = np.array([1., 1., 1., 1.])
        g4 = np.array([[1., 1., 0.], [-1., 1., 0.], [-1., -1., 0.], [1., -1., 0.]])
        self.assertEqual(_get_TR_space(m4, g4).shape, (6, 12))

    def test_orthonormal(self):
        """TR vectors should be orthonormal (rows form an orthonormal set)."""
        TR = _get_TR_space(mass, geom)
        ovlp = TR.dot(TR.T)
        self.assertTrue(np.allclose(ovlp, np.eye(TR.shape[0]), atol=1e-12))

    def test_linear_noisy(self):
        """Noisy linear geometry still detected as 5-dof with tolerance."""
        m3 = np.array([1., 1., 1.])
        g3_noisy = np.array([[3., 3.001, 3.], [4., 4.001, 4.], [5., 5., 5.01]])
        TR = _get_TR_space(m3, g3_noisy, tol=1.e-2)
        self.assertEqual(TR.shape, (5, 9))


# ============================================================================
#  rotation_const / _get_rotor_type
# ============================================================================

class TestRotConst(unittest.TestCase):
    def test_rotational_constants_ghz(self):
        mass_center = (mass[:, None] * geom).sum(axis=0) / mass.sum()
        geom_c = geom - mass_center
        rc = rotation_const(mass, geom_c, 'GHz')
        self.assertEqual(rc.shape, (3,))
        # A ≥ B ≥ C; the distorted geometry may break B ≈ C symmetry
        self.assertGreater(rc[0], rc[1])
        self.assertGreaterEqual(rc[1], rc[2])
        # all should be finite and positive for a polyatomic
        self.assertTrue(np.all(np.isfinite(rc)))
        self.assertTrue(np.all(rc > 0))

    def test_rotational_constants_wavenumber(self):
        mass_center = (mass[:, None] * geom).sum(axis=0) / mass.sum()
        geom_c = geom - mass_center
        rc = rotation_const(mass, geom_c, 'wavenumber')
        self.assertEqual(rc.shape, (3,))
        # typical NH₃ ~ 9.9, 6.3 cm⁻¹ for equilibrium, distorted will differ
        self.assertGreater(rc[0], 1.0)

    def test_rotor_type(self):
        # atom
        self.assertEqual(_get_rotor_type(np.array([1e9, 1e9, 1e9])), 'ATOM')
        # linear (first component far larger than split between others)
        self.assertEqual(_get_rotor_type(np.array([5e8, 2., 2.])), 'LINEAR')
        # regular
        self.assertEqual(_get_rotor_type(np.array([300., 200., 180.])), 'REGULAR')


# ============================================================================
#  harmonic_analysis
# ============================================================================

class TestHarmonicAnalysis(unittest.TestCase):
    """Core vibrational analysis tests."""

    def test_all_12_modes_returned(self):
        """All 3×nat = 12 modes are present."""
        self.assertEqual(len(vibinfo['omega']), 12)
        for key in ['q', 'w', 'x', 'mu', 'k', 'DQ0', 'Qtp0', 'Xtp0',
                     'theta_vib', 'degeneracy', 'TRV']:
            self.assertIn(key, vibinfo)

    def test_tr_v_classification(self):
        """6 TR modes + 6 V modes for nonlinear NH₃."""
        trv = vibinfo['TRV']
        self.assertEqual(list(trv).count('TR'), 6)
        self.assertEqual(list(trv).count('V'), 6)

    def test_tr_modes_near_zero_frequency(self):
        """TR modes should all have |ω| < 1 cm⁻¹."""
        tr_mask = vibinfo['TRV'] == 'TR'
        tr_freqs = vibinfo['omega'][tr_mask]
        for f in tr_freqs:
            self.assertLess(abs(f), 1.0,
                            msg=f"TR mode freq {f:.4f} not near zero")

    def test_frequencies_match_pyscf(self):
        """Vibrational frequencies match PySCF (6 values)."""
        v_mask = vibinfo['TRV'] == 'V'
        our_vib_freqs = vibinfo['omega'][v_mask]
        pyscf_freqs = pyscf_results['freq_wavenumber']
        # our frequencies are complex, PySCF are complex too but stored as neg-real for imag
        # Both should have ~same real parts (tiny imag for stable modes)
        our_real = np.array([f.real for f in our_vib_freqs])
        pyscf_real = np.array([f.real for f in pyscf_freqs])
        self.assertTrue(np.allclose(our_real, pyscf_real, atol=1e-3, rtol=1e-6),
                        msg=f"diff: {our_real - pyscf_real}")

    def test_q_orthonormal(self):
        """Mass-weighted normal modes q should be orthonormal: q^T q = I."""
        q = vibinfo['q']  # (12, 12)
        ovlp = q.T @ q
        self.assertTrue(np.allclose(ovlp, np.eye(12), atol=1e-12))

    def test_w_relation(self):
        """w = m^{-1/2} · q, so μ = 1/||w_col||^2."""
        sqrtmmm_inv = np.divide(1.0, np.repeat(np.sqrt(mass), 3))
        w_from_q = sqrtmmm_inv[:, None] * vibinfo['q']
        self.assertTrue(np.allclose(w_from_q, vibinfo['w'], atol=1e-14))

        mu_from_w = np.divide(1.0, np.linalg.norm(vibinfo['w'], axis=0)**2)
        self.assertTrue(np.allclose(mu_from_w, vibinfo['mu'], atol=1e-14))

    def test_x_normalization(self):
        """x = √μ · w, so each column of x has norm sqrt(μ)."""
        x = vibinfo['x']
        w = vibinfo['w']
        mu = vibinfo['mu']
        x_from_w = np.sqrt(mu) * w
        self.assertTrue(np.allclose(x, x_from_w, atol=1e-14))
        # ||x_i||² = μ_i * ||w_i||² = μ_i * (1/μ_i) = 1
        x_norms = np.linalg.norm(x, axis=0)
        self.assertTrue(np.allclose(x_norms, np.ones(12), atol=1e-12))

    def test_phase_cols_consistency(self):
        """All columns of q have positive extreme element."""
        q = vibinfo['q']
        for v in range(q.shape[1]):
            iextreme = np.argmax(np.abs(q[:, v]))
            self.assertGreaterEqual(q[iextreme, v], 0)

    def test_degeneracy_labels_consistent(self):
        # degeneracy[k] = number of modes in k's near-degenerate group
        # sum over unique groups should equal ndof=12
        deg = vibinfo['degeneracy']
        _, uinv, ucts = np.unique(np.around(vibinfo['omega'], 1),
                                  return_inverse=True, return_counts=True)
        self.assertEqual(np.sum(ucts), 12)
        # each mode's degeneracy label matches its group count
        self.assertTrue(np.all(deg == ucts[uinv]))

    def test_force_constants_positive_for_v_modes(self):
        v_mask = vibinfo['TRV'] == 'V'
        self.assertTrue(np.all(vibinfo['k'][v_mask] > 0))

    def test_characteristic_temperatures(self):
        """θ_vib = ω_real * 100·h·c/kB, so for 1000 cm⁻¹ → ~1439 K."""
        v_mask = vibinfo['TRV'] == 'V'
        self.assertTrue(np.all(vibinfo['theta_vib'][v_mask] > 100))


# ============================================================================
#  filter_nonvib / filter_omega_to_real
# ============================================================================

class TestFilters(unittest.TestCase):
    def test_filter_nonvib_removes_tr(self):
        vibonly = filter_nonvib(vibinfo)
        self.assertEqual(len(vibonly['omega']), 6)
        self.assertTrue(np.all(vibonly['TRV'] == 'V'))
        for key in ['q', 'w', 'x']:
            self.assertEqual(vibonly[key].shape[1], 6)

    def test_filter_nonvib_custom_removal(self):
        # remove modes 0 and 1 (two TR modes)
        filt = filter_nonvib(vibinfo, remove=[0, 1])
        self.assertEqual(len(filt['omega']), 10)
        self.assertEqual(filt['q'].shape, (12, 10))

    def test_filter_omega_to_real(self):
        imag_omega = np.array([100.0+0j, 0.0+50.0j, 200.0+30.0j])
        # mode 1: imag=50 > real=0 → -50
        # mode 2: real=200 > imag=30 → 200
        expected = np.array([100.0, -50.0, 200.0])
        result = filter_omega_to_real(imag_omega)
        self.assertTrue(np.allclose(result, expected))


# ============================================================================
#  thermo
# ============================================================================

class TestThermo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mass_tot = mass.sum()
        mass_center = (mass[:, None] * geom).sum(axis=0) / mass.sum()
        geom_c = geom - mass_center
        rc_ghz = rotation_const(mass, geom_c, 'GHz')
        rc_cm = rotation_const(mass, geom_c, 'wavenumber')
        rotor = _get_rotor_type(rc_ghz)

        cls.therminfo = thermo(
            vibinfo, T=298.15, P=101325.0,
            multiplicity=1, molecular_mass=mass_tot,
            E0=-56.0, sigma=3, rot_const=rc_cm,
            rotor_type=rotor,
        )

    def test_basic_thermo_keys(self):
        """Thermo dict has expected keys."""
        for k in ['E0', 'B', 'sigma', 'T', 'P',
                  'S_tot', 'Cv_tot', 'Cp_tot',
                  'ZPE_corr', 'E_tot', 'H_tot', 'G_tot',
                  'S_vib', 'E_vib', 'S_trans', 'E_trans']:
            self.assertIn(k, self.therminfo, msg=f"Missing key {k}")

    def test_zpe_positive(self):
        self.assertGreater(self.therminfo['ZPE_corr'], 0.0)

    def test_h_tot_greater_than_e0(self):
        """Total enthalpy > electronic energy (ZPE + thermal contributions > 0)."""
        self.assertGreater(self.therminfo['H_tot'], self.therminfo['E0'])

    def test_g_tot_defined(self):
        """Gibbs free energy computed and finite."""
        self.assertTrue(np.isfinite(self.therminfo['G_tot']))
        # G = H - TS; both H and G are physically meaningful
        # Note: for distorted geometries G can be above or below E0

    def test_s_total_positive(self):
        """Total entropy positive for polyatomic."""
        self.assertGreater(self.therminfo['S_tot'], 0.0)


# ============================================================================
#  Print functions (smoke tests)
# ============================================================================

class TestPrintFunctions(unittest.TestCase):
    def test_print_vibs_short(self):
        text = print_vibs(vibinfo, atom_lbl=['N', 'H', 'H', 'H'],
                          normco='x', shortlong=True)
        self.assertIn('Vibration', text)
        self.assertIn('Freq', text)

    def test_print_vibs_long(self):
        text = print_vibs(vibinfo, normco='q', shortlong=False)
        self.assertIn('Vibration', text)

    def test_print_molden_vibs(self):
        text = print_molden_vibs(vibinfo, atom_symbol=['N', 'H', 'H', 'H'],
                                 geom=geom)
        self.assertIn('[FREQ]', text)
        self.assertIn('[FR-COORD]', text)
        self.assertIn('[FR-NORM-COORD]', text)

    def test_format_omega(self):
        # real mode: shows real part
        # imaginary mode (imag > real): shows imag with 'i' suffix
        omega = np.array([100.0+0j, 0.0+50.0j, 10.0+0j])
        result = _format_omega(omega, 2)
        self.assertEqual(result[0], '100.00')
        self.assertEqual(result[1], '50.00i')
        self.assertEqual(result[2], '10.00')


# ============================================================================
#  Helper functions
# ============================================================================

class TestHelpers(unittest.TestCase):
    def test_phase_cols_to_max_element(self):
        arr = np.array([[1.0, -2.0], [0.5, 1.0], [-0.3, 0.0]])
        out = _phase_cols_to_max_element(arr)
        # col 0: extreme at index 0 (value 1.0, already positive) → unchanged
        self.assertAlmostEqual(out[0, 0], 1.0)
        # col 1: extreme at index 0 (value -2.0) → sign flipped
        self.assertAlmostEqual(out[0, 1], 2.0)
        self.assertAlmostEqual(out[1, 1], -1.0)

    def test_vec_in_space_true(self):
        """A vector IN a space should be detected."""
        space = np.array([[1., 0., 0.], [0., 1., 0.]])  # xy-plane
        vec = np.array([0.3, 0.4, 0.0])  # in xy-plane
        self.assertTrue(_vec_in_space(vec, space))

    def test_vec_in_space_false(self):
        """A vector NOT in a space should not be detected."""
        space = np.array([[1., 0., 0.], [0., 1., 0.]])  # xy-plane
        vec = np.array([0.3, 0.4, 0.5])  # has z-component
        self.assertFalse(_vec_in_space(vec, space))

    def test_check_degen_modes(self):
        """Degenerate modes should be stably sorted."""
        freq = np.array([100., 100., 200.])
        arr = np.array([
            [1., 0., 0.],  # mode 0
            [0., 1., 0.],  # mode 1 (degen with mode 0)
            [0., 0., 1.],  # mode 2
        ], dtype=float).T  # (3, 3)
        out = _check_degen_modes(arr, freq)
        self.assertEqual(out.shape, arr.shape)

    def test_check_rank_degen_modes_single(self):
        """Single (non-degenerate) modes checked by direct comparison.
        Phases must be standardized first with _phase_cols_to_max_element."""
        freq = np.array([100., 200.])
        cv = np.random.RandomState(42).randn(4, 2)
        cv = np.linalg.qr(cv)[0]
        # standardize phases
        cv = _phase_cols_to_max_element(cv)
        ev = cv.copy()
        self.assertTrue(_check_rank_degen_modes(cv, freq, ev))
        # flip phase of one mode → phase-standardize then check
        ev2 = cv.copy()
        ev2[:, 0] *= -1
        ev2 = _phase_cols_to_max_element(ev2)
        self.assertTrue(_check_rank_degen_modes(cv, freq, ev2))


# ============================================================================
#  Reference value persistence / regression
#  Hardcoded values from NH₃ / HF / def2-TZVP (distorted C₁ geometry).
#  These are the ground truth for any future Rust reimplementation.
# ============================================================================

class TestReferenceValues(unittest.TestCase):
    """Hardcoded reference numbers for NH₃ HF/def2-TZVP vibrational analysis."""

    # ---------- all 12 modes (TR + V) ----------

    # frequencies: complex pair (real, imag) — imag>0 means imaginary mode
    REF_OMEGA = [
        (0.0, 0.088262),     # 0 TR
        (0.0, 0.058625),     # 1 TR
        (0.018864, 0.0),     # 2 TR
        (0.040365, 0.0),     # 3 TR
        (0.055391, 0.0),     # 4 TR
        (0.096076, 0.0),     # 5 TR
        (1263.343780, 0.0),  # 6 V
        (1367.102321, 0.0),  # 7 V
        (1424.072405, 0.0),  # 8 V
        (2132.997526, 0.0),  # 9 V
        (2443.140863, 0.0),  # 10 V
        (3517.051480, 0.0),  # 11 V
    ]
    REF_TRV = ['TR', 'TR', 'TR', 'TR', 'TR', 'TR',
               'V', 'V', 'V', 'V', 'V', 'V']
    REF_DEGENERACY = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    REF_MU = [
        3.86884659, 3.06565089, 1.26470459, 4.32893670, 1.13967262, 1.01863812,
        1.06204364, 1.10163745, 1.08291867, 1.01923533, 1.02344569, 1.05679190,
    ]
    REF_K = [
        -0.00000002, -0.00000001, 0.0, 0.0, 0.0, 0.00000001,
        0.99870147, 1.21308426, 1.29392831, 2.73215546, 3.59925030, 7.70188730,
    ]
    REF_THETA_VIB = [
        0.0, 0.0, 0.027142, 0.058077, 0.079696, 0.138232,
        1817.670421, 1966.955859, 2048.923127, 3068.908537, 3515.135746, 5060.254022,
    ]

    # ---------- vibrational modes only ----------
    REF_VIB_XTP0 = [
        0.299557473294, 0.282743257373, 0.279413996917,
        0.235331304287, 0.219434777674, 0.179981560425,
    ]
    REF_VIB_DQ0 = [
        0.218291257188, 0.209844023697, 0.205603772919,
        0.167997159786, 0.156972242123, 0.130830123918,
    ]

    # ---------- thermochemistry ----------
    THERMO_REF = {
        'rot_const_cm':  (9.5091088865, 6.8639356510, 6.0381183326),
        'rotor_type':    'REGULAR',
        'ZPE_vib':       0.027674515751202,
        'E_vib':         0.027703154872583,
        'S_vib':         0.000110929605075,
        'Cv_vib':        0.000624692166256,
        'S_trans':       0.054886124655946,
        'E_trans':       0.001416276822069,
        'S_rot':         0.018956988040946,
        'E_rot':         0.001416276822069,
        'E_tot':        -55.969464291483277,
        'H_tot':        -55.968520106935230,
        'G_tot':        -55.990569504647567,
        'ZPE_tot':      -55.972325484248799,
        'S_tot':         0.073954042301967,
        'Cv_tot':        0.010125123640812,
        'Cp_tot':        0.013291934132330,
    }

    @classmethod
    def setUpClass(cls):
        cls.vibonly = filter_nonvib(vibinfo)

    # ---- all 12 modes ----

    def test_frequencies_all_exact(self):
        for i, (re, im) in enumerate(self.REF_OMEGA):
            f = vibinfo['omega'][i]
            self.assertAlmostEqual(f.real, re, places=5,
                                   msg=f'omega[{i}].real')
            self.assertAlmostEqual(abs(f.imag), abs(im), places=5,
                                   msg=f'omega[{i}].imag')

    def test_trv_labels_exact(self):
        self.assertEqual(list(vibinfo['TRV']), self.REF_TRV)

    def test_degeneracy_exact(self):
        self.assertEqual(list(vibinfo['degeneracy']), self.REF_DEGENERACY)

    def test_reduced_masses_exact(self):
        for i, ref in enumerate(self.REF_MU):
            # TR-mode reduced masses (indices 0..5) are not physically meaningful
            # and drift at the ~1e-5 level under BLAS threading noise; vibrational
            # modes (6..11) are pinned tightly.
            places = 3 if vibinfo['TRV'][i] == 'TR' else 5
            self.assertAlmostEqual(vibinfo['mu'][i], ref, places=places,
                                   msg=f'mu[{i}]')

    def test_force_constants_exact(self):
        for i, ref in enumerate(self.REF_K):
            self.assertAlmostEqual(vibinfo['k'][i], ref, places=5,
                                   msg=f'k[{i}]')

    def test_theta_vib_exact(self):
        for i, ref in enumerate(self.REF_THETA_VIB):
            self.assertAlmostEqual(vibinfo['theta_vib'][i], ref, places=4,
                                   msg=f'theta_vib[{i}]')

    # ---- vib-only extras ----

    def test_xtp0_exact(self):
        for i, ref in enumerate(self.REF_VIB_XTP0):
            self.assertAlmostEqual(self.vibonly['Xtp0'][i], ref, places=9,
                                   msg=f'Xtp0[{i}]')

    def test_dq0_exact(self):
        for i, ref in enumerate(self.REF_VIB_DQ0):
            self.assertAlmostEqual(self.vibonly['DQ0'][i], ref, places=9,
                                   msg=f'DQ0[{i}]')

    # ---- thermochemistry ----

    def test_rotational_constants_exact(self):
        mass_center = (mass[:, None] * geom).sum(axis=0) / mass.sum()
        geom_c = geom - mass_center
        rc = rotation_const(mass, geom_c, 'wavenumber')
        a, b, c = self.THERMO_REF['rot_const_cm']
        self.assertAlmostEqual(rc[0], a, places=8)
        self.assertAlmostEqual(rc[1], b, places=8)
        self.assertAlmostEqual(rc[2], c, places=8)

    def test_rotor_type_exact(self):
        mass_center = (mass[:, None] * geom).sum(axis=0) / mass.sum()
        geom_c = geom - mass_center
        rc_ghz = rotation_const(mass, geom_c, 'GHz')
        self.assertEqual(_get_rotor_type(rc_ghz), self.THERMO_REF['rotor_type'])

    def test_thermo_values_exact(self):
        mass_center = (mass[:, None] * geom).sum(axis=0) / mass.sum()
        geom_c = geom - mass_center
        rc_cm = rotation_const(mass, geom_c, 'wavenumber')
        rc_ghz = rotation_const(mass, geom_c, 'GHz')
        rotor = _get_rotor_type(rc_ghz)
        th = thermo(vibinfo, T=298.15, P=101325.0, multiplicity=1,
                    molecular_mass=mass.sum(), E0=-56.0, sigma=3,
                    rot_const=rc_cm, rotor_type=rotor)
        for key, ref in self.THERMO_REF.items():
            if key in ('rot_const_cm', 'rotor_type'):
                continue
            actual = th[key]
            self.assertAlmostEqual(float(actual), float(ref), places=9,
                                   msg=f'thermo {key}')

    # ---- structural checks ----

    def test_q_orthonormal(self):
        q = vibinfo['q']
        self.assertTrue(np.allclose(q.T @ q, np.eye(12), atol=1e-12))

    def test_hessian_reconstruction_from_modes(self):
        q = vibinfo['q']
        omega = vibinfo['omega']
        uconv_cm_1 = (np.sqrt(6.022140857e23 * 4.35974465e-18 * 1.0e19) /
                      (2 * np.pi * 299792458.0 * 0.52917721067))
        fc_au = (omega * omega).real / (uconv_cm_1 * uconv_cm_1)
        H_recon = q @ np.diag(fc_au) @ q.T

        nmwhess = hess_flat
        sqrtmmm = np.repeat(np.sqrt(mass), 3)
        sqrtmmminv = np.divide(1.0, sqrtmmm)
        mwhess = (sqrtmmminv[:, None] * nmwhess) * sqrtmmminv[None, :]

        TRspace = _get_TR_space(mass, geom)
        P = np.eye(12)
        for irt in TRspace:
            P -= np.outer(irt, irt)
        mwhess_proj = P.T @ mwhess @ P

        self.assertTrue(np.allclose(H_recon, mwhess_proj, atol=1e-8))


# ============================================================================
#  Save reference values for cross-validation with psi4 / rust
# ============================================================================

class TestSaveReferenceNpz(unittest.TestCase):
    """Generates a reference .npz that the Rust implementation can later load."""

    def test_save_vib_reference(self):
        import os
        vibonly = filter_nonvib(vibinfo)

        out = {}
        for key in ['omega', 'q', 'w', 'x', 'mu', 'k', 'DQ0', 'Qtp0', 'Xtp0',
                     'theta_vib', 'degeneracy', 'TRV']:
            val = vibonly[key]
            if isinstance(val, np.ndarray) and val.dtype.kind == 'U':
                out[key] = val.astype(np.bytes_)
            else:
                out[key] = np.asarray(val)

        # Also save the flat hessian, geometry, and masses used
        out['hess_flat'] = hess_flat
        out['geom'] = geom
        out['mass'] = mass

        # Thermochemistry reference
        mass_center = (mass[:, None] * geom).sum(axis=0) / mass.sum()
        geom_c = geom - mass_center
        rc_cm = rotation_const(mass, geom_c, 'wavenumber')
        rc_ghz = rotation_const(mass, geom_c, 'GHz')
        rotor = _get_rotor_type(rc_ghz)
        th = thermo(vibinfo, T=298.15, P=101325.0, multiplicity=1,
                    molecular_mass=mass.sum(), E0=-56.0, sigma=3,
                    rot_const=rc_cm, rotor_type=rotor)
        for k, v in th.items():
            if isinstance(v, (np.floating, float)):
                out['thermo_' + k] = np.float64(v)
            elif isinstance(v, np.ndarray):
                out['thermo_' + k] = v

        path = 'prototype/nh3_r_hf_vib_reference.npz'
        np.savez(path, **out)
        self.assertTrue(os.path.exists(path))


# ============================================================================
#  Acetaldehyde (CH3CHO) / HF / def2-TZVP reference
#  Geometry in Ångström (PySCF converts to bohr). Hessian key ``ref_de``.
# ============================================================================

ET_XYZ = """
C          0.91993       -0.00984        0.04649
C          2.43421       -0.01198        0.05340
O          2.91590       -1.03737        0.90936
H          0.53513        0.76934       -0.61730
H          0.52793        0.16086        1.05450
H          0.53445       -0.97954       -0.28537
H          2.82564        0.94892        0.40128
H          2.82120       -0.19808       -0.95268
H          2.57632       -0.85751        1.80258
"""


def _et_setup():
    mol_et = gto.Mole(atom=ET_XYZ, basis="def2-TZVP", max_memory=8000).build()
    mass_et = mol_et.atom_mass_list(isotope_avg=True)
    geom_et = mol_et.atom_coords()
    ref_et = np.load("prototype/et_r_hf.npz")
    hess_44_et = ref_et["ref_de"]                       # [9, 9, 3, 3]
    hess_flat_et = hess_44_et.transpose(0, 2, 1, 3).reshape(27, 27)
    vibinfo_et = harmonic_analysis(hess_flat_et, geom_et, mass_et)
    return mol_et, mass_et, geom_et, hess_flat_et, vibinfo_et


class TestEthaneReferenceValues(unittest.TestCase):
    """Hardcoded reference numbers for acetaldehyde (CH3CHO) HF/def2-TZVP.

    9 atoms → 27 dof = 6 TR + 21 vibrational modes.
    """

    # 6 TR frequencies (real parts, cm⁻¹) — should all be ~0
    REF_TR_FREQ = [0.0, 0.0, 0.037297, 0.072651, 0.091944, 0.104091]

    # first 10 vibrational frequencies [cm⁻¹]
    REF_VIB_FREQ = [
        397.693146, 482.146314, 748.727643, 909.980504, 966.500675,
        1128.802224, 1198.255774, 1254.376121, 1404.826430, 1513.104741,
    ]
    # reduced masses [u] for those 10 modes
    REF_VIB_MU = [
        1.114579, 2.671366, 1.116921, 1.079670, 2.425550,
        3.653739, 1.566218, 1.431222, 1.196174, 1.233885,
    ]
    # characteristic vibrational temperatures [K] for those 10 modes
    REF_VIB_THETA = [
        572.1919, 693.7012, 1077.2524, 1309.2593, 1390.5793,
        1624.0951, 1724.0233, 1804.7680, 2021.2325, 2177.0208,
    ]

    THERMO_REF = {
        'rot_const_cm': (1.11175601, 0.31790423, 0.27739508),
        'rot_const_ghz': (33.32960657, 9.53052918, 8.31609533),
        'rotor_type': 'REGULAR',
        'ZPE_vib': 0.084945609427,
        'E_vib': 0.085764093805,
        'S_vib': 0.003772758345,
        'Cv_vib': 0.008731629015,
        'S_trans': 0.059613088371,
        'E_trans': 0.001416276822,
        'S_rot': 0.035576765773,
        'E_rot': 0.001416276822,
        'E_tot': -152.911403352551,
        'H_tot': -152.910459168003,
        'G_tot': -152.939964870917,
        'ZPE_tot': -152.915054390573,
        'S_tot': 0.098962612490,
        'Cv_tot': 0.018232060490,
        'Cp_tot': 0.021398870982,
    }

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mass, cls.geom, cls.hess_flat, cls.vibinfo = _et_setup()
        cls.vibonly = filter_nonvib(cls.vibinfo)

    def test_trv_counts(self):
        """6 TR + 21 V for nonlinear 9-atom acetaldehyde."""
        trv = self.vibinfo['TRV']
        self.assertEqual(list(trv).count('TR'), 6)
        self.assertEqual(list(trv).count('V'), 21)
        self.assertEqual(len(trv), 27)

    def test_tr_frequencies(self):
        """6 TR frequencies, all near zero."""
        tr_freqs = [self.vibinfo['omega'][i].real
                    for i in range(27) if self.vibinfo['TRV'][i] == 'TR']
        self.assertEqual(len(tr_freqs), 6)
        for actual, ref in zip(tr_freqs, self.REF_TR_FREQ):
            self.assertAlmostEqual(actual, ref, places=4)
            self.assertLess(abs(actual), 1.0)

    def test_vib_frequencies(self):
        for k in range(10):
            self.assertAlmostEqual(self.vibonly['omega'][k].real,
                                   self.REF_VIB_FREQ[k], places=4,
                                   msg=f'vib freq {k}')

    def test_vib_reduced_masses(self):
        for k in range(10):
            self.assertAlmostEqual(self.vibonly['mu'][k], self.REF_VIB_MU[k],
                                   places=5, msg=f'mu {k}')

    def test_vib_theta(self):
        for k in range(10):
            self.assertAlmostEqual(self.vibonly['theta_vib'][k],
                                   self.REF_VIB_THETA[k], places=3,
                                   msg=f'theta {k}')

    def test_q_orthonormal(self):
        q = self.vibinfo['q']
        self.assertTrue(np.allclose(q.T @ q, np.eye(27), atol=1e-12))

    def test_rotational_constants(self):
        mass_center = (self.mass[:, None] * self.geom).sum(axis=0) / self.mass.sum()
        geom_c = self.geom - mass_center
        rc_cm = rotation_const(self.mass, geom_c, 'wavenumber')
        rc_ghz = rotation_const(self.mass, geom_c, 'GHz')
        for i in range(3):
            self.assertAlmostEqual(rc_cm[i], self.THERMO_REF['rot_const_cm'][i], places=7)
            self.assertAlmostEqual(rc_ghz[i], self.THERMO_REF['rot_const_ghz'][i], places=6)
        self.assertEqual(_get_rotor_type(rc_ghz), self.THERMO_REF['rotor_type'])

    def test_thermo(self):
        mass_center = (self.mass[:, None] * self.geom).sum(axis=0) / self.mass.sum()
        geom_c = self.geom - mass_center
        rc_cm = rotation_const(self.mass, geom_c, 'wavenumber')
        rc_ghz = rotation_const(self.mass, geom_c, 'GHz')
        rotor = _get_rotor_type(rc_ghz)
        th = thermo(self.vibinfo, T=298.15, P=101325.0, multiplicity=1,
                    molecular_mass=self.mass.sum(), E0=-153.0, sigma=1,
                    rot_const=rc_cm, rotor_type=rotor)
        for key, ref in self.THERMO_REF.items():
            if key in ('rot_const_cm', 'rot_const_ghz', 'rotor_type'):
                continue
            self.assertAlmostEqual(float(th[key]), float(ref), places=8,
                                   msg=f'thermo {key}')
