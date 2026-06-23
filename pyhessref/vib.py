"""Vibrational (harmonic) analysis module.

Closely follows Psi4's ``psi4/driver/qcdb/vib.py``, but:
- No ``Datum`` — plain dicts of numpy arrays.
- No irrep/symmetry (C₁ only).
- No psi4 runtime dependencies; numpy + scipy + stdlib only.

Physical constants are from CODATA2014 (via qcelemental).
"""

import math
import numpy as np

__all__ = [
    "harmonic_analysis",
    "thermo",
    "filter_nonvib",
    "filter_omega_to_real",
    "print_vibs",
    "print_molden_vibs",
    "_get_TR_space",
]

# ---------------------------------------------------------------------------
# Physical constants (CODATA2014, same source as qcelemental / Psi4)
# ---------------------------------------------------------------------------
_na = 6.022140857e23               # Avogadro constant
_hartree2J = 4.35974465e-18         # hartree → J
_c = 299792458.0                    # speed of light [m/s]
_bohr2angstroms = 0.52917721067     # a₀ → Å
_h = 6.62607004e-34                 # Planck constant [J·s]
_kb = 1.38064852e-23                # Boltzmann constant [J/K]
_R = 8.3144598                      # gas constant [J/(mol·K)]
_hartree2kJmol = 2625.4996382852164 # hartree → kJ/mol
_amu2kg = 1.66053904e-27            # u → kg
_hartree2wavenumbers = 219474.6313702  # hartree → cm⁻¹
_hartree2kcalmol = 627.5094737775374   # hartree → kcal/mol
_electron_mass_u = 5.48579909065e-4    # electron mass [u]
_fine_structure = 0.0072973525664      # fine-structure constant α
_a0 = 5.2917721067e-11                 # atomic unit of length [m]
_dipmom_au2debye = 2.5417464157449032  # e·a₀ → Debye

# tolerance for detecting nearly-linear geometries in _get_TR_space
LINEAR_A_TOL = 1.0E-2

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _phase_cols_to_max_element(arr, tol=1.e-2):
    """Return copy of *arr* where each column is scaled so that its
    element of maximum absolute value is positive.
    """
    arr2 = np.copy(arr)
    for v in range(arr2.shape[1]):
        vextreme = np.max(np.abs(arr2[:, v]))
        # find the first index whose fabs equals that value, within tol
        mask = (vextreme - np.abs(arr2[:, v])) < tol
        iextreme = np.argmax(mask)  # first True
        sign = np.sign(arr2[iextreme, v])
        if sign == -1.:
            arr2[:, v] *= -1.
    return arr2


def _check_degen_modes(arr, freq):
    """Sort degenerate columns of *arr* (eigenvectors) identified by *freq*.

    Within each near-degenerate subspace (frequencies equal to 0.1 cm⁻¹),
    columns are lexsorted by (value of extreme element, index of extreme element)
    to obtain a deterministic ordering.
    """
    rounded = np.around(freq, 1)
    dfreq, didx, dinv, dcts = np.unique(rounded, return_index=True,
                                        return_inverse=True, return_counts=True)
    idx_max_elem = np.argmax(np.around(arr, 2), axis=0)
    max_elem = np.amax(np.around(arr, 2), axis=0)
    idx_reordering = np.empty_like(idx_max_elem)

    for idegen, istart in enumerate(didx):
        degree = dcts[idegen]
        idx_sort = np.lexsort(
            (idx_max_elem[istart:istart + degree],
             max_elem[istart:istart + degree]))
        idx_reordering[istart:istart + degree] = (
            np.arange(istart, istart + degree)[idx_sort])

    return arr[:, idx_reordering]


def _check_rank_degen_modes(cvecs, cfreq, evecs, difftol=1e-4, svdtol=1e-6):
    """Check that *cvecs* and *evecs* span the same space within
    each degenerate subspace (from *cfreq*).  Returns *True* if ok.
    """
    rounded = np.around(cfreq, 1)
    dfreq, didx, dinv, dcts = np.unique(rounded, return_index=True,
                                        return_inverse=True, return_counts=True)
    ok = True
    for idegen, istart in enumerate(didx):
        degree = dcts[idegen]
        cv = cvecs[:, istart:istart + degree]
        ev = evecs[:, istart:istart + degree]
        if degree == 1:
            ok = ok and np.allclose(ev, cv, atol=difftol)
        else:
            cevecs = np.concatenate((cv, ev), axis=1)
            rank_cv = np.linalg.matrix_rank(cv)
            rank_ev = np.linalg.matrix_rank(ev)
            CE = np.linalg.svd(cevecs, compute_uv=False)
            rank_ce = np.count_nonzero(CE > svdtol)
            ok = ok and (rank_cv == rank_ev == rank_ce)
    return ok


def _vec_in_space(vec, space, tol=1.0e-4):
    """Return *True* if *vec* lies in the subspace spanned by *space* (rows)."""
    merged = np.vstack((space, vec))
    _, s, _ = np.linalg.svd(merged)
    return s[-1] < tol


def _format_omega(omega, decimals):
    """Convert complex frequencies to ``str``; imaginary modes get 'i' suffix."""
    out = []
    for fr in omega:
        if fr.imag > fr.real:
            out.append("{:.{prec}f}i".format(fr.imag, prec=decimals))
        else:
            out.append("{:.{prec}f}".format(fr.real, prec=decimals))
    return np.array(out)


# ---------------------------------------------------------------------------
# Translation / rotation space
# ---------------------------------------------------------------------------

def _get_TR_space(mass, geom, space='TR', tol=None):
    """Idealized translation + rotation basis vectors.

    Parameters
    ----------
    mass : (nat,) ndarray
        Atomic masses [u].
    geom : (nat, 3) ndarray
        Cartesian geometry [a₀].
    space : str
        ``'T'``, ``'R'``, or ``'TR'``.
    tol : float or None
        SVD tolerance for linear-dependence detection.

    Returns
    -------
    TRindep : (nrt, 3*nat) ndarray
        Orthonormal basis vectors (rows).
    """
    sqrtmmm = np.repeat(np.sqrt(mass), 3)
    xxx = np.repeat(geom[:, 0], 3)
    yyy = np.repeat(geom[:, 1], 3)
    zzz = np.repeat(geom[:, 2], 3)

    z = np.zeros_like(mass)
    i = np.ones_like(mass)
    ux = np.ravel([i, z, z], order='F')
    uy = np.ravel([z, i, z], order='F')
    uz = np.ravel([z, z, i], order='F')

    # translation
    T1 = sqrtmmm * ux
    T2 = sqrtmmm * uy
    T3 = sqrtmmm * uz
    # rotation
    R4 = sqrtmmm * (yyy * uz - zzz * uy)
    R5 = sqrtmmm * (zzz * ux - xxx * uz)
    R6 = sqrtmmm * (xxx * uy - yyy * ux)

    TRvecs = []
    if 'T' in space:
        TRvecs.extend([T1, T2, T3])
    if 'R' in space:
        TRvecs.extend([R4, R5, R6])
    if not TRvecs:
        ZZ = np.zeros_like(T1)
        TRvecs = [ZZ]

    TR = np.vstack(TRvecs)

    # orthogonalise via SVD
    def _orth(A, tol=tol):
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        M, N = A.shape
        if tol is None:
            tol_eff = max(M, N) * np.amax(s) * np.finfo(float).eps
        else:
            tol_eff = tol
        num = np.sum(s > tol_eff, dtype=int)
        return u[:, :num]

    TRindep = _orth(TR.T).T
    return TRindep


# ---------------------------------------------------------------------------
# Rotation constants and rotor type
# ---------------------------------------------------------------------------

def rotation_const(mass, atom_coords, unit='GHz'):
    """Rotational constants [*unit*].

    Parameters
    ----------
    mass : (nat,) ndarray
        Atomic masses [u].
    atom_coords : (nat, 3) ndarray
        **Mass-centred** Cartesian geometry [a₀].
    unit : str
        ``'GHz'`` or ``'wavenumber'``.

    Returns
    -------
    e : (3,) ndarray
        Rotational constants sorted (A ≥ B ≥ C, but largest inertia → smallest).
    """
    r = atom_coords
    # I_rs = Σ m_a (δ_rs |r_a|² - r_{a,r} r_{a,s})
    # compute Σ m_a r_{a,r} r_{a,s} first, then subtract from trace
    im = (mass[:, None, None] * r[:, :, None] * r[:, None, :]).sum(axis=0)
    im = np.eye(3) * im.trace() - im
    e = np.sort(np.linalg.eigvalsh(im))
    e[abs(e) < 1e-9] = 0.0

    unit_im = 1.660539040427164e-27 * (5.2917721092e-11)**2  # u·a₀² → kg·m²
    unit_hz = 1.0545718001391127e-34 / (4 * np.pi * unit_im)  # ℏ/(4πI)
    with np.errstate(divide='ignore'):
        if unit.lower() == 'ghz':
            e = unit_hz / e * 1e-9
        elif unit.lower() == 'wavenumber':
            e = unit_hz / e / 299792458.0 * 1e-2
        else:
            raise ValueError('Unsupported unit ' + unit)
    return e


def _get_rotor_type(rot_const):
    """Classify rotor type from rotational constants [GHz].

    Returns ``'ATOM'``, ``'LINEAR'``, or ``'REGULAR'``.
    """
    if np.all(rot_const > 1e8):
        return 'ATOM'
    elif rot_const[0] > 1e8 and (rot_const[1] - rot_const[2] < 1e-3):
        return 'LINEAR'
    else:
        return 'REGULAR'


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def harmonic_analysis(hess, geom, mass, dipder=None,
                      project_trans=True, project_rot=True):
    """Extract frequencies, normal modes, and other properties from an
    electronic Hessian.

    Parameters
    ----------
    hess : (3*nat, 3*nat) ndarray
        Non-mass-weighted Cartesian Hessian [E_h / a₀²].
    geom : (nat, 3) ndarray
        Cartesian geometry [a₀].
    mass : (nat,) ndarray
        Atomic masses [u].
    dipder : (3, 3*nat) ndarray or None
        Dipole derivatives [E_h·a₀ / u] (optional).
    project_trans : bool
        Project out idealized translations.
    project_rot : bool
        Project out idealized rotations.

    Returns
    -------
    vibinfo : dict
        Keys with plain data arrays:
        - ``omega`` : (ndof,) complex — frequencies [cm⁻¹]
        - ``q`` : (ndof, ndof) — mass-weighted normal modes [a₀·u^½]
        - ``w`` : (ndof, ndof) — un-mass-weighted normal modes [a₀]
        - ``x`` : (ndof, ndof) — normalized un-mass-weighted modes [a₀]
        - ``degeneracy`` : (ndof,) int
        - ``TRV`` : (ndof,) str — 'TR' or 'V' or '-'
        - ``mu`` : (ndof,) — reduced mass [u]
        - ``k`` : (ndof,) — force constant [mDyne/Å]
        - ``DQ0`` : (ndof,) — RMS deviation v=0 [a₀·u^½]
        - ``Qtp0`` : (ndof,) — turning point v=0 (mass-weighted) [a₀·u^½]
        - ``Xtp0`` : (ndof,) — turning point v=0 (Cartesian) [a₀]
        - ``theta_vib`` : (ndof,) — characteristic vibrational temperature [K]
        - ``IR_intensity`` : (ndof,) — infrared intensity [km/mol] (if *dipder* given)
    """
    # validate dimensions
    nat = len(mass)
    ndof = 3 * nat
    if not (mass.shape[0] == geom.shape[0] == hess.shape[0] // 3 == hess.shape[1] // 3
            and geom.shape[1] == 3):
        raise ValueError(
            f"Dimension mismatch: mass {mass.shape}, geom {geom.shape}, hess {hess.shape}")

    vibinfo = {}

    nmwhess = np.asarray(hess, dtype=np.float64).reshape(ndof, ndof)

    # expected number of translation + rotation dof
    if nat == 1:
        nrt_expected = 3
    elif np.linalg.matrix_rank(geom) == 1:
        nrt_expected = 5
    else:
        nrt_expected = 6

    # --------------- translation / rotation projector ---------------
    space = ('T' if project_trans else '') + ('R' if project_rot else '')
    TRspace = _get_TR_space(mass, geom, space=space, tol=LINEAR_A_TOL)
    nrt = TRspace.shape[0]

    # projector  P = I - Σ|tr⟩⟨tr|
    P = np.identity(ndof)
    for irt in TRspace:
        P -= np.outer(irt, irt)

    # --------------- mass-weight & solve ---------------
    sqrtmmm = np.repeat(np.sqrt(mass), 3)
    sqrtmmminv = np.divide(1.0, sqrtmmm)
    # mwhess[i,j] = hess[i,j] / sqrt(m_i * m_j)
    mwhess = (sqrtmmminv[:, None] * nmwhess) * sqrtmmminv[None, :]

    # pre-projection estimate (for diagnostics)
    pre_fc_au = np.linalg.eigvalsh(mwhess)
    pre_fc_au = pre_fc_au[np.argsort(pre_fc_au)]
    uconv_cm_1 = (np.sqrt(_na * _hartree2J * 1.0e19) /
                  (2 * np.pi * _c * _bohr2angstroms))
    pre_freq = np.lib.scimath.sqrt(pre_fc_au) * uconv_cm_1

    # project & diagonalise
    mwhess_proj = np.dot(P.T, mwhess).dot(P)
    force_constant_au, qL = np.linalg.eigh(mwhess_proj)

    # sort ascending (steepest downhill → steepest uphill)
    idx_sort = np.argsort(force_constant_au)
    force_constant_au = force_constant_au[idx_sort]
    qL = qL[:, idx_sort]

    # phase convention: extreme element positive
    qL = _phase_cols_to_max_element(qL)
    vibinfo['q'] = qL  # (ndof, ndof)

    # --------------- frequencies ---------------
    frequency_cm_1 = np.lib.scimath.sqrt(force_constant_au) * uconv_cm_1
    vibinfo['omega'] = frequency_cm_1

    # --------------- degeneracies ---------------
    ufreq, uinv, ucts = np.unique(np.around(frequency_cm_1, 1),
                                  return_inverse=True, return_counts=True)
    vibinfo['degeneracy'] = ucts[uinv]

    # --------------- TR / V classification (no irrep) ---------------
    trv = []
    for idx_vib, vib in enumerate(frequency_cm_1):
        if _vec_in_space(qL[:, idx_vib], TRspace, 1.0e-4):
            trv.append('TR')
        else:
            if np.linalg.norm(vib) < 1.e-3:
                trv.append('-')
            else:
                trv.append('V')
    vibinfo['TRV'] = np.array(trv)

    # --------------- conversion factors ---------------
    # force constant: [cm⁻¹] → [mDyne/Å]  LAB II.16
    uconv_mdyne_a = (0.1 * (2 * np.pi * _c)**2) / _na

    # turning point helper
    uconv_S = np.sqrt((_c * (2 * np.pi * _bohr2angstroms)**2) /
                      (_h * _na * 1.0e21))

    # --------------- normal modes & reduced mass ---------------
    # un-mass-weighted: w = m^{-½}·q   LAB II.14
    # wL[a,i] = qL[a,i] / sqrt(m_a)
    wL = sqrtmmminv[:, None] * qL
    vibinfo['w'] = wL

    reduced_mass_u = np.divide(1.0, np.linalg.norm(wL, axis=0)**2)
    vibinfo['mu'] = reduced_mass_u

    # normalized un-mass-weighted: x = √μ · w   LAB II.15
    xL = np.sqrt(reduced_mass_u) * wL
    vibinfo['x'] = xL

    # --------------- IR intensities ---------------
    if dipder is not None and dipder.size > 0:
        uconv_kmmol = (_na * np.pi * 1.e-3 * _electron_mass_u *
                       _fine_structure**2 * _a0 / 3)
        qDD = dipder.dot(wL)
        ir_intensity = np.zeros(qDD.shape[1])
        for i in range(qDD.shape[1]):
            ir_intensity[i] = qDD[:, i].dot(qDD[:, i])
        ir_intensity_kmmol = ir_intensity * uconv_kmmol
        vibinfo['IR_intensity'] = ir_intensity_kmmol

    # --------------- force constants ---------------
    force_constant_mdyne_a = (reduced_mass_u *
                              (frequency_cm_1 * frequency_cm_1).real *
                              uconv_mdyne_a)
    vibinfo['k'] = force_constant_mdyne_a

    # --------------- turning points (v=0) ---------------
    nu = 0
    tp_rnc = np.sqrt(2.0 * nu + 1.0)
    omega_real = frequency_cm_1.real

    with np.errstate(divide='ignore'):
        Qtp0 = tp_rnc / (np.sqrt(omega_real) * uconv_S)
    Qtp0[Qtp0 == np.inf] = 0.0
    vibinfo['Qtp0'] = Qtp0

    with np.errstate(divide='ignore'):
        Xtp0 = tp_rnc / (np.sqrt(omega_real * reduced_mass_u) * uconv_S)
    Xtp0[Xtp0 == np.inf] = 0.0
    vibinfo['Xtp0'] = Xtp0

    # RMS deviation v=0
    vibinfo['DQ0'] = Qtp0 / np.sqrt(2.0)

    # --------------- characteristic vibrational temperature ---------------
    uconv_K = 100 * _h * _c / _kb
    vibinfo['theta_vib'] = omega_real * uconv_K

    return vibinfo


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def filter_nonvib(vibinfo, remove=None):
    """Return a copy of *vibinfo* with specified modes removed.

    Parameters
    ----------
    vibinfo : dict
        Output of :func:`harmonic_analysis`.
    remove : list of int or None
        0-indexed mode indices to remove.  If *None*, all non-'V' modes
        (from ``vibinfo['TRV']``) are removed.

    Returns
    -------
    dict
        Filtered copy.
    """
    if remove is None:
        remove = [idx for idx, tag in enumerate(vibinfo['TRV']) if tag != 'V']
    work = {}
    for key, arr in vibinfo.items():
        if key in ('q', 'w', 'x'):
            axis = 1
        else:
            axis = 0
        work[key] = np.delete(arr, remove, axis=axis)
    return work


def filter_omega_to_real(omega):
    """Convert complex *omega* to real, representing imaginary modes as
    negative real numbers.
    """
    out = []
    for fr in omega:
        if fr.imag > fr.real:
            out.append(-fr.imag)
        else:
            out.append(fr.real)
    return np.array(out)


# ---------------------------------------------------------------------------
# Thermochemistry
# ---------------------------------------------------------------------------

def thermo(vibinfo, T, P, multiplicity, molecular_mass, E0,
           sigma, rot_const, rotor_type=None):
    """Thermochemical analysis from harmonic vibrational output.

    Parameters
    ----------
    vibinfo : dict
        Output of :func:`harmonic_analysis`.
    T : float
        Temperature [K].
    P : float
        Pressure [Pa].
    multiplicity : int
        Spin multiplicity.
    molecular_mass : float
        Total molecular mass [u].
    E0 : float
        Electronic energy at well bottom [E_h].
    sigma : int
        Rotational (external) symmetry number.
    rot_const : (3,) ndarray
        Rotational constants [cm⁻¹].
    rotor_type : str or None
        ``'RT_ATOM'``, ``'RT_LINEAR'``, or *None* (nonlinear).

    Returns
    -------
    therminfo : dict
        All thermochemical components in atomic units plus input conditions.
    """
    sm = {}  # (quantity, term) → float (unitless or [K] before conversion)

    # conditions
    therminfo = {}
    therminfo['E0'] = E0
    therminfo['B'] = rot_const
    therminfo['sigma'] = sigma
    therminfo['T'] = T
    therminfo['P'] = P

    # ---------- electronic ----------
    q_elec = multiplicity
    sm[('S', 'elec')] = math.log(q_elec)

    # ---------- translational ----------
    beta = 1.0 / (_kb * T)
    q_trans = ((2.0 * np.pi * molecular_mass * _amu2kg /
                (beta * _h * _h))**1.5 * _na / (beta * P))
    sm[('S', 'trans')] = 2.5 + math.log(q_trans / _na)
    sm[('Cv', 'trans')] = 1.5
    sm[('Cp', 'trans')] = 2.5
    sm[('E', 'trans')] = 1.5 * T
    sm[('H', 'trans')] = 2.5 * T

    # ---------- rotational ----------
    if rotor_type is None:
        # infer from rot_const [cm⁻¹]; 1 cm⁻¹ = c [Hz] * 1e-9 [GHz] * 1e2 [cm/m]
        # = 29.9792458 GHz
        rot_ghz = rot_const * 29.9792458
        rotor_type = _get_rotor_type(rot_ghz)

    if rotor_type == 'RT_ATOM':
        pass
    elif rotor_type == 'LINEAR':
        q_rot = 1.0 / (beta * sigma * 100 * _c * _h * rot_const[1])
        sm[('S', 'rot')] = 1.0 + math.log(q_rot)
        sm[('Cv', 'rot')] = 1.0
        sm[('Cp', 'rot')] = 1.0
        sm[('E', 'rot')] = T
    else:
        phi_A, phi_B, phi_C = rot_const * 100 * _c * _h / _kb
        q_rot = (math.sqrt(math.pi) * T**1.5 /
                 (sigma * math.sqrt(phi_A * phi_B * phi_C)))
        sm[('S', 'rot')] = 1.5 + math.log(q_rot)
        sm[('Cv', 'rot')] = 1.5
        sm[('Cp', 'rot')] = 1.5
        sm[('E', 'rot')] = 1.5 * T
    sm[('H', 'rot')] = sm.get(('E', 'rot'), 0.0)

    # ---------- vibrational ----------
    vibonly = filter_nonvib(vibinfo)
    omega_vib = vibonly['omega']
    theta_vib = vibonly['theta_vib']

    # exclude imaginary modes
    imag_mask = omega_vib.imag > omega_vib.real
    filtered_theta = theta_vib[~imag_mask]
    rT = filtered_theta / max(1e-14, T)  # reduced temperature

    sm[('S', 'vib')] = np.sum(rT / np.expm1(rT) - np.log(1.0 - np.exp(-rT)))
    sm[('Cv', 'vib')] = np.sum(np.exp(rT) * (rT / np.expm1(rT))**2)
    sm[('Cp', 'vib')] = sm[('Cv', 'vib')]
    sm[('ZPE', 'vib')] = np.sum(rT) * T / 2.0
    sm[('E', 'vib')] = sm[('ZPE', 'vib')] + np.sum(rT * T / np.expm1(rT))
    sm[('H', 'vib')] = sm[('E', 'vib')]

    # ---------- Gibbs ----------
    for term in ['elec', 'trans', 'rot', 'vib']:
        H = sm.get(('H', term), 0.0)
        S = sm.get(('S', term), 0.0)
        sm[('G', term)] = H - T * S

    # ---------- convert to atomic units ----------
    uconv_R_EhK = _R / _hartree2kJmol  # R [Eh/K] = [mEh/K] / 1000

    for term in ['elec', 'trans', 'rot', 'vib']:
        for piece in ['S', 'Cv', 'Cp']:
            key = (piece, term)
            if key in sm:
                sm[key] *= uconv_R_EhK  # [mEh/K]
        for piece in ['ZPE', 'E', 'H', 'G']:
            key = (piece, term)
            if key in sm:
                sm[key] *= uconv_R_EhK * 0.001  # [Eh]

    # ---------- totals ----------
    for piece in ['S', 'Cv', 'Cp']:
        sm[(piece, 'tot')] = sum(sm.get((piece, t), 0.0)
                                 for t in ['elec', 'trans', 'rot', 'vib'])
    for piece in ['ZPE', 'E', 'H', 'G']:
        sm[(piece, 'corr')] = sum(sm.get((piece, t), 0.0)
                                  for t in ['elec', 'trans', 'rot', 'vib'])
        sm[(piece, 'tot')] = E0 + sm[(piece, 'corr')]

    # package: flat keys like 'S_elec', 'E_tot', etc.
    for (piece, term), val in sm.items():
        therminfo[f'{piece}_{term}'] = val

    return therminfo


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_vibs(vibinfo, atom_lbl=None, normco='x', shortlong=True,
               groupby=None, prec=4, ncprec=None):
    """Pretty-print vibrational analysis results.

    Parameters
    ----------
    vibinfo : dict
        Output of :func:`harmonic_analysis`.
    atom_lbl : list of str or None
        Atomic symbols.  If *None*, integers are used.
    normco : str
        ``'q'`` (mass-weighted), ``'w'`` (un-mass-weighted), or ``'x'``
        (normalized un-mass-weighted, default).
    shortlong : bool
        *True* for ``(nat, 3)`` layout, *False* for ``(3*nat, 1)``.
    groupby : int or None
        Modes per row.  Default 3 (short) or 6 (long); ``-1`` for all.
    prec : int
        Decimal places for scalar properties.
    ncprec : int or None
        Decimal places for normal coordinates.

    Returns
    -------
    str
        Formatted string.
    """
    import itertools

    if normco not in ('q', 'w', 'x'):
        raise ValueError("normco must be 'q', 'w', or 'x'")

    nat = len(vibinfo['q'][:, 0]) // 3
    if atom_lbl is None:
        atom_lbl = [''] * nat

    active = [idx for idx, trv in enumerate(vibinfo['TRV']) if trv == 'V']

    presp = 2
    colsp = 2
    if shortlong:
        groupby = groupby if groupby else 3
        ncprec = ncprec if ncprec else 2
        width = (ncprec + 4) * 3
        prewidth = 24
    else:
        groupby = groupby if groupby else 6
        ncprec = ncprec if ncprec else 4
        width = ncprec + 8
        prewidth = 24
    if groupby == -1:
        groupby = len(active)

    def _br(s):
        return '[' + s + ']'

    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    omega_str = _format_omega(vibinfo['omega'], decimals=prec)

    lines = []
    for row in grouper(active, groupby):
        # header: Vibration
        lines.append('{:>{presp}}{:{prewidth}}'.format(
            '', 'Vibration', prewidth=prewidth, presp=presp) +
            ''.join("{:^{width}d}{:{colsp}}".format(
                vib + 1, '', width=width, colsp=colsp)
                for vib in row if vib is not None))

        # freq
        lines.append('{:>{presp}}{:{prewidth}}'.format(
            '', 'Freq [cm^-1]', prewidth=prewidth, presp=presp) +
            ''.join("{:^{width}}  ".format(omega_str[vib], width=width)
                    for vib in row if vib is not None))

        # reduced mass
        lines.append('{:>{presp}}{:{prewidth}}'.format(
            '', 'Reduced mass [u]', prewidth=prewidth, presp=presp) +
            ''.join("{:^{width}.{prec}f}{:{colsp}}".format(
                vibinfo['mu'][vib], '', width=width, prec=prec, colsp=colsp)
                for vib in row if vib is not None))

        # force const
        lines.append('{:>{presp}}{:{prewidth}}'.format(
            '', 'Force const [mDyne/A]', prewidth=prewidth, presp=presp) +
            ''.join("{:^{width}.{prec}f}{:{colsp}}".format(
                vibinfo['k'][vib], '', width=width, prec=prec, colsp=colsp)
                for vib in row if vib is not None))

        # turning point v=0
        lines.append('{:>{presp}}{:{prewidth}}'.format(
            '', 'Turning point v=0 [a0]', prewidth=prewidth, presp=presp) +
            ''.join("{:^{width}.{prec}f}{:{colsp}}".format(
                vibinfo['Xtp0'][vib], '', width=width, prec=prec, colsp=colsp)
                for vib in row if vib is not None))

        # RMS dev v=0
        lines.append('{:>{presp}}{:{prewidth}}'.format(
            '', 'RMS dev v=0 [a0 u^1/2]', prewidth=prewidth, presp=presp) +
            ''.join("{:^{width}.{prec}f}{:{colsp}}".format(
                vibinfo['DQ0'][vib], '', width=width, prec=prec, colsp=colsp)
                for vib in row if vib is not None))

        # IR intensity
        if 'IR_intensity' in vibinfo:
            lines.append('{:>{presp}}{:{prewidth}}'.format(
                '', 'IR activ [km/mol]', prewidth=prewidth, presp=presp) +
                ''.join("{:^{width}.{prec}f}{:{colsp}}".format(
                    vibinfo['IR_intensity'][vib], '', width=width,
                    prec=prec, colsp=colsp)
                    for vib in row if vib is not None))

        # characteristic temperature
        if 'theta_vib' in vibinfo:
            lines.append('{:>{presp}}{:{prewidth}}'.format(
                '', 'Char temp [K]', prewidth=prewidth, presp=presp) +
                ''.join("{:^{width}.{prec}f}{:{colsp}}".format(
                    vibinfo['theta_vib'][vib], '', width=width,
                    prec=prec, colsp=colsp)
                    for vib in row if vib is not None))

        # separator
        lines.append(' ' * presp + '-' * (prewidth + groupby * (width + colsp) - colsp))

        # normal coordinate values
        if shortlong:
            for at in range(nat):
                line = '{:{presp}}{:5d}   {:{width}}'.format(
                    '', at + 1, atom_lbl[at],
                    width=prewidth - 8, presp=presp)
                for vib in row:
                    if vib is None:
                        break
                    vals = vibinfo[normco][:, vib].reshape(nat, 3)[at]
                    line += (("{:^{w}.{prec}f}" * 3).format(
                        *vals, w=int(width / 3), prec=ncprec) +
                        '{:{colsp}}'.format('', colsp=colsp))
                lines.append(line)
        else:
            for at in range(nat):
                for xyz in range(3):
                    line = '{:{presp}}{:5d}    {}    {:{width}}'.format(
                        '', at + 1, 'XYZ'[xyz], atom_lbl[at],
                        width=prewidth - 14, presp=presp)
                    for vib in row:
                        if vib is None:
                            break
                        val = vibinfo[normco][3 * at + xyz, vib]
                        line += ('{:^{width}.{prec}f}'.format(
                            val, width=width, prec=ncprec) +
                            '{:{colsp}}'.format('', colsp=colsp))
                    lines.append(line)

    return '\n'.join(lines)


def print_molden_vibs(vibinfo, atom_symbol, geom, standalone=True):
    """Format vibrational analysis for Molden.

    Parameters
    ----------
    vibinfo : dict
        Output of :func:`harmonic_analysis`.
    atom_symbol : (nat,) list of str
        Element symbols.
    geom : (nat, 3) ndarray
        Cartesian geometry [a₀].
    standalone : bool
        Prepend ``[Molden Format]`` header.

    Returns
    -------
    str
        Molden-format text with ``[FREQ]``, ``[FR-COORD]``,
        and ``[FR-NORM-COORD]`` sections.
    """
    nat = len(vibinfo['q'][:, 0]) // 3
    active = [idx for idx, trv in enumerate(vibinfo['TRV']) if trv == 'V']

    lines = []
    if standalone:
        lines.append('[Molden Format]')

    # [FREQ] section
    lines.append('\n[FREQ]')
    for vib in active:
        if vibinfo['omega'][vib].imag > vibinfo['omega'][vib].real:
            freq = -vibinfo['omega'][vib].imag
        else:
            freq = vibinfo['omega'][vib].real
        lines.append('   {:20.10f}'.format(freq))

    # [FR-COORD] section
    lines.append('\n[FR-COORD]')
    for at in range(nat):
        lines.append(('{:3}' + '{:20.10f}' * 3).format(
            atom_symbol[at], *geom[at]))

    # [FR-NORM-COORD] section
    lines.append('\n[FR-NORM-COORD]')
    for idx_v, vib in enumerate(active):
        lines.append('vibration {}'.format(idx_v + 1))
        for at in range(nat):
            disp = vibinfo['x'][:, vib].reshape(nat, 3)[at].real
            lines.append(('   ' + '{:20.10f}' * 3).format(*disp))

    return '\n'.join(lines)
