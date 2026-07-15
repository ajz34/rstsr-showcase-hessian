"""Reference (Python) implementation of the Becke multicenter partitioning.

This is the *reference* implementation that mirrors the analytical derivation worked out in the
``prototype/10-1`` (deriv 0), ``10-2`` (deriv 1) and ``10-3`` (deriv 2) notebooks.  It favours
clarity over micro-efficiency: the per-grid computation is fully vectorised with NumPy einsums,
and only the atomic-coordinate loop (which cannot be vectorised without a large 6D tensor) is left
explicit.  Grids are processed in batches of ``nbatch`` purely to bound the size of the
``ddR_log_P`` intermediate; the result is independent of ``nbatch``.

Only the partition weights ``w``/``dw``/``ddw`` are returned.  The contraction machinery that
exists in the Rust port (``BeckeDerivArg``) is intentionally not implemented here; ``deriv_arg`` is
accepted only for API parity and currently ignored.

Reference
---------
A. D. Becke, "A multicenter numerical integration scheme for polyatomic molecules",
J. Chem. Phys. 88, 2547 (1988), doi:10.1063/1.454033.
"""

import numpy as np

# Switch cutoff: grid points where ``|s_{MN}|`` is below this are treated as lying on the
# switch-function zero, where the *log* derivative is ill-defined.  There the divisor is floored
# to 1.0 (NOT to a small ``INVTOL`` constant) so that ``P * dmu_log_s`` evaluates to ``P * dmu_s``
# rather than ``P * dmu_s / 1e-14``; the latter blows up the 2nd-order accumulation.  cf the
# ``10-3``/``10-5`` notebooks and the ``becke_rsprep_deriv2_port`` memory note.
S_SAFE_TOL = 1e-14


def _becke_s_derivs(mu, a, hardness):
    """Becke switch ``s(mu)`` together with its 1st/2nd ``mu``-derivatives.

    The switch function is

    .. math::
        \\nu = \\mu + a (1 - \\mu^2)                              \\quad\\text{(eq A2)} \\\\
        f = p^{\\circ\\mathrm{hardness}}(\\nu),\\;
        p(x) = \\tfrac32 x - \\tfrac12 x^3                       \\quad\\text{(eq 19, 20)} \\\\
        s = \\tfrac12 (1 - f)                                      \\quad\\text{(eq 21)}

    where ``mu`` and ``a`` broadcast against each other (typically ``(M, N, ngrids)`` and
    ``(M, N, 1)``).  Derivatives use the chain-rule recurrence with ``p'(x) = 3/2 (1 - x^2)`` and
    ``p''(x) = -3 x``; for ``hardness = 3`` this reduces to the closed forms of the ``10-3``
    notebook.

    Returns ``(s, ds/dmu, d2s/dmu2)``, all with the broadcast shape of ``mu``.
    """
    nu = mu + a * (1.0 - mu * mu)  # eq (A2)
    dnu = 1.0 - 2.0 * a * mu  # d nu / d mu
    ddnu = -2.0 * a  # d2 nu / d mu2 (constant in mu)

    f = nu
    df = np.ones_like(mu)
    ddf = np.zeros_like(mu)
    for _ in range(hardness):
        g = 1.5 * (1.0 - f * f)  # p'(f)         (uses old f)
        ddf = -3.0 * f * df * df + g * ddf  # f''(nu) recurrence (old f, df, ddf)
        df = g * df  # f'(nu)  recurrence (old df)
        f = 1.5 * f - 0.5 * f * f * f  # f = p(f)          (old f)

    s = 0.5 * (1.0 - f)  # eq (21)
    dmu_s = -0.5 * df * dnu  # ds/dmu = -1/2 f'(nu) nu'
    ddmu_s = -0.5 * (ddf * dnu * dnu + df * ddnu)  # d2s/dmu2 = -1/2 [f''(nu)(nu')^2 + f'(nu) nu'']
    return s, dmu_s, ddmu_s


def becke_partition(
    grid_coords: np.ndarray,
    atm_coords: np.ndarray,
    atm_indices: np.ndarray,
    quadrature_weights: np.ndarray,
    adjustment_factor: np.ndarray,
    hardness: int,
    nbatch: int,
    deriv: int,
    deriv_arg: dict,
):
    """Becke multicenter partitioning and its atomic-coordinate derivatives.

    Parameters
    ----------
    grid_coords : ndarray, shape (ngrids, 3)
        Cartesian coordinates of the (already atom-centred and concatenated) grid points.
    atm_coords : ndarray, shape (natm, 3)
        Cartesian coordinates of the atoms.
    atm_indices : ndarray, shape (ngrids,)
        Index of the atom that *generated* each grid point (the Lebedev shell centre).  Values are
        in ``[0, natm)``; entries ``>= natm`` are treated as padding and skipped.
    quadrature_weights : ndarray, shape (ngrids,)
        Original (pre-partition) Lebedev quadrature weights.
    adjustment_factor : ndarray, shape (natm, natm)
        Anti-symmetric radii-adjustment table ``a_{AB}`` (cf Becke 1988 eqs A2-A6).  Indexed
        ``a[A, B]`` together with ``mu_{AB} = (r_A - r_B) / R_{AB}``.
    hardness : int
        Number of switch-function iterations (eq 20).  Most commonly 3.
    nbatch : int
        Grid batch size.  Used only to bound the size of the ``ddR_log_P`` intermediate; the
        returned weights are independent of this value.
    deriv : int
        Derivative order: 0, 1 or 2.
    deriv_arg : dict
        Currently ignored (contraction is not implemented in the reference).  Accepted for API
        parity with the Rust port.

    Returns
    -------
    dict
        ``{"w": w, "dw": dw, "ddw": ddw}`` where

        - ``w``  has shape ``(ngrids,)`` (always returned);
        - ``dw`` has shape ``(natm, 3, ngrids)`` (``None`` when ``deriv < 1``);
        - ``ddw`` has shape ``(natm, 3, natm, 3, ngrids)`` (``None`` when ``deriv < 2``).
    """
    # ----- input normalisation ----- #
    grid_coords = np.asarray(grid_coords, dtype=float)
    atm_coords = np.asarray(atm_coords, dtype=float)
    atm_indices = np.asarray(atm_indices).astype(int)
    wquad = np.asarray(quadrature_weights, dtype=float)
    a = np.asarray(adjustment_factor, dtype=float)
    natm = atm_coords.shape[0]
    ngrids = grid_coords.shape[0]
    assert deriv in (0, 1, 2), "deriv must be 0, 1, or 2"
    assert a.shape == (natm, natm), "adjustment_factor must have shape (natm, natm)"
    assert nbatch >= 1, "nbatch must be >= 1"
    # deriv_arg is accepted for API parity but the reference does not contract.
    del deriv_arg

    # ----- atom-only preparations (no grid dimension) ----- #
    # interatomic distances R_{AB} (inf on the diagonal so mu_{AA} = 0).
    atom_dist = np.linalg.norm(atm_coords[:, None, :] - atm_coords[None, :, :], axis=-1)
    np.fill_diagonal(atom_dist, np.inf)
    # d R_{AB} / d R_A = (R_A - R_B) / R_{AB}; anti-symmetric in (A, B).  Shape (M, N, t).
    dR_atom_dist = (atm_coords[:, None, :] - atm_coords[None, :, :]) / atom_dist[:, :, None]
    # finite-diagonal copy so the 2nd-order mu-derivatives do not emit nan on M = N; the diagonal
    # is zeroed again below regardless of its value here.
    atom_dist_safe = atom_dist.copy()
    np.fill_diagonal(atom_dist_safe, 1.0)

    # ----- output buffers ----- #
    w = np.zeros(ngrids)
    dw = np.zeros((natm, 3, ngrids)) if deriv >= 1 else None
    ddw = np.zeros((natm, 3, natm, 3, ngrids)) if deriv >= 2 else None

    eye3 = np.eye(3)

    for g0 in range(0, ngrids, nbatch):
        g1 = min(g0 + nbatch, ngrids)
        ng = g1 - g0
        sl = slice(g0, g1)
        gc = grid_coords[g0:g1]  # (ng, 3)
        ag = atm_indices[g0:g1]  # (ng,)
        wg = wquad[g0:g1]  # (ng,)
        g_idx = np.arange(ng)

        # ===== deriv 0 : partition function ===== #
        # grid_dist[A, g] = |r_g - R_A|;  mu[M, N, g] = (r_M - r_N) / R_{MN}  (eq 11).
        grid_dist = np.linalg.norm(gc[None, :, :] - atm_coords[:, None, :], axis=-1)  # (natm, ng)
        mu = (grid_dist[:, None, :] - grid_dist[None, :, :]) / atom_dist[:, :, None]  # (M, N, ng)
        s, dmu_s, ddmu_s = _becke_s_derivs(mu, a[:, :, None], hardness)  # (M, N, ng)
        for M in range(natm):
            s[M, M] = 1.0  # prod over N != M (eq 13); diagonal is undefined

        P = s.prod(axis=1)  # (natm, ng), eq (13)
        Z = P.sum(axis=0)  # (ng,)
        Pg = P[ag, g_idx]  # (ng,)
        w[sl] = wg * Pg / Z  # eq (22)

        if deriv >= 1:
            # ===== deriv 1 : first atomic-coordinate derivative ===== #
            # d r_A / d R_A = (R_A - r_g) / |r_g - R_A|  (unit vec, grid held fixed).
            dR_grid_dist = (atm_coords[:, :, None] - gc.T[None, :, :]) / grid_dist[:, None, :]  # (A, t, ng)
            # role A = d/dR_M, role B = d/dR_N for the pair (M, N).
            dR_mu_roleA = (dR_grid_dist[:, None, :, :] - mu[:, :, None, :] * dR_atom_dist[:, :, :, None]) / atom_dist[
                :, :, None, None
            ]
            dR_mu_roleB = (-dR_grid_dist[None, :, :, :] + mu[:, :, None, :] * dR_atom_dist[:, :, :, None]) / atom_dist[
                :, :, None, None
            ]

            # log derivative of s (computed directly, not as dmu_s / s, to retain precision when s is tiny).
            s_safe_mask = np.abs(s) > S_SAFE_TOL
            s_safe = s.copy()
            s_safe[~s_safe_mask] = 1.0
            dmu_log_s = dmu_s / s_safe
            for M in range(natm):
                dmu_log_s[M, M] = 0.0
            if deriv >= 2:
                ddmu_log_s = np.where(s_safe_mask, ddmu_s / s_safe, 0.0) - dmu_log_s**2
                for M in range(natm):
                    ddmu_log_s[M, M] = 0.0

            # dR_P[M, A, t, g] = d P_M / d R_{A,t}; role A (A = M) sums over N, role B (A = N) per pair.
            dR_P_roleA = np.einsum("Ag, ANg, ANtg -> Atg", P, dmu_log_s, dR_mu_roleA)
            dR_P_roleB = np.einsum("Mg, MAg, MAtg -> MAtg", P, dmu_log_s, dR_mu_roleB)
            dR_P = dR_P_roleB.copy()
            for A in range(natm):
                dR_P[A, A] = dR_P_roleA[A]
            dR_Z = dR_P.sum(axis=0)  # (t, ng)
            dR_Pg = dR_P[ag, :, :, g_idx].transpose(1, 2, 0)  # (A, t, ng)

            # quotient rule (partial, grid held fixed): dw = wquad * (dR_Pg / Z - Pg / Z^2 * dR_Z).
            dw_batch = wg * (dR_Pg / Z - Pg / Z**2 * dR_Z)  # (A, t, ng)
            # translation invariance: the partial is correct only for A != A_g; for A = A_g the
            # atom-centred grid moves with the atom, enforced by sum_A dw = 0  ->  dw[A_g] = -sum_{A'!=A_g}.
            dw_batch[ag, :, g_idx] = 0.0
            dw_batch[ag, :, g_idx] = -dw_batch.sum(axis=0).T
            dw[:, :, sl] = dw_batch

            if deriv >= 2:
                # ===== deriv 2 : second atomic-coordinate derivative ===== #
                # 2nd role derivatives of mu_{MN} via the quotient rule d2(f/g); the four role blocks
                # (AA, AB, BA = AB^T, BB) cover the (dR_A, dR_B) outer products where A, B in {M, N}.
                uA = dR_grid_dist[:, None, :, :]  # (M, N, t, ng) = r_A unit vec for pair (M, N)
                uB = dR_grid_dist[None, :, :, :]  # (M, N, t, ng) = r_B unit vec (= dR_grid_dist[N])
                U = dR_atom_dist[:, :, :, None]  # (M, N, t, ng) = R_{MN} unit vec, broadcast over g
                f_ab = grid_dist[:, None, :] - grid_dist[None, :, :]  # (M, N, ng) = |r_A| - |r_B| (= mu * R_{MN})

                def proj(v):
                    # Proj(v) = I - v v^T, outer over (t, s) with g shared.  v: (M, N, t, ng) -> (M, N, t, s, ng)
                    return eye3[None, None, :, :, None] - v[:, :, :, None, :] * v[:, :, None, :, :]

                PuA, PuB, PU = proj(uA), proj(uB), proj(U)
                Rn5 = atom_dist_safe[:, :, None, None, None]  # (M, N, 1, 1, 1) = |R_{MN}|
                f5 = f_ab[..., None, None, :]  # (M, N, 1, 1, ng) = f

                def d2mu(fX, fY, fXY, gX, gY, gXY):
                    # d2(f/g) = [f_xy g - (f_x g_y + g_x f_y) - f g_xy] / g^2 + 2 f g_x g_y / g^3
                    ofg = fX[:, :, :, None, :] * gY[:, :, None, :, :] + gX[:, :, :, None, :] * fY[:, :, None, :, :]
                    ogg = gX[:, :, :, None, :] * gY[:, :, None, :, :]
                    return (fXY * Rn5 - ofg - f5 * gXY) / Rn5**2 + 2.0 * f5 * ogg / Rn5**3

                ddR_mu_roleAA = d2mu(uA, uA, PuA / grid_dist[:, None, None, None, :], U, U, PU / Rn5)
                ddR_mu_roleAB = d2mu(uA, -uB, np.zeros_like(PuA), U, -U, -PU / Rn5)
                ddR_mu_roleBB = d2mu(-uB, -uB, -PuB / grid_dist[None, :, None, None, :], -U, -U, PU / Rn5)
                for A in range(natm):  # diagonal M = N is undefined; zero out
                    ddR_mu_roleAA[A, A] = 0.0
                    ddR_mu_roleAB[A, A] = 0.0
                    ddR_mu_roleBB[A, A] = 0.0
                ddR_mu_roleBA = ddR_mu_roleAB.transpose(0, 1, 3, 2, 4)  # role BA = role AB transposed in (t, s)

                # log 1st derivative dR_log_P[M, A, t, g] = d log P_M / d R_{A,t}.
                dR_log_P_roleA = np.einsum("MNg, MNtg -> Mtg", dmu_log_s, dR_mu_roleA)  # role A (A = M), sum over N
                dR_log_P_roleB = dmu_log_s[:, :, None, :] * dR_mu_roleB  # role B (A = N), (M, N, t, ng)
                dR_log_P = dR_log_P_roleB.copy()
                for M in range(natm):
                    dR_log_P[M, M] = dR_log_P_roleA[M]

                # log 2nd derivative ddR_log_P[M, A, t, B, s, g] = d2 log P_M / dR_A dR_B,
                # = sum_{N!=M} [ ddmu_log_s (d_A mu)(d_B mu) + dmu_log_s d2_AB mu ]  (sparse over (A,B) in {M,N}).
                L2_AA = np.einsum("MNg, MNtg, MNsg -> Mtsg", ddmu_log_s, dR_mu_roleA, dR_mu_roleA) + np.einsum(
                    "MNg, MNtsg -> Mtsg", dmu_log_s, ddR_mu_roleAA
                )  # (M, t, s, ng), summed over N
                L2_AB = np.einsum("MNg, MNtg, MNsg -> MNtsg", ddmu_log_s, dR_mu_roleA, dR_mu_roleB) + np.einsum(
                    "MNg, MNtsg -> MNtsg", dmu_log_s, ddR_mu_roleAB
                )  # (M, N, t, s, ng)
                L2_BA = np.einsum("MNg, MNtg, MNsg -> MNtsg", ddmu_log_s, dR_mu_roleB, dR_mu_roleA) + np.einsum(
                    "MNg, MNtsg -> MNtsg", dmu_log_s, ddR_mu_roleBA
                )
                L2_BB = np.einsum("MNg, MNtg, MNsg -> MNtsg", ddmu_log_s, dR_mu_roleB, dR_mu_roleB) + np.einsum(
                    "MNg, MNtsg -> MNtsg", dmu_log_s, ddR_mu_roleBB
                )
                # scatter into ddR_log_P[M, A, t, B, s, ng]; += so the shared diagonal (M, M, M) accumulates.
                ddR_log_P = np.zeros((natm, natm, 3, natm, 3, ng))
                idx = np.arange(natm)
                Mi, Ni = np.indices((natm, natm))
                mi, ni = Mi.ravel(), Ni.ravel()
                ddR_log_P[idx, idx, :, idx, :, :] += L2_AA  # role AA: (A, B) = (M, M)
                ddR_log_P[mi, ni, :, ni, :, :] += L2_BB.reshape(natm * natm, 3, 3, ng)  # role BB: (A, B) = (N, N)
                ddR_log_P[mi, mi, :, ni, :, :] += L2_AB.reshape(natm * natm, 3, 3, ng)  # role AB: (A, B) = (M, N)
                ddR_log_P[mi, ni, :, mi, :, :] += L2_BA.reshape(natm * natm, 3, 3, ng)  # role BA: (A, B) = (N, M)

                # d2 P_M = P_M (ddR_log_P + dR_log_P_A (x) dR_log_P_B).
                ddR_P = P[:, None, None, None, None, :] * (
                    ddR_log_P + dR_log_P[:, :, :, None, None, :] * dR_log_P[:, None, None, :, :, :]
                )  # (M, A, t, B, s, ng)
                ddR_Z = ddR_P.sum(axis=0)  # (A, t, B, s, ng)
                ddR_Pg = ddR_P[ag, :, :, :, :, g_idx].transpose(1, 2, 3, 4, 0)  # (A, t, B, s, ng)

                # quotient rule (partial, grid held fixed): q = Pg / Z.
                q = Pg / Z
                dq = (dR_Pg - q * dR_Z) / Z  # (A, t, ng)
                term1 = np.einsum("Bsg, Atg -> AtBsg", dq, dR_Z)  # (dq_B)(dZ_A)
                term2 = np.einsum("Atg, Bsg -> AtBsg", dq, dR_Z)  # (dq_A)(dZ_B)
                d2q = (ddR_Pg - term1 - q * ddR_Z) / Z - term2 / Z
                ddw_partial = wg * d2q  # (A, t, B, s, ng)

                # translation invariance: the partial is correct for A, B != A_g; the A = A_g row,
                # B = A_g column and the (A_g, A_g) corner are filled from the axis sums so that
                # sum_A ddw = sum_B ddw = 0.  The sums are computed once (vectorised); only the
                # per-grid row/column/corner assignment remains.
                ddw_batch = ddw_partial.copy()
                fullA = ddw_partial.sum(axis=0)  # (t, B, s, ng) = sum_A ddw_partial[A, t, B, s, ng]
                fullB = ddw_partial.sum(axis=2)  # (A, t, s, ng) = sum_B ddw_partial[A, t, B, s, ng]
                fullAB = ddw_partial.sum(axis=(0, 2))  # (t, s, ng)    = sum_A sum_B
                for g in range(ng):
                    Ag = int(ag[g])
                    ddw_batch[Ag, :, :, :, g] = -fullA[:, :, :, g] + ddw_partial[Ag, :, :, :, g]  # row A = A_g
                    ddw_batch[:, :, Ag, :, g] = -fullB[:, :, :, g] + ddw_partial[:, :, Ag, :, g]  # col B = A_g
                    ddw_batch[Ag, :, Ag, :, g] = (
                        fullAB[:, :, g]  # corner (A_g, A_g)
                        - fullB[Ag, :, :, g]
                        - fullA[:, Ag, :, g]
                        + ddw_partial[Ag, :, Ag, :, g]
                    )
                ddw[:, :, :, :, sl] = ddw_batch

    return {"w": w, "dw": dw, "ddw": ddw}
