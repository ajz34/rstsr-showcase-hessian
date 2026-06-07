#!/usr/bin/env python3
"""Clean block krylov solver implementation.

This can replace krylov_glm in the CPHF notebook. It solves (1+a)*x = b
using a block Krylov approach that adds nroots directions per cycle,
matching PySCF's lib.krylov convergence behavior.

Key differences from krylov_glm (GMRES):
1. Block approach: adds all RHS directions per cycle, not one per iteration
2. Non-normalized basis: trial vectors deflate naturally as solution converges
3. Identity handled analytically: Krylov subspace built for `a` only

Usage:
    # Instead of:
    mo1 = krylov_glm(vind_vo_plus, mo1_base, tol=1e-8)

    # Use:
    mo1 = krylov_block(vind_vo, mo1_base, tol=1e-8)

    # Note: pass vind_vo (without +I), not vind_vo_plus!
"""

import numpy as np


def krylov_block(aop, b, x0=None, tol=1e-10, max_cycle=30, lindep=1e-13):
    """Block Krylov subspace method to solve (1+a)*x = b.

    At each cycle, the current trial block is multiplied by aop and
    orthogonalized against the full subspace.  This adds nroots new
    directions per cycle (one per surviving right-hand side), making
    it much faster than single-vector GMRES for multi-RHS problems.

    The basis vectors are kept non-normalized (orthogonal but not
    orthonormal), and their squared norms are tracked in `innerprod`.
    This causes trial vectors to naturally "deflate" as the solution
    converges, providing an accurate and inexpensive convergence signal
    without requiring the actual residual to be computed.

    Parameters
    ----------
    aop : callable
        Linear operator a, mapping (nblock, n) -> (nblock, n).
        The equation solved is (1 + aop) * x = b.
    b : ndarray, shape (nset, n) or (n,)
        Right-hand sides.
    x0 : ndarray, shape (nset, n) or (n,), optional
        Initial guess.
    tol : float
        Convergence tolerance.  Iteration stops when
        max(||new_trial_vec_i||^2) < max(lindep, tol^2).
    max_cycle : int
        Maximum number of block cycles.
    lindep : float
        Linear dependency threshold.  Vectors with ||v||^2 < lindep
        are dropped from the subspace.

    Returns
    -------
    x : ndarray, same shape as b
        Approximate solution of (1 + aop) * x = b.
    """
    if b.ndim == 1:
        b = b.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    nset_l, ndim_l = b.shape

    if x0 is not None:
        if x0.ndim == 1:
            x0 = x0.reshape(1, -1)
        b = b - (x0 + aop(x0))

    # MGS that keeps non-normalized vectors and tracks ||v||^2
    def _orth_block(vec_list):
        result = []
        norms_sq = []
        for vi in vec_list:
            vi = vi.copy()
            for j in range(len(result)):
                coeff = np.dot(vi, result[j]) / norms_sq[j]
                vi -= coeff * result[j]
            nsq = np.dot(vi, vi).real
            if nsq > lindep:
                result.append(vi)
                norms_sq.append(nsq)
        return result, norms_sq

    # Initialize: orthogonalize b
    x1_list, innerprod = _orth_block([b[i] for i in range(nset_l)])

    if not x1_list:
        result = np.zeros((nset_l, ndim_l))
        if x0 is not None:
            result += x0
        return result[0] if was_1d else result

    xs = []              # basis vectors (orthogonal, non-normalized)
    ax_list = []         # aop(xs[i]) for each basis vector
    all_innerprod = []    # ||xs[i]||^2

    for cycle in range(max_cycle):
        # Apply operator to current trial block
        x1_arr = np.array(x1_list)
        axt = aop(x1_arr)
        if axt.ndim == 1:
            axt = axt.reshape(1, -1)

        # Store basis vectors and aop results
        for i in range(len(x1_list)):
            xs.append(x1_list[i].copy())
            ax_list.append(axt[i].copy())
            all_innerprod.append(innerprod[i])

        # Orthogonalize axt against full subspace
        # For non-normalized orthogonal xs[i] with ||xs[i]||^2 = all_innerprod[i]:
        #   proj_coeff = dot(v, xs[i]) / all_innerprod[i]
        x1_new = axt.copy()
        for i in range(len(xs)):
            xsi = xs[i]
            w = x1_new @ xsi / all_innerprod[i]    # (nblock,) coefficients
            x1_new -= np.outer(w, xsi)

        # Orthogonalize among themselves via MGS
        x1_list, innerprod = _orth_block(x1_new)

        max_innerprod = max(innerprod) if innerprod else 0
        r = np.sqrt(max_innerprod)

        print(f"Cycle {cycle+1}: max(||new_trial_vec_i||^2) = {max_innerprod:.3e}, max(||new_trial_vec_i||) = {r:.3e}")

        # Convergence check (same as PySCF)
        if max_innerprod < max(lindep, tol**2):
            break

    # Build and solve small system: (I + A_projected) * c = b_projected
    nd = len(xs)
    Xs = np.array(xs)
    AX = np.array(ax_list)

    # h[i,j] = dot(xs[i], aop(xs[j])) + delta_ij * ||xs[i]||^2
    h = Xs @ AX.T
    for i in range(nd):
        h[i, i] += all_innerprod[i]

    # g[i,k] = dot(b[k], xs[i])
    g = Xs @ b.T

    c = np.linalg.solve(h, g)
    x = c.T @ Xs

    if x0 is not None:
        x += x0

    return x[0] if was_1d else x
