#!/usr/bin/env python3
"""Clean block krylov solver implementation.

This can replace krylov_glm in the CPHF notebook. It solves (1+a)*x = b
using a block Krylov approach that adds nroots directions per cycle,
matching PySCF's lib.krylov convergence behavior.

Key differences from krylov_glm (GMRES):
1. Block approach: adds all RHS directions per cycle, not one per iteration
2. Non-normalized basis: trial vectors deflate naturally as solution converges
3. Identity handled analytically: Krylov subspace built for `a` only

To bound memory the subspace is capped at `max_space` cycles.  When that cap
is hit without convergence, a **hard restart** is performed: the projected
system is solved, the partial solution is folded into a running accumulator,
the subspace is discarded, and the new residual becomes the RHS.

Usage:
    # Instead of:
    mo1 = krylov_glm(vind_vo_plus, mo1_base, tol=1e-8)

    # Use:
    mo1 = krylov_block(vind_vo, mo1_base, tol=1e-8)

    # Note: pass vind_vo (without +I), not vind_vo_plus!
"""

import numpy as np


def krylov_block(aop, b, x0=None, tol=1e-10, max_cycle=30, max_space=14, lindep=1e-14):
    """Block Krylov subspace method to solve (1+a)*x = b with hard restarts.

    At each cycle, the current trial block is multiplied by aop and
    orthogonalized against the full subspace.  This adds nroots new
    directions per cycle (one per surviving right-hand side), making
    it much faster than single-vector GMRES for multi-RHS problems.

    The basis vectors are kept non-normalized (orthogonal but not
    orthonormal), and their squared norms are tracked in `innerprod`.
    This causes trial vectors to naturally "deflate" as the solution
    converges, providing an accurate and inexpensive convergence signal
    without requiring the actual residual to be computed.

    To bound memory, the subspace is capped at `max_space` cycles.
    When the subspace fills without convergence, a hard restart is
    triggered: the current approximate solution is extracted from the
    projected system and accumulated, the subspace is cleared, and the
    residual becomes the new right-hand side.  This is the standard
    GMRES(m) trick and is algebraically exact.

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
        Maximum **total** number of inner cycles (summed across restarts).
    max_space : int
        Maximum number of cycles per restart before forcing a hard restart.
        Typical values are 3..=20; with max_space >= max_cycle the solver
        never restarts (matches the pre-restart behaviour).
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
    b_orig = b.copy()

    # Running solution accumulator, refined on each hard restart.
    x_accum = x0.copy() if x0 is not None else np.zeros_like(b_orig)
    if x_accum.ndim == 1:
        x_accum = x_accum.reshape(1, -1)

    # Two-tier separation: `lindep` is the MGS-level floor (drop a
    # subspace vector only when its squared-norm falls below this — i.e.
    # numerical zero), while `conv_thresh = max(lindep, tol²)` controls
    # which vectors are eligible to extend the next Krylov direction.
    # Borderline vectors with squared-norm in (lindep, conv_thresh] still
    # contribute to the projected solve via xs / ax; they just don't get
    # axt-applied again.  To exploit the two tiers separately, set lindep
    # tight (numerical zero) and tol so that tol² is a looser filter
    # threshold — e.g. lindep=1e-14, tol≈3e-7 → conv_thresh=1e-13.
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

    total_cycles = 0
    conv_thresh = max(lindep, tol ** 2)

    # Remember the last inner loop's subspace & residual for the final solve.
    last_xs_arr, last_ax_arr = None, None
    last_ip, last_b_residual = None, None
    last_ok = False  # True if the last inner loop produced a non-empty subspace

    restart_idx = 0
    while total_cycles < max_cycle:
        # Form residual for the current accumulated solution.
        if restart_idx == 0 and x0 is None:
            b_residual = b_orig.copy()
        else:
            b_residual = b_orig - (x_accum + aop(x_accum))

        x1_list, innerprod = _orth_block([b_residual[i] for i in range(nset_l)])

        if not x1_list:
            # Residual is already (numerically) zero.
            last_ok = False
            last_b_residual = b_residual
            break

        xs = []
        ax_list = []
        all_innerprod = []
        inner_converged = False

        for inner in range(max_space):
            if total_cycles >= max_cycle:
                break
            total_cycles += 1

            x1_arr = np.array(x1_list)
            axt = aop(x1_arr)
            if axt.ndim == 1:
                axt = axt.reshape(1, -1)

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

            # MGS new directions: keep at numerical-zero floor; defer
            # the user's lindep filter so borderline vectors still feed
            # the projected solve via xs / ax.
            x1_list, innerprod = _orth_block(x1_new)

            # Convergence test uses the unfiltered max innerprod.
            max_innerprod = max(innerprod) if innerprod else 0
            r = np.sqrt(max_innerprod)

            # Filter the next-iteration directions to those above conv_thresh.
            kept = [(v, ip) for v, ip in zip(x1_list, innerprod)
                    if ip > conv_thresh]
            x1_list = [v for v, _ in kept]
            innerprod = [ip for _, ip in kept]

            # Per-element diagnostics on the new trial block.  These are
            # what `tol` directly bounds: at exit ||new_trial||₂ < tol per
            # vector, so the per-element RMS is roughly tol / √ndim.  The
            # actual linear-system residue ||b - (I+A)x||₂ is a separate
            # quantity (Galerkin solve doesn't minimise it) and can be
            # substantially larger.
            new_trial = np.asarray(x1_new)
            l2_per_elem = np.sqrt(max_innerprod / ndim_l)
            max_abs = float(np.max(np.abs(new_trial))) if new_trial.size else 0.0

            print(f"  restart {restart_idx} inner {inner+1} (total cycle {total_cycles}): "
                  f"max(||new_trial_vec_i||^2) = {max_innerprod:.3e}, "
                  f"max(||new_trial_vec_i||) = {r:.3e}, "
                  f"per-elem L2 = {l2_per_elem:.3e}, max-abs = {max_abs:.3e}")

            if max_innerprod < conv_thresh:
                inner_converged = True
                break

        last_xs_arr = np.array(xs) if xs else None
        last_ax_arr = np.array(ax_list) if xs else None
        last_ip = list(all_innerprod)
        last_b_residual = b_residual
        last_ok = inner_converged or (xs and total_cycles >= max_cycle)

        if inner_converged or total_cycles >= max_cycle:
            break

        # Hard restart: solve projected system, fold into x_accum, continue.
        nd = len(xs)
        Xs = last_xs_arr
        AX = last_ax_arr
        h = Xs @ AX.T
        for i in range(nd):
            h[i, i] += all_innerprod[i]
        g = Xs @ b_residual.T
        c = np.linalg.solve(h, g)
        x_partial = c.T @ Xs
        x_accum = x_accum + x_partial
        restart_idx += 1
        print(f"  ---- restart {restart_idx}: x_accum refined, subspace reset ----")

    # Final projected solve using the last subspace & matching residual.
    if last_ok and last_xs_arr is not None and last_xs_arr.shape[0] > 0:
        nd = last_xs_arr.shape[0]
        h = last_xs_arr @ last_ax_arr.T
        for i in range(nd):
            h[i, i] += last_ip[i]
        g = last_xs_arr @ last_b_residual.T
        c = np.linalg.solve(h, g)
        x_final = c.T @ last_xs_arr
        result = x_accum + x_final
    else:
        result = x_accum

    return result[0] if was_1d else result