"""Prototype hard-restart Krylov: add max_space, restart when subspace fills.

Strategy:
- Outer "restart" loop, inner loop accumulates up to max_space cycles of basis.
- On hitting max_space without convergence, solve projected system to get x_partial,
  accumulate it into x_accum, recompute residual b' = b_orig - (I+A) x_accum,
  reset subspace, continue.
- Total inner cycles capped at max_cycle.

Validates against the npz fixture used by the Rust test.
"""
import numpy as np
from pathlib import Path


def krylov_block(aop, b, x0=None, tol=1e-10, max_cycle=30, max_space=14, lindep=1e-13, verbose=True):
    if b.ndim == 1:
        b = b.reshape(1, -1)
        was_1d = True
    else:
        was_1d = False
    nset_l, ndim_l = b.shape
    b_orig = b.copy()

    # x_accum plays the role of the running initial guess; refined each restart.
    x_accum = x0.copy() if x0 is not None else np.zeros_like(b_orig)
    if x_accum.ndim == 1:
        x_accum = x_accum.reshape(1, -1)

    def _orth_block(vec_list):
        result, norms_sq = [], []
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
    converged = False
    conv_thresh = max(lindep, tol ** 2)

    # Final x = x_accum + Xs @ c from the LAST inner loop (whether converged or hit max_cycle).
    # We need to keep the last loop's Xs/ax/all_innerprod around for that.
    last_xs, last_ax, last_ip = None, None, None
    last_b_residual = None

    restart_idx = 0
    while total_cycles < max_cycle and not converged:
        # Compute residual for the current accumulated solution: b' = b - (I+A) x_accum
        if restart_idx == 0 and x0 is None:
            b_residual = b_orig.copy()
        else:
            b_residual = b_orig - (x_accum + aop(x_accum))

        x1_list, innerprod = _orth_block([b_residual[i] for i in range(nset_l)])
        if not x1_list:
            converged = True
            last_b_residual = b_residual
            break

        xs, ax_list, all_innerprod = [], [], []

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

            x1_new = axt.copy()
            for i in range(len(xs)):
                xsi = xs[i]
                w = x1_new @ xsi / all_innerprod[i]
                x1_new -= np.outer(w, xsi)

            x1_list, innerprod = _orth_block(x1_new)
            max_innerprod = max(innerprod) if innerprod else 0.0
            if verbose:
                print(f"  restart {restart_idx} inner {inner+1} (total cycle {total_cycles}): "
                      f"max||v||^2 = {max_innerprod:.3e}")

            if max_innerprod < conv_thresh:
                inner_converged = True
                break

        last_xs = np.array(xs)
        last_ax = np.array(ax_list)
        last_ip = list(all_innerprod)
        last_b_residual = b_residual

        if inner_converged:
            converged = True
            break

        # Hard restart: solve projected system, fold x_partial into x_accum, restart.
        nd = len(xs)
        Xs = last_xs
        AX = last_ax
        h = Xs @ AX.T
        for i in range(nd):
            h[i, i] += last_ip[i]
        g = Xs @ b_residual.T
        c = np.linalg.solve(h, g)
        x_partial = c.T @ Xs
        x_accum = x_accum + x_partial
        restart_idx += 1
        if verbose:
            print(f"  ---- restart {restart_idx}: x_accum updated, subspace reset ----")

    # Final solve using the LAST loop's subspace and the LAST residual b.
    if last_xs is None or last_xs.shape[0] == 0:
        # Either b was zero from the start, or we did 0 cycles. x_accum is the answer.
        result = x_accum
    else:
        nd = last_xs.shape[0]
        h = last_xs @ last_ax.T
        for i in range(nd):
            h[i, i] += last_ip[i]
        g = last_xs @ last_b_residual.T
        c = np.linalg.solve(h, g)
        x_final = c.T @ last_xs
        result = x_accum + x_final

    if verbose:
        print(f"Total inner cycles: {total_cycles}, restarts: {restart_idx}, converged: {converged}")
    return result[0] if was_1d else result


if __name__ == "__main__":
    # Load the same fixture used by tests/test_krylov_block.rs.
    fixture = Path(__file__).parent / "02-7-krylov_testing_data.npz"
    data = np.load(fixture)
    A = data["A"]
    b = data["b"]
    x_ref = data["x_ref"]

    def aop(x):
        return x @ A.T  # row-major: x is [nset, n], A is [n, n]

    print(f"A shape {A.shape}, b shape {b.shape}, x_ref shape {x_ref.shape}")

    print("\n=== max_space = 30 (effectively no restart) ===")
    x_a = krylov_block(aop, b, tol=1e-10, max_cycle=30, max_space=30)
    print(f"max |x - x_ref| = {np.abs(x_a - x_ref).max():.3e}")

    print("\n=== max_space = 6 ===")
    x_b = krylov_block(aop, b, tol=1e-10, max_cycle=30, max_space=6)
    print(f"max |x - x_ref| = {np.abs(x_b - x_ref).max():.3e}")

    print("\n=== max_space = 3 (forces restarts) ===")
    x_c = krylov_block(aop, b, tol=1e-10, max_cycle=60, max_space=3)
    print(f"max |x - x_ref| = {np.abs(x_c - x_ref).max():.3e}")

    print("\n=== max_space = 2 (heavy restart) ===")
    x_d = krylov_block(aop, b, tol=1e-10, max_cycle=120, max_space=2)
    print(f"max |x - x_ref| = {np.abs(x_d - x_ref).max():.3e}")
