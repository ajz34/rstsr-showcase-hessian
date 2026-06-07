from krylov_block import krylov_block
import numpy as np


def build_test_system(n=12, nrhs=2):
    """Build a deterministic, moderately challenging linear system for testing
    `krylov_block`. Solves (I + A) x = b.

    Design goals:
    - Deterministic (no RNG).
    - Size around 10x10 so iterations are clearly visible.
    - Spectrum of A spread across a wide range, with some eigenvalues close
      to (but not equal to) -1, so that (I + A) is well-defined but not
      trivially diagonal-dominant. This forces the block Krylov method to
      take several cycles (~6-10) instead of converging in 2.
    - Non-symmetric (CPHF-style: A is generally not symmetric).
    - Multiple distinct RHS that excite different eigendirections.

    Construction:
    - Eigenvalues lambda_k spread geometrically between -0.95 and +5.
      The near-(-1) eigenvalue makes (I+A) ill-conditioned along that
      direction; the large positive eigenvalues stretch the spectrum.
    - A non-orthogonal but well-conditioned basis V is built from a
      deterministic Hilbert-like matrix; A = V diag(lambda) V^{-1}.
    - RHS b is built so that each row has nontrivial overlap with every
      eigenvector (no accidental subspace).
    """
    # Eigenvalues chosen so that (I+A) has:
    #   - one eigenvalue close to zero (lambda ≈ -0.9 → 1+lambda = 0.1),
    #   - one large eigenvalue (lambda = 5 → 1+lambda = 6),
    #   - a tight cluster of eigenvalues in between.
    # Krylov methods converge fast when the spectrum has few distinct
    # clusters; a *clustered* spectrum with one isolated extreme on each
    # side makes the method need many cycles before the small eigenvalue
    # is resolved. The cluster gives n-2 nearly-degenerate directions
    # which the block iteration cannot deflate in one go.
    eigs = np.empty(n)
    eigs[0] = -0.9                                     # near-singular direction
    eigs[-1] = 5.0                                     # stretched top
    cluster = 1.0 + 0.05 * np.cos(np.arange(n - 2) * 1.7 + 0.3)
    eigs[1:-1] = cluster
    # Deterministic permutation so the small/large eigenvalues are not
    # exposed in the first/last basis index.
    perm = np.argsort(np.sin(np.arange(n) * 1.3 + 0.5))
    eigs = eigs[perm]

    # Deterministic non-orthogonal basis V: Hilbert-like + identity shift
    # to ensure good conditioning of V.
    i_idx = np.arange(n).reshape(-1, 1)
    j_idx = np.arange(n).reshape(1, -1)
    V = 1.0 / (i_idx + j_idx + 1.0) + 0.5 * np.eye(n)

    # A = V diag(eigs) V^{-1}; non-symmetric in general.
    Vinv = np.linalg.inv(V)
    A = V @ np.diag(eigs) @ Vinv

    # RHS: deterministic, with components in every eigendirection.
    # Use a smooth but non-trivial pattern.
    k = np.arange(nrhs).reshape(-1, 1) + 1.0
    cols = np.arange(n).reshape(1, -1) + 1.0
    b = np.cos(0.7 * k * cols) + 0.3 * np.sin(1.1 * k + 0.2 * cols)

    return A, b, eigs


if __name__ == "__main__":
    n = 40
    nrhs = 12
    A, b, eigs = build_test_system(n=n, nrhs=nrhs)

    print(f"Problem size: n = {n}, nrhs = {nrhs}")
    print(f"Eigenvalues of A:        {np.sort(eigs)}")
    print(f"Eigenvalues of (I + A):  {np.sort(1.0 + eigs)}")
    cond_IA = np.linalg.cond(np.eye(n) + A)
    print(f"cond(I + A) = {cond_IA:.3e}")

    call_count = [0]
    def aop(x):
        call_count[0] += 1
        print(f"  [aop call {call_count[0]}] x shape: {x.shape}")
        return (A @ x.T).T

    x = krylov_block(aop, b, max_cycle=30, tol=1e-10)

    # Reference solution via direct solve, for ground-truth comparison.
    x_ref = np.linalg.solve(np.eye(n) + A, b.T).T

    residue = x + (A @ x.T).T - b
    print(f"\nmax |residue|         = {np.max(np.abs(residue)):.3e}")
    print(f"max |x - x_ref|       = {np.max(np.abs(x - x_ref)):.3e}")
    print(f"total aop applications = {call_count[0]}")

    # Persist the test case so it can be reloaded by other scripts/notebooks
    # without needing to re-solve. `x_ref` is the direct-solve ground truth;
    # `x` is the krylov_block result (they should agree to ~1e-10).
    # Force C-contiguous so downstream loaders (Rust `read_npz`) that assume
    # row-major raw layout get the indices they expect.
    npz_path = "02-7-krylov_testing_data.npz"
    np.savez(
        npz_path,
        A=np.ascontiguousarray(A),
        b=np.ascontiguousarray(b),
        x=np.ascontiguousarray(x),
        x_ref=np.ascontiguousarray(x_ref),
        eigs=np.ascontiguousarray(eigs),
    )
    print(f"\nSaved test data to {npz_path}")
