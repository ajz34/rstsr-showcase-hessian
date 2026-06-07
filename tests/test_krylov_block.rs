mod test_util;

use rstsr::prelude::*;
use rstsr_showcase_hessian::util::krylov_block::krylov_block;
use test_util::{Tsr, TsrView, fp, read_npz_dict};

#[test]
fn test_krylov_block_small_deterministic() {
    let dict = read_npz_dict("02-7-krylov_testing_data.npz");
    let a: Tsr = dict["A"].to_owned();
    // Python stores b and x_ref as [nset, n] (row-major NumPy convention).
    // The col-major Rust API expects [n, nset] (each RHS is a column), so
    // transpose at load. `into_reverse_axes` is a cheap layout-only swap.
    let b: Tsr = dict["b"].to_owned().into_reverse_axes();
    let x_ref: Tsr = dict["x_ref"].to_owned().into_reverse_axes();

    let mut call_count: usize = 0;
    let aop = |x: TsrView| -> Tsr {
        call_count += 1;
        // x has shape [n, nblock]; A is [n, n]. For each column v of x, the
        // corresponding output column should be A @ v, i.e. A % x.
        &a % &x
    };

    let x = krylov_block(aop, b.view(), None, 1e-10, 30, 1e-13);

    let diff: Tsr = &x - &x_ref;
    let max_diff = diff.abs().max();
    println!("aop call count = {call_count}, max |x - x_ref| = {max_diff:.3e}");

    // Compare against the direct-solve reference saved in the npz. Tolerance
    // matches the krylov_block convergence tol of 1e-10; the residual of the
    // projected solve typically lands a few times that.
    assert!(rt::allclose(x.view(), x_ref.view(), (1e-6, 1e-8)));

    // Residue check: x + A x - b ≈ 0.
    let residue: Tsr = &x + (&a % &x) - &b;
    let max_residue = residue.abs().max();
    println!("max |residue| = {max_residue:.3e}");
    assert!(max_residue < 1e-6, "residue too large: {max_residue:.3e}");

    // Fingerprint of the solution, for regression detection on top of the
    // numeric closeness check.
    println!("fp(x) = {:.12}", fp(x.view()));
}
