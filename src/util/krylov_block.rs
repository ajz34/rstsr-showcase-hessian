//! Block Krylov subspace solver for `(1 + A) x = b`.
//!
//! Direct translation of `prototype/krylov_block.py`. The convergence behavior
//! matches PySCF's `lib.krylov` block algorithm: each cycle adds the surviving
//! trial directions to the subspace, and the basis is kept non-normalized so
//! that the squared norms of the new trial vectors act as the convergence
//! signal without requiring an explicit residual evaluation.
//!
//! Layout convention is col-major: each right-hand side / basis vector is a
//! **column** of the corresponding matrix. So `b` is shaped `[n, nset]`,
//! the operator maps `[n, nblock] -> [n, nblock]`, and the basis matrices
//! `xs`, `ax` are `[n, nd]` with new vectors appended along axis 1.

use crate::prelude::*;

/// Solve `(I + aop) x = b` by a block Krylov subspace method.
///
/// # Parameters
///
/// - `aop` : Linear operator. Given a `[n, nblock]` input it must return a `[n, nblock]` output
///   (the action of `A` applied column-wise).
/// - `b` : Right-hand sides, shape `[n, nset]`. Each column is one RHS.
/// - `x0` : Optional initial guess, shape `[n, nset]`.
/// - `tol` : Convergence tolerance on `max(||new_trial_vec_i||)`.
/// - `max_cycle` : Maximum number of block cycles.
/// - `lindep` : Vectors with `||v||^2 < lindep` are dropped from the subspace.
///
/// # Returns
///
/// `x` of shape `[n, nset]`, an approximate solution of `(I + aop) x = b`.
pub fn krylov_block(
    mut aop: impl FnMut(TsrView) -> Tsr,
    b: TsrView,
    x0: Option<TsrView>,
    tol: f64,
    max_cycle: usize,
    lindep: f64,
) -> Tsr {
    let device = b.device().clone();
    let ndim_l = b.shape()[0];
    let nset_l = b.shape()[1];

    // Subtract the contribution of the initial guess from the RHS.
    let b: Tsr = match x0.as_ref() {
        Some(x0v) => &b - (&x0v.to_owned() + aop(x0v.view())),
        None => b.to_owned(),
    };

    // Initialize: orthogonalize the columns of b.
    let (mut x1, mut innerprod) = orth_block(b.view(), lindep);

    if x1.shape()[1] == 0 {
        let mut result: Tsr = rt::zeros(([ndim_l, nset_l], &device));
        if let Some(x0v) = x0 {
            result += x0v;
        }
        return result;
    }

    // Pre-allocate basis storage: at most nset_l vectors per cycle plus the
    // initial block, so cap at nset_l * (max_cycle + 1) columns.
    let max_basis = nset_l * (max_cycle + 1);
    let mut xs: Tsr = rt::zeros(([ndim_l, max_basis], &device));
    let mut ax: Tsr = rt::zeros(([ndim_l, max_basis], &device));
    let mut all_innerprod: Vec<f64> = Vec::with_capacity(max_basis);
    let mut nd: usize = 0;

    let conv_thresh = lindep.max(tol * tol);

    for cycle in 0..max_cycle {
        let nblock = x1.shape()[1];

        // Apply operator to current trial block.
        let axt = aop(x1.view());

        // Append current (x1, axt, innerprod) to the subspace.
        xs.i_mut((.., nd..nd + nblock)).assign(&x1);
        ax.i_mut((.., nd..nd + nblock)).assign(&axt);
        all_innerprod.extend_from_slice(&innerprod);
        nd += nblock;

        // Orthogonalize axt against the full subspace. For non-normalized
        // orthogonal columns xs[:, i] with ||xs[:, i]||^2 = all_innerprod[i]:
        //   coeffs[i, k] = (xs[:, i] . axt[:, k]) / all_innerprod[i]
        //              = (xs_slc.T @ axt)[i, k] / all_innerprod[i]
        //   x1_new[:, k] = axt[:, k] - sum_i coeffs[i, k] * xs[:, i]
        //              = axt - xs_slc @ coeffs
        let xs_slc = xs.i((.., ..nd));
        let ip_vec: Tsr = rt::asarray((all_innerprod.clone(), &device));
        let coeffs = (xs_slc.t() % &axt) / ip_vec.i((.., None));
        let x1_new = axt - &xs_slc % &coeffs;

        // Orthogonalize the new trial block among itself.
        let (next_x1, next_ip) = orth_block(x1_new.view(), lindep);

        let max_innerprod = next_ip.iter().copied().fold(0.0_f64, f64::max);
        let r = max_innerprod.sqrt();
        println!(
            "Cycle {}: max(||new_trial_vec_i||^2) = {:.3e}, max(||new_trial_vec_i||) = {:.3e}",
            cycle + 1,
            max_innerprod,
            r
        );

        x1 = next_x1;
        innerprod = next_ip;

        if max_innerprod < conv_thresh {
            break;
        }
    }

    // Build and solve the projected system: (I + A_projected) c = b_projected.
    let xs_slc = xs.i((.., ..nd));
    let ax_slc = ax.i((.., ..nd));

    // h[i, j] = dot(xs[:, i], ax[:, j]) + delta_ij * ||xs[:, i]||^2
    //        = (xs_slc.T @ ax_slc)[i, j] + diagonal correction
    let mut h: Tsr = xs_slc.t() % &ax_slc;
    for i in 0..nd {
        h[[i, i]] += all_innerprod[i];
    }

    // g[i, k] = dot(b[:, k], xs[:, i]) = (xs_slc.T @ b)[i, k]
    let g: Tsr = xs_slc.t() % &b;

    // Solve h c = g, then reconstruct x[:, k] = sum_i c[i, k] * xs[:, i] = xs c.
    let c = rt::linalg::solve_general((h.view(), g.view()));
    let mut x: Tsr = &xs_slc % &c;

    if let Some(x0v) = x0 {
        x += x0v;
    }
    x
}

/// Modified Gram-Schmidt over the **columns** of `vec`, keeping non-normalized
/// orthogonal vectors and tracking their squared norms. Mirrors `_orth_block`
/// in the Python prototype (which operated on rows; here we operate on columns
/// because the rest of the solver is col-major).
///
/// Columns whose remaining squared norm falls below `lindep` are dropped.
///
/// Returns `(out, norms_sq)` where `out` has shape `[n, m]` with `m <= nblock`.
fn orth_block(vec: TsrView, lindep: f64) -> (Tsr, Vec<f64>) {
    let device = vec.device().clone();
    let n = vec.shape()[0];
    let nblock = vec.shape()[1];

    let mut result: Vec<Tsr> = Vec::with_capacity(nblock);
    let mut norms_sq: Vec<f64> = Vec::with_capacity(nblock);

    for i in 0..nblock {
        let mut vi: Tsr = vec.i((.., i)).to_owned();
        for j in 0..result.len() {
            let coeff = (&vi % &result[j]).to_scalar() / norms_sq[j];
            vi -= coeff * &result[j];
        }
        let nsq = (&vi % &vi).to_scalar();
        if nsq > lindep {
            result.push(vi);
            norms_sq.push(nsq);
        }
    }

    let m = result.len();
    let mut out: Tsr = rt::zeros(([n, m], &device));
    for (i, vi) in result.into_iter().enumerate() {
        out.i_mut((.., i)).assign(&vi);
    }
    (out, norms_sq)
}
