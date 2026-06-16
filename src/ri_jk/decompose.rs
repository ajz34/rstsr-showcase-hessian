use crate::prelude::*;
use serde::{Deserialize, Serialize};
use serde_inline_default::serde_inline_default;

pub const J2C_THRESH: f64 = 1e-13;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum J2CDecompPolicy {
    /// Cholesky decomposition (give upper/lower triangular decomposition of matrix).
    #[serde(alias = "cholesky", alias = "cd")]
    Cd,
    /// Eigen decomposition (give symmetric decomposition of matrix).
    #[serde(alias = "eigen", alias = "eig", alias = "eigenvalue")]
    Eig,
}

/// Policy for 2c-2e ERI (j2c) decomposition.
///
/// - `Cd`: Cholesky decomposition
///   - None: no threshold, fails if j2c is not positive-definite
///   - threshold: if the diagonal elements of the Cholesky factor are smaller than the threshold,
///     will make matrix to be sufficiently positive-definite with the given threshold, then perform
///     Cholesky decomposition again.
/// - `Eig`: Eigen decomposition, make matrix power -1/2 by strict way with eigenvalues that larger
///   than given threshold, a more orthogonal but costly way than Cholesky decomposition.
#[serde_inline_default]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct J2CDecompOption {
    /// The policy for 2c-2e ERI decomposition. Default to `Eig`.
    #[serde_inline_default(J2CDecompPolicy::Eig)]
    pub policy: J2CDecompPolicy,
    /// The threshold for 2c-2e ERI decomposition. Default to `1e-13`.
    #[serde_inline_default(Some(J2C_THRESH))]
    pub threshold: Option<f64>,
    /// The flag indicating whether the Cholesky factor is upper or lower triangular. Default to
    /// `Upper`.
    ///
    /// This is developer option. In most cases, col-major uses upper triangular.
    /// Lower triangular is only for debug and testing purposes.
    ///
    /// This field is only used for Cholesky decomposition, and will be ignored for eigen
    /// decomposition.
    #[serde_inline_default(Upper)]
    pub uplo: FlagUpLo,
}

impl Default for J2CDecompOption {
    fn default() -> Self {
        J2CDecompOption { policy: J2CDecompPolicy::Eig, threshold: Some(J2C_THRESH), uplo: Upper }
    }
}

/// Output of decomposed intermediates for 2c-2e ERI matrix (j2c).
pub enum J2CDecompose {
    /// Cholesky decomposition
    ///
    /// Required fields:
    /// - `j2c_l`: the Cholesky factor of 2c-2e ERI
    /// - `uplo`: the flag indicating whether the Cholesky factor is upper or lower triangular
    ///
    /// Optional field:
    /// - `j2c_l_inv`: the inverse of the Cholesky factor, only for debugging and testing purposes;
    ///   use TRSM with `j2c_l` is recommended
    Cd { j2c_l: Tsr<f64>, uplo: FlagUpLo, j2c_l_inv: Option<Tsr<f64>> },

    /// Eigen decomposition
    ///
    /// Required fields:
    /// - `j2c_l_inv`: the -1/2 power of the 2c-2e ERI matrix
    /// - `threshold`: the threshold for eigenvalue decomposition
    ///
    /// Optional field:
    /// - `j2c_e`: the eigenvalues of the 2c-2e ERI matrix
    /// - `j2c_v`: the eigenvectors of the 2c-2e ERI matrix
    Eig { j2c_l_inv: Tsr<f64>, j2c_e: Option<Tsr<f64>>, j2c_v: Option<Tsr<f64>> },
}
