pub const AO_DERIV_DIM: [usize; 5] = [1, 4, 10, 20, 35];

/// Density type for numint.
///
/// - RHO: only density
/// - SIGMA: density + gradient
/// - TAU: density + gradient + kinetic energy density
/// - LAPL: density + gradient + kinetic energy density + laplacian
///
/// Note for this enum, each higher-level density type also contains all components of the
/// lower-level types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NIDenType {
    RHO,
    SIGMA,
    TAU,
    LAPL,
}

impl NIDenType {
    /// Returns the number of components in the output density for this XC type.
    ///
    /// - RHO: 1 component (density)
    /// - SIGMA: 4 components (density + 3 gradient components)
    /// - TAU: 5 components (density + 3 gradient components + kinetic energy density)
    /// - LAPL: 6 components (density + 3 gradient components + kinetic energy density + laplacian)
    pub fn num_nvar(&self) -> usize {
        match self {
            NIDenType::RHO => 1,
            NIDenType::SIGMA => 4,
            NIDenType::TAU => 5,
            NIDenType::LAPL => 6,
        }
    }

    /// Returns the required AO derivative level for this XC type.
    ///
    /// - RHO: 0th order
    /// - SIGMA: 1st order (gradient)
    /// - TAU: 1st order (gradient)
    /// - LAPL: 2nd order (Laplacian)
    pub fn num_ao_deriv(&self) -> usize {
        match self {
            NIDenType::RHO => 0,
            NIDenType::SIGMA => 1,
            NIDenType::TAU => 1,
            NIDenType::LAPL => 2,
        }
    }

    /// Returns the number of AO components needed for this XC type
    ///
    /// - RHO: 1 component (AO value)
    /// - SIGMA: 4 components (AO value + 3 gradient components)
    /// - TAU: 4 components (AO value + 3 gradient components) [
    /// - LAPL: 10 components (AO value + 3 gradient components + 6 second derivative components)
    pub fn num_ao_comp(&self) -> usize {
        AO_DERIV_DIM[self.num_ao_deriv()]
    }
}

/// Parallelization strategy for numint.
///
/// This enum allows three kinds of parallelization strategies by `From` trait implementations:
///
/// - usize number : parallel with given chunk size;
/// - None : Use default chunk size determined by the implementation function;
/// - bool : parallel with auto-chunking if true, or serial if false.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NIPar {
    Par { chunk_size: Option<usize> },
    Serial,
}

impl From<usize> for NIPar {
    fn from(chunk_size: usize) -> Self {
        NIPar::Par { chunk_size: Some(chunk_size) }
    }
}

impl From<Option<usize>> for NIPar {
    fn from(chunk_size: Option<usize>) -> Self {
        NIPar::Par { chunk_size }
    }
}

impl From<bool> for NIPar {
    fn from(parallel: bool) -> Self {
        if parallel {
            NIPar::Par { chunk_size: None }
        } else {
            NIPar::Serial
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NISpin {
    Unpolarized,
    Polarized,
}
