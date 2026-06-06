#![allow(clippy::deref_addrof)]
#![allow(clippy::manual_is_multiple_of)]

pub mod util;

pub mod prelude {
    #![allow(unused)]

    use super::*;

    pub(crate) use crate::check_shape;
    pub(crate) use util::prelude::*;

    pub(crate) use core::assert_matches;
    pub(crate) use itertools::Itertools;
    pub(crate) use libcint::prelude::*;
    pub(crate) use rayon::prelude::*;
    pub(crate) use rstsr::prelude::*;
    pub(crate) use std::collections::HashMap;

    pub(crate) type DeviceTsr = DeviceFaer;
    pub(crate) type Tsr<T = f64> = Tensor<T, DeviceTsr, IxD>;
    pub(crate) type TsrView<'a, T = f64> = TensorView<'a, T, DeviceTsr, IxD>;
    pub(crate) type TsrMut<'a, T = f64> = TensorMut<'a, T, DeviceTsr, IxD>;
    pub(crate) type TsrCow<'a, T = f64> = TensorCow<'a, T, DeviceTsr, IxD>;
}
