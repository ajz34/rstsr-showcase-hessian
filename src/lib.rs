#![allow(clippy::deref_addrof)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(mixed_script_confusables)]

pub mod hessian;
pub mod ri_jk;
pub mod util;

pub mod prelude {
    #![allow(unused)]

    pub use crate::hessian::prelude::*;
    pub use crate::ri_jk::prelude::*;

    pub(crate) use crate::check_shape;
    pub(crate) use crate::prelude_dev::*;
    pub(crate) use crate::util::prelude::*;
}

pub mod prelude_dev {
    pub use core::assert_matches;
    pub use itertools::Itertools;
    pub use libcint::prelude::*;
    pub use rayon::prelude::*;
    pub use rstsr::prelude::*;
    pub use std::collections::HashMap;

    pub type DeviceTsr = DeviceFaer;
    pub type Tsr<T = f64> = Tensor<T, DeviceTsr, IxD>;
    pub type TsrView<'a, T = f64> = TensorView<'a, T, DeviceTsr, IxD>;
    pub type TsrMut<'a, T = f64> = TensorMut<'a, T, DeviceTsr, IxD>;
    pub type TsrCow<'a, T = f64> = TensorCow<'a, T, DeviceTsr, IxD>;
}
