//! Matrix multiplication driver for DFT numerical integration.
#![doc = include_str!("docs/mod.md")]

pub mod nimatmul;
pub mod pure_eval_rho;
pub mod pure_xcpot;

#[allow(unused)]
pub mod prelude {
    pub use crate::prelude::*;

    pub(crate) use super::nimatmul::*;
    pub(crate) use super::pure_eval_rho::*;
    pub(crate) use super::pure_xcpot::*;

    pub(crate) use std::sync::Mutex;
}

#[allow(unused)]
use prelude::*;
