#![warn(missing_docs)]

//! Qudit-Core is the core package in the OpenQudit library.
mod radices;
mod scalar;
mod system;
mod function;
mod radix;
mod perm;
mod bitwidth;

pub mod accel;
pub mod matrix;
pub mod memory;
pub mod unitary;

pub use radices::QuditRadices;
pub use radices::ToRadices;
pub use radix::ToRadix;
pub use scalar::ComplexScalar;
pub use scalar::RealScalar;
pub use system::QuditSystem;
pub use system::HybridSystem;
pub use system::ClassicalSystem;
pub use function::HasBounds;
pub use function::HasPeriods;
pub use function::HasParams;
pub use perm::QuditPermutation;
pub use perm::calc_index_permutation;
pub use bitwidth::BitWidthConvertible;

////////////////////////////////////////////////////////////////////////
/// Complex number types.
////////////////////////////////////////////////////////////////////////
pub use faer::c32;
pub use faer::c64;
////////////////////////////////////////////////////////////////////////
