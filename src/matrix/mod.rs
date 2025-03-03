//! Matrix and Matrix-like types for use in the Openqudit library.

mod matvec;
mod matmat;

/// Re-export of the macro constructors from the faer_core crate. 
pub use faer::mat as mat;

pub use faer::Row as Row;
pub use faer::Col as Col;

/// Re-export of the basic matrix types from the faer_core crate.
pub use faer::Mat as Mat;
pub use faer::MatMut as MatMut;
pub use faer::MatRef as MatRef;

/// Matrix Vector (3d tensor) type.
pub use matvec::MatVec;
pub use matvec::MatVecMut;
pub use matvec::MatVecRef;

/// Matrix Matrix (4d tensor) type.
pub use matmat::SymSqMatMat;
pub use matmat::SymSqMatMatMut;
pub use matmat::SymSqMatMatRef;

