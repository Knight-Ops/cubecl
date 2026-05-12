#[macro_use]
extern crate derive_new;

pub mod shared;

pub use shared::ComputeKernel;
pub use shared::register_supported_types;
pub use shared::{Dialect, DialectWmmaCompiler};

/// Format CPP code.
pub mod formatter;

//#[cfg(feature = "cuda")]
pub mod cuda;
//#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "tt_metal")]
pub mod tt_metal;

#[cfg(feature = "metal")]
pub type MslCompiler = shared::CppCompiler<metal::MslDialect>;
#[cfg(feature = "tt_metal")]
pub type TtCompiler = shared::CppCompiler<tt_metal::TtMetalDialect>;
