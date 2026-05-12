#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod compute;
pub mod device;
pub mod runtime;
pub use device::*;
pub use runtime::TtRuntime;

/// The default WMMA compiler for TT-Metal (no-op).
pub type TtWmmaCompiler = cubecl_cpp::tt_metal::TtNoWmma;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::TtRuntime;

    cubecl_std::testgen!();
    cubecl_core::testgen_all!(f32: [f32]);
}
