use std::fmt;

use crate::shared::{
    DialectWmmaCompiler, Flags, Fragment, FragmentIdent, FragmentLayout, ManualMma,
    SupportedMmaCombinations, SupportedScaledMmaCombinations, Variable, WmmaInstruction,
};

use super::TtArchitecture;
use super::dialect::TtMetalDialect;

/// No-op WMMA compiler for TT-Metal.
///
/// Tensix cores do not have tensor cores in the CUDA/HIP sense.
/// Matrix operations are performed by the FPU (matrix engine).
/// WMMA and MMA combinations are empty; compilation methods panic
/// if called (CubeCL will not generate WMMA ops without registered support).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TtNoWmma;

impl DialectWmmaCompiler<TtMetalDialect<TtNoWmma>> for TtNoWmma {
    fn compile_wmma_fragment_declaration(
        _f: &mut fmt::Formatter<'_>,
        _var: &Variable<TtMetalDialect<TtNoWmma>>,
    ) -> fmt::Result {
        unimplemented!("TT-Metal does not support WMMA fragment operations")
    }

    fn compile_wmma_instruction(
        _f: &mut fmt::Formatter<'_>,
        _instruction: &WmmaInstruction<TtMetalDialect<TtNoWmma>>,
    ) -> fmt::Result {
        unimplemented!("TT-Metal does not support WMMA instructions")
    }

    fn compile_manual_mma(
        _f: &mut fmt::Formatter<'_>,
        _mma: ManualMma<TtMetalDialect<TtNoWmma>>,
    ) -> fmt::Result {
        unimplemented!("TT-Metal does not support manual MMA")
    }

    fn compile_scaled_mma(
        _f: &mut fmt::Formatter<'_>,
        _mma: ManualMma<TtMetalDialect<TtNoWmma>>,
        _scales_a: Variable<TtMetalDialect<TtNoWmma>>,
        _scales_b: Variable<TtMetalDialect<TtNoWmma>>,
        _scales_factor: u32,
    ) -> fmt::Result {
        unimplemented!("TT-Metal does not support scaled MMA")
    }

    fn supported_wmma_combinations(_arch: &TtArchitecture) -> SupportedMmaCombinations {
        Vec::new()
    }

    fn supported_mma_combinations(_arch: &TtArchitecture) -> SupportedMmaCombinations {
        Vec::new()
    }

    fn supported_scaled_mma_combinations(
        _arch: &TtArchitecture,
    ) -> SupportedScaledMmaCombinations {
        Vec::new()
    }
}
