use cubecl_core::ir::{ElemType, StorageType, UIntKind};
use cubecl_core::prelude::KernelDefinition;
use cubecl_core::server::ExecutionMode;
use cubecl_runtime::compiler::{CompilationError, Compiler};

use crate::shared::{CompilationOptions, CppCompiler};
use crate::tt_metal::dialect::TtMetalDialect;

use super::detect::detect_op_kind;
use super::kernel::TtKernelSources;

/// Compile a `CubeCL` `KernelDefinition` into `TT-Metal` kernel sources.
///
/// Runs the IR through `CppCompiler` for validation, then uses template-driven
/// codegen to produce the reader, compute, and writer kernel sources.
///
/// The `num_tiles` and `tile_size_bytes` parameters specify the work size.
/// These typically come from the tensor dimensions and element type.
pub fn compile_to_tt_sources(
    kernel: &KernelDefinition,
    num_tiles: u32,
    tile_size_bytes: u32,
) -> Result<TtKernelSources, CompilationError> {
    // Validate the IR through the CppCompiler pipeline.
    // This catches errors in our Dialect trait implementations.
    // We don't use the formatted output — the compute kernel comes from templates.
    let mut compiler: CppCompiler<TtMetalDialect> = Default::default();
    let _ = compiler.compile(
        kernel.clone(),
        &CompilationOptions::default(),
        ExecutionMode::Checked,
        StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
    )?;

    // Detect the operation type
    let op_kind = detect_op_kind(kernel);

    // Generate kernel sources based on operation type
    match op_kind {
        super::detect::TtOpKind::Copy | super::detect::TtOpKind::Unknown => {
            Ok(TtKernelSources::copy_kernel(num_tiles, tile_size_bytes))
        }
        super::detect::TtOpKind::EltwiseBinaryAdd => {
            Ok(TtKernelSources::add_kernel(num_tiles, tile_size_bytes))
        }
    }
}
