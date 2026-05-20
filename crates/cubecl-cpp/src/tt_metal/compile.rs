use cubecl_core::backtrace::BackTrace;
use cubecl_core::prelude::Visibility;
use cubecl_runtime::compiler::{CompilationError, Compiler};

use crate::shared::{CompilationOptions, ComputeKernel, CppCompiler, Elem, Instruction, Item};
use crate::tt_metal::dialect::TtMetalDialect;

use super::kernel::{TtBinaryComputeOp, TtKernelSources, TtUnaryComputeOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SupportedComputeKind {
    Copy,
    Add,
    Sub,
    Mul,
    Abs,
    Sqrt,
    Rsqrt,
    Generic,
}

#[derive(Debug, Clone)]
pub(crate) struct TtKernelAnalysis {
    pub kind: SupportedComputeKind,
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub num_tiles: u32,
    pub tile_size_bytes: u32,
    pub data_format_tt: u8,
    pub unit_item: Item<TtMetalDialect>,
    pub buffer_item_sizes: Vec<u32>,
    pub info_static_len: usize,
}

/// Compile a `CubeCL` `KernelDefinition` into `TT-Metal` kernel sources.
///
/// Runs the IR through `CppCompiler`, validates that the kernel shape fits the
/// current single-core tiled TT-Metal backend, and emits the three TT kernel
/// sources needed by the runtime.
pub fn compile_to_tt_sources(
    kernel: &cubecl_core::prelude::KernelDefinition,
    num_tiles: u32,
    tile_size_bytes: u32,
) -> Result<TtKernelSources, CompilationError> {
    let mut compiler: CppCompiler<TtMetalDialect> = Default::default();
    let repr = compiler.compile(
        kernel.clone(),
        &CompilationOptions::default(),
        cubecl_core::server::ExecutionMode::Checked,
        cubecl_core::ir::StorageType::Scalar(cubecl_core::ir::ElemType::UInt(
            cubecl_core::ir::UIntKind::U32,
        )),
    )?;

    let sources = sources_from_repr(&repr, num_tiles)?;
    if tile_size_bytes != sources.tile_size_bytes {
        return Err(CompilationError::Generic {
            reason: format!(
                "TT-Metal tile size mismatch: caller requested {tile_size_bytes} bytes, but kernel element type requires {} bytes",
                sources.tile_size_bytes
            ),
            backtrace: BackTrace::capture(),
        });
    }

    Ok(sources)
}

pub fn sources_from_repr(
    repr: &ComputeKernel<TtMetalDialect>,
    num_tiles: u32,
) -> Result<TtKernelSources, CompilationError> {
    let analysis = analyze_kernel(repr, num_tiles)?;

    let mut sources = match analysis.kind {
        SupportedComputeKind::Copy => {
            if analysis.num_inputs != 1 || analysis.num_outputs != 1 {
                return Err(CompilationError::UnsupportedInstruction {
                    reason: format!(
                        "TT-Metal copy kernels currently require 1 input and 1 output buffer, found {} inputs and {} outputs",
                        analysis.num_inputs, analysis.num_outputs,
                    ),
                    backtrace: BackTrace::capture(),
                });
            }
            TtKernelSources::copy_kernel_with_format(
                analysis.num_tiles,
                analysis.tile_size_bytes,
                analysis.data_format_tt,
            )
        }
        SupportedComputeKind::Add | SupportedComputeKind::Sub | SupportedComputeKind::Mul => {
            if analysis.num_inputs != 2 || analysis.num_outputs != 1 {
                return Err(CompilationError::UnsupportedInstruction {
                    reason: format!(
                        "TT-Metal binary native kernels currently require 2 input buffers and 1 output buffer, found {} inputs and {} outputs",
                        analysis.num_inputs, analysis.num_outputs,
                    ),
                    backtrace: BackTrace::capture(),
                });
            }
            let op = match analysis.kind {
                SupportedComputeKind::Add => TtBinaryComputeOp::Add,
                SupportedComputeKind::Sub => TtBinaryComputeOp::Sub,
                SupportedComputeKind::Mul => TtBinaryComputeOp::Mul,
                _ => unreachable!("binary match arm only accepts binary ops"),
            };
            TtKernelSources::binary_kernel_with_format(
                op,
                analysis.num_tiles,
                analysis.tile_size_bytes,
                analysis.data_format_tt,
            )
        }
        SupportedComputeKind::Abs | SupportedComputeKind::Sqrt | SupportedComputeKind::Rsqrt => {
            if analysis.num_inputs != 1 || analysis.num_outputs != 1 {
                return Err(CompilationError::UnsupportedInstruction {
                    reason: format!(
                        "TT-Metal unary native kernels currently require 1 input buffer and 1 output buffer, found {} inputs and {} outputs",
                        analysis.num_inputs, analysis.num_outputs,
                    ),
                    backtrace: BackTrace::capture(),
                });
            }
            let op = match analysis.kind {
                SupportedComputeKind::Abs => TtUnaryComputeOp::Abs,
                SupportedComputeKind::Sqrt => TtUnaryComputeOp::Sqrt,
                SupportedComputeKind::Rsqrt => TtUnaryComputeOp::Rsqrt,
                _ => unreachable!("unary match arm only accepts unary ops"),
            };
            TtKernelSources::unary_kernel_with_format(
                op,
                analysis.num_tiles,
                analysis.tile_size_bytes,
                analysis.data_format_tt,
            )
        }
        SupportedComputeKind::Generic => TtKernelSources::new(
            super::reader::generate_scalar_reader_source(analysis.num_inputs),
            super::writer::generate_noop_compute_source(),
            super::writer::generate_scalar_writer_source(repr, &analysis),
            analysis.num_inputs,
            analysis.num_outputs,
            analysis.num_tiles,
            analysis.tile_size_bytes,
            analysis.data_format_tt,
        ),
    };

    sources = sources
        .with_buffer_item_sizes(analysis.buffer_item_sizes)
        .with_info_static_len(analysis.info_static_len);

    Ok(sources)
}

fn analyze_kernel(
    repr: &ComputeKernel<TtMetalDialect>,
    requested_num_tiles: u32,
) -> Result<TtKernelAnalysis, CompilationError> {
    if !repr.tensor_maps.is_empty() {
        return Err(unsupported_reason(
            "TT-Metal Phase 1 does not support tensor maps",
        ));
    }
    if !repr.body.shared_memories.is_empty() {
        return Err(unsupported_reason(
            "TT-Metal Phase 1 does not support shared memory kernels",
        ));
    }
    if !repr.body.local_arrays.is_empty() {
        return Err(unsupported_reason(
            "TT-Metal Phase 1 does not support local array kernels",
        ));
    }
    if !repr.body.pipelines.is_empty() || !repr.body.barriers.is_empty() {
        return Err(unsupported_reason(
            "TT-Metal Phase 1 does not support pipeline or barrier operations",
        ));
    }
    if repr.cube_dim.y != 1 || repr.cube_dim.z != 1 {
        return Err(unsupported_reason(
            "TT-Metal Phase 1 only supports single-axis cube dimensions",
        ));
    }
    if let Some(cluster_dim) = repr.cluster_dim {
        if cluster_dim.x != 1 || cluster_dim.y != 1 || cluster_dim.z != 1 {
            return Err(unsupported_reason(
                "TT-Metal Phase 1 does not support distributed or clustered execution",
            ));
        }
    }

    let num_inputs = repr
        .buffers
        .iter()
        .filter(|binding| matches!(binding.vis, Visibility::Read))
        .count() as u32;
    let num_outputs = repr
        .buffers
        .iter()
        .filter(|binding| matches!(binding.vis, Visibility::ReadWrite))
        .count() as u32;

    if num_outputs == 0 {
        return Err(unsupported_reason(
            "TT-Metal backend currently requires at least one writable output buffer",
        ));
    }

    for binding in &repr.buffers {
        let elem = binding.item.elem.unpacked();
        if !is_supported_buffer_elem(elem) {
            return Err(CompilationError::UnsupportedInstruction {
                reason: format!(
                    "TT-Metal Phase 1 supports BF16/F16/F32 and reinterpret-friendly integer buffer elements only, found {}",
                    elem.ident()
                ),
                backtrace: BackTrace::capture(),
            });
        }
    }

    let unit_item = select_unit_item(repr)?;
    let (data_format_tt, tile_size_bytes) =
        infer_data_format_and_tile_size(unit_item.elem.unpacked())?;
    let buffer_item_sizes = repr
        .buffers
        .iter()
        .map(|binding| binding.item.size() as u32)
        .collect::<Vec<_>>();

    let kind = detect_supported_compute_kind(repr)?;
    let num_tiles = match kind {
        SupportedComputeKind::Copy
        | SupportedComputeKind::Add
        | SupportedComputeKind::Sub
        | SupportedComputeKind::Mul
        | SupportedComputeKind::Abs
        | SupportedComputeKind::Sqrt
        | SupportedComputeKind::Rsqrt => requested_num_tiles.max(1),
        SupportedComputeKind::Generic => {
            let tile_units = (tile_size_bytes as usize / unit_item.size()).max(1) as u32;
            repr.cube_dim.x.div_ceil(tile_units).max(1)
        }
    };

    Ok(TtKernelAnalysis {
        kind,
        num_inputs,
        num_outputs,
        num_tiles,
        tile_size_bytes,
        data_format_tt,
        unit_item,
        buffer_item_sizes,
        info_static_len: repr.body.info_static_len,
    })
}

fn select_unit_item(
    repr: &ComputeKernel<TtMetalDialect>,
) -> Result<Item<TtMetalDialect>, CompilationError> {
    let preferred = repr
        .buffers
        .iter()
        .find(|binding| {
            matches!(binding.vis, Visibility::ReadWrite) && binding.item.vectorization == 1
        })
        .or_else(|| {
            repr.buffers.iter().find(|binding| {
                matches!(binding.vis, Visibility::Read) && binding.item.vectorization == 1
            })
        })
        .or_else(|| {
            repr.buffers
                .iter()
                .find(|binding| matches!(binding.vis, Visibility::ReadWrite))
        })
        .or_else(|| repr.buffers.first());

    preferred
        .map(|binding| binding.item)
        .ok_or_else(|| unsupported_reason("TT-Metal backend requires at least one buffer binding"))
}

fn infer_data_format_and_tile_size(
    elem: Elem<TtMetalDialect>,
) -> Result<(u8, u32), CompilationError> {
    let data_format_tt = match elem {
        Elem::BF16 => 5,
        Elem::F16 => 1,
        Elem::F32 => 0,
        Elem::I8 | Elem::U8 | Elem::Bool => 30,
        Elem::I16 | Elem::U16 => 9,
        Elem::I32 => 8,
        Elem::U32 => 24,
        other => {
            return Err(CompilationError::UnsupportedInstruction {
                reason: format!(
                    "TT-Metal Phase 1 cannot infer a tile data format for {}",
                    other.ident()
                ),
                backtrace: BackTrace::capture(),
            });
        }
    };

    Ok((data_format_tt, (32 * 32 * elem.size()) as u32))
}

fn is_supported_buffer_elem(elem: Elem<TtMetalDialect>) -> bool {
    matches!(
        elem,
        Elem::BF16
            | Elem::F16
            | Elem::F32
            | Elem::I8
            | Elem::U8
            | Elem::I16
            | Elem::U16
            | Elem::I32
            | Elem::U32
            | Elem::Bool
    )
}

fn detect_supported_compute_kind(
    repr: &ComputeKernel<TtMetalDialect>,
) -> Result<SupportedComputeKind, CompilationError> {
    let mut compute_kind = SupportedComputeKind::Copy;

    fn set_native_kind(
        compute_kind: &mut SupportedComputeKind,
        new_kind: SupportedComputeKind,
    ) {
        if matches!(*compute_kind, SupportedComputeKind::Copy) {
            *compute_kind = new_kind;
        } else if *compute_kind != new_kind {
            *compute_kind = SupportedComputeKind::Generic;
        }
    }

    for instruction in &repr.body.instructions {
        match instruction {
            Instruction::DeclareVariable { .. }
            | Instruction::Assign(_)
            | Instruction::Metadata { .. }
            | Instruction::ExtendedMetadata { .. }
            | Instruction::ConstLength { .. }
            | Instruction::SliceLength { .. }
            | Instruction::Comment { .. }
            | Instruction::Line { .. }
            | Instruction::Return => {}
            Instruction::Add(_) => set_native_kind(&mut compute_kind, SupportedComputeKind::Add),
            Instruction::Sub(_) => set_native_kind(&mut compute_kind, SupportedComputeKind::Sub),
            Instruction::Mul(_) => set_native_kind(&mut compute_kind, SupportedComputeKind::Mul),
            Instruction::Abs(_) => set_native_kind(&mut compute_kind, SupportedComputeKind::Abs),
            Instruction::Sqrt(_) | Instruction::FastSqrt(_) => {
                set_native_kind(&mut compute_kind, SupportedComputeKind::Sqrt)
            }
            Instruction::InverseSqrt(_) | Instruction::FastInverseSqrt(_) => {
                set_native_kind(&mut compute_kind, SupportedComputeKind::Rsqrt)
            }
            Instruction::Index(_)
            | Instruction::IndexAssign(_)
            | Instruction::Bitcast(_)
            | Instruction::Select { .. }
            | Instruction::Equal(_)
            | Instruction::NotEqual(_)
            | Instruction::Lower(_)
            | Instruction::Greater(_)
            | Instruction::LowerEqual(_)
            | Instruction::GreaterEqual(_)
            | Instruction::Not(_)
            | Instruction::Or(_)
            | Instruction::And(_)
            | Instruction::Min(_)
            | Instruction::Max(_)
            | Instruction::If { .. }
            | Instruction::IfElse { .. }
            | Instruction::Slice { .. }
            | Instruction::CheckedSlice { .. }
            | Instruction::ReinterpretSlice { .. }
            | Instruction::Copy { .. }
            | Instruction::CopyBulk { .. } => {
                compute_kind = SupportedComputeKind::Generic;
            }
            Instruction::RangeLoop { .. }
            | Instruction::Loop { .. }
            | Instruction::Switch { .. }
            | Instruction::Break
            | Instruction::SyncThreads
            | Instruction::SyncWarp
            | Instruction::ThreadFence
            | Instruction::ProxyAsyncToSharedFence
            | Instruction::BulkCommitGroup
            | Instruction::BulkWaitGroup { .. }
            | Instruction::BulkWaitGroupRead { .. }
            | Instruction::Warp(_)
            | Instruction::Wmma(_)
            | Instruction::AtomicLoad(_)
            | Instruction::AtomicStore(_)
            | Instruction::AtomicSwap(_)
            | Instruction::AtomicAdd(_)
            | Instruction::AtomicSub(_)
            | Instruction::AtomicMax(_)
            | Instruction::AtomicMin(_)
            | Instruction::AtomicAnd(_)
            | Instruction::AtomicOr(_)
            | Instruction::AtomicXor(_)
            | Instruction::AtomicCAS { .. }
            | Instruction::SpecialCast(_)
            | Instruction::Neg(_)
            | Instruction::Magnitude(_)
            | Instruction::FastMagnitude(_)
            | Instruction::Normalize(_)
            | Instruction::FastNormalize(_)
            | Instruction::Div(_)
            | Instruction::FastDiv(_)
            | Instruction::FastRecip(_)
            | Instruction::Modulo(_)
            | Instruction::Remainder(_)
            | Instruction::Exp(_)
            | Instruction::FastExp(_)
            | Instruction::Log(_)
            | Instruction::FastLog(_)
            | Instruction::Log1p(_)
            | Instruction::Cos(_)
            | Instruction::Sin(_)
            | Instruction::Tan(_)
            | Instruction::Tanh(_)
            | Instruction::Sinh(_)
            | Instruction::Cosh(_)
            | Instruction::ArcCos(_)
            | Instruction::ArcSin(_)
            | Instruction::ArcTan(_)
            | Instruction::ArcSinh(_)
            | Instruction::ArcCosh(_)
            | Instruction::ArcTanh(_)
            | Instruction::Degrees(_)
            | Instruction::Radians(_)
            | Instruction::ArcTan2(_)
            | Instruction::FastSin(_)
            | Instruction::FastCos(_)
            | Instruction::FastTanh(_)
            | Instruction::Powf(_)
            | Instruction::FastPowf(_)
            | Instruction::Powi(_)
            | Instruction::Hypot(_)
            | Instruction::Rhypot(_)
            | Instruction::Clamp { .. }
            | Instruction::IsNan(_)
            | Instruction::IsInf(_)
            | Instruction::Round(_)
            | Instruction::Ceil(_)
            | Instruction::Trunc(_)
            | Instruction::Floor(_)
            | Instruction::Fma { .. }
            | Instruction::VecInit { .. }
            | Instruction::BitwiseOr(_)
            | Instruction::BitwiseAnd(_)
            | Instruction::BitwiseXor(_)
            | Instruction::CountBits(_)
            | Instruction::ReverseBits(_)
            | Instruction::ShiftLeft(_)
            | Instruction::ShiftRight(_)
            | Instruction::BitwiseNot(_)
            | Instruction::LeadingZeros(_)
            | Instruction::TrailingZeros(_)
            | Instruction::FindFirstSet(_)
            | Instruction::SaturatingAdd(_)
            | Instruction::SaturatingSub(_)
            | Instruction::HiMul(_)
            | Instruction::Dot(_)
            | Instruction::VectorSum(_)
            | Instruction::TmaReplacePointer { .. } => {
                return Err(unsupported_compute_instruction(instruction));
            }
            _ => return Err(unsupported_compute_instruction(instruction)),
        }
    }

    Ok(compute_kind)
}

fn unsupported_compute_instruction(instruction: &Instruction<TtMetalDialect>) -> CompilationError {
    CompilationError::UnsupportedInstruction {
        reason: format!("TT-Metal Phase 1 does not support instruction: {instruction:?}"),
        backtrace: BackTrace::capture(),
    }
}

fn unsupported_reason(reason: &str) -> CompilationError {
    CompilationError::UnsupportedInstruction {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}
