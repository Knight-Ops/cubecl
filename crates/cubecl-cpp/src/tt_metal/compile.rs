use cubecl_core::backtrace::BackTrace;
use cubecl_core::ir::BarrierLevel;
use cubecl_core::prelude::Visibility;
use cubecl_runtime::compiler::{CompilationError, Compiler};

use crate::shared::{AtomicKind, BarrierOps, CompilationOptions, ComputeKernel, Component, CppCompiler, Elem, Instruction, Item, Variable};
use crate::tt_metal::dialect::TtMetalDialect;

use super::kernel::{TtBinaryComputeOp, TtKernelSources, TtUnaryComputeOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SupportedComputeKind {
    Copy,
    Add,
    Sub,
    Mul,
    Div,
    Abs,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tan,
    Tanh,
    Exp,
    Log,
    Generic,
}

#[derive(Debug, Clone)]
pub(crate) struct TtKernelAnalysis {
    pub kind: SupportedComputeKind,
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub input_binding_indices: Vec<usize>,
    pub output_binding_indices: Vec<usize>,
    pub num_tiles: u32,
    pub tile_size_bytes: u32,
    pub data_format_tt: u8,
    pub unit_item: Item<TtMetalDialect>,
    pub unit_item_size_bytes: u32,
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
                analysis.unit_item_size_bytes,
            )
        }
        SupportedComputeKind::Add
        | SupportedComputeKind::Sub
        | SupportedComputeKind::Mul
        | SupportedComputeKind::Div => {
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
                SupportedComputeKind::Div => TtBinaryComputeOp::Div,
                _ => unreachable!("binary match arm only accepts binary ops"),
            };
            TtKernelSources::binary_kernel_with_format(
                op,
                analysis.num_tiles,
                analysis.tile_size_bytes,
                analysis.data_format_tt,
                analysis.unit_item_size_bytes,
            )
        }
        SupportedComputeKind::Abs
        | SupportedComputeKind::Sqrt
        | SupportedComputeKind::Rsqrt
        | SupportedComputeKind::Sin
        | SupportedComputeKind::Cos
        | SupportedComputeKind::Tan
        | SupportedComputeKind::Tanh
        | SupportedComputeKind::Exp
        | SupportedComputeKind::Log => {
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
                SupportedComputeKind::Sin => TtUnaryComputeOp::Sin,
                SupportedComputeKind::Cos => TtUnaryComputeOp::Cos,
                SupportedComputeKind::Tan => TtUnaryComputeOp::Tan,
                SupportedComputeKind::Tanh => TtUnaryComputeOp::Tanh,
                SupportedComputeKind::Exp => TtUnaryComputeOp::Exp,
                SupportedComputeKind::Log => TtUnaryComputeOp::Log,
                _ => unreachable!("unary match arm only accepts unary ops"),
            };
            TtKernelSources::unary_kernel_with_format(
                op,
                analysis.num_tiles,
                analysis.tile_size_bytes,
                analysis.data_format_tt,
                analysis.unit_item_size_bytes,
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
            analysis.unit_item_size_bytes,
        ),
    };

    sources = sources
        .with_binding_indices(
            analysis.input_binding_indices,
            analysis.output_binding_indices,
        )
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
    let atomic_read_writeback = uses_atomic_writer_output(&repr.body.instructions);
    if atomic_read_writeback && repr.cube_dim.num_elems() != 1 {
        return Err(unsupported_reason(
            "TT-Metal atomic bring-up currently only supports single-unit cube dimensions",
        ));
    }
    if !repr.body.pipelines.is_empty() {
        return Err(unsupported_reason(
            "TT-Metal Phase 1 does not support pipeline operations",
        ));
    }
    if repr.cube_dim.z != 1 {
        return Err(unsupported_reason(
            "TT-Metal Phase 1 currently only supports z=1 cube dimensions",
        ));
    }
    if let Some(cluster_dim) = repr.cluster_dim {
        if cluster_dim.x != 1 || cluster_dim.y != 1 || cluster_dim.z != 1 {
            return Err(unsupported_reason(
                "TT-Metal Phase 1 does not support distributed or clustered execution",
            ));
        }
    }

    let input_binding_indices = repr
        .buffers
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| {
            let needs_atomic_writeback = atomic_read_writeback
                && matches!(binding.vis, Visibility::Read)
                && binding.item.is_atomic();
            (!needs_atomic_writeback && matches!(binding.vis, Visibility::Read)).then_some(index)
        })
        .collect::<Vec<_>>();
    let output_binding_indices = repr
        .buffers
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| {
            let needs_atomic_writeback = atomic_read_writeback
                && matches!(binding.vis, Visibility::Read)
                && binding.item.is_atomic();
            (matches!(binding.vis, Visibility::ReadWrite) || needs_atomic_writeback)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let num_inputs = input_binding_indices.len() as u32;
    let num_outputs = output_binding_indices.len() as u32;

    if num_outputs == 0 {
        return Err(unsupported_reason(
            "TT-Metal backend currently requires at least one writable output buffer",
        ));
    }

    for binding in &repr.buffers {
        let elem = normalize_buffer_elem(binding.item.elem.unpacked());
        if !is_supported_buffer_elem(elem) {
            return Err(CompilationError::UnsupportedInstruction {
                reason: format!(
                    "TT-Metal Phase 1 supports BF16/F16/F32, reinterpret-friendly integer buffer elements, and the proven Atomic<u32> load/store buffer shape only, found {}",
                    elem.ident()
                ),
                backtrace: BackTrace::capture(),
            });
        }
    }

    let unit_item = select_unit_item(repr)?;
    let (data_format_tt, tile_size_bytes) =
        infer_data_format_and_tile_size(normalize_buffer_elem(unit_item.elem.unpacked()))?;
    let buffer_item_sizes = repr
        .buffers
        .iter()
        .map(|binding| binding.item.size() as u32)
        .collect::<Vec<_>>();

    let supports_cube_shared_barrier_subset = supports_cube_shared_barrier_subset(repr);
    let mut kind = detect_supported_compute_kind(repr)?;
    if !repr.body.shared_memories.is_empty() {
        if repr.cube_dim.num_elems() != 1 && !supports_cube_shared_barrier_subset {
            return Err(unsupported_reason(
                "TT-Metal shared-memory bring-up currently only supports single-unit scratch or the proven cube-shared barrier memcpy subset",
            ));
        }
        // The current TT shared-memory slice is emitted by the generic scalar writer.
        // Keep all native tiled paths blocked until we have a truthful shared-layout model.
        kind = SupportedComputeKind::Generic;
    }
    kind = match kind {
        SupportedComputeKind::Copy if num_inputs == 1 && num_outputs == 1 => {
            SupportedComputeKind::Copy
        }
        SupportedComputeKind::Add
        | SupportedComputeKind::Sub
        | SupportedComputeKind::Mul
        | SupportedComputeKind::Div
            if num_inputs == 2 && num_outputs == 1 =>
        {
            kind
        }
        SupportedComputeKind::Abs
        | SupportedComputeKind::Sqrt
        | SupportedComputeKind::Rsqrt
        | SupportedComputeKind::Sin
        | SupportedComputeKind::Cos
        | SupportedComputeKind::Tan
        | SupportedComputeKind::Tanh
        | SupportedComputeKind::Exp
        | SupportedComputeKind::Log
            if num_inputs == 1 && num_outputs == 1 =>
        {
            kind
        }
        SupportedComputeKind::Copy
        | SupportedComputeKind::Add
        | SupportedComputeKind::Sub
        | SupportedComputeKind::Mul
        | SupportedComputeKind::Div
        | SupportedComputeKind::Abs
        | SupportedComputeKind::Sqrt
        | SupportedComputeKind::Rsqrt
        | SupportedComputeKind::Sin
        | SupportedComputeKind::Cos
        | SupportedComputeKind::Tan
        | SupportedComputeKind::Tanh
        | SupportedComputeKind::Exp
        | SupportedComputeKind::Log => SupportedComputeKind::Generic,
        SupportedComputeKind::Generic => SupportedComputeKind::Generic,
    };
    let num_tiles = match kind {
        SupportedComputeKind::Copy
        | SupportedComputeKind::Add
        | SupportedComputeKind::Sub
        | SupportedComputeKind::Mul
        | SupportedComputeKind::Div
        | SupportedComputeKind::Abs
        | SupportedComputeKind::Sqrt
        | SupportedComputeKind::Rsqrt
        | SupportedComputeKind::Sin
        | SupportedComputeKind::Cos
        | SupportedComputeKind::Tan
        | SupportedComputeKind::Tanh
        | SupportedComputeKind::Exp
        | SupportedComputeKind::Log => requested_num_tiles.max(1),
        SupportedComputeKind::Generic => {
            let tile_units = (tile_size_bytes as usize / unit_item.size()).max(1) as u32;
            repr.cube_dim.num_elems().div_ceil(tile_units).max(1)
        }
    };

    Ok(TtKernelAnalysis {
        kind,
        num_inputs,
        num_outputs,
        input_binding_indices,
        output_binding_indices,
        num_tiles,
        tile_size_bytes,
        data_format_tt,
        unit_item,
        unit_item_size_bytes: unit_item.size() as u32,
        buffer_item_sizes,
        info_static_len: repr.body.info_static_len,
    })
}

fn uses_atomic_writer_output(instructions: &[Instruction<TtMetalDialect>]) -> bool {
    instructions.iter().any(|instruction| match instruction {
        Instruction::AtomicLoad(_)
        | Instruction::AtomicStore(_)
                | Instruction::AtomicSub(_)
        | Instruction::AtomicMax(_)
        | Instruction::AtomicMin(_) => true,
        Instruction::RangeLoop { instructions, .. }
        | Instruction::Loop { instructions }
        | Instruction::If { instructions, .. } => uses_atomic_writer_output(instructions),
        Instruction::IfElse {
            instructions_if,
            instructions_else,
            ..
        } => {
            uses_atomic_writer_output(instructions_if)
                || uses_atomic_writer_output(instructions_else)
        }
        Instruction::Switch {
            instructions_default,
            instructions_cases,
            ..
        } => {
            uses_atomic_writer_output(instructions_default)
                || instructions_cases
                    .iter()
                    .any(|(_, instructions)| uses_atomic_writer_output(instructions))
        }
        _ => false,
    })
}

fn is_supported_single_unit_atomic_kind(kind: AtomicKind<TtMetalDialect>) -> bool {
    matches!(kind, AtomicKind::I32 | AtomicKind::U32 | AtomicKind::F32)
}

fn matches_atomic_plain_value(
    atomic_item: &Item<TtMetalDialect>,
    plain_item: &Item<TtMetalDialect>,
) -> bool {
    matches!(atomic_item.elem, Elem::Atomic(kind) if is_supported_single_unit_atomic_kind(kind)
        && plain_item.elem == kind.as_elem()
        && atomic_item.vectorization == plain_item.vectorization)
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

fn normalize_buffer_elem(elem: Elem<TtMetalDialect>) -> Elem<TtMetalDialect> {
    match elem {
        Elem::Atomic(kind) => kind.as_elem(),
        other => other,
    }
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

    let tile_size_bytes = match elem {
        // TT's logical scalar path for 32-bit integers currently aligns with the
        // 2 KiB MeshBuffer page granularity rather than a 4 KiB 32x32 scalar tile.
        Elem::I32 | Elem::U32 => 2048,
        _ => (32 * 32 * elem.size()) as u32,
    };

    Ok((data_format_tt, tile_size_bytes))
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

fn supports_cube_shared_barrier_subset(repr: &ComputeKernel<TtMetalDialect>) -> bool {
    repr.body.instructions.iter().all(|instruction| {
        matches!(
            instruction,
            Instruction::Barrier(BarrierOps::Declare { .. })
                | Instruction::Barrier(BarrierOps::Init { .. })
                | Instruction::Barrier(BarrierOps::ArriveAndWait { .. })
                | Instruction::Barrier(BarrierOps::MemCopyAsync {
                    cooperative: false,
                    ..
                })
                | Instruction::SyncThreads
                | Instruction::Metadata { .. }
                | Instruction::ExtendedMetadata { .. }
                | Instruction::ConstLength { .. }
                | Instruction::SliceLength { .. }
                | Instruction::DeclareVariable { .. }
                | Instruction::Assign(_)
                | Instruction::Index(_)
                | Instruction::IndexAssign(_)
                | Instruction::Add(_)
                | Instruction::Mul(_)
                | Instruction::RangeLoop { .. }
        ) || !matches!(instruction, Instruction::Barrier(_) | Instruction::SyncThreads)
    })
}


fn detect_supported_compute_kind(
    repr: &ComputeKernel<TtMetalDialect>,
) -> Result<SupportedComputeKind, CompilationError> {
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

    fn is_bounds_check_compare<'a>(
        instruction: &'a Instruction<TtMetalDialect>,
        next: Option<&'a Instruction<TtMetalDialect>>,
    ) -> Option<&'a [Instruction<TtMetalDialect>]> {
        let (out, lhs) = match instruction {
            Instruction::Equal(op)
            | Instruction::NotEqual(op)
            | Instruction::Lower(op)
            | Instruction::Greater(op)
            | Instruction::LowerEqual(op)
            | Instruction::GreaterEqual(op) => (&op.out, &op.lhs),
            _ => return None,
        };

        if !matches!(lhs, Variable::AbsolutePos(_)) {
            return None;
        }

        match next {
            Some(Instruction::If { cond, instructions }) if cond == out => Some(instructions),
            _ => None,
        }
    }

    fn is_tt_native_elem(elem: &Elem<TtMetalDialect>) -> bool {
        matches!(elem, Elem::BF16 | Elem::F32)
    }

    fn visit_instructions(
        instructions: &[Instruction<TtMetalDialect>],
        compute_kind: &mut SupportedComputeKind,
    ) -> Result<(), CompilationError> {
        let mut index = 0;
        while index < instructions.len() {
            let instruction = &instructions[index];
            if let Some(nested) = is_bounds_check_compare(instruction, instructions.get(index + 1)) {
                visit_instructions(nested, compute_kind)?;
                index += 2;
                continue;
            }

            match instruction {
                Instruction::DeclareVariable { .. }
                | Instruction::Assign(_)
                | Instruction::Metadata { .. }
                | Instruction::ExtendedMetadata { .. }
                | Instruction::ConstLength { .. }
                | Instruction::SliceLength { .. }
                | Instruction::Comment { .. }
                | Instruction::Line { .. }
                | Instruction::Return
                | Instruction::Index(_)
                | Instruction::IndexAssign(_)
                | Instruction::Bitcast(_)
                | Instruction::SpecialCast(_)
                | Instruction::Unreachable => {}
                Instruction::Add(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Add)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Sub(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Sub)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Mul(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Mul)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Div(op) | Instruction::FastDiv(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Div)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Abs(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Abs)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Sqrt(op) | Instruction::FastSqrt(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Sqrt)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::InverseSqrt(op) | Instruction::FastInverseSqrt(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Rsqrt)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Sin(op) | Instruction::FastSin(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Sin)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Cos(op) | Instruction::FastCos(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Cos)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Tan(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Tan)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Tanh(op) | Instruction::FastTanh(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Tanh)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Exp(op) | Instruction::FastExp(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Exp)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::Log(op) | Instruction::FastLog(op) => {
                    if is_tt_native_elem(&op.out.item().elem) {
                        set_native_kind(compute_kind, SupportedComputeKind::Log)
                    } else {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                }
                Instruction::If { instructions, .. } => {
                    visit_instructions(instructions, compute_kind)?;
                }
                Instruction::RangeLoop { instructions, .. } => {
                    *compute_kind = SupportedComputeKind::Generic;
                    visit_instructions(instructions, compute_kind)?;
                }
                Instruction::Switch {
                    instructions_default,
                    instructions_cases,
                    ..
                } => {
                    *compute_kind = SupportedComputeKind::Generic;
                    visit_instructions(instructions_default, compute_kind)?;
                    for (_, instructions) in instructions_cases {
                        visit_instructions(instructions, compute_kind)?;
                    }
                }
                Instruction::Select { .. }
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
                | Instruction::IfElse { .. }
                | Instruction::Slice { .. }
                | Instruction::CheckedSlice { .. }
                | Instruction::ReinterpretSlice { .. }
                | Instruction::Copy { .. }
                | Instruction::CopyBulk { .. }
                | Instruction::SaturatingAdd(_)
                | Instruction::SaturatingSub(_)
                | Instruction::HiMul(_)
                | Instruction::CountBits(_)
                | Instruction::ReverseBits(_)
                | Instruction::BitwiseNot(_)
                | Instruction::LeadingZeros(_)
                | Instruction::TrailingZeros(_)
                | Instruction::FindFirstSet(_)
                | Instruction::VectorSum(_)
                | Instruction::Break
                | Instruction::Modulo(_)
                | Instruction::BitwiseAnd(_)
                | Instruction::BitwiseOr(_)
                | Instruction::BitwiseXor(_)
                | Instruction::ShiftLeft(_)
                | Instruction::ShiftRight(_) => {
                    *compute_kind = SupportedComputeKind::Generic;
                }
                Instruction::Loop { instructions } => {
                    *compute_kind = SupportedComputeKind::Generic;
                    visit_instructions(instructions, compute_kind)?;
                }
                Instruction::Barrier(barrier_ops) => match barrier_ops {
                    BarrierOps::Declare { .. }
                    | BarrierOps::Init {
                        level: BarrierLevel::Unit | BarrierLevel::Cube,
                        ..
                    }
                    | BarrierOps::ArriveAndWait {
                        level: BarrierLevel::Unit | BarrierLevel::Cube,
                        ..
                    }
                    | BarrierOps::MemCopyAsync {
                        cooperative: false,
                        ..
                    } => {
                        *compute_kind = SupportedComputeKind::Generic;
                    }
                    _ => {
                        return Err(CompilationError::UnsupportedInstruction {
                            reason: "TT-Metal shared-memory/barrier bring-up currently only supports unit-local barriers and the proven cube-shared memcpy_async subset".into(),
                            backtrace: BackTrace::capture(),
                        });
                    }
                },
                Instruction::AtomicLoad(op) => {
                    if matches!(op.input.item().elem, Elem::Atomic(AtomicKind::U32)) {
                        *compute_kind = SupportedComputeKind::Generic;
                    } else {
                        return Err(CompilationError::UnsupportedInstruction {
                            reason: "TT-Metal atomic bring-up currently only supports single-unit Atomic<i32>/Atomic<u32>/Atomic<f32> load/store/add/min/max".into(),
                            backtrace: BackTrace::capture(),
                        });
                    }
                }
                Instruction::AtomicStore(op) => {
                    if matches!(op.out.item().elem, Elem::Atomic(AtomicKind::U32)) {
                        *compute_kind = SupportedComputeKind::Generic;
                    } else {
                        return Err(CompilationError::UnsupportedInstruction {
                            reason: "TT-Metal atomic bring-up currently only supports single-unit Atomic<i32>/Atomic<u32>/Atomic<f32> load/store/add/min/max".into(),
                            backtrace: BackTrace::capture(),
                        });
                    }
                }
                Instruction::AtomicAdd(op) => {
                    if matches_atomic_plain_value(&op.lhs.item(), &op.rhs.item())
                        && op.out.item() == op.rhs.item()
                    {
                        *compute_kind = SupportedComputeKind::Generic;
                    } else {
                        return Err(CompilationError::UnsupportedInstruction {
                            reason: "TT-Metal atomic bring-up currently only supports single-unit Atomic<i32>/Atomic<u32>/Atomic<f32> load/store/add/min/max".into(),
                            backtrace: BackTrace::capture(),
                        });
                    }
                }
                Instruction::AtomicMax(op) | Instruction::AtomicMin(op) => {
                    if matches_atomic_plain_value(&op.lhs.item(), &op.rhs.item())
                        && op.out.item() == op.rhs.item()
                    {
                        *compute_kind = SupportedComputeKind::Generic;
                    } else {
                        return Err(CompilationError::UnsupportedInstruction {
                            reason: "TT-Metal atomic bring-up currently only supports single-unit Atomic<i32>/Atomic<u32>/Atomic<f32> load/store/add/min/max".into(),
                            backtrace: BackTrace::capture(),
                        });
                    }
                }
                Instruction::SyncThreads | Instruction::SyncWarp => {
                    *compute_kind = SupportedComputeKind::Generic;
                }
                Instruction::Warp(_) => {
                    *compute_kind = SupportedComputeKind::Generic;
                }
                Instruction::ThreadFence
                | Instruction::ProxyAsyncToSharedFence
                | Instruction::BulkCommitGroup
                | Instruction::BulkWaitGroup { .. }
                | Instruction::BulkWaitGroupRead { .. }
                | Instruction::Wmma(_)
                | Instruction::AtomicSwap(_)
                | Instruction::AtomicSub(_)
                | Instruction::AtomicAnd(_)
                | Instruction::AtomicOr(_)
                | Instruction::AtomicXor(_)
                | Instruction::AtomicCAS { .. }
                | Instruction::Neg(_)
                | Instruction::Magnitude(_)
                | Instruction::FastMagnitude(_)
                | Instruction::Normalize(_)
                | Instruction::FastNormalize(_)
                | Instruction::FastRecip(_)
                | Instruction::Remainder(_)
                | Instruction::Log1p(_)
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
                | Instruction::Dot(_)
                | Instruction::TmaReplacePointer { .. } => {
                    return Err(unsupported_compute_instruction(instruction));
                }
                _ => return Err(unsupported_compute_instruction(instruction)),
            }
            index += 1;
        }

        Ok(())
    }

    let mut compute_kind = SupportedComputeKind::Copy;
    visit_instructions(&repr.body.instructions, &mut compute_kind)?;
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
