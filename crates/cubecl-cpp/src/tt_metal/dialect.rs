use std::collections::HashSet;
use std::fmt;
use std::fmt::Display;
use std::marker::PhantomData;

use cubecl_core::ir::Processor;

use crate::shared::{
    self, Component, DialectBindings, DialectCubeBuiltins, DialectIncludes,
    DialectInstructions, DialectProcessors, DialectTypes, DialectWarpReduceCompiler,
    DialectWmmaCompiler, Elem, Flags, Fragment, FragmentIdent, FragmentLayout, Item, KernelArg,
    ManualMma, SupportedMmaCombinations, SupportedScaledMmaCombinations, Variable, WarpInstruction,
    WmmaInstruction,
};
use crate::Dialect;
use crate::shared::Instruction;

use super::arch::TtArchitecture;

// ── Dialect struct ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TtMetalDialect<Wmma = super::wmma::TtNoWmma> {
    _wmma_compiler: PhantomData<Wmma>,
}

// ── Base Dialect ──────────────────────────────────────────────────────────

impl<Wmma: DialectWmmaCompiler<Self>> Dialect for TtMetalDialect<Wmma> {
    type Architecture = TtArchitecture;
}

impl<Wmma: DialectWmmaCompiler<Self>> DialectWarpReduceCompiler<Self> for TtMetalDialect<Wmma> {}

// ── Includes ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TtExtension {
    NoExtension,
}

impl<Wmma: DialectWmmaCompiler<Self>> DialectIncludes<Self> for TtMetalDialect<Wmma> {
    type Extension = TtExtension;

    fn compile_includes(f: &mut fmt::Formatter<'_>, _flags: &Flags<Self>) -> fmt::Result {
        write!(
            f,
            "#include \"compute_kernel_api.h\"\n#include \"compute_kernel_api/common.h\"\n#include \"compute_kernel_api/eltwise_binary.h\"\n"
        )
    }

    fn compile_extensions(_f: &mut fmt::Formatter<'_>, _extensions: &[Self::Extension]) -> fmt::Result {
        Ok(())
    }

    fn register_instruction_extension(
        _extensions: &mut Vec<Self::Extension>,
        _instruction: &Instruction<Self>,
    ) {
    }

    fn register_warp_instruction_extension(
        _extensions: &mut Vec<Self::Extension>,
        _instruction: &WarpInstruction<Self>,
    ) {
    }
}

// ── Types ─────────────────────────────────────────────────────────────────

impl<Wmma: DialectWmmaCompiler<Self>> DialectTypes<Self> for TtMetalDialect<Wmma> {
    fn item_can_be_optimized() -> bool {
        false
    }

    fn compile_elem(
        f: &mut fmt::Formatter<'_>,
        elem: &shared::Elem<Self>,
        _words: bool,
    ) -> fmt::Result {
        match elem {
            shared::Elem::F16 | shared::Elem::F16x2 => f.write_str("bfloat16"),
            shared::Elem::F32 => f.write_str("float"),
            shared::Elem::F64 => f.write_str("double"),
            shared::Elem::BF16 | shared::Elem::BF16x2 => f.write_str("bfloat16"),
            shared::Elem::TF32 => f.write_str("float"),
            shared::Elem::I8 => f.write_str("int8_t"),
            shared::Elem::I16 => f.write_str("int16_t"),
            shared::Elem::I32 => f.write_str("int32_t"),
            shared::Elem::I64 => f.write_str("int64_t"),
            shared::Elem::U8 => f.write_str("uint8_t"),
            shared::Elem::U16 => f.write_str("uint16_t"),
            shared::Elem::U32 => f.write_str("uint32_t"),
            shared::Elem::U64 => f.write_str("uint64_t"),
            shared::Elem::Bool => f.write_str("bool"),
            shared::Elem::Atomic(inner) => inner.fmt(f),
            shared::Elem::_Dialect(_) => Ok(()),
            _ => f.write_str("uint32_t"),
        }
    }

    fn compile_item(f: &mut fmt::Formatter<'_>, item: &Item<Self>) -> fmt::Result {
        if 1 == item.vectorization {
            return write!(f, "{}", item.elem);
        }
        Self::compile_elem(f, &item.elem, true)?;
        write!(f, "{}", item.vectorization)
    }

    fn compile_type_definitions(
        _f: &mut fmt::Formatter<'_>,
        _items: &HashSet<Item<Self>>,
        _scalars: &[(shared::Elem<Self>, usize)],
        _info: &cubecl_core::Info,
        _flags: &Flags<Self>,
    ) -> fmt::Result {
        Ok(())
    }

    fn compile_local_memory_qualifier(_f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

// ── Kernel Bindings ───────────────────────────────────────────────────────

impl<Wmma: DialectWmmaCompiler<Self>> DialectBindings<Self> for TtMetalDialect<Wmma> {
    fn compile_kernel_signature(
        f: &mut fmt::Formatter<'_>,
        _kernel_name: &str,
        _tensor_maps: &[KernelArg<Self>],
        _buffers: &[KernelArg<Self>],
        _flags: &Flags<Self>,
    ) -> fmt::Result {
        writeln!(f, "void kernel_main()")
    }
}

// ── Cube Builtins ─────────────────────────────────────────────────────────
// TT-Metal doesn't have threadIdx/blockIdx/blockDim.
// These are mapped to loop variables and compile-time constants.

impl<Wmma: DialectWmmaCompiler<Self>> DialectCubeBuiltins<Self> for TtMetalDialect<Wmma> {
    fn compile_absolute_pos_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "absolute_pos") }
    fn compile_absolute_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "absolute_pos") }
    fn compile_absolute_pos_x(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "i") }
    fn compile_absolute_pos_y(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
    fn compile_absolute_pos_z(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }

    fn compile_cube_count_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "cube_count") }
    fn compile_cube_count(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "cube_count") }
    fn compile_cube_count_x(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "1") }
    fn compile_cube_count_y(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "1") }
    fn compile_cube_count_z(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "1") }

    fn compile_cube_dim_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "cube_dim") }
    fn compile_cube_dim(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "cube_dim") }
    fn compile_cube_dim_x(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "num_tiles") }
    fn compile_cube_dim_y(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "1") }
    fn compile_cube_dim_z(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "1") }

    fn compile_cube_pos_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "cube_pos") }
    fn compile_cube_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "cube_pos") }
    fn compile_cube_pos_x(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
    fn compile_cube_pos_y(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
    fn compile_cube_pos_z(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }

    fn compile_unit_pos_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "unit_pos") }
    fn compile_unit_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "unit_pos") }
    fn compile_unit_pos_x(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "i") }
    fn compile_unit_pos_y(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
    fn compile_unit_pos_z(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }

    fn compile_plane_dim(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "32") }
    fn compile_plane_dim_checked(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "32") }
    fn compile_plane_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
    fn compile_unit_pos_plane(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "i % 32") }

    fn compile_cluster_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
    fn compile_cluster_pos_x(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
    fn compile_cluster_pos_y(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
    fn compile_cluster_pos_z(f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "0") }
}

// ── Instructions ──────────────────────────────────────────────────────────

impl<Wmma: DialectWmmaCompiler<Self>> DialectInstructions<Self> for TtMetalDialect<Wmma> {
    fn compile_saturating_add(
        _f: &mut fmt::Formatter<'_>,
        _lhs: impl Display,
        _rhs: impl Display,
        _item: Item<Self>,
    ) -> fmt::Result {
        unimplemented!("saturating_add not yet implemented for TT-Metal")
    }

    fn compile_saturating_sub(
        _f: &mut fmt::Formatter<'_>,
        _lhs: impl Display,
        _rhs: impl Display,
        _item: Item<Self>,
    ) -> fmt::Result {
        unimplemented!("saturating_sub not yet implemented for TT-Metal")
    }

    fn compile_instruction_sync_threads(_f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }

    fn compile_instruction_sync_warp(_f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }

    fn compile_instruction_thread_fence(_f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }

    fn compile_instruction_find_first_set<T: Component<Self>>(
        _f: &mut fmt::Formatter<'_>,
        _input: T,
        _out_elem: Elem<Self>,
    ) -> fmt::Result {
        unimplemented!("find_first_set not yet implemented for TT-Metal")
    }

    fn compile_instruction_leading_zeros_scalar<T: Component<Self>>(
        _f: &mut fmt::Formatter<'_>,
        _input: T,
        _out_elem: Elem<Self>,
    ) -> fmt::Result {
        unimplemented!("leading_zeros not yet implemented for TT-Metal")
    }

    fn compile_instruction_trailing_zeros_scalar<T: Component<Self>>(
        _f: &mut fmt::Formatter<'_>,
        _input: T,
        _out_elem: Elem<Self>,
    ) -> fmt::Result {
        unimplemented!("trailing_zeros not yet implemented for TT-Metal")
    }

    fn compile_instruction_max_function_name(
        f: &mut fmt::Formatter<'_>,
        _item: Item<Self>,
    ) -> fmt::Result {
        write!(f, "max")
    }

    fn compile_instruction_min_function_name(
        f: &mut fmt::Formatter<'_>,
        _item: Item<Self>,
    ) -> fmt::Result {
        write!(f, "min")
    }

    fn compile_warp_shuffle(
        _f: &mut fmt::Formatter<'_>,
        _var: &str,
        _source: &str,
    ) -> fmt::Result {
        unimplemented!("warp_shuffle not yet implemented for TT-Metal")
    }

    fn compile_warp_shuffle_xor(
        _f: &mut fmt::Formatter<'_>,
        _var: &str,
        _elem: &Elem<Self>,
        _offset: &str,
    ) -> fmt::Result {
        unimplemented!("warp_shuffle_xor not yet implemented for TT-Metal")
    }

    fn compile_warp_shuffle_up(
        _f: &mut fmt::Formatter<'_>,
        _var: &str,
        _offset: &str,
    ) -> fmt::Result {
        unimplemented!("warp_shuffle_up not yet implemented for TT-Metal")
    }

    fn compile_warp_shuffle_down(
        _f: &mut fmt::Formatter<'_>,
        _var: &str,
        _offset: &str,
    ) -> fmt::Result {
        unimplemented!("warp_shuffle_down not yet implemented for TT-Metal")
    }

    fn compile_warp_all<T: Component<Self>>(
        _f: &mut fmt::Formatter<'_>,
        _input: &T,
    ) -> fmt::Result {
        unimplemented!("warp_all not yet implemented for TT-Metal")
    }

    fn compile_warp_any<T: Component<Self>>(
        _f: &mut fmt::Formatter<'_>,
        _input: &T,
    ) -> fmt::Result {
        unimplemented!("warp_any not yet implemented for TT-Metal")
    }

    fn compile_warp_ballot(
        _f: &mut fmt::Formatter<'_>,
        _input: &Variable<Self>,
        _out_elem: &Elem<Self>,
    ) -> fmt::Result {
        unimplemented!("warp_ballot not yet implemented for TT-Metal")
    }

    fn compile_unreachable(_f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!("unreachable not yet implemented for TT-Metal")
    }
}

// ── WMMA ──────────────────────────────────────────────────────────────────

impl<Wmma: DialectWmmaCompiler<Self>> DialectWmmaCompiler<Self> for TtMetalDialect<Wmma> {
    fn compile_wmma_includes(f: &mut fmt::Formatter<'_>, flags: &Flags<Self>) -> fmt::Result {
        Wmma::compile_wmma_includes(f, flags)
    }
    fn compile_wmma_type_definitions(f: &mut fmt::Formatter<'_>, flags: &Flags<Self>) -> fmt::Result {
        Wmma::compile_wmma_type_definitions(f, flags)
    }
    fn compile_wmma_local_variables(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Wmma::compile_wmma_local_variables(f)
    }
    fn compile_wwma_fragment_ident(f: &mut fmt::Formatter<'_>, ident: &FragmentIdent<Self>) -> fmt::Result {
        Wmma::compile_wwma_fragment_ident(f, ident)
    }
    fn compile_wmma_fragment_layout(f: &mut fmt::Formatter<'_>, layout: &FragmentLayout<Self>) -> fmt::Result {
        Wmma::compile_wmma_fragment_layout(f, layout)
    }
    fn compile_wmma_fragment(f: &mut fmt::Formatter<'_>, fragment: &Fragment<Self>) -> fmt::Result {
        Wmma::compile_wmma_fragment(f, fragment)
    }
    fn compile_wmma_fragment_declaration(f: &mut fmt::Formatter<'_>, var: &Variable<Self>) -> fmt::Result {
        Wmma::compile_wmma_fragment_declaration(f, var)
    }
    fn compile_wmma_instruction(f: &mut fmt::Formatter<'_>, instruction: &WmmaInstruction<Self>) -> fmt::Result {
        Wmma::compile_wmma_instruction(f, instruction)
    }
    fn compile_manual_mma(f: &mut fmt::Formatter<'_>, mma: ManualMma<Self>) -> fmt::Result {
        Wmma::compile_manual_mma(f, mma)
    }
    fn compile_scaled_mma(
        f: &mut fmt::Formatter<'_>,
        mma: ManualMma<Self>,
        scales_a: Variable<Self>,
        scales_b: Variable<Self>,
        scales_factor: u32,
    ) -> fmt::Result {
        Wmma::compile_scaled_mma(f, mma, scales_a, scales_b, scales_factor)
    }
    fn supported_wmma_combinations(arch: &TtArchitecture) -> SupportedMmaCombinations {
        Wmma::supported_wmma_combinations(arch)
    }
    fn supported_mma_combinations(arch: &TtArchitecture) -> SupportedMmaCombinations {
        Wmma::supported_mma_combinations(arch)
    }
    fn supported_scaled_mma_combinations(arch: &TtArchitecture) -> SupportedScaledMmaCombinations {
        Wmma::supported_scaled_mma_combinations(arch)
    }
}

// ── Processors ────────────────────────────────────────────────────────────

impl<Wmma: DialectWmmaCompiler<Self>> DialectProcessors<Self> for TtMetalDialect<Wmma> {
    fn processors() -> Vec<Box<dyn Processor>> {
        Vec::new()
    }
}
