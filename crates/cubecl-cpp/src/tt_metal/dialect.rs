use std::collections::HashSet;
use std::fmt;
use std::fmt::Display;
use std::marker::PhantomData;

use cubecl_core::ir::Processor;

use crate::Dialect;
use crate::shared::Instruction;
use crate::shared::binary::{Add, Binary, Max, Min};
use crate::shared::{
    self, Component, DialectBindings, DialectCubeBuiltins, DialectIncludes, DialectInstructions,
    DialectProcessors, DialectTypes, DialectWarpReduceCompiler, DialectWmmaCompiler, Elem, Flags,
    FmtLeft, Fragment, FragmentIdent, FragmentLayout, Item, KernelArg, ManualMma,
    SupportedMmaCombinations, SupportedScaledMmaCombinations, Variable, WarpInstruction,
    WmmaInstruction,
};

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
            "#include <limits>\n#include \"api/compute/common.h\"\n#include \"api/compute/compute_kernel_api.h\"\n"
        )
    }

    fn compile_extensions(
        _f: &mut fmt::Formatter<'_>,
        _extensions: &[Self::Extension],
    ) -> fmt::Result {
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
            shared::Elem::F16 => f.write_str("uint16_t"),
            shared::Elem::F16x2 => f.write_str("uint32_t"),
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

    fn compile_shared_memory_declaration(
        f: &mut fmt::Formatter<'_>,
        shared: &shared::SharedMemory<Self>,
    ) -> fmt::Result {
        match shared {
            shared::SharedMemory::Array {
                index,
                item,
                length,
                align,
                ..
            } => {
                let size_bytes = length * item.size();
                let align = (*align).max(1);
                writeln!(
                    f,
                    "// TT shared scratch array size: {length}, {size_bytes} bytes"
                )?;
                writeln!(
                    f,
                    "alignas({align}) {item} shared_memory_{index}[{length}];"
                )
            }
            shared::SharedMemory::Value {
                index, item, align, ..
            } => {
                let size_bytes = item.size();
                let align = (*align).max(1);
                writeln!(f, "// TT shared scratch value size: {size_bytes} bytes")?;
                writeln!(f, "alignas({align}) {item} shared_memory_{index};")
            }
        }
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
    fn compile_absolute_pos_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "absolute_pos")
    }
    fn compile_absolute_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "absolute_pos")
    }
    fn compile_absolute_pos_x(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i")
    }
    fn compile_absolute_pos_y(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }
    fn compile_absolute_pos_z(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }

    fn compile_cube_count_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_count")
    }
    fn compile_cube_count(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_count")
    }
    fn compile_cube_count_x(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_count_x")
    }
    fn compile_cube_count_y(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_count_y")
    }
    fn compile_cube_count_z(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_count_z")
    }

    fn compile_cube_dim_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_dim")
    }
    fn compile_cube_dim(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_dim")
    }
    fn compile_cube_dim_x(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_dim_x")
    }
    fn compile_cube_dim_y(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_dim_y")
    }
    fn compile_cube_dim_z(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_dim_z")
    }

    fn compile_cube_pos_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_pos")
    }
    fn compile_cube_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_pos")
    }
    fn compile_cube_pos_x(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_pos_x")
    }
    fn compile_cube_pos_y(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_pos_y")
    }
    fn compile_cube_pos_z(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cube_pos_z")
    }

    fn compile_unit_pos_base_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unit_pos")
    }
    fn compile_unit_pos_computation(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let variable = Variable::<Self>::UnitPos;
        let ty = variable.item();
        writeln!(f, "{ty} {variable} = unit_idx;")
    }

    fn compile_unit_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unit_pos")
    }
    fn compile_unit_pos_x(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unit_pos_x")
    }
    fn compile_unit_pos_y(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unit_pos_y")
    }
    fn compile_unit_pos_z(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unit_pos_z")
    }

    fn compile_plane_dim(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "32")
    }
    fn compile_plane_dim_checked(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "32")
    }
    fn compile_plane_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }
    fn compile_unit_pos_plane(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "i % 32")
    }

    fn compile_cluster_pos(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_x(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_y(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_z(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }
}

// ── Instructions ──────────────────────────────────────────────────────────

impl<Wmma: DialectWmmaCompiler<Self>> DialectInstructions<Self> for TtMetalDialect<Wmma> {
    fn compile_atomic_load(
        f: &mut fmt::Formatter<'_>,
        input: &Variable<Self>,
        out: &Variable<Self>,
    ) -> fmt::Result {
        writeln!(f, "{} = *{};", out.fmt_left(), input)
    }

    fn compile_atomic_store(
        f: &mut fmt::Formatter<'_>,
        input: &Variable<Self>,
        out: &Variable<Self>,
    ) -> fmt::Result {
        writeln!(f, "*{} = {};", out, input)
    }

    fn compile_atomic_add(
        f: &mut fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> fmt::Result {
        writeln!(f, "{} = *{};", out.fmt_left(), lhs)?;
        let tmp = Variable::tmp(out.item());
        <Add as Binary<Self>>::format(f, out, rhs, &tmp)?;
        writeln!(f, "*{} = {};", lhs, tmp)
    }

    fn compile_atomic_max(
        f: &mut fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> fmt::Result {
        writeln!(f, "{} = *{};", out.fmt_left(), lhs)?;
        let tmp = Variable::tmp(out.item());
        <Max as Binary<Self>>::format(f, out, rhs, &tmp)?;
        writeln!(f, "*{} = {};", lhs, tmp)
    }

    fn compile_atomic_min(
        f: &mut fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> fmt::Result {
        writeln!(f, "{} = *{};", out.fmt_left(), lhs)?;
        let tmp = Variable::tmp(out.item());
        <Min as Binary<Self>>::format(f, out, rhs, &tmp)?;
        writeln!(f, "*{} = {};", lhs, tmp)
    }

    fn compile_saturating_add(
        f: &mut fmt::Formatter<'_>,
        lhs: impl Display,
        rhs: impl Display,
        item: Item<Self>,
    ) -> fmt::Result {
        let lhs = lhs.to_string();
        let rhs = rhs.to_string();
        match item.elem {
            shared::Elem::U8 => write!(
                f,
                "(({lhs}) > (std::numeric_limits<uint8_t>::max() - ({rhs})) ? std::numeric_limits<uint8_t>::max() : ({lhs}) + ({rhs}))"
            ),
            shared::Elem::U16 => write!(
                f,
                "(({lhs}) > (std::numeric_limits<uint16_t>::max() - ({rhs})) ? std::numeric_limits<uint16_t>::max() : ({lhs}) + ({rhs}))"
            ),
            shared::Elem::U32 => write!(
                f,
                "(({lhs}) > (std::numeric_limits<uint32_t>::max() - ({rhs})) ? std::numeric_limits<uint32_t>::max() : ({lhs}) + ({rhs}))"
            ),
            shared::Elem::U64 => write!(
                f,
                "(({lhs}) > (std::numeric_limits<uint64_t>::max() - ({rhs})) ? std::numeric_limits<uint64_t>::max() : ({lhs}) + ({rhs}))"
            ),
            shared::Elem::I8 => write!(
                f,
                "((({rhs}) > 0 && ({lhs}) > (std::numeric_limits<int8_t>::max() - ({rhs}))) ? std::numeric_limits<int8_t>::max() : ((({rhs}) < 0 && ({lhs}) < (std::numeric_limits<int8_t>::min() - ({rhs}))) ? std::numeric_limits<int8_t>::min() : ({lhs}) + ({rhs})))"
            ),
            shared::Elem::I16 => write!(
                f,
                "((({rhs}) > 0 && ({lhs}) > (std::numeric_limits<int16_t>::max() - ({rhs}))) ? std::numeric_limits<int16_t>::max() : ((({rhs}) < 0 && ({lhs}) < (std::numeric_limits<int16_t>::min() - ({rhs}))) ? std::numeric_limits<int16_t>::min() : ({lhs}) + ({rhs})))"
            ),
            shared::Elem::I32 => write!(
                f,
                "((({rhs}) > 0 && ({lhs}) > (std::numeric_limits<int32_t>::max() - ({rhs}))) ? std::numeric_limits<int32_t>::max() : ((({rhs}) < 0 && ({lhs}) < (std::numeric_limits<int32_t>::min() - ({rhs}))) ? std::numeric_limits<int32_t>::min() : ({lhs}) + ({rhs})))"
            ),
            shared::Elem::I64 => write!(
                f,
                "((({rhs}) > 0 && ({lhs}) > (std::numeric_limits<int64_t>::max() - ({rhs}))) ? std::numeric_limits<int64_t>::max() : ((({rhs}) < 0 && ({lhs}) < (std::numeric_limits<int64_t>::min() - ({rhs}))) ? std::numeric_limits<int64_t>::min() : ({lhs}) + ({rhs})))"
            ),
            _ => write!(f, "({lhs}) + ({rhs})"),
        }
    }

    fn compile_saturating_sub(
        f: &mut fmt::Formatter<'_>,
        lhs: impl Display,
        rhs: impl Display,
        item: Item<Self>,
    ) -> fmt::Result {
        let lhs = lhs.to_string();
        let rhs = rhs.to_string();
        match item.elem {
            shared::Elem::U8 => write!(f, "(({lhs}) < ({rhs}) ? uint8_t(0) : ({lhs}) - ({rhs}))"),
            shared::Elem::U16 => write!(f, "(({lhs}) < ({rhs}) ? uint16_t(0) : ({lhs}) - ({rhs}))"),
            shared::Elem::U32 => write!(f, "(({lhs}) < ({rhs}) ? uint32_t(0) : ({lhs}) - ({rhs}))"),
            shared::Elem::U64 => write!(f, "(({lhs}) < ({rhs}) ? uint64_t(0) : ({lhs}) - ({rhs}))"),
            shared::Elem::I8 => write!(
                f,
                "((({rhs}) > 0 && ({lhs}) < (std::numeric_limits<int8_t>::min() + ({rhs}))) ? std::numeric_limits<int8_t>::min() : ((({rhs}) < 0 && ({lhs}) > (std::numeric_limits<int8_t>::max() + ({rhs}))) ? std::numeric_limits<int8_t>::max() : ({lhs}) - ({rhs})))"
            ),
            shared::Elem::I16 => write!(
                f,
                "((({rhs}) > 0 && ({lhs}) < (std::numeric_limits<int16_t>::min() + ({rhs}))) ? std::numeric_limits<int16_t>::min() : ((({rhs}) < 0 && ({lhs}) > (std::numeric_limits<int16_t>::max() + ({rhs}))) ? std::numeric_limits<int16_t>::max() : ({lhs}) - ({rhs})))"
            ),
            shared::Elem::I32 => write!(
                f,
                "((({rhs}) > 0 && ({lhs}) < (std::numeric_limits<int32_t>::min() + ({rhs}))) ? std::numeric_limits<int32_t>::min() : ((({rhs}) < 0 && ({lhs}) > (std::numeric_limits<int32_t>::max() + ({rhs}))) ? std::numeric_limits<int32_t>::max() : ({lhs}) - ({rhs})))"
            ),
            shared::Elem::I64 => write!(
                f,
                "((({rhs}) > 0 && ({lhs}) < (std::numeric_limits<int64_t>::min() + ({rhs}))) ? std::numeric_limits<int64_t>::min() : ((({rhs}) < 0 && ({lhs}) > (std::numeric_limits<int64_t>::max() + ({rhs}))) ? std::numeric_limits<int64_t>::max() : ({lhs}) - ({rhs})))"
            ),
            _ => write!(f, "({lhs}) - ({rhs})"),
        }
    }

    fn compile_instruction_popcount_scalar<T: Component<Self>>(
        f: &mut fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> fmt::Result {
        write!(f, "{out_elem}(")?;
        match input.elem() {
            shared::Elem::I32 => write!(f, "__builtin_popcount(uint32_t({input}))"),
            shared::Elem::U32 => write!(f, "__builtin_popcount(uint32_t({input}))"),
            shared::Elem::I64 => write!(f, "__builtin_popcountll(uint64_t({input}))"),
            shared::Elem::U64 => write!(f, "__builtin_popcountll(uint64_t({input}))"),
            _ => write!(f, "__builtin_popcount(uint32_t({input}))"),
        }?;
        write!(f, ")")
    }

    fn compile_instruction_reverse_bits_scalar<T: Component<Self>>(
        f: &mut fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> fmt::Result {
        match out_elem {
            shared::Elem::I64 | shared::Elem::U64 => write!(
                f,
                "([&]() -> {out_elem} {{ uint64_t v = uint64_t({input}); v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1); v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2); v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4); v = ((v >> 8) & 0x00FF00FF00FF00FFull) | ((v & 0x00FF00FF00FF00FFull) << 8); v = ((v >> 16) & 0x0000FFFF0000FFFFull) | ((v & 0x0000FFFF0000FFFFull) << 16); v = (v >> 32) | (v << 32); return {out_elem}(v); }}())"
            ),
            _ => {
                let shift = (size_of::<u32>() - out_elem.size()) * 8;
                write!(
                    f,
                    "([&]() -> {out_elem} {{ uint32_t v = uint32_t({input}); v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1); v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2); v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4); v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8); v = (v >> 16) | (v << 16); return {out_elem}(v >> {shift}); }}())"
                )
            }
        }
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
        f: &mut fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> fmt::Result {
        match input.elem() {
            shared::Elem::I64 | shared::Elem::U64 => {
                write!(f, "{out_elem}(__builtin_ffsll(uint64_t({input})))")
            }
            _ => write!(f, "{out_elem}(__builtin_ffs(uint32_t({input})))"),
        }
    }

    fn compile_instruction_leading_zeros_scalar<T: Component<Self>>(
        f: &mut fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> fmt::Result {
        match input.elem() {
            shared::Elem::I64 | shared::Elem::U64 => write!(
                f,
                "(({input}) == 0 ? {out_elem}(64) : {out_elem}(__builtin_clzll(uint64_t({input}))))"
            ),
            shared::Elem::I32 | shared::Elem::U32 => write!(
                f,
                "(({input}) == 0 ? {out_elem}(32) : {out_elem}(__builtin_clz(uint32_t({input}))))"
            ),
            shared::Elem::I16 | shared::Elem::U16 => write!(
                f,
                "(({input}) == 0 ? {out_elem}(16) : {out_elem}(__builtin_clz(uint32_t({input})) - 16))"
            ),
            _ => write!(
                f,
                "(({input}) == 0 ? {out_elem}(8) : {out_elem}(__builtin_clz(uint32_t({input})) - 24))"
            ),
        }
    }

    fn compile_instruction_trailing_zeros_scalar<T: Component<Self>>(
        f: &mut fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> fmt::Result {
        match input.elem() {
            shared::Elem::I64 | shared::Elem::U64 => write!(
                f,
                "(({input}) == 0 ? {out_elem}(64) : {out_elem}(__builtin_ctzll(uint64_t({input}))))"
            ),
            shared::Elem::I32 | shared::Elem::U32 => write!(
                f,
                "(({input}) == 0 ? {out_elem}(32) : {out_elem}(__builtin_ctz(uint32_t({input}))))"
            ),
            shared::Elem::I16 | shared::Elem::U16 => write!(
                f,
                "(({input}) == 0 ? {out_elem}(16) : {out_elem}(__builtin_ctz(uint32_t({input}))))"
            ),
            _ => write!(
                f,
                "(({input}) == 0 ? {out_elem}(8) : {out_elem}(__builtin_ctz(uint32_t({input}))))"
            ),
        }
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

    fn compile_warp_shuffle(f: &mut fmt::Formatter<'_>, var: &str, _source: &str) -> fmt::Result {
        // TT-Metal has no SIMT warp model — return own value (identity shuffle)
        write!(f, "{var}")
    }

    fn compile_warp_shuffle_xor(
        f: &mut fmt::Formatter<'_>,
        var: &str,
        _elem: &Elem<Self>,
        _offset: &str,
    ) -> fmt::Result {
        write!(f, "{var}")
    }

    fn compile_warp_shuffle_up(
        f: &mut fmt::Formatter<'_>,
        var: &str,
        _offset: &str,
    ) -> fmt::Result {
        write!(f, "{var}")
    }

    fn compile_warp_shuffle_down(
        f: &mut fmt::Formatter<'_>,
        var: &str,
        _offset: &str,
    ) -> fmt::Result {
        write!(f, "{var}")
    }

    fn compile_warp_all<T: Component<Self>>(f: &mut fmt::Formatter<'_>, input: &T) -> fmt::Result {
        // TT-Metal runs single-thread per core — all/any returns own value
        write!(f, "{input}")
    }

    fn compile_warp_any<T: Component<Self>>(f: &mut fmt::Formatter<'_>, input: &T) -> fmt::Result {
        write!(f, "{input}")
    }

    fn compile_warp_ballot(
        f: &mut fmt::Formatter<'_>,
        _input: &Variable<Self>,
        _out_elem: &Elem<Self>,
    ) -> fmt::Result {
        // TT-Metal has no thread mask concept — return 0
        write!(f, "0")
    }

    fn compile_unreachable(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "__builtin_unreachable();")
    }
}

// ── WMMA ──────────────────────────────────────────────────────────────────

impl<Wmma: DialectWmmaCompiler<Self>> DialectWmmaCompiler<Self> for TtMetalDialect<Wmma> {
    fn compile_wmma_includes(f: &mut fmt::Formatter<'_>, flags: &Flags<Self>) -> fmt::Result {
        Wmma::compile_wmma_includes(f, flags)
    }
    fn compile_wmma_type_definitions(
        f: &mut fmt::Formatter<'_>,
        flags: &Flags<Self>,
    ) -> fmt::Result {
        Wmma::compile_wmma_type_definitions(f, flags)
    }
    fn compile_wmma_local_variables(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Wmma::compile_wmma_local_variables(f)
    }
    fn compile_wwma_fragment_ident(
        f: &mut fmt::Formatter<'_>,
        ident: &FragmentIdent<Self>,
    ) -> fmt::Result {
        Wmma::compile_wwma_fragment_ident(f, ident)
    }
    fn compile_wmma_fragment_layout(
        f: &mut fmt::Formatter<'_>,
        layout: &FragmentLayout<Self>,
    ) -> fmt::Result {
        Wmma::compile_wmma_fragment_layout(f, layout)
    }
    fn compile_wmma_fragment(f: &mut fmt::Formatter<'_>, fragment: &Fragment<Self>) -> fmt::Result {
        Wmma::compile_wmma_fragment(f, fragment)
    }
    fn compile_wmma_fragment_declaration(
        f: &mut fmt::Formatter<'_>,
        var: &Variable<Self>,
    ) -> fmt::Result {
        Wmma::compile_wmma_fragment_declaration(f, var)
    }
    fn compile_wmma_instruction(
        f: &mut fmt::Formatter<'_>,
        instruction: &WmmaInstruction<Self>,
    ) -> fmt::Result {
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
