use std::collections::HashSet;
use std::fmt::{self, Display, Write};

use cubecl_core::prelude::Visibility;

use crate::shared::{self, ComputeKernel, Item};

use super::compile::TtKernelAnalysis;
use super::dialect::TtMetalDialect;
use super::kernel::{TtBinaryComputeOp, TtUnaryComputeOp};

/// Generate the C++ source for a TT-Metal writer (dataflow) kernel.
///
/// Uses `TensorAccessorArgs` with `noc_async_write_tile` for correct
/// interleaved DRAM tile addressing. Each output buffer needs 2 compile-time
/// args: [`ArgConfig`] flags, [`AlignedPageSize`].
pub fn generate_writer_source(num_outputs: u32) -> String {
    let mut src = String::new();
    src.push_str("#include <cstdint>\n\n");
    src.push_str("void kernel_main() {\n");

    if num_outputs == 0 {
        src.push_str("    (void)get_arg_val<uint32_t>(0);\n");
        src.push_str("}\n");
        return src;
    }

    // Runtime args: dst_addr_0, ..., dst_addr_N, num_tiles
    for i in 0..num_outputs {
        src.push_str(&format!(
            "    uint32_t dst{}_addr = get_arg_val<uint32_t>({});\n",
            i, i
        ));
    }
    let num_tiles_idx = num_outputs;
    src.push_str(&format!(
        "    uint32_t num_tiles = get_arg_val<uint32_t>({});\n\n",
        num_tiles_idx
    ));

    // Output circular buffers start at index 16
    for i in 0..num_outputs {
        let cb_idx = 16 + i;
        src.push_str(&format!(
            "    constexpr uint32_t cb_out{} = {};\n",
            i, cb_idx
        ));
    }

    // Declare TensorAccessor for each output buffer.
    src.push('\n');
    for i in 0..num_outputs {
        if i == 0 {
            src.push_str(&format!(
                "    constexpr auto c{}_args = TensorAccessorArgs<0>();\n",
                i
            ));
        } else {
            src.push_str(&format!(
                "    constexpr auto c{}_args = TensorAccessorArgs<c{}_args.next_compile_time_args_offset()>();\n",
                i, i - 1
            ));
        }
        src.push_str(&format!(
            "    const auto c{} = TensorAccessor(c{}_args, dst{}_addr);\n",
            i, i, i
        ));
    }

    src.push_str("\n    for (uint32_t i = 0; i < num_tiles; i++) {\n");
    for idx in 0..num_outputs {
        src.push_str(&format!("        cb_wait_front(cb_out{idx}, 1);\n"));
        src.push_str(&format!(
            "        uint32_t l1_addr_out{idx} = get_read_ptr(cb_out{idx});\n"
        ));
        src.push_str(&format!(
            "        noc_async_write_tile(i, c{idx}, l1_addr_out{idx});\n"
        ));
    }
    src.push_str("        noc_async_write_barrier();\n");
    for idx in 0..num_outputs {
        src.push_str(&format!("        cb_pop_front(cb_out{idx}, 1);\n"));
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    src
}

/// Generate a single-kernel dram_loopback-style copy source.
///
/// Uses a circular buffer as scratch space for the DRAM→DRAM copy.
/// Reads tiles from input DRAM into the CB, then writes from the CB
/// to output DRAM. No compute kernel, no separate reader/writer.
/// This matches the TT-Metal `dram_loopback` example pattern adapted to use CBs.
pub fn generate_dram_loopback_source() -> String {
    r#"#include <cstdint>

void kernel_main() {
    uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scratch = tt::CBIndex::c_0;

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_scratch, 1);
        uint32_t l1_addr = get_write_ptr(cb_scratch);
        noc_async_read_tile(i, in0, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_scratch, 1);

        cb_wait_front(cb_scratch, 1);
        l1_addr = get_read_ptr(cb_scratch);
        noc_async_write_tile(i, out0, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_scratch, 1);
    }
}
"#
    .to_string()
}

/// Generate the C++ source for a simple copy compute kernel.
pub fn generate_copy_compute_source() -> String {
    r#"#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    binary_op_init_common(cb_in0, cb_in0, cb_out0);
    copy_tile_init(cb_in0);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(cb_in0, 1);
        copy_tile(cb_in0, 0, dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_pop_front(cb_in0, 1);

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_push_back(cb_out0, 1);

        tile_regs_release();
    }
}
"#
    .to_string()
}

fn binary_compute_init_and_op(op: TtBinaryComputeOp) -> (&'static str, &'static str) {
    match op {
        TtBinaryComputeOp::Add => ("add_tiles_init", "add_tiles"),
        TtBinaryComputeOp::Sub => ("sub_tiles_init", "sub_tiles"),
        TtBinaryComputeOp::Mul => ("mul_tiles_init", "mul_tiles"),
    }
}

pub fn generate_binary_compute_source(op: TtBinaryComputeOp) -> String {
    let (init_fn, op_fn) = binary_compute_init_and_op(op);
    format!(
        "#include \"api/compute/common.h\"
#include \"api/compute/eltwise_binary.h\"
#include \"api/compute/compute_kernel_api.h\"

void kernel_main() {{
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    {init_fn}(cb_in0, cb_in1);

    for (uint32_t i = 0; i < num_tiles; i++) {{
        tile_regs_acquire();
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        {op_fn}(cb_in0, cb_in1, 0, 0, dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_push_back(cb_out0, 1);

        tile_regs_release();
    }}
}}
"
    )
}

/// Generate the C++ source for an element-wise addition compute kernel.
pub fn generate_add_compute_source() -> String {
    generate_binary_compute_source(TtBinaryComputeOp::Add)
}

fn unary_compute_spec(op: TtUnaryComputeOp) -> (&'static str, &'static str, &'static str) {
    match op {
        TtUnaryComputeOp::Abs => ("abs", "abs_tile_init", "abs_tile"),
        TtUnaryComputeOp::Sqrt => ("sqrt", "sqrt_tile_init", "sqrt_tile"),
        TtUnaryComputeOp::Rsqrt => ("rsqrt", "rsqrt_tile_init", "rsqrt_tile"),
    }
}

pub fn generate_unary_compute_source(op: TtUnaryComputeOp) -> String {
    let (header, init_fn, op_fn) = unary_compute_spec(op);
    format!(
        "#include \"api/compute/common.h\"
#include \"api/compute/tile_move_copy.h\"
#include \"api/compute/eltwise_unary/eltwise_unary.h\"
#include \"api/compute/eltwise_unary/{header}.h\"
#include \"api/compute/compute_kernel_api.h\"

void kernel_main() {{
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    init_sfpu(cb_in0, cb_out0);
    copy_tile_init(cb_in0);
    {init_fn}();

    for (uint32_t i = 0; i < num_tiles; i++) {{
        cb_wait_front(cb_in0, 1);
        tile_regs_acquire();
        copy_tile(cb_in0, 0, dst_reg);
        {op_fn}(dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_pop_front(cb_in0, 1);
        tile_regs_release();
        cb_push_back(cb_out0, 1);
    }}
}}
"
    )
}

pub fn generate_noop_compute_source() -> String {
    "void kernel_main() {}
".to_string()
}

pub(crate) fn generate_scalar_writer_source(
    repr: &ComputeKernel<TtMetalDialect>,
    analysis: &TtKernelAnalysis,
) -> String {
    let tile_units = (analysis.tile_size_bytes as usize / analysis.unit_item.size()).max(1) as u32;
    let mut src = String::new();
    let num_tiles_idx = analysis.num_outputs;
    let static_arg_offset_words = num_tiles_idx + 1;
    let dynamic_meta_offset_words = (repr.info.dynamic_meta_offset / core::mem::size_of::<u32>()) as u32;

    let _ = writeln!(src, "#include <cstdint>\n#include <algorithm>\n");
    let _ = writeln!(src, "using std::max;\nusing std::min;\n");
    let _ = writeln!(
        src,
        "{}",
        ScalarWriterTypeDefinitions {
            items: &repr.items,
            scalars: &repr.scalars,
            info: &repr.info,
            address_type: repr.flags.address_type,
        }
    );
    let _ = writeln!(src, "void kernel_main() {{");

    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(
            src,
            "    uint32_t dst{output_idx}_addr = get_arg_val<uint32_t>({output_idx});"
        );
    }
    let _ = writeln!(
        src,
        "    uint32_t num_tiles = get_arg_val<uint32_t>({num_tiles_idx});"
    );
    let _ = writeln!(src, "    constexpr uint32_t tile_units = {tile_units};");
    let _ = writeln!(
        src,
        "    constexpr uint32_t cube_dim_x = {};",
        repr.cube_dim.x
    );
    let _ = writeln!(
        src,
        "    constexpr uint32_t cube_dim_y = {};",
        repr.cube_dim.y
    );
    let _ = writeln!(
        src,
        "    constexpr uint32_t cube_dim_z = {};",
        repr.cube_dim.z
    );
    let _ = writeln!(src, "    constexpr uint32_t cube_count_x = 1;");
    let _ = writeln!(src, "    constexpr uint32_t cube_count_y = 1;");
    let _ = writeln!(src, "    constexpr uint32_t cube_count_z = 1;");
    let _ = writeln!(src, "    constexpr uint32_t cube_pos_x = 0;");
    let _ = writeln!(src, "    constexpr uint32_t cube_pos_y = 0;");
    let _ = writeln!(src, "    constexpr uint32_t cube_pos_z = 0;");
    let _ = writeln!(src, "    constexpr uint32_t num_units = cube_dim_x;");

    for input_idx in 0..analysis.num_inputs {
        let _ = writeln!(
            src,
            "    constexpr uint32_t cb_in{input_idx} = {input_idx};"
        );
    }
    for output_idx in 0..analysis.num_outputs {
        let cb = 16 + output_idx;
        let _ = writeln!(src, "    constexpr uint32_t cb_out{output_idx} = {cb};");
    }

    src.push('\n');
    for output_idx in 0..analysis.num_outputs {
        if output_idx == 0 {
            let _ = writeln!(
                src,
                "    constexpr auto c{output_idx}_args = TensorAccessorArgs<0>();"
            );
        } else {
            let prev = output_idx - 1;
            let _ = writeln!(
                src,
                "    constexpr auto c{output_idx}_args = TensorAccessorArgs<c{prev}_args.next_compile_time_args_offset()>();"
            );
        }
        let _ = writeln!(
            src,
            "    const auto c{output_idx} = TensorAccessor(c{output_idx}_args, dst{output_idx}_addr);"
        );
    }

    if repr.body.info_static_len > 0 || !repr.info.scalars.is_empty() {
        let _ = writeln!(src, "    info_st info = {{}};");
        let _ = writeln!(
            src,
            "    constexpr uint32_t info_arg_offset_words = {static_arg_offset_words};"
        );
        let _ = writeln!(
            src,
            "    const uint8_t* runtime_info_bytes = reinterpret_cast<const uint8_t*>(get_arg_addr(info_arg_offset_words));",
        );

        for (field, (ty, _)) in repr.info.scalars.iter().zip(repr.scalars.iter()) {
            let _ = writeln!(
                src,
                "    __builtin_memcpy(info.scalars_{ty}, runtime_info_bytes + {}, sizeof(info.scalars_{ty}));",
                field.offset,
            );
        }

        if let Some(static_meta) = repr.info.sized_meta {
            let _ = writeln!(
                src,
                "    __builtin_memcpy(info.static_meta, runtime_info_bytes + {}, sizeof(info.static_meta));",
                static_meta.offset,
            );
        }
    }

    if repr.body.has_dynamic_meta {
        let _ = writeln!(
            src,
            "    constexpr uint32_t dynamic_meta_arg_offset_words = {};",
            static_arg_offset_words + dynamic_meta_offset_words,
        );
        let _ = writeln!(
            src,
            "    const tt_l1_ptr {}* dynamic_meta = reinterpret_cast<tt_l1_ptr const {}*>(get_arg_addr(dynamic_meta_arg_offset_words));",
            repr.body.address_type,
            repr.body.address_type,
        );
    }

    for const_array in &repr.body.const_arrays {
        let _ = write!(
            src,
            "    const {} arrays_{}[{}] = {{",
            const_array.item,
            const_array.index,
            const_array.size,
        );
        let item = const_array.item;
        for value in const_array.values.iter().copied() {
            let value = match value {
                shared::Variable::Constant(value, _) => shared::Variable::Constant(value, item),
                _ => unreachable!("constant arrays only contain constant values"),
            };
            let _ = write!(src, "{value},");
        }
        let _ = writeln!(src, "}};");
    }

    let _ = writeln!(
        src,
        "\n    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {{"
    );
    for input_idx in 0..analysis.num_inputs {
        let _ = writeln!(src, "        cb_wait_front(cb_in{input_idx}, 1);");
    }
    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(src, "        cb_reserve_back(cb_out{output_idx}, 1);");
    }
    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(
            src,
            "        uint32_t l1_addr_out{output_idx}_prefill = get_write_ptr(cb_out{output_idx});"
        );
        let _ = writeln!(
            src,
            "        noc_async_read_tile(tile_idx, c{output_idx}, l1_addr_out{output_idx}_prefill);"
        );
    }
    if analysis.num_outputs > 0 {
        let _ = writeln!(src, "        noc_async_read_barrier();");
    }

    let mut read_idx = 0u32;
    let mut write_idx = 0u32;
    for binding in &repr.buffers {
        match binding.vis {
            Visibility::Read => {
                let _ = writeln!(
                    src,
                    "        const {}* buffer_{} = reinterpret_cast<const {}*>(get_read_ptr(cb_in{}));",
                    binding.item, binding.id, binding.item, read_idx,
                );
                read_idx += 1;
            }
            Visibility::ReadWrite => {
                let _ = writeln!(
                    src,
                    "        {}* buffer_{} = reinterpret_cast<{}*>(get_write_ptr(cb_out{}));",
                    binding.item, binding.id, binding.item, write_idx,
                );
                write_idx += 1;
            }
        }
    }

    let _ = writeln!(
        src,
        "\n        for (uint32_t i = 0; i < tile_units; ++i) {{"
    );
    let _ = writeln!(
        src,
        "            uint32_t unit_idx = tile_idx * tile_units + i;"
    );
    let _ = writeln!(src, "            if (unit_idx >= num_units) break;");
    emit_builtin_locals(&mut src, &repr.flags.indexes);
    for instruction in &repr.body.instructions {
        let _ = write!(src, "            {instruction}");
    }
    let _ = writeln!(src, "        }}");

    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(src, "        cb_push_back(cb_out{output_idx}, 1);");
        let _ = writeln!(src, "        cb_wait_front(cb_out{output_idx}, 1);");
        let _ = writeln!(
            src,
            "        uint32_t l1_addr_out{output_idx} = get_read_ptr(cb_out{output_idx});"
        );
        let _ = writeln!(
            src,
            "        uint32_t page_bytes_out{output_idx} = get_tile_size(cb_out{output_idx});"
        );
        let _ = writeln!(
            src,
            "        uint64_t dst_noc_addr_out{output_idx} = c{output_idx}.get_noc_addr(tile_idx);"
        );
        let _ = writeln!(
            src,
            "        noc_async_write(l1_addr_out{output_idx}, dst_noc_addr_out{output_idx}, page_bytes_out{output_idx});"
        );
    }
    let _ = writeln!(src, "        noc_async_write_barrier();");
    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(src, "        cb_pop_front(cb_out{output_idx}, 1);");
    }
    for input_idx in 0..analysis.num_inputs {
        let _ = writeln!(src, "        cb_pop_front(cb_in{input_idx}, 1);");
    }
    let _ = writeln!(src, "    }}");
    let _ = writeln!(src, "}}");

    src
}

fn emit_builtin_locals(src: &mut String, indexes: &crate::shared::CubeIndexFlags) {
    if indexes.unit_pos {
        let _ = writeln!(src, "            uint32_t unit_pos = unit_idx;");
    }
    if indexes.absolute_pos {
        let _ = writeln!(src, "            uint32_t absolute_pos = unit_idx;");
    }
    if indexes.cube_dim {
        let _ = writeln!(
            src,
            "            uint32_t cube_dim = cube_dim_x * cube_dim_y * cube_dim_z;"
        );
    }
    if indexes.cube_count {
        let _ = writeln!(
            src,
            "            uint32_t cube_count = cube_count_x * cube_count_y * cube_count_z;"
        );
    }
    if indexes.cube_pos {
        let _ = writeln!(src, "            uint32_t cube_pos = 0;");
    }
    if indexes.plane_dim {
        let _ = writeln!(src, "            uint32_t plane_dim = 32;");
    }
    if indexes.plane_dim_checked {
        let _ = writeln!(
            src,
            "            uint32_t plane_dim_checked = num_units < 32 ? num_units : 32;"
        );
    }
    if indexes.plane_pos {
        let _ = writeln!(src, "            uint32_t plane_pos = 0;");
    }
    if indexes.unit_pos_plane {
        let _ = writeln!(src, "            uint32_t unit_pos_plane = unit_idx % 32;");
    }
}

struct ScalarWriterTypeDefinitions<'a> {
    items: &'a HashSet<Item<TtMetalDialect>>,
    scalars: &'a [(crate::shared::Elem<TtMetalDialect>, usize)],
    info: &'a cubecl_core::Info,
    address_type: Item<TtMetalDialect>,
}

impl Display for ScalarWriterTypeDefinitions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        scalar_writer_vectorized_definitions(f, self.items)?;
        shared::type_info_definition_sized::<TtMetalDialect>(
            f,
            self.info,
            self.scalars,
            self.address_type,
        )
    }
}

fn scalar_writer_vectorized_definitions(
    f: &mut fmt::Formatter<'_>,
    items: &HashSet<Item<TtMetalDialect>>,
) -> fmt::Result {
    for item in items.iter() {
        let elem = item.elem;
        let size = item.vectorization;
        let alignment = elem.size() * size;
        if size > 1 {
            write!(f, "
struct alignas({alignment}) {item} {{")?;
            for i in 0..size {
                write!(f, "
    {elem} i_{i};")?;
            }
            f.write_str("
};")?;
        }
    }
    Ok(())
}
