/// Generate the C++ source for a TT-Metal writer (dataflow) kernel.
///
/// Uses `TensorAccessorArgs` with `noc_async_write_tile` for correct
/// interleaved DRAM tile addressing. Each output buffer needs 2 compile-time
/// args: [ArgConfig flags, AlignedPageSize].
pub fn generate_writer_source(num_outputs: u32) -> String {
    assert!(num_outputs >= 1, "at least one output buffer required");

    let mut src = String::new();
    src.push_str("#include <cstdint>\n\n");
    src.push_str("void kernel_main() {\n");

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

    // Declare TensorAccessor for each output buffer
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

    // Tile writing loop
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

/// Generate the C++ source for a simple copy compute kernel.
pub fn generate_copy_compute_source() -> String {
    r#"#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

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
